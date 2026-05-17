# -*- coding: utf-8 -*-
"""
Fair MS-GWCN-Cheb baseline for Houston 2013 HSI-LiDAR classification.

This script rewrites the original Indian Pines MS-GWCN code for Houston 2013 and follows
the fair-comparison premise used in houston_slimamba_msgw_v5_urbanfocus_5runs_maps_direct_rgb_palette.py.

Fair protocol:
    1. Use Houston 2013 files: HSI.mat, LiDAR.mat, gt.mat, TRLabel.mat, TSLabel.mat.
    2. Fit HSI standardization, LiDAR standardization, and PCA only on the current training mask.
    3. Use official TRLabel.mat only for training and spatial validation.
    4. Use official TSLabel.mat only once for final test evaluation.
    5. Do not use TSLabel for preprocessing, model selection, hyperparameter selection, or early stopping.
    6. Retrain on all official training labels after spatial-validation epoch selection.
    7. Report 5-run OA, AA, Kappa, and per-class accuracy as mean ± std.
    8. Draw land-cover maps with the exact Houston color list through direct RGB lookup.

This is a fair baseline, not a performance-maximized method.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch_geometric.nn import ChebConv
    from torch_geometric.utils import to_undirected, add_self_loops
except Exception as exc:
    raise ImportError(
        "This script requires torch-geometric. Install a PyTorch-compatible PyG build first."
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ==============================================================================
# 1. Houston categories and exact color palette from houston_slimamba_msgw_v5
# ==============================================================================

categories = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Tree", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway",
    "Railway", "Parking lot 1", "Parking lot 2", "Tennis court", "Running track"
]
colors = [
    "#006400", "#008000", "#00FF00", "#008080", "#8B4513",
    "#0000FF", "#FFFF00", "#FFD700", "#808080", "#A9A9A9",
    "#696969", "#FFA500", "#FF8C00", "#FF0000", "#FF1493"
]
cmap = ListedColormap(["#000000"] + colors)

CLASS_NAMES = categories
CLASS_COLORS = colors


# ==============================================================================
# 2. Utilities
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_first_numeric_array(mat_dict: Dict, preferred_key: Optional[str] = None) -> np.ndarray:
    if preferred_key is not None and preferred_key in mat_dict:
        return np.asarray(mat_dict[preferred_key])
    candidates = []
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            candidates.append((k, v))
    if not candidates:
        raise ValueError("No numeric ndarray found in .mat file.")
    candidates.sort(key=lambda kv: (kv[1].ndim >= 2, kv[1].size), reverse=True)
    return np.asarray(candidates[0][1])


def load_mat_array(path: str, preferred_key: Optional[str] = None) -> np.ndarray:
    return np.asarray(find_first_numeric_array(sio.loadmat(path), preferred_key=preferred_key))


def ensure_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {x.shape}")
    if x.shape[0] < 20 and x.shape[-1] > 20:
        x = np.transpose(x, (1, 2, 0))
    return x


def load_houston_2013(data_root: str) -> Dict[str, np.ndarray]:
    paths = {
        "hsi": os.path.join(data_root, "HSI.mat"),
        "lidar": os.path.join(data_root, "LiDAR.mat"),
        "gt": os.path.join(data_root, "gt.mat"),
        "tr": os.path.join(data_root, "TRLabel.mat"),
        "ts": os.path.join(data_root, "TSLabel.mat"),
    }
    missing = [v for v in paths.values() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f"Missing Houston 2013 files: {missing}")

    hsi = ensure_hwc(load_mat_array(paths["hsi"], preferred_key="HSI")).astype(np.float32)
    lidar = ensure_hwc(load_mat_array(paths["lidar"], preferred_key="LiDAR")).astype(np.float32)
    gt = load_mat_array(paths["gt"], preferred_key="gt").squeeze().astype(np.int64)
    tr = load_mat_array(paths["tr"], preferred_key="TRLabel").squeeze().astype(np.int64)
    ts = load_mat_array(paths["ts"], preferred_key="TSLabel").squeeze().astype(np.int64)

    if hsi.shape[:2] != lidar.shape[:2] or hsi.shape[:2] != gt.shape:
        raise ValueError(f"Spatial mismatch: HSI={hsi.shape}, LiDAR={lidar.shape}, gt={gt.shape}")

    return {"hsi": hsi, "lidar": lidar, "gt": gt, "tr": tr, "ts": ts}


def mask_to_coords_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    labels = mask[mask > 0].astype(np.int64) - 1
    return coords.astype(np.int64), labels.astype(np.int64)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# 3. Leakage-controlled preprocessing
# ==============================================================================

def fit_preprocess_train_only(
    hsi: np.ndarray,
    lidar: np.ndarray,
    train_mask: np.ndarray,
    pca_dim: int,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit all data-dependent transforms on train_mask only.

    This is the key anti-leakage step:
    - no all-pixel standardization,
    - no test-pixel PCA fitting,
    - no use of TSLabel for preprocessing.
    """
    train_bool = train_mask > 0
    if train_bool.sum() == 0:
        raise ValueError("Empty train_mask in preprocessing.")

    h, w, b = hsi.shape
    hsi_train = hsi[train_bool].reshape(-1, b)
    mean_h = hsi_train.mean(axis=0, keepdims=True)
    std_h = hsi_train.std(axis=0, keepdims=True) + 1e-6

    hsi_flat_norm = ((hsi.reshape(-1, b) - mean_h) / std_h).astype(np.float32)
    hsi_train_norm = ((hsi_train - mean_h) / std_h).astype(np.float32)

    n_comp = min(int(pca_dim), b, hsi_train_norm.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="full", whiten=False)
    pca.fit(hsi_train_norm)

    hsi_pca = pca.transform(hsi_flat_norm).reshape(h, w, n_comp).astype(np.float32)

    lidar = ensure_hwc(lidar).astype(np.float32)
    lc = lidar.shape[-1]
    lidar_train = lidar[train_bool].reshape(-1, lc)
    mean_l = lidar_train.mean(axis=0, keepdims=True)
    std_l = lidar_train.std(axis=0, keepdims=True) + 1e-6
    lidar_norm = ((lidar.reshape(-1, lc) - mean_l) / std_l).reshape(h, w, lc).astype(np.float32)

    if lidar_norm.shape[-1] != 1:
        lidar_norm = lidar_norm[..., :1]

    features = np.concatenate([hsi_pca, lidar_norm], axis=-1).astype(np.float32)
    return features, lidar_norm[..., 0].astype(np.float32), pca


# ==============================================================================
# 4. Spatial validation split
# ==============================================================================

def classwise_spatial_train_val_split(
    tr_label: np.ndarray,
    val_ratio: float,
    block_size: int,
    seed: int,
    min_train_per_class: int = 16,
    min_val_per_class: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    coords, labels = mask_to_coords_labels(tr_label)
    if val_ratio <= 0.0:
        return tr_label.copy(), np.zeros_like(tr_label, dtype=np.int64)

    num_classes = int(labels.max()) + 1
    rng = np.random.RandomState(seed)
    train_mask = np.zeros_like(tr_label, dtype=np.int64)
    val_mask = np.zeros_like(tr_label, dtype=np.int64)

    for cls in range(num_classes):
        cls_all = np.where(labels == cls)[0]
        n_cls = len(cls_all)
        if n_cls == 0:
            continue

        target_val = int(round(n_cls * val_ratio))
        target_val = max(target_val, min_val_per_class)
        target_val = min(target_val, max(0, n_cls - min_train_per_class))

        if target_val <= 0:
            chosen_val = np.array([], dtype=np.int64)
            chosen_train = cls_all
        else:
            block_map: Dict[Tuple[int, int], List[int]] = {}
            for gidx in cls_all:
                y, x = coords[gidx]
                key = (int(y // block_size), int(x // block_size))
                block_map.setdefault(key, []).append(gidx)

            block_keys = list(block_map.keys())
            rng.shuffle(block_keys)
            block_keys.sort(key=lambda k: len(block_map[k]))

            chosen = []
            cur = 0
            for key in block_keys:
                idxs = block_map[key]
                if cur >= target_val:
                    break
                remaining = n_cls - (cur + len(idxs))
                if remaining < min_train_per_class:
                    continue
                chosen.extend(idxs)
                cur += len(idxs)

            if cur < min(min_val_per_class, target_val):
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                if fallback_target > 0:
                    chosen = rng.permutation(cls_all)[:fallback_target].tolist()

            chosen_val = np.array(sorted(set(chosen)), dtype=np.int64)
            val_set = set(chosen_val.tolist())
            chosen_train = np.array([idx for idx in cls_all if idx not in val_set], dtype=np.int64)

            if len(chosen_train) < min_train_per_class or len(chosen_val) < min_val_per_class:
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                perm = rng.permutation(cls_all)
                chosen_val = perm[:fallback_target]
                chosen_train = perm[fallback_target:]

        train_mask[coords[chosen_train, 0], coords[chosen_train, 1]] = cls + 1
        if len(chosen_val) > 0:
            val_mask[coords[chosen_val, 0], coords[chosen_val, 1]] = cls + 1

    if (val_mask > 0).sum() == 0:
        raise RuntimeError("Spatial validation split failed.")
    return train_mask, val_mask


# ==============================================================================
# 5. Graph construction for a given mask
# ==============================================================================

@dataclass
class GraphPack:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    coords: np.ndarray


def build_knn_edges(coords: np.ndarray, k: int) -> torch.Tensor:
    n = coords.shape[0]
    if n <= 1:
        edge_index = torch.arange(n, dtype=torch.long).view(1, -1).repeat(2, 1)
        return edge_index

    kk = min(int(k) + 1, n)
    nn_model = NearestNeighbors(n_neighbors=kk, algorithm="auto")
    nn_model.fit(coords.astype(np.float32))
    _dist, ind = nn_model.kneighbors(coords.astype(np.float32), return_distance=True)

    src_list = []
    dst_list = []
    for i in range(n):
        for j in ind[i]:
            if int(j) == i:
                continue
            src_list.append(i)
            dst_list.append(int(j))

    if len(src_list) == 0:
        edge_index = torch.arange(n, dtype=torch.long).view(1, -1).repeat(2, 1)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=n)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)

    return edge_index.contiguous()


def make_graph_pack(features: np.ndarray, label_mask: np.ndarray, k: int) -> GraphPack:
    coords, labels = mask_to_coords_labels(label_mask)
    if len(labels) == 0:
        raise ValueError("Empty label mask for graph pack.")

    x_np = features[coords[:, 0], coords[:, 1], :].astype(np.float32)
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(labels.astype(np.int64)).long()
    edge_index = build_knn_edges(coords, k=k)

    return GraphPack(x=x, y=y, edge_index=edge_index, coords=coords)


# ==============================================================================
# 6. Model
# ==============================================================================

class GraphChebMultiScale(nn.Module):
    """
    Multi-scale Chebyshev block.
    Different Chebyshev orders act as different graph receptive-field scales.
    """
    def __init__(self, in_channels: int, out_channels: int, cheb_k_list=(2, 4, 6), dropout: float = 0.35):
        super().__init__()
        self.k_list = tuple(int(k) for k in cheb_k_list)
        self.convs = nn.ModuleList([
            ChebConv(in_channels, out_channels, K=k, normalization="sym")
            for k in self.k_list
        ])
        self.norm = nn.LayerNorm(out_channels * len(self.k_list))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        outs = []
        for conv in self.convs:
            outs.append(conv(x, edge_index))
        x = torch.cat(outs, dim=-1)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class MS_GWCN_Cheb_Houston(nn.Module):
    """
    Fair Houston baseline adapted from the Indian Pines MS-GWCN-Cheb code.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        cheb_k_list=(2, 4, 6),
        hidden_dims=(64, 96, 128),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.k_list = tuple(int(k) for k in cheb_k_list)
        scales = len(self.k_list)

        dims = [int(d) for d in hidden_dims]
        self.blocks = nn.ModuleList()
        last_dim = in_channels
        for d in dims:
            self.blocks.append(GraphChebMultiScale(last_dim, d, cheb_k_list=self.k_list, dropout=dropout))
            last_dim = d * scales

        self.head = nn.Sequential(
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, edge_index)
        return self.head(x)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            src = v.detach().cpu()
            if k not in self.shadow:
                self.shadow[k] = src.clone()
            elif torch.is_floating_point(self.shadow[k]):
                self.shadow[k].mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(src)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


# ==============================================================================
# 7. Metrics and evaluation
# ==============================================================================

@dataclass
class EvalResult:
    oa: float
    aa: float
    kappa: float
    per_class: np.ndarray
    conf_mat: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> EvalResult:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    oa = accuracy_score(y_true, y_pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    aa = per_class.mean()
    kappa = cohen_kappa_score(y_true, y_pred, labels=np.arange(num_classes))
    return EvalResult(float(oa), float(aa), float(kappa), per_class, cm, y_true, y_pred)


@torch.no_grad()
def predict_graph(model: nn.Module, pack: GraphPack, device: torch.device) -> np.ndarray:
    model.eval()
    x = pack.x.to(device)
    edge_index = pack.edge_index.to(device)
    logits = model(x, edge_index)
    return logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def evaluate_graph(model: nn.Module, pack: GraphPack, device: torch.device, num_classes: int) -> EvalResult:
    pred = predict_graph(model, pack, device)
    true = pack.y.cpu().numpy().astype(np.int64)
    return compute_metrics(true, pred, num_classes)


# ==============================================================================
# 8. Visualization with direct RGB lookup
# ==============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PALETTE_RGB = np.array([[0, 0, 0]] + [list(hex_to_rgb(c)) for c in colors], dtype=np.uint8)


def label_map_to_rgb(label_img: np.ndarray) -> np.ndarray:
    label_img = np.asarray(label_img)
    out = np.zeros(label_img.shape + (3,), dtype=np.uint8)
    valid = (label_img >= 0) & (label_img < len(PALETTE_RGB))
    out[valid] = PALETTE_RGB[label_img[valid].astype(np.int64)]
    return out


def save_rgb_map(
    label_img: np.ndarray,
    title: str,
    path: str,
    with_legend: bool = True,
    figsize: Tuple[float, float] = (18.0, 4.8),
    dpi: int = 300,
) -> None:
    rgb = label_map_to_rgb(label_img)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")

    if with_legend:
        handles = [
            Patch(facecolor=colors[i], edgecolor="k", label=categories[i])
            for i in range(len(categories) - 1, -1, -1)
        ]
        labels = [categories[i] for i in range(len(categories) - 1, -1, -1)]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)

    plt.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=dpi)
    plt.close(fig)


def save_paper_map_pair(gt_img: np.ndarray, pred_img: np.ndarray, path: str, run_id: int, method_name: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18.0, 7.2))
    axes[0].imshow(label_map_to_rgb(gt_img), interpolation="nearest")
    axes[0].set_title("Ground Truth", fontsize=18)
    axes[0].axis("off")

    axes[1].imshow(label_map_to_rgb(pred_img), interpolation="nearest")
    axes[1].set_title(f"Classification Result (Run {run_id}) by {method_name}", fontsize=18)
    axes[1].axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, hspace=0.18)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_per_class_bar(per_class: np.ndarray, out_png: str, title: str, std: Optional[np.ndarray] = None) -> None:
    x = np.arange(1, len(per_class) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, per_class, color=list(colors)[: len(per_class)], edgecolor="black", alpha=0.85)
    if std is not None:
        ax.errorbar(x, per_class, yerr=std, fmt="none", capsize=4, ecolor="black", elinewidth=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories[: len(per_class)], rotation=45, ha="right")
    for xi, yi in zip(x, per_class):
        ax.text(xi, min(1.02, yi + 0.015), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fill_label_image(shape_hw: Tuple[int, int], coords: np.ndarray, pred_zero_based: np.ndarray) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    out[coords[:, 0], coords[:, 1]] = pred_zero_based.astype(np.uint8) + 1
    return out


# ==============================================================================
# 9. Training helpers
# ==============================================================================

def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def train_graph_model(
    args,
    train_pack: GraphPack,
    val_pack: Optional[GraphPack],
    in_channels: int,
    num_classes: int,
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[MS_GWCN_Cheb_Houston, int, Optional[EvalResult]]:
    set_seed(seed)

    model = MS_GWCN_Cheb_Houston(
        in_channels=in_channels,
        num_classes=num_classes,
        cheb_k_list=parse_int_list(args.cheb_k_list),
        hidden_dims=parse_int_list(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=args.lr * 0.05)

    class_weights = compute_class_weights(train_pack.y.numpy(), num_classes).to(device)
    x_train = train_pack.x.to(device)
    y_train = train_pack.y.to(device)
    ei_train = train_pack.edge_index.to(device)

    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None

    best_score = -1.0
    best_epoch = epochs
    best_state = None
    best_val = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train, ei_train)
        loss = F.cross_entropy(
            logits,
            y_train,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        if val_pack is not None and (epoch == 1 or epoch % args.eval_interval == 0 or epoch == epochs):
            eval_model = model
            if ema is not None:
                eval_state = ema.state_dict()
                eval_model.load_state_dict(eval_state, strict=True)

            val_res = evaluate_graph(eval_model, val_pack, device, num_classes)
            score = val_res.oa + 0.35 * val_res.aa + 0.55 * val_res.kappa

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val = val_res
                if ema is not None:
                    best_state = ema.state_dict()
                else:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"epoch {epoch:03d}/{epochs} | loss={float(loss.item()):.4f} | "
                f"val_OA={val_res.oa:.4f} | val_AA={val_res.aa:.4f} | "
                f"val_Kappa={val_res.kappa:.4f} | best_epoch={best_epoch}"
            )
        elif epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}/{epochs} | loss={float(loss.item()):.4f}")

    if best_state is None:
        if ema is not None:
            best_state = ema.state_dict()
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    return model, best_epoch, best_val


# ==============================================================================
# 10. Run protocol
# ==============================================================================

def run_one(args, data: Dict[str, np.ndarray], run_id: int, base_seed: int, device: torch.device) -> Dict:
    hsi = data["hsi"]
    lidar = data["lidar"]
    gt = data["gt"]
    tr_label = data["tr"]
    ts_label = data["ts"]
    num_classes = int(max(gt.max(), tr_label.max(), ts_label.max()))

    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"[RUN {run_id}/{args.runs}] seed={base_seed}")

    cv_best_epochs = []
    cv_results = []

    print("[Stage A] Spatial validation inside official training labels only")
    for fold in range(args.cv_folds):
        fold_seed = base_seed + 100 * fold
        print("\n" + "-" * 88)
        print(f"[CV FOLD {fold + 1}/{args.cv_folds}] seed={fold_seed}")

        cv_train_mask, cv_val_mask = classwise_spatial_train_val_split(
            tr_label,
            val_ratio=args.cv_val_ratio,
            block_size=args.block_size,
            seed=fold_seed,
            min_train_per_class=args.min_train_per_class,
            min_val_per_class=args.min_val_per_class,
        )
        tr_coords, tr_y = mask_to_coords_labels(cv_train_mask)
        va_coords, va_y = mask_to_coords_labels(cv_val_mask)
        print(f"[INFO] split sizes -> train={len(tr_y)} | val={len(va_y)}")

        features_cv, _lidar_cv, pca_cv = fit_preprocess_train_only(hsi, lidar, cv_train_mask, args.pca_dim)
        train_pack = make_graph_pack(features_cv, cv_train_mask, args.graph_k)
        val_pack = make_graph_pack(features_cv, cv_val_mask, args.graph_k)

        model_cv, best_epoch, best_val = train_graph_model(
            args=args,
            train_pack=train_pack,
            val_pack=val_pack,
            in_channels=features_cv.shape[-1],
            num_classes=num_classes,
            device=device,
            seed=fold_seed,
            epochs=args.cv_epochs,
        )

        cv_best_epochs.append(best_epoch)
        cv_results.append(best_val)
        print(
            f"[CV BEST] epoch={best_epoch} | OA={best_val.oa:.4f} | "
            f"AA={best_val.aa:.4f} | Kappa={best_val.kappa:.4f}"
        )

        del model_cv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_epoch = int(round(float(np.median(cv_best_epochs)) * args.final_epoch_scale))
    selected_epoch = max(args.epochs_min, min(selected_epoch, int(round(args.cv_epochs * 1.35))))

    cv_oa = np.array([m.oa for m in cv_results], dtype=np.float64)
    cv_aa = np.array([m.aa for m in cv_results], dtype=np.float64)
    cv_k = np.array([m.kappa for m in cv_results], dtype=np.float64)
    print("\n" + "-" * 88)
    print(f"[CV SUMMARY] median best epoch={np.median(cv_best_epochs):.1f} | selected final epoch={selected_epoch}")
    print(f"[CV SUMMARY] OA={cv_oa.mean():.4f} ± {cv_oa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] AA={cv_aa.mean():.4f} ± {cv_aa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] Kappa={cv_k.mean():.4f} ± {cv_k.std(ddof=0):.4f}")

    print("\n[Stage B] Retrain on full official training labels")
    features_full, _lidar_full, pca_full = fit_preprocess_train_only(hsi, lidar, tr_label, args.pca_dim)
    full_train_pack = make_graph_pack(features_full, tr_label, args.graph_k)
    final_model, _best_epoch_unused, _best_val_unused = train_graph_model(
        args=args,
        train_pack=full_train_pack,
        val_pack=None,
        in_channels=features_full.shape[-1],
        num_classes=num_classes,
        device=device,
        seed=base_seed + 777,
        epochs=selected_epoch,
    )
    print(f"[INFO] full-train PCA dims={pca_full.n_components_}")

    print("\n[Stage C] Final official test evaluation")
    test_pack = make_graph_pack(features_full, ts_label, args.graph_k)
    test_res = evaluate_graph(final_model, test_pack, device, num_classes)

    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")
    print("[TEST] Per-class accuracy")
    for i, acc in enumerate(test_res.per_class):
        print(f"  {i + 1:02d}. {categories[i]:<18s} : {acc:.4f}")

    # Fair map generation.
    # Official test map uses test graph only. All-labeled map is composed from train-graph prediction
    # and test-graph prediction separately, avoiding train-test message passing.
    train_pred = predict_graph(final_model, full_train_pack, device)
    test_pred = test_res.y_pred.astype(np.int64)

    pred_train_img = fill_label_image(gt.shape, full_train_pack.coords, train_pred)
    pred_test_img = fill_label_image(gt.shape, test_pack.coords, test_pred)
    pred_all_img = np.zeros_like(gt, dtype=np.uint8)
    pred_all_img[full_train_pack.coords[:, 0], full_train_pack.coords[:, 1]] = train_pred.astype(np.uint8) + 1
    pred_all_img[test_pack.coords[:, 0], test_pack.coords[:, 1]] = test_pred.astype(np.uint8) + 1

    save_rgb_map(gt.astype(np.uint8), "Ground Truth Land-cover Map (All Labeled Pixels)",
                 os.path.join(run_dir, "ground_truth_all_labels.png"), with_legend=True)
    save_rgb_map(ts_label.astype(np.uint8), "Ground Truth Land-cover Map (Official Test Pixels)",
                 os.path.join(run_dir, "ground_truth_test_labels.png"), with_legend=True)
    save_rgb_map(pred_all_img, "Predicted Land-cover Classification Map (All Labeled Pixels)",
                 os.path.join(run_dir, "prediction_all_labels.png"), with_legend=True)
    save_rgb_map(pred_test_img, "Predicted Land-cover Classification Map (Official Test Pixels)",
                 os.path.join(run_dir, "prediction_test_labels.png"), with_legend=True)
    save_paper_map_pair(ts_label.astype(np.uint8), pred_test_img,
                        os.path.join(run_dir, "paper_landcover_map_test_labels.png"),
                        run_id=run_id, method_name=args.method_name)
    plot_per_class_bar(test_res.per_class, os.path.join(run_dir, "per_class_accuracy.png"),
                       f"Per-class Accuracy (Run {run_id})")

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "OA": test_res.oa,
                "AA": test_res.aa,
                "Kappa": test_res.kappa,
                "selected_epoch": selected_epoch,
                "cv_best_epochs": [int(e) for e in cv_best_epochs],
                "per_class": test_res.per_class.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(run_dir, "per_class_accuracy.csv"), "w", encoding="utf-8") as f:
        f.write("class_id,class_name,accuracy\n")
        for i, acc in enumerate(test_res.per_class):
            f.write(f"{i + 1},{categories[i]},{acc:.6f}\n")

    return {
        "oa": test_res.oa,
        "aa": test_res.aa,
        "kappa": test_res.kappa,
        "per_class": test_res.per_class.copy(),
        "run_dir": run_dir,
        "selected_epoch": selected_epoch,
    }


# ==============================================================================
# 11. Summary
# ==============================================================================

def summarize_runs(run_results: List[Dict], output_dir: str) -> None:
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    oa = np.array([r["oa"] for r in run_results], dtype=np.float64)
    aa = np.array([r["aa"] for r in run_results], dtype=np.float64)
    kappa = np.array([r["kappa"] for r in run_results], dtype=np.float64)
    per_class = np.stack([r["per_class"] for r in run_results], axis=0)

    mean_pc = per_class.mean(axis=0)
    std_pc = per_class.std(axis=0, ddof=0)

    summary = {
        "runs": len(run_results),
        "OA_mean": float(oa.mean()),
        "OA_std": float(oa.std(ddof=0)),
        "AA_mean": float(aa.mean()),
        "AA_std": float(aa.std(ddof=0)),
        "Kappa_mean": float(kappa.mean()),
        "Kappa_std": float(kappa.std(ddof=0)),
        "per_class_mean": mean_pc.tolist(),
        "per_class_std": std_pc.tolist(),
    }

    with open(os.path.join(summary_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(summary_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"OA     : {summary['OA_mean']:.4f} ± {summary['OA_std']:.4f}\n")
        f.write(f"AA     : {summary['AA_mean']:.4f} ± {summary['AA_std']:.4f}\n")
        f.write(f"Kappa  : {summary['Kappa_mean']:.4f} ± {summary['Kappa_std']:.4f}\n")
        f.write("Per-class accuracy mean ± std\n")
        for i in range(len(mean_pc)):
            f.write(f"{i + 1:02d}. {categories[i]:<18s} : {mean_pc[i]:.4f} ± {std_pc[i]:.4f}\n")

    pd.DataFrame([{
        "OA Mean": float(oa.mean()),
        "OA Std": float(oa.std(ddof=0)),
        "AA Mean": float(aa.mean()),
        "AA Std": float(aa.std(ddof=0)),
        "Kappa Mean": float(kappa.mean()),
        "Kappa Std": float(kappa.std(ddof=0)),
    }]).to_csv(os.path.join(summary_dir, "overall_metrics.csv"), index=False)

    rows = []
    for i in range(len(mean_pc)):
        rows.append({
            "Class ID": i + 1,
            "Class Name": categories[i],
            "Mean": float(mean_pc[i]),
            "Std": float(std_pc[i]),
        })
    pd.DataFrame(rows).to_csv(os.path.join(summary_dir, "per_class_mean_std.csv"), index=False)

    plot_per_class_bar(mean_pc, os.path.join(summary_dir, "per_class_mean_std.png"),
                       "Per-class Accuracy Mean ± Std", std=std_pc)

    best_idx = int(np.argmax(oa))
    src = os.path.join(run_results[best_idx]["run_dir"], "paper_landcover_map_test_labels.png")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, os.path.join(summary_dir, "best_run_paper_landcover_map_test_labels.png"))

    print("\n" + "#" * 88)
    print(f"Final summary across {len(run_results)} runs")
    print(f"OA     : {summary['OA_mean']:.4f} ± {summary['OA_std']:.4f}")
    print(f"AA     : {summary['AA_mean']:.4f} ± {summary['AA_std']:.4f}")
    print(f"Kappa  : {summary['Kappa_mean']:.4f} ± {summary['Kappa_std']:.4f}")
    print("Per-class accuracy mean ± std")
    for i in range(len(mean_pc)):
        print(f"  {i + 1:02d}. {categories[i]:<18s} : {mean_pc[i]:.4f} ± {std_pc[i]:.4f}")
    print(f"[INFO] Summary saved to: {summary_dir}")


# ==============================================================================
# 12. Main
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser("Fair MS-GWCN-Cheb baseline for Houston 2013 HSI-LiDAR classification")
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")
    parser.add_argument("--output-dir", type=str, default="houston_msgwcn_cheb_fair_runs")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-val-ratio", type=float, default=0.20)
    parser.add_argument("--block-size", type=int, default=18)
    parser.add_argument("--min-train-per-class", type=int, default=16)
    parser.add_argument("--min-val-per-class", type=int, default=8)

    parser.add_argument("--cv-epochs", type=int, default=180)
    parser.add_argument("--epochs-min", type=int, default=50)
    parser.add_argument("--final-epoch-scale", type=float, default=1.08)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--cheb-k-list", type=str, default="2,4,6")
    parser.add_argument("--hidden-dims", type=str, default="64,96,128")
    parser.add_argument("--eval-interval", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--method-name", type=str, default="MS-GWCN-Cheb")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    data = load_houston_2013(args.data_root)
    print(f"[INFO] data-root={args.data_root}")
    print(f"[INFO] HSI shape={data['hsi'].shape}, LiDAR shape={data['lidar'].shape}")
    print(f"[INFO] total train labels={(data['tr'] > 0).sum()}, total test labels={(data['ts'] > 0).sum()}")
    print(f"[INFO] device={device}")
    print(f"[INFO] output-dir={args.output_dir}")
    print("[INFO] Fair baseline protocol: train-only preprocessing, spatial CV, full official-train retraining, final TSLabel-only test.")
    print("[INFO] Graphs are built separately for train, validation, and test masks to avoid train-test message passing.")

    run_results: List[Dict] = []
    for run_id in range(1, args.runs + 1):
        base_seed = args.seed + 10000 * (run_id - 1)
        result = run_one(args, data, run_id, base_seed, device)
        run_results.append(result)

    summarize_runs(run_results, args.output_dir)


if __name__ == "__main__":
    main()

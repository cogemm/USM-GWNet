# -*- coding: utf-8 -*-
"""
Multimodal Prompt Tuning for Houston 2013 HSI-LiDAR Classification.

Paper-inspired implementation:
- Grouped PCA: split HSI bands into T groups and extract L PCA components per group.
- Stage I: HSI-only transformer representation learning.
- Stage II: LiDAR prompt tuning through cross-attention in transformer blocks.
- Evaluation: OA, AA, Kappa, per-class accuracy, 5-run mean ± std.
- Visualization: uses the exact user-provided Houston color list by direct RGB lookup.

Expected local files under --data-root:
    HSI.mat, LiDAR.mat, gt.mat, TRLabel.mat, TSLabel.mat

Default:
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")

Strict protocol:
    TRLabel.mat is used for training/validation only.
    TSLabel.mat is used only for final testing.
    Standardization and grouped PCA are fitted only on training pixels.
    No test labels are used for preprocessing, model selection, or training.

Note:
    The original paper uses external HSI-only datasets in Stage I.
    Those external datasets are not present in your Houston folder.
    This script implements the same mechanism with Houston-only Stage I pretraining.
"""

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_first_numeric_array(mat_dict: Dict) -> np.ndarray:
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


def load_mat_array(path: str) -> np.ndarray:
    return np.asarray(find_first_numeric_array(sio.loadmat(path)))


def ensure_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape={x.shape}.")
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
        raise FileNotFoundError(f"Missing files: {missing}")

    hsi = ensure_hwc(load_mat_array(paths["hsi"])).astype(np.float32)
    lidar = ensure_hwc(load_mat_array(paths["lidar"])).astype(np.float32)
    gt = load_mat_array(paths["gt"]).squeeze().astype(np.int64)
    tr = load_mat_array(paths["tr"]).squeeze().astype(np.int64)
    ts = load_mat_array(paths["ts"]).squeeze().astype(np.int64)

    if hsi.shape[:2] != lidar.shape[:2] or hsi.shape[:2] != gt.shape:
        raise ValueError(f"Spatial mismatch: HSI={hsi.shape}, LiDAR={lidar.shape}, gt={gt.shape}")
    return {"hsi": hsi, "lidar": lidar, "gt": gt, "tr": tr, "ts": ts}


def mask_to_coords_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    labels = mask[mask > 0] - 1
    return coords.astype(np.int64), labels.astype(np.int64)


def stratified_train_val_split(tr_label: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    coords, labels = mask_to_coords_labels(tr_label)
    if val_ratio <= 0:
        return tr_label.copy(), np.zeros_like(tr_label, dtype=np.int64)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(coords, labels))
    train_mask = np.zeros_like(tr_label, dtype=np.int64)
    val_mask = np.zeros_like(tr_label, dtype=np.int64)
    train_mask[coords[train_idx, 0], coords[train_idx, 1]] = labels[train_idx] + 1
    val_mask[coords[val_idx, 0], coords[val_idx, 1]] = labels[val_idx] + 1
    return train_mask, val_mask


def fit_standardize_train_only(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train_pixels = x[train_mask > 0]
    mean = train_pixels.mean(axis=0, keepdims=True)
    std = train_pixels.std(axis=0, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def fit_groupwise_pca_train_only(
    hsi_norm: np.ndarray,
    train_mask: np.ndarray,
    groups: int = 4,
    components_per_group: int = 8,
) -> np.ndarray:
    h, w, c = hsi_norm.shape
    train_pixels = hsi_norm[train_mask > 0]
    band_groups = np.array_split(np.arange(c), groups)
    flat = hsi_norm.reshape(-1, c)
    transformed_groups = []

    for gidx in band_groups:
        x_train = train_pixels[:, gidx]
        n_comp = min(components_per_group, x_train.shape[1], x_train.shape[0])
        pca = PCA(n_components=n_comp, svd_solver="full", whiten=False)
        pca.fit(x_train)
        z_all = pca.transform(flat[:, gidx]).astype(np.float32)

        if n_comp < components_per_group:
            pad = np.zeros((z_all.shape[0], components_per_group - n_comp), dtype=np.float32)
            z_all = np.concatenate([z_all, pad], axis=1)
        transformed_groups.append(z_all)

    out = np.concatenate(transformed_groups, axis=1).reshape(h, w, groups * components_per_group)
    return out.astype(np.float32)


def random_patch_augment(hsi_pca: np.ndarray, lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = np.random.randint(0, 4)
    if k:
        hsi_pca = np.rot90(hsi_pca, k=k, axes=(0, 1)).copy()
        lidar = np.rot90(lidar, k=k, axes=(0, 1)).copy()
    if np.random.rand() < 0.5:
        hsi_pca = np.flip(hsi_pca, axis=1).copy()
        lidar = np.flip(lidar, axis=1).copy()
    if np.random.rand() < 0.5:
        hsi_pca = np.flip(hsi_pca, axis=0).copy()
        lidar = np.flip(lidar, axis=0).copy()
    if np.random.rand() < 0.35:
        hsi_pca = hsi_pca + np.random.normal(0.0, 0.01, size=hsi_pca.shape).astype(np.float32)
    if np.random.rand() < 0.20:
        lidar = lidar + np.random.normal(0.0, 0.01, size=lidar.shape).astype(np.float32)
    return hsi_pca, lidar


class HoustonPromptDataset(Dataset):
    def __init__(
        self,
        hsi_pca: np.ndarray,
        lidar: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.hsi_pca = hsi_pca
        self.lidar = lidar
        self.coords = coords
        self.labels = labels
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.augment = augment

        self.hsi_pad = np.pad(
            hsi_pca,
            ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
            mode="reflect",
        )
        self.lidar_pad = np.pad(
            lidar,
            ((self.radius, self.radius), (self.radius, self.radius), (0, 0)),
            mode="reflect",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        y, x = self.coords[idx]
        ps = self.patch_size
        hsi_patch = self.hsi_pad[y:y + ps, x:x + ps, :].copy()
        lidar_patch = self.lidar_pad[y:y + ps, x:x + ps, :].copy()

        if self.augment:
            hsi_patch, lidar_patch = random_patch_augment(hsi_patch, lidar_patch)

        hsi_patch = torch.from_numpy(np.transpose(hsi_patch, (2, 0, 1))).float()
        lidar_patch = torch.from_numpy(np.transpose(lidar_patch, (2, 0, 1))).float()
        label = int(self.labels[idx])
        coord = torch.tensor([y, x], dtype=torch.long)
        return hsi_patch, lidar_patch, label, coord


def make_loader(
    hsi_pca: np.ndarray,
    lidar: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
    pin_memory: bool,
) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    coords, labels = mask_to_coords_labels(mask)
    ds = HoustonPromptDataset(hsi_pca, lidar, coords, labels, patch_size=patch_size, augment=augment)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, coords, labels


# =========================
# Model
# =========================

class LiDARPromptEncoder(nn.Module):
    """
    Lightweight LiDAR prompt encoder:
        Conv2D(1 -> 32), MaxPool, Conv2D(32 -> 64), GAP.
    """
    def __init__(self, in_ch: int = 1, prompt_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, prompt_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prompt_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.net(lidar).flatten(1)


class PromptTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        prompt_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_prompt = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.prompt_to_k = nn.Linear(prompt_dim, embed_dim)
        self.prompt_to_v = nn.Linear(prompt_dim, embed_dim)
        self.prompt_gate = nn.Sequential(nn.Linear(prompt_dim, embed_dim), nn.Sigmoid())
        self.cross_out = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, prompt: torch.Tensor, use_prompt: bool = True) -> torch.Tensor:
        x = self.norm1(tokens)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        tokens = tokens + attn_out

        if use_prompt:
            # Q from HSI tokens, K/V from LiDAR prompt.
            x = self.norm_prompt(tokens)
            q = self.q_proj(x)
            k = self.prompt_to_k(prompt).unsqueeze(1)
            v = self.prompt_to_v(prompt).unsqueeze(1)

            score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            attn = torch.softmax(score, dim=-1)
            prompt_out = torch.matmul(attn, v)

            # A gate is needed because one prompt token alone gives a trivial softmax.
            gate = self.prompt_gate(prompt).unsqueeze(1)
            tokens = tokens + self.cross_out(prompt_out * gate)

        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MultimodalPromptTransformer(nn.Module):
    """
    HSI branch:
        grouped PCA channels -> spectral tokens -> class token -> transformer.

    LiDAR branch:
        CNN prompt token -> cross-attention injection in each transformer block.
    """
    def __init__(
        self,
        hsi_pca_channels: int,
        lidar_channels: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        prompt_dim: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads.")

        self.hsi_pca_channels = hsi_pca_channels
        self.patch_size = patch_size
        spatial_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(spatial_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, hsi_pca_channels + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.lidar_encoder = LiDARPromptEncoder(in_ch=lidar_channels, prompt_dim=prompt_dim)

        self.blocks = nn.ModuleList([
            PromptTransformerBlock(
                embed_dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                prompt_dim=prompt_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, hsi_pca: torch.Tensor, lidar: torch.Tensor, use_prompt: bool = True) -> torch.Tensor:
        b, c, h, w = hsi_pca.shape
        x = hsi_pca.reshape(b, c, h * w)
        tokens = self.patch_embed(x)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.pos_drop(tokens)

        prompt = self.lidar_encoder(lidar)
        for blk in self.blocks:
            tokens = blk(tokens, prompt, use_prompt=use_prompt)

        cls_out = self.norm(tokens[:, 0])
        return self.head(cls_out)


# =========================
# Metrics and visualization
# =========================

@dataclass
class EvalResult:
    oa: float
    aa: float
    kappa: float
    per_class: np.ndarray
    cm: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> EvalResult:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    oa = float((y_true == y_pred).mean())
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    aa = float(per_class.mean())
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=np.arange(num_classes)))
    return EvalResult(oa, aa, kappa, per_class, cm, y_true, y_pred)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) != 6:
        raise ValueError(f"Invalid color: {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PALETTE_RGB = np.array([[0, 0, 0]] + [list(hex_to_rgb(c)) for c in colors], dtype=np.uint8)


def label_map_to_rgb(label_img: np.ndarray) -> np.ndarray:
    label_img = np.asarray(label_img)
    out = np.zeros(label_img.shape + (3,), dtype=np.uint8)
    valid = (label_img >= 0) & (label_img < len(PALETTE_RGB))
    out[valid] = PALETTE_RGB[label_img[valid].astype(np.int64)]
    return out


def build_label_image(shape_hw: Tuple[int, int], coords: np.ndarray, zero_based_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    if len(coords) > 0:
        out[coords[:, 0], coords[:, 1]] = zero_based_pred.astype(np.uint8) + 1
    return out


def save_single_map(label_img: np.ndarray, title: str, path: str, with_legend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.imshow(label_map_to_rgb(label_img), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")

    if with_legend:
        handles = [Patch(facecolor=colors[i], edgecolor="k", label=categories[i]) for i in range(14, -1, -1)]
        labels = [categories[i] for i in range(14, -1, -1)]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

    plt.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def save_paper_map_pair(gt_img: np.ndarray, pred_img: np.ndarray, path: str, run_id: int, method_name: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 7.2))
    axes[0].imshow(label_map_to_rgb(gt_img), interpolation="nearest")
    axes[0].set_title("Ground Truth", fontsize=18)
    axes[0].axis("off")

    axes[1].imshow(label_map_to_rgb(pred_img), interpolation="nearest")
    axes[1].set_title(f"Classification Result (Run {run_id}) by {method_name}", fontsize=18)
    axes[1].axis("off")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, hspace=0.18)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_per_class_bar(per_class: np.ndarray, path: str, title: str) -> None:
    x = np.arange(1, len(per_class) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, per_class, color=colors[: len(per_class)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories[: len(per_class)], rotation=45, ha="right")
    for xi, yi in zip(x, per_class):
        ax.text(xi, min(0.985, yi + 0.015), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_mean_std(mean_pc: np.ndarray, std_pc: np.ndarray, path: str) -> None:
    x = np.arange(1, len(mean_pc) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, mean_pc, color=colors[: len(mean_pc)])
    ax.errorbar(x, mean_pc, yerr=std_pc, fmt="none", capsize=4, ecolor="black", elinewidth=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Land-cover Classification Accuracy Mean ± Std")
    ax.set_xticks(x)
    ax.set_xticklabels(categories[: len(mean_pc)], rotation=45, ha="right")
    for xi, m, s in zip(x, mean_pc, std_pc):
        ax.text(xi, min(0.985, m + s + 0.015), f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Training and inference
# =========================

def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast("cuda", enabled=(enabled and device.type == "cuda"))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    use_prompt: bool,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for hsi_pca, lidar, labels, _coords in loader:
        hsi_pca = hsi_pca.to(device, non_blocking=True)
        lidar = lidar.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = model(hsi_pca, lidar, use_prompt=use_prompt)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(total, 1)


@torch.no_grad()
def infer_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_prompt: bool,
    tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    labels_all = []
    coords_all = []

    tta_fns = [
        lambda h, l: (h, l),
        lambda h, l: (torch.flip(h, dims=[-1]), torch.flip(l, dims=[-1])),
        lambda h, l: (torch.flip(h, dims=[-2]), torch.flip(l, dims=[-2])),
        lambda h, l: (torch.rot90(h, 1, dims=[-2, -1]), torch.rot90(l, 1, dims=[-2, -1])),
        lambda h, l: (torch.rot90(h, 2, dims=[-2, -1]), torch.rot90(l, 2, dims=[-2, -1])),
        lambda h, l: (torch.rot90(h, 3, dims=[-2, -1]), torch.rot90(l, 3, dims=[-2, -1])),
    ]

    for hsi_pca, lidar, labels, coords in loader:
        hsi_pca = hsi_pca.to(device, non_blocking=True)
        lidar = lidar.to(device, non_blocking=True)

        if tta:
            acc = None
            for fn in tta_fns:
                h_aug, l_aug = fn(hsi_pca, lidar)
                out = model(h_aug, l_aug, use_prompt=use_prompt)
                acc = out if acc is None else (acc + out)
            logits = acc / float(len(tta_fns))
        else:
            logits = model(hsi_pca, lidar, use_prompt=use_prompt)

        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.cpu())
        coords_all.append(coords.cpu())

    return (
        torch.cat(logits_all, dim=0).numpy(),
        torch.cat(labels_all, dim=0).numpy(),
        torch.cat(coords_all, dim=0).numpy(),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_prompt: bool,
    tta: bool,
) -> EvalResult:
    logits, labels, _coords = infer_logits(model, loader, device, use_prompt=use_prompt, tta=tta)
    preds = logits.argmax(axis=1).astype(np.int64)
    return compute_metrics(labels.astype(np.int64), preds, num_classes)


def run_one(args, data: Dict[str, np.ndarray], run_id: int, base_seed: int, device: torch.device) -> Dict:
    set_seed(base_seed)

    hsi = data["hsi"]
    lidar = data["lidar"]
    gt = data["gt"]
    tr_label = data["tr"]
    ts_label = data["ts"]
    num_classes = int(max(gt.max(), tr_label.max(), ts_label.max()))

    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    train_mask, val_mask = stratified_train_val_split(tr_label, args.val_ratio, base_seed)
    train_coords, train_labels = mask_to_coords_labels(train_mask)
    val_coords, val_labels = mask_to_coords_labels(val_mask)
    test_coords, test_labels = mask_to_coords_labels(ts_label)

    hsi_norm = fit_standardize_train_only(hsi, train_mask)
    lidar_norm = fit_standardize_train_only(lidar, train_mask)
    hsi_pca = fit_groupwise_pca_train_only(
        hsi_norm,
        train_mask,
        groups=args.pca_groups,
        components_per_group=args.pca_per_group,
    )

    pin = (device.type == "cuda")

    train_loader, _, _ = make_loader(
        hsi_pca, lidar_norm, train_mask,
        args.patch_size, args.batch_size, args.num_workers,
        augment=True, shuffle=True, pin_memory=pin,
    )
    val_loader, _, _ = make_loader(
        hsi_pca, lidar_norm, val_mask,
        args.patch_size, args.batch_size, args.num_workers,
        augment=False, shuffle=False, pin_memory=pin,
    )
    test_loader, _, _ = make_loader(
        hsi_pca, lidar_norm, ts_label,
        args.patch_size, args.batch_size, args.num_workers,
        augment=False, shuffle=False, pin_memory=pin,
    )
    all_loader, _, _ = make_loader(
        hsi_pca, lidar_norm, gt,
        args.patch_size, args.batch_size, args.num_workers,
        augment=False, shuffle=False, pin_memory=pin,
    )

    model = MultimodalPromptTransformer(
        hsi_pca_channels=args.pca_groups * args.pca_per_group,
        lidar_channels=lidar_norm.shape[-1],
        patch_size=args.patch_size,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        prompt_dim=args.prompt_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = max(1, args.pretrain_epochs + args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=args.lr * 0.05,
    )

    print("\n" + "=" * 88)
    print(f"[RUN {run_id}/{args.runs}] seed={base_seed}")
    print(f"[INFO] train={len(train_labels)} | val={len(val_labels)} | test={len(test_labels)}")
    print(f"[INFO] grouped PCA dims={args.pca_groups * args.pca_per_group} | patch={args.patch_size}")

    # Stage I. HSI-only representation learning.
    for ep in range(1, args.pretrain_epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_prompt=False, amp_enabled=args.amp
        )
        scheduler.step()

        if ep == 1 or ep % args.log_interval == 0 or ep == args.pretrain_epochs:
            val_res = evaluate(model, val_loader, device, num_classes, use_prompt=False, tta=False)
            print(
                f"[Stage I HSI-only] epoch {ep:03d}/{args.pretrain_epochs} | "
                f"loss={loss:.4f} | val_OA={val_res.oa:.4f} | "
                f"val_AA={val_res.aa:.4f} | val_Kappa={val_res.kappa:.4f}"
            )

    # Stage II. LiDAR prompt tuning.
    best_score = -1.0
    best_state = None
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_prompt=True, amp_enabled=args.amp
        )
        scheduler.step()

        if ep == 1 or ep % args.eval_interval == 0 or ep == args.epochs:
            val_res = evaluate(model, val_loader, device, num_classes, use_prompt=True, tta=False)
            score = 0.45 * val_res.oa + 0.35 * val_res.aa + 0.20 * val_res.kappa

            if score > best_score:
                best_score = score
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(
                f"[Stage II LPT] epoch {ep:03d}/{args.epochs} | "
                f"loss={loss:.4f} | val_OA={val_res.oa:.4f} | "
                f"val_AA={val_res.aa:.4f} | val_Kappa={val_res.kappa:.4f} | "
                f"best_epoch={best_epoch}"
            )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_res = evaluate(model, test_loader, device, num_classes, use_prompt=True, tta=args.tta)

    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")
    print("[TEST] Per-class accuracy")
    for i, acc in enumerate(test_res.per_class):
        print(f"  {i + 1:02d}. {categories[i]:<18s} : {acc:.4f}")

    all_logits, _all_y, all_coords_infer = infer_logits(model, all_loader, device, use_prompt=True, tta=args.tta)
    all_pred = all_logits.argmax(axis=1).astype(np.int64)

    test_logits, _test_y, test_coords_infer = infer_logits(model, test_loader, device, use_prompt=True, tta=args.tta)
    test_pred = test_logits.argmax(axis=1).astype(np.int64)

    pred_all_img = build_label_image(gt.shape, all_coords_infer, all_pred)
    pred_test_img = build_label_image(gt.shape, test_coords_infer, test_pred)

    save_single_map(
        gt.astype(np.uint8),
        "Ground Truth Land-cover Map (All Labeled Pixels)",
        os.path.join(run_dir, "ground_truth_all_labels.png"),
        with_legend=True,
    )
    save_single_map(
        ts_label.astype(np.uint8),
        "Ground Truth Land-cover Map (Official Test Pixels)",
        os.path.join(run_dir, "ground_truth_test_labels.png"),
        with_legend=True,
    )
    save_single_map(
        pred_all_img,
        "Predicted Land-cover Classification Map (All Labeled Pixels)",
        os.path.join(run_dir, "prediction_all_labels.png"),
        with_legend=True,
    )
    save_single_map(
        pred_test_img,
        "Predicted Land-cover Classification Map (Official Test Pixels)",
        os.path.join(run_dir, "prediction_test_labels.png"),
        with_legend=True,
    )
    save_paper_map_pair(
        gt.astype(np.uint8),
        pred_all_img,
        os.path.join(run_dir, "paper_landcover_map_all_labels.png"),
        run_id,
        args.method_name,
    )
    save_paper_map_pair(
        ts_label.astype(np.uint8),
        pred_test_img,
        os.path.join(run_dir, "paper_landcover_map_test_labels.png"),
        run_id,
        args.method_name,
    )
    save_per_class_bar(
        test_res.per_class,
        os.path.join(run_dir, "per_class_accuracy.png"),
        f"Per-class Accuracy (Run {run_id})",
    )

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "OA": test_res.oa,
                "AA": test_res.aa,
                "Kappa": test_res.kappa,
                "best_epoch": best_epoch,
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
        "best_epoch": best_epoch,
    }


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

    with open(os.path.join(summary_dir, "per_class_overall.csv"), "w", encoding="utf-8") as f:
        f.write("class_id,class_name,mean,std\n")
        for i in range(len(mean_pc)):
            f.write(f"{i + 1},{categories[i]},{mean_pc[i]:.6f},{std_pc[i]:.6f}\n")

    save_per_class_mean_std(mean_pc, std_pc, os.path.join(summary_dir, "per_class_mean_std.png"))

    best_idx = int(np.argmax(oa))
    best_dir = run_results[best_idx]["run_dir"]
    src = os.path.join(best_dir, "paper_landcover_map_test_labels.png")
    if os.path.exists(src):
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


def main() -> None:
    parser = argparse.ArgumentParser("Multimodal Prompt Tuning for Houston 2013 HSI-LiDAR classification")
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")
    parser.add_argument("--output-dir", type=str, default="houston_mpt_results")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--pca-groups", type=int, default=4)
    parser.add_argument("--pca-per-group", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.20)

    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--prompt-dim", type=int, default=64)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--method-name", type=str, default="MPT-LPT")

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
    print("[INFO] Stage I uses Houston-only HSI pretraining because external HSI pretraining datasets are not supplied.")

    run_results: List[Dict] = []
    for run_id in range(1, args.runs + 1):
        base_seed = args.seed + 10000 * (run_id - 1)
        result = run_one(args, data, run_id, base_seed, device)
        run_results.append(result)

    summarize_runs(run_results, args.output_dir)


if __name__ == "__main__":
    main()

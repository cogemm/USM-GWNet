# -*- coding: utf-8 -*-
"""
Fair GS-WCN-SOA-Transformer baseline for Houston 2013 HSI-LiDAR classification.

This version is modified according to the no-cheating comparison protocol used in
`houston_slimamba_msgw_v5_urbanfocus_5runs_maps_direct_rgb_palette.py`.

Key fairness constraints
------------------------
1. The local data path and Houston color list are kept unchanged.
2. TRLabel.mat is used for training/model selection only.
3. TSLabel.mat is used only for final testing and map rendering.
4. Standardization and optional PCA are fitted only on the active training split.
5. Spatial validation is performed only inside the official training labels.
6. The final model is retrained on all official training labels before testing.
7. No test-label-aware label propagation, no test-set hyperparameter search, and
   no all-pixel preprocessing are used.
8. Classification maps are rendered by direct RGB lookup using the provided palette.

Expected local files
--------------------
E:/PythonProject1/Houston/
    gt.mat
    HSI.mat
    LiDAR.mat
    TRLabel.mat
    TSLabel.mat

Default quick test
------------------
python GS_WCN_SOA_Transformer_Houston_fair_v2.py --runs 1 --cv-folds 1 --cv-epochs 20 --epochs-min 20

Formal run
----------
python GS_WCN_SOA_Transformer_Houston_fair_v2.py --runs 5 --cv-folds 3 --cv-epochs 180 --tta
"""

import argparse
import copy
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import softmax
    from torch_geometric.nn import MessagePassing
except Exception as e:
    raise ImportError(
        "This script requires torch-geometric. Install PyG in the same environment "
        "used by the original GS_WCN_SOA_Transformer code."
    ) from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# =================== Global and fixed Houston palette ===================

BASE_SEED = 42

# Houston categories (15 classes)
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
cmap = ListedColormap(['#000000'] + colors)

CLASS_NAMES = categories
CLASS_COLORS = colors

# zero-based ids. They are used only for reporting and optional focus weighting.
URBAN_IDS = [6, 7, 8, 9, 10, 11, 12]
FOCUS_IDS = [7, 8, 9, 11]  # Commercial, Road, Highway, Parking lot 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# =================== Data loading and leakage-controlled preprocessing ===================


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
        raise ValueError(f"Expected a 2-D or 3-D array, got shape={x.shape}")
    # Defensive conversion for C,H,W input. Houston normally arrives as H,W,C.
    if x.shape[0] < 20 and x.shape[-1] > 20:
        x = np.transpose(x, (1, 2, 0))
    return x


def load_houston_2013(data_root: str) -> Dict[str, np.ndarray]:
    paths = {
        "gt": os.path.join(data_root, "gt.mat"),
        "hsi": os.path.join(data_root, "HSI.mat"),
        "lidar": os.path.join(data_root, "LiDAR.mat"),
        "tr": os.path.join(data_root, "TRLabel.mat"),
        "ts": os.path.join(data_root, "TSLabel.mat"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing Houston files: {missing}")

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


def fit_standardize_train_only(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pixels = x[train_mask > 0]
    mean = train_pixels.mean(axis=0, keepdims=True)
    std = train_pixels.std(axis=0, keepdims=True) + 1e-6
    out = (x - mean) / std
    return out.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_feature_cube_train_only(
    hsi: np.ndarray,
    lidar: np.ndarray,
    train_mask: np.ndarray,
    pca_dim: int = 0,
) -> Tuple[np.ndarray, Optional[PCA]]:
    """Fit all preprocessing only on train_mask and apply it to the whole image."""
    hsi_norm, _, _ = fit_standardize_train_only(hsi, train_mask)
    lidar_norm, _, _ = fit_standardize_train_only(lidar, train_mask)

    pca_model: Optional[PCA] = None
    if pca_dim and pca_dim > 0:
        train_pixels = hsi_norm[train_mask > 0]
        n_components = min(int(pca_dim), train_pixels.shape[0], train_pixels.shape[1])
        pca_model = PCA(n_components=n_components, svd_solver="full", whiten=False)
        pca_model.fit(train_pixels)
        h, w, b = hsi_norm.shape
        hsi_feat = pca_model.transform(hsi_norm.reshape(-1, b)).reshape(h, w, n_components).astype(np.float32)
    else:
        hsi_feat = hsi_norm.astype(np.float32)

    feat = np.concatenate([hsi_feat, lidar_norm.astype(np.float32)], axis=-1)
    return feat.astype(np.float32), pca_model


# =================== Spatial validation split ===================


def stratified_train_val_split(tr_label: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    coords, labels = mask_to_coords_labels(tr_label)
    if val_ratio <= 0:
        return tr_label.copy(), np.zeros_like(tr_label, dtype=np.int64)
    rng = np.random.RandomState(seed)
    train_mask = np.zeros_like(tr_label, dtype=np.int64)
    val_mask = np.zeros_like(tr_label, dtype=np.int64)
    for cls in np.unique(labels):
        ids = np.where(labels == cls)[0]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids = ids[:n_val]
        tr_ids = ids[n_val:]
        train_mask[coords[tr_ids, 0], coords[tr_ids, 1]] = cls + 1
        val_mask[coords[val_ids, 0], coords[val_ids, 1]] = cls + 1
    return train_mask, val_mask


def classwise_spatial_train_val_split(
    tr_label: np.ndarray,
    val_ratio: float,
    block_size: int,
    seed: int,
    min_train_per_class: int = 16,
    min_val_per_class: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Class-wise spatial block split inside official training labels only."""
    coords, labels = mask_to_coords_labels(tr_label)
    if val_ratio <= 0:
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
            keys = list(block_map.keys())
            rng.shuffle(keys)
            keys.sort(key=lambda k: len(block_map[k]))

            chosen: List[int] = []
            cur = 0
            for key in keys:
                if cur >= target_val:
                    break
                idxs = block_map[key]
                remaining = n_cls - (cur + len(idxs))
                if remaining < min_train_per_class:
                    continue
                chosen.extend(idxs)
                cur += len(idxs)

            if cur < min(min_val_per_class, target_val):
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                if fallback_target > 0:
                    perm = rng.permutation(cls_all)
                    chosen = perm[:fallback_target].tolist()

            chosen_val = np.array(sorted(set(chosen)), dtype=np.int64)
            chosen_train = np.setdiff1d(cls_all, chosen_val, assume_unique=False)
            if len(chosen_train) < min_train_per_class or len(chosen_val) < min_val_per_class:
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                if fallback_target > 0:
                    perm = rng.permutation(cls_all)
                    chosen_val = perm[:fallback_target]
                    chosen_train = perm[fallback_target:]
                else:
                    chosen_val = np.array([], dtype=np.int64)
                    chosen_train = cls_all

        train_mask[coords[chosen_train, 0], coords[chosen_train, 1]] = cls + 1
        if len(chosen_val) > 0:
            val_mask[coords[chosen_val, 0], coords[chosen_val, 1]] = cls + 1

    if (val_mask > 0).sum() == 0:
        return stratified_train_val_split(tr_label, val_ratio, seed)
    return train_mask, val_mask


# =================== Graph construction ===================


def _auto_temperature(vals: np.ndarray, eps: float = 1e-8) -> float:
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > eps]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals) + eps)


def build_graph_data(
    feature_cube: np.ndarray,
    label_mask: np.ndarray,
    graph_k: int,
    tau_f: Optional[float],
    tau_p: Optional[float],
    num_classes: int,
) -> Tuple[Data, np.ndarray, np.ndarray]:
    """Build an inductive graph for the given mask only.

    The graph uses the features of the current split only. Thus training, validation,
    and test graphs are separated and no test nodes participate in training.
    """
    coords, labels = mask_to_coords_labels(label_mask)
    if coords.shape[0] == 0:
        raise ValueError("Cannot build graph from an empty label mask.")

    x_np = feature_cube[coords[:, 0], coords[:, 1], :].astype(np.float32)
    n = coords.shape[0]
    k_eff = min(max(1, int(graph_k)), max(1, n - 1))

    if n == 1:
        edges_np = np.array([[0, 0]], dtype=np.int64)
    else:
        # Spatial KNN makes the graph usable under strict train-only graphs, where
        # fixed 7x7 windows can be nearly empty because the official labels are sparse.
        coords_float = coords.astype(np.float32)
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto")
        nn.fit(coords_float)
        neigh = nn.kneighbors(coords_float, return_distance=False)
        src_list = []
        dst_list = []
        for i in range(n):
            for j in neigh[i, 1:]:
                src_list.append(i)
                dst_list.append(int(j))
                src_list.append(int(j))
                dst_list.append(i)
        src_list.extend(range(n))
        dst_list.extend(range(n))
        edges_np = np.stack([np.asarray(src_list, dtype=np.int64), np.asarray(dst_list, dtype=np.int64)], axis=1)
        edges_np = np.unique(edges_np, axis=0)

    row = edges_np[:, 0]
    col = edges_np[:, 1]

    # Feature and spatial Gaussian weights. Temperatures are chosen from the
    # current training/evaluation graph only and are never fitted on test labels.
    df = np.sum((x_np[row] - x_np[col]) ** 2, axis=1)
    pos_np = coords.astype(np.float32)
    if pos_np.shape[0] > 1:
        pos_norm = (pos_np - pos_np.min(axis=0, keepdims=True)) / (pos_np.max(axis=0, keepdims=True) - pos_np.min(axis=0, keepdims=True) + 1e-6)
    else:
        pos_norm = np.zeros_like(pos_np)
    dp = np.sum((pos_norm[row] - pos_norm[col]) ** 2, axis=1)

    tau_f_val = float(tau_f) if tau_f is not None and tau_f > 0 else _auto_temperature(df)
    tau_p_val = float(tau_p) if tau_p is not None and tau_p > 0 else _auto_temperature(dp)
    ew_np = np.exp(-df / tau_f_val - dp / tau_p_val).astype(np.float32)
    ew_np[row == col] = 1.0

    row_t = torch.from_numpy(row).long()
    col_t = torch.from_numpy(col).long()
    ew = torch.from_numpy(ew_np).float()
    deg = torch.zeros(n, dtype=torch.float32).scatter_add_(0, row_t, ew)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[~torch.isfinite(deg_inv_sqrt)] = 0
    norm_w = deg_inv_sqrt[row_t] * ew * deg_inv_sqrt[col_t]

    edge_index = torch.stack([row_t, col_t], dim=0).long()
    L = torch.sparse_coo_tensor(edge_index, norm_w, (n, n), dtype=torch.float32).coalesce()
    try:
        L = L.to_sparse_csr()
    except Exception:
        pass

    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(labels.astype(np.int64)).long()
    pos = torch.from_numpy(pos_norm.astype(np.float32)).float()

    data = Data(x=x, y=y, edge_index=edge_index, pos=pos)
    data.L = L
    data.num_classes = int(num_classes)
    data.coords = coords
    return data, coords, labels


# =================== Model blocks from GS-WCN-SOA-Transformer ===================


def _coalesce_if_coo(M: torch.Tensor) -> torch.Tensor:
    if isinstance(M, torch.Tensor) and M.layout == torch.sparse_coo:
        return M.coalesce()
    return M


class GraphWaveletConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scales: int = 6, K: int = 5, p_drop: float = 0.20):
        super().__init__()
        self.scales = int(scales)
        self.K = int(K)
        self.s = nn.Parameter(torch.linspace(0.5, 2.0, self.scales), requires_grad=True)
        self.register_buffer("alpha", self._cheby(self.K))
        self.linear = nn.Linear(in_ch * self.scales, out_ch)
        self.ln = nn.LayerNorm(out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p_drop)
        self.res = nn.Sequential(nn.Linear(in_ch, out_ch), nn.LayerNorm(out_ch)) if in_ch != out_ch else None

    @staticmethod
    def _wave(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) * torch.exp(-x ** 2 / 2)

    def _cheby(self, K: int) -> torch.Tensor:
        j = torch.arange(0, K + 1, dtype=torch.float32)
        t = torch.cos(torch.pi * (j + 0.5) / (K + 1))
        g = self._wave(t)
        a = torch.zeros(K + 1, dtype=torch.float32)
        for k in range(K + 1):
            cosk = torch.cos(torch.pi * k * (j + 0.5) / (K + 1))
            a[k] = ((1.0 if k == 0 else 2.0) / (K + 1)) * torch.sum(g * cosk)
        return a

    @staticmethod
    def _spmm(L: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if L.layout == torch.sparse_csr:
            return torch.matmul(L, X)
        return torch.sparse.mm(L, X)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        res_in = x
        x32 = x.float()
        L_ = _coalesce_if_coo(L.float())
        T0 = x32
        T1 = self._spmm(L_, x32)
        Ts = [T0, T1]
        for _ in range(2, self.K + 1):
            Ts.append(2 * self._spmm(L_, Ts[-1]) - Ts[-2])
        stack = torch.stack(Ts, dim=0)
        alpha = self.alpha.to(stack.device).float().view(self.K + 1, 1, 1)
        gx = torch.sum(alpha * stack, dim=0)

        W = self.linear.weight
        b = self.linear.bias
        cout, cin_s = W.shape
        cin = gx.shape[1]
        if cin_s != cin * self.scales:
            raise RuntimeError(f"Linear shape mismatch: expected {cin * self.scales}, got {cin_s}")
        W_blocks = W.view(cout, self.scales, cin)
        scale = self.s.to(gx.device).float().view(1, self.scales, 1)
        W_eff = torch.sum(W_blocks * scale, dim=1)
        out = F.linear(gx, W_eff, b)
        out = self.ln(out)
        out = self.act(out)
        out = self.drop(out)
        if self.res is not None:
            out = out + self.res(res_in.float())
        return out


class EdgeTransformerBlock(MessagePassing):
    """Sparse neighborhood multi-head attention implemented with PyG MessagePassing."""
    def __init__(self, d_model: int, n_heads: int = 4, attn_drop: float = 0.10, ff_mult: int = 4, ff_drop: float = 0.10):
        super().__init__(aggr="add", node_dim=0)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.dh = self.d_model // self.n_heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(attn_drop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_drop),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(ff_drop),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x1 = self.ln1(x)
        n, d = x1.size()
        qkv = self.qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(n, self.n_heads, self.dh)
        k = k.view(n, self.n_heads, self.dh)
        v = v.view(n, self.n_heads, self.dh)
        out = self.propagate(edge_index, size=(n, n), q=q, k=k, v=v)
        out = out.reshape(n, d)
        x = x + self.proj_drop(self.proj(out))
        x = x + self.ff(self.ln2(x))
        return x

    def message(self, q_i: torch.Tensor, k_j: torch.Tensor, v_j: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        score = (q_i * k_j).sum(-1) * self.scale
        attn = softmax(score, index)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)


class GWCNTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        scales: int = 6,
        K: int = 5,
        p_drop: float = 0.20,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 2,
        attn_drop: float = 0.10,
        ff_mult: int = 4,
        ff_drop: float = 0.10,
    ):
        super().__init__()
        self.g1 = GraphWaveletConv(in_channels, d_model, scales=scales, K=K, p_drop=p_drop)
        self.g2 = GraphWaveletConv(d_model, d_model, scales=scales, K=K, p_drop=p_drop)
        self.blocks = nn.ModuleList([
            EdgeTransformerBlock(d_model, n_heads=n_heads, attn_drop=attn_drop, ff_mult=ff_mult, ff_drop=ff_drop)
            for _ in range(int(n_layers))
        ])
        self.pos_mlp = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        self._last_feat: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, L: torch.Tensor, edge_index: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.gelu(self.g1(x, L))
        x = F.gelu(self.g2(x, L))
        if pos is not None:
            x = x + self.pos_mlp(pos)
        for blk in self.blocks:
            x = blk(x, edge_index)
        feat = F.normalize(x, dim=-1)
        self._last_feat = feat
        return self.head(feat)


# =================== Metrics, loss, and EMA ===================


@dataclass
class EvalResult:
    oa: float
    aa: float
    kappa: float
    per_class: np.ndarray
    conf_mat: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])
            else:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> EvalResult:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    oa = float((y_true == y_pred).mean())
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    aa = float(per_class.mean())
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=np.arange(num_classes)))
    return EvalResult(oa, aa, kappa, per_class, cm, y_true, y_pred)


def class_group_mean(per_class: np.ndarray, class_ids: Sequence[int]) -> float:
    ids = [i for i in class_ids if 0 <= i < len(per_class)]
    return float(np.mean(per_class[ids])) if ids else 0.0


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def move_data_to_device(data: Data, device: torch.device) -> Data:
    data = copy.copy(data)
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    data.edge_index = data.edge_index.to(device)
    data.pos = data.pos.to(device)
    data.L = data.L.to(device)
    return data


def weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float,
    focus_boost: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none", label_smoothing=label_smoothing)
    if focus_boost > 0:
        focus_mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls in FOCUS_IDS:
            focus_mask = focus_mask | (labels == cls)
        mult = torch.ones_like(ce)
        mult[focus_mask] = mult[focus_mask] * (1.0 + focus_boost)
        ce = ce * mult
    return ce.mean()


@torch.no_grad()
def evaluate_model(model: nn.Module, data: Data, device: torch.device, num_classes: int) -> EvalResult:
    model.eval()
    data_dev = move_data_to_device(data, device)
    logits = model(data_dev.x, data_dev.L, data_dev.edge_index, data_dev.pos)
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = data.y.detach().cpu().numpy()
    return compute_metrics(y_true, pred, num_classes)


@torch.no_grad()
def predict_model(model: nn.Module, data: Data, device: torch.device) -> np.ndarray:
    model.eval()
    data_dev = move_data_to_device(data, device)
    logits = model(data_dev.x, data_dev.L, data_dev.edge_index, data_dev.pos)
    return logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)


def make_model(args, in_channels: int, num_classes: int) -> GWCNTransformer:
    return GWCNTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        scales=args.scales,
        K=args.cheby_k,
        p_drop=args.p_drop,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        attn_drop=args.attn_drop,
        ff_mult=args.ff_mult,
        ff_drop=args.ff_drop,
    )


def train_graph_model(
    args,
    train_data: Data,
    val_data: Optional[Data],
    num_classes: int,
    in_channels: int,
    device: torch.device,
    seed: int,
    epochs: int,
) -> Tuple[Dict[str, torch.Tensor], int, EvalResult]:
    set_seed(seed)
    model = make_model(args, in_channels, num_classes).to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    train_dev = move_data_to_device(train_data, device)
    class_weights = compute_class_weights(train_data.y.numpy(), num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=args.lr * 0.05)

    best_score = -1.0
    best_epoch = epochs
    best_state: Optional[Dict[str, torch.Tensor]] = None
    empty_val = EvalResult(0, 0, 0, np.zeros(num_classes), np.zeros((num_classes, num_classes)), np.array([]), np.array([]))
    best_val = empty_val

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_dev.x, train_dev.L, train_dev.edge_index, train_dev.pos)
        loss = weighted_ce_loss(
            logits,
            train_dev.y,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            focus_boost=args.focus_boost,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        if val_data is not None:
            eval_model = ema.shadow if ema is not None else model
            val_res = evaluate_model(eval_model, val_data, device, num_classes)
            focus4 = class_group_mean(val_res.per_class, FOCUS_IDS)
            urban7 = class_group_mean(val_res.per_class, URBAN_IDS)
            score = val_res.oa + 0.55 * val_res.kappa + 0.22 * val_res.aa + 0.20 * focus4 + 0.08 * urban7
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val = val_res
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else:
            eval_model = ema.shadow if ema is not None else model
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            best_epoch = epoch

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            if val_data is not None:
                print(
                    f"epoch {epoch:03d}/{epochs} | loss={loss.item():.4f} | "
                    f"val_OA={val_res.oa:.4f} | val_AA={val_res.aa:.4f} | "
                    f"val_Kappa={val_res.kappa:.4f} | best_epoch={best_epoch} | lr={lr_now:.2e}"
                )
            else:
                print(f"epoch {epoch:03d}/{epochs} | loss={loss.item():.4f} | lr={lr_now:.2e}")

    if best_state is None:
        eval_model = ema.shadow if ema is not None else model
        best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    return best_state, best_epoch, best_val


# =================== Visualization ===================


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


def build_result_image(shape_hw: Tuple[int, int], coords: np.ndarray, zero_based_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    if len(coords) > 0:
        out[coords[:, 0], coords[:, 1]] = zero_based_pred.astype(np.uint8) + 1
    return out


def save_rgb_map(label_img: np.ndarray, title: str, path: str, with_legend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.imshow(label_map_to_rgb(label_img), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    if with_legend:
        handles = [Patch(facecolor=colors[i], edgecolor="k", label=categories[i]) for i in range(len(categories) - 1, -1, -1)]
        labels = [categories[i] for i in range(len(categories) - 1, -1, -1)]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
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


def plot_per_class_bar(per_class: np.ndarray, out_png: str, title: str) -> None:
    x = np.arange(1, len(per_class) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, per_class, color=colors[: len(per_class)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories[: len(per_class)], rotation=45, ha="right")
    for xi, yi in zip(x, per_class):
        ax.text(xi, min(0.985, yi + 0.015), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_mean_std(mean_pc: np.ndarray, std_pc: np.ndarray, out_png: str) -> None:
    x = np.arange(1, len(mean_pc) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, mean_pc, color=colors[: len(mean_pc)])
    ax.errorbar(x, mean_pc, yerr=std_pc, fmt="none", capsize=4, ecolor="black", elinewidth=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class Accuracy Mean ± Std over 5 Runs")
    ax.set_xticks(x)
    ax.set_xticklabels(categories[: len(mean_pc)], rotation=45, ha="right")
    for xi, m, s in zip(x, mean_pc, std_pc):
        ax.text(xi, min(0.985, m + s + 0.015), f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================== One run and summary ===================


def run_one(args, data_dict: Dict[str, np.ndarray], run_id: int, seed: int, device: torch.device) -> Dict:
    hsi = data_dict["hsi"]
    lidar = data_dict["lidar"]
    gt = data_dict["gt"]
    tr_label = data_dict["tr"]
    ts_label = data_dict["ts"]
    num_classes = int(max(gt.max(), tr_label.max(), ts_label.max()))

    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"[RUN {run_id}/{args.runs}] seed={seed}")

    best_epochs: List[int] = []
    cv_metrics: List[EvalResult] = []

    if args.cv_folds > 0 and args.cv_val_ratio > 0:
        print("[STAGE A] Spatial CV inside official training labels only")
        for fold in range(args.cv_folds):
            fold_seed = seed + 100 * fold
            train_mask, val_mask = classwise_spatial_train_val_split(
                tr_label,
                val_ratio=args.cv_val_ratio,
                block_size=args.block_size,
                seed=fold_seed,
                min_train_per_class=args.min_train_per_class,
                min_val_per_class=args.min_val_per_class,
            )
            tr_coords, tr_labels = mask_to_coords_labels(train_mask)
            va_coords, va_labels = mask_to_coords_labels(val_mask)
            print("\n" + "-" * 88)
            print(f"[CV {fold + 1}/{args.cv_folds}] train={len(tr_labels)} | val={len(va_labels)}")

            feat_cube, _ = build_feature_cube_train_only(hsi, lidar, train_mask, pca_dim=args.pca_dim)
            train_graph, _, _ = build_graph_data(feat_cube, train_mask, args.graph_k, args.tau_f, args.tau_p, num_classes)
            val_graph, _, _ = build_graph_data(feat_cube, val_mask, args.graph_k, args.tau_f, args.tau_p, num_classes)
            in_channels = train_graph.x.shape[1]

            _state, best_epoch, best_val = train_graph_model(
                args,
                train_graph,
                val_graph,
                num_classes,
                in_channels,
                device,
                seed=fold_seed,
                epochs=args.cv_epochs,
            )
            best_epochs.append(best_epoch)
            cv_metrics.append(best_val)
            print(
                f"[CV BEST] epoch={best_epoch} | OA={best_val.oa:.4f} | "
                f"AA={best_val.aa:.4f} | Kappa={best_val.kappa:.4f} | "
                f"focus4={class_group_mean(best_val.per_class, FOCUS_IDS):.4f}"
            )
        selected_epoch = int(round(np.median(best_epochs) * args.final_epoch_scale))
        selected_epoch = max(args.epochs_min, min(selected_epoch, args.epochs_max))
    else:
        selected_epoch = args.cv_epochs

    print("\n" + "-" * 88)
    print(f"[STAGE B] Retrain on all official training labels | epochs={selected_epoch}")

    full_train_mask = tr_label.copy()
    feat_cube, pca_model = build_feature_cube_train_only(hsi, lidar, full_train_mask, pca_dim=args.pca_dim)
    train_graph, _, train_labels = build_graph_data(feat_cube, full_train_mask, args.graph_k, args.tau_f, args.tau_p, num_classes)
    test_graph, test_coords, test_labels = build_graph_data(feat_cube, ts_label, args.graph_k, args.tau_f, args.tau_p, num_classes)
    in_channels = train_graph.x.shape[1]
    if pca_model is not None:
        print(f"[INFO] train-only PCA dims={pca_model.n_components_}")
    print(f"[INFO] full train nodes={len(train_labels)} | test nodes={len(test_labels)} | in_channels={in_channels}")

    final_state, _epoch, _ = train_graph_model(
        args,
        train_graph,
        None,
        num_classes,
        in_channels,
        device,
        seed=seed + 1000,
        epochs=selected_epoch,
    )

    model = make_model(args, in_channels, num_classes).to(device)
    model.load_state_dict(final_state, strict=True)
    test_res = evaluate_model(model, test_graph, device, num_classes)
    test_pred = predict_model(model, test_graph, device)

    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")
    print(f"[TEST] focus4={class_group_mean(test_res.per_class, FOCUS_IDS):.4f} | urban7={class_group_mean(test_res.per_class, URBAN_IDS):.4f}")
    print("[TEST] Per-class accuracy")
    for i, acc in enumerate(test_res.per_class):
        print(f"  {i + 1:02d}. {categories[i]:<18s} : {acc:.4f}")

    pred_test_img = build_result_image(gt.shape, test_coords, test_pred)
    gt_test_img = ts_label.astype(np.uint8)
    save_rgb_map(gt_test_img, "Ground Truth Land-cover Map (Official Test Pixels)", os.path.join(run_dir, "ground_truth_test_labels.png"), with_legend=True)
    save_rgb_map(pred_test_img, "Predicted Land-cover Classification Map (Official Test Pixels)", os.path.join(run_dir, "prediction_test_labels.png"), with_legend=True)
    save_paper_map_pair(gt_test_img, pred_test_img, os.path.join(run_dir, "paper_landcover_map_test_labels.png"), run_id, args.method_name)
    plot_per_class_bar(test_res.per_class, os.path.join(run_dir, "per_class_accuracy.png"), f"Per-class Accuracy (Run {run_id})")

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "OA": test_res.oa,
                "AA": test_res.aa,
                "Kappa": test_res.kappa,
                "selected_epoch": selected_epoch,
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
        "selected_epoch": selected_epoch,
        "run_dir": run_dir,
    }


def summarize_runs(results: List[Dict], output_dir: str) -> None:
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    oa = np.array([r["oa"] for r in results], dtype=np.float64)
    aa = np.array([r["aa"] for r in results], dtype=np.float64)
    kappa = np.array([r["kappa"] for r in results], dtype=np.float64)
    per = np.stack([r["per_class"] for r in results], axis=0)
    mean_pc = per.mean(axis=0)
    std_pc = per.std(axis=0, ddof=0)

    summary = {
        "runs": len(results),
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

    with open(os.path.join(summary_dir, "per_class_mean_std.csv"), "w", encoding="utf-8") as f:
        f.write("class_id,class_name,mean,std\n")
        for i in range(len(mean_pc)):
            f.write(f"{i + 1},{categories[i]},{mean_pc[i]:.6f},{std_pc[i]:.6f}\n")

    plot_per_class_mean_std(mean_pc, std_pc, os.path.join(summary_dir, "per_class_mean_std.png"))

    best_idx = int(np.argmax(oa))
    best_run_dir = results[best_idx]["run_dir"]
    src = os.path.join(best_run_dir, "paper_landcover_map_test_labels.png")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(summary_dir, "best_run_paper_landcover_map_test_labels.png"))

    print("\n" + "#" * 88)
    print(f"Final summary across {len(results)} runs")
    print(f"OA     : {summary['OA_mean']:.4f} ± {summary['OA_std']:.4f}")
    print(f"AA     : {summary['AA_mean']:.4f} ± {summary['AA_std']:.4f}")
    print(f"Kappa  : {summary['Kappa_mean']:.4f} ± {summary['Kappa_std']:.4f}")
    print("Per-class accuracy mean ± std")
    for i in range(len(mean_pc)):
        print(f"  {i + 1:02d}. {categories[i]:<18s} : {mean_pc[i]:.4f} ± {std_pc[i]:.4f}")
    print(f"[INFO] Summary saved to: {summary_dir}")


# =================== Main ===================


def main() -> None:
    parser = argparse.ArgumentParser("Fair GS-WCN-SOA-Transformer for Houston 2013")
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")
    parser.add_argument("--output-dir", type=str, default="gs_wcn_soa_transformer_houston_fair_runs")
    parser.add_argument("--method-name", type=str, default="GS-WCN-SOA-Transformer")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=BASE_SEED)

    # Fair preprocessing and validation protocol.
    parser.add_argument("--pca-dim", type=int, default=0, help="0 keeps raw standardized HSI bands; >0 uses train-only HSI PCA plus LiDAR.")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-val-ratio", type=float, default=0.20)
    parser.add_argument("--cv-epochs", type=int, default=180)
    parser.add_argument("--final-epoch-scale", type=float, default=1.12)
    parser.add_argument("--epochs-min", type=int, default=50)
    parser.add_argument("--epochs-max", type=int, default=260)
    parser.add_argument("--block-size", type=int, default=18)
    parser.add_argument("--min-train-per-class", type=int, default=16)
    parser.add_argument("--min-val-per-class", type=int, default=8)

    # Graph construction.
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--tau-f", type=float, default=0.0, help="<=0 uses graph-wise median feature distance.")
    parser.add_argument("--tau-p", type=float, default=0.0, help="<=0 uses graph-wise median spatial distance.")

    # Model hyperparameters. Keep moderate defaults so CPU/GPU runs remain feasible.
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--scales", type=int, default=6)
    parser.add_argument("--cheby-k", type=int, default=5)
    parser.add_argument("--p-drop", type=float, default=0.20)
    parser.add_argument("--attn-drop", type=float, default=0.10)
    parser.add_argument("--ff-drop", type=float, default=0.10)
    parser.add_argument("--ff-mult", type=int, default=4)

    # Optimization.
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--focus-boost", type=float, default=0.0, help="Optional focus multiplier for Commercial/Road/Highway/Parking lot 1.")
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--log-interval", type=int, default=10)

    # Runtime.
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--tta", action="store_true", help="Reserved for interface compatibility. This strict graph version does not use test-time graph augmentation.")
    args = parser.parse_args()

    if args.tau_f <= 0:
        args.tau_f = None
    if args.tau_p <= 0:
        args.tau_p = None

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_houston_2013(args.data_root)
    print(f"[INFO] data-root={args.data_root}")
    print(f"[INFO] HSI shape={data['hsi'].shape}, LiDAR shape={data['lidar'].shape}")
    print(f"[INFO] total train labels={int((data['tr'] > 0).sum())}, total test labels={int((data['ts'] > 0).sum())}")
    print(f"[INFO] device={device}")
    print("[INFO] fairness: train-only preprocessing, spatial-CV model selection, full official-train retraining, TSLabel used only for final testing.")

    results: List[Dict] = []
    for run_id in range(1, args.runs + 1):
        run_seed = args.seed + 10000 * (run_id - 1)
        out = run_one(args, data, run_id, run_seed, device)
        results.append(out)

    summarize_runs(results, args.output_dir)


if __name__ == "__main__":
    main()

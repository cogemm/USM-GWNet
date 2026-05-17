import argparse
import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
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

CLASS_NAMES = categories
CLASS_COLORS = colors

# zero-based ids
URBAN_IDS = [6, 7, 8, 9, 10, 11, 12]
FOCUS_IDS = [7, 8, 9, 11]  # Commercial, Road, Highway, Parking lot 1


# =========================
# Utilities
# =========================


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
        raise ValueError("No numeric ndarray found in .mat file")
    candidates.sort(key=lambda kv: (kv[1].ndim >= 2, kv[1].size), reverse=True)
    return np.asarray(candidates[0][1])



def load_mat_array(path: str) -> np.ndarray:
    return np.asarray(find_first_numeric_array(sio.loadmat(path)))



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
        raise FileNotFoundError(f"Missing files: {missing}")

    hsi = ensure_hwc(load_mat_array(paths["hsi"])).astype(np.float32)
    lidar = ensure_hwc(load_mat_array(paths["lidar"])).astype(np.float32)
    gt = load_mat_array(paths["gt"]).squeeze().astype(np.int64)
    tr = load_mat_array(paths["tr"]).squeeze().astype(np.int64)
    ts = load_mat_array(paths["ts"]).squeeze().astype(np.int64)

    if hsi.shape[:2] != lidar.shape[:2] or hsi.shape[:2] != gt.shape:
        raise ValueError(f"Spatial mismatch: HSI={hsi.shape}, LiDAR={lidar.shape}, gt={gt.shape}")
    return {"hsi": hsi, "lidar": lidar, "gt": gt, "tr": tr, "ts": ts}



def fit_standardize_train_only(x: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pixels = x[train_mask > 0]
    mean = train_pixels.mean(axis=0, keepdims=True)
    std = train_pixels.std(axis=0, keepdims=True) + 1e-6
    out = (x - mean) / std
    return out.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)



def fit_pca_train_only(hsi_norm: np.ndarray, train_mask: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    train_pixels = hsi_norm[train_mask > 0]
    n_components = min(n_components, train_pixels.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
    pca.fit(train_pixels)
    h, w, c = hsi_norm.shape
    flat = hsi_norm.reshape(-1, c)
    out = pca.transform(flat).reshape(h, w, n_components)
    return out.astype(np.float32), pca



def mask_to_coords_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask > 0)
    labels = mask[mask > 0] - 1
    return coords.astype(np.int64), labels.astype(np.int64)



def stratified_train_val_split(tr_label: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    coords, labels = mask_to_coords_labels(tr_label)
    if val_ratio <= 0.0:
        return tr_label.copy(), np.zeros_like(tr_label, dtype=np.int64)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(splitter.split(coords, labels))
    train_mask = np.zeros_like(tr_label, dtype=np.int64)
    val_mask = np.zeros_like(tr_label, dtype=np.int64)
    train_mask[coords[tr_idx, 0], coords[tr_idx, 1]] = labels[tr_idx] + 1
    val_mask[coords[va_idx, 0], coords[va_idx, 1]] = labels[va_idx] + 1
    return train_mask, val_mask



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
                    perm = rng.permutation(cls_all)
                    chosen = perm[:fallback_target].tolist()

            chosen_val = np.array(sorted(set(chosen)), dtype=np.int64)
            train_flags = np.ones(n_cls, dtype=bool)
            if len(chosen_val) > 0:
                pos = {idx: i for i, idx in enumerate(cls_all.tolist())}
                for idx in chosen_val.tolist():
                    if idx in pos:
                        train_flags[pos[idx]] = False
            chosen_train = cls_all[train_flags]

            if len(chosen_train) < min_train_per_class or len(chosen_val) < min_val_per_class:
                perm = rng.permutation(cls_all)
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                chosen_val = perm[:fallback_target]
                chosen_train = perm[fallback_target:]

        if len(chosen_train) == 0:
            chosen_train = cls_all
            chosen_val = np.array([], dtype=np.int64)

        train_mask[coords[chosen_train, 0], coords[chosen_train, 1]] = cls + 1
        if len(chosen_val) > 0:
            val_mask[coords[chosen_val, 0], coords[chosen_val, 1]] = cls + 1

    if (val_mask > 0).sum() == 0:
        return stratified_train_val_split(tr_label, val_ratio, seed)
    return train_mask, val_mask



def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)



def compute_subset_class_weights(labels: np.ndarray, class_ids: Sequence[int]) -> torch.Tensor:
    subset = np.asarray(class_ids, dtype=np.int64)
    remap = {cls: i for i, cls in enumerate(subset.tolist())}
    sub_labels = np.array([remap[y] for y in labels if y in remap], dtype=np.int64)
    if len(sub_labels) == 0:
        return torch.ones(len(subset), dtype=torch.float32)
    counts = np.bincount(sub_labels, minlength=len(subset)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)



def random_patch_augment(hsi: np.ndarray, pca: np.ndarray, lidar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.random.randint(0, 4)
    if k:
        hsi = np.rot90(hsi, k=k, axes=(0, 1)).copy()
        pca = np.rot90(pca, k=k, axes=(0, 1)).copy()
        lidar = np.rot90(lidar, k=k, axes=(0, 1)).copy()
    if np.random.rand() < 0.5:
        hsi = np.flip(hsi, axis=1).copy()
        pca = np.flip(pca, axis=1).copy()
        lidar = np.flip(lidar, axis=1).copy()
    if np.random.rand() < 0.5:
        hsi = np.flip(hsi, axis=0).copy()
        pca = np.flip(pca, axis=0).copy()
        lidar = np.flip(lidar, axis=0).copy()
    if np.random.rand() < 0.50:
        hsi = hsi + np.random.normal(0.0, 0.008, size=hsi.shape).astype(np.float32)
    if np.random.rand() < 0.25:
        lidar = lidar + np.random.normal(0.0, 0.008, size=lidar.shape).astype(np.float32)
    return hsi, pca, lidar


# =========================
# Dataset
# =========================


class HoustonPatchDataset(Dataset):
    def __init__(
        self,
        hsi: np.ndarray,
        pca_hsi: np.ndarray,
        lidar: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.hsi = hsi
        self.pca_hsi = pca_hsi
        self.lidar = lidar
        self.coords = coords
        self.labels = labels
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.augment = augment

        self.hsi_pad = np.pad(hsi, ((self.radius, self.radius), (self.radius, self.radius), (0, 0)), mode="reflect")
        self.pca_pad = np.pad(pca_hsi, ((self.radius, self.radius), (self.radius, self.radius), (0, 0)), mode="reflect")
        self.lidar_pad = np.pad(lidar, ((self.radius, self.radius), (self.radius, self.radius), (0, 0)), mode="reflect")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        y, x = self.coords[idx]
        ps = self.patch_size

        hsi_patch = self.hsi_pad[y:y + ps, x:x + ps, :].copy()
        pca_patch = self.pca_pad[y:y + ps, x:x + ps, :].copy()
        lidar_patch = self.lidar_pad[y:y + ps, x:x + ps, :].copy()

        if self.augment:
            hsi_patch, pca_patch, lidar_patch = random_patch_augment(hsi_patch, pca_patch, lidar_patch)

        hsi_patch = torch.from_numpy(np.transpose(hsi_patch, (2, 0, 1))).float()
        pca_patch = torch.from_numpy(np.transpose(pca_patch, (2, 0, 1))).float()
        lidar_patch = torch.from_numpy(np.transpose(lidar_patch, (2, 0, 1))).float()
        label = int(self.labels[idx])
        coord = torch.tensor([y, x], dtype=torch.long)
        return hsi_patch, pca_patch, lidar_patch, label, coord


# =========================
# Model blocks
# =========================


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k=3, s=1, p=1, groups: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        rnd.floor_()
        return x * rnd / keep_prob


class SpectralSelectionConv(nn.Module):
    def __init__(self, channels: int, groups: int = 8, reduction: int = 4):
        super().__init__()
        groups = groups if channels % groups == 0 else 1
        hidden = max(channels // reduction, 16)
        self.pre = ConvBNAct(channels, channels, k=1, p=0, groups=groups)
        self.dw = ConvBNAct(channels, channels, k=3, p=1, groups=channels)
        self.post = ConvBNAct(channels, channels, k=1, p=0, groups=groups, act=False)
        self.center_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        center = x[:, :, h // 2, w // 2]
        c_gate = self.center_mlp(center).view(b, c, 1, 1)
        s_gate = self.spatial_gate(x.mean(dim=1, keepdim=True))
        y = self.post(self.dw(self.pre(x)))
        y = y * c_gate * s_gate
        return self.act(self.norm(x + y))


class CenterGuidedScanBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.row_conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.col_conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.center_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.drop_path = DropPath(dropout)

    def scan_seq(self, seq: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        fwd = conv(seq)
        bwd = torch.flip(conv(torch.flip(seq, dims=[-1])), dims=[-1])
        return 0.5 * (fwd + bwd)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x0 = self.norm(x)
        row_seq = x0.reshape(b, c, h * w)
        col_seq = x0.permute(0, 1, 3, 2).contiguous().reshape(b, c, h * w)
        row_feat = self.scan_seq(row_seq, self.row_conv).reshape(b, c, h, w)
        col_feat = self.scan_seq(col_seq, self.col_conv).reshape(b, c, w, h).permute(0, 1, 3, 2).contiguous()
        center = x0[:, :, h // 2, w // 2]
        center_gate = self.center_gate(center).view(b, c, 1, 1)
        ctx = self.ctx_gate(context)
        y = self.mix((row_feat + col_feat) * center_gate * ctx)
        return x + self.drop_path(y)


class MultiScaleWaveletOp(nn.Module):
    def __init__(self, patch_size: int, scales: Sequence[float]):
        super().__init__()
        n = patch_size * patch_size
        coords = np.asarray([(i, j) for i in range(patch_size) for j in range(patch_size)], dtype=np.float32)
        a = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dy = abs(coords[i, 0] - coords[j, 0])
                dx = abs(coords[i, 1] - coords[j, 1])
                if max(dy, dx) == 1:
                    dist2 = dy * dy + dx * dx
                    a[i, j] = math.exp(-dist2 / 1.5)
        d = np.diag(a.sum(axis=1) + 1e-8)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(d)))
        l = np.eye(n, dtype=np.float32) - d_inv_sqrt @ a @ d_inv_sqrt
        lam, u = np.linalg.eigh(l)
        lam = np.clip(lam, 0.0, None)
        self.register_buffer("low_pass", torch.tensor(u @ np.diag(np.exp(-0.8 * lam)) @ u.T, dtype=torch.float32))
        for idx, s in enumerate(scales):
            band = s * lam * np.exp(-s * lam)
            self.register_buffer(f"wavelet_{idx}", torch.tensor(u @ np.diag(band) @ u.T, dtype=torch.float32))
        self.num_scales = len(scales)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = [torch.einsum("nm,bmc->bnc", self.low_pass, x)]
        for i in range(self.num_scales):
            psi = getattr(self, f"wavelet_{i}")
            outs.append(torch.einsum("nm,bmc->bnc", psi, x))
        return outs


class GraphWaveletFusion(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, patch_size: int, scales: Sequence[float]):
        super().__init__()
        self.proj = ConvBNAct(in_ch, hidden_ch, k=1, p=0)
        self.op = MultiScaleWaveletOp(patch_size, scales)
        self.linears = nn.ModuleList([nn.Linear(hidden_ch, hidden_ch) for _ in range(len(scales) + 1)])
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.GELU(),
            nn.Linear(hidden_ch, len(scales) + 1),
        )
        self.center_idx = (patch_size * patch_size) // 2
        self.out = nn.Sequential(
            ConvBNAct(hidden_ch, hidden_ch, k=1, p=0),
            ConvBNAct(hidden_ch, hidden_ch, k=3, p=1, groups=hidden_ch),
            ConvBNAct(hidden_ch, hidden_ch, k=1, p=0, act=False),
        )
        self.norm = nn.BatchNorm2d(hidden_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        nodes = x.flatten(2).transpose(1, 2).contiguous()
        waves = self.op(nodes)
        center = nodes[:, self.center_idx, :]
        attn = torch.softmax(self.scale_gate(center), dim=-1)
        acc = 0.0
        for i, (wave, lin) in enumerate(zip(waves, self.linears)):
            acc = acc + attn[:, i].view(b, 1, 1) * lin(wave)
        acc = acc.transpose(1, 2).reshape(b, c, h, w)
        y = self.out(acc)
        return self.act(self.norm(x + y))


class LiDARReliefStem(nn.Module):
    def __init__(self, out_ch: int):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float32),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=torch.float32),
        )
        self.register_buffer(
            "lap",
            torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float32),
        )
        self.net = nn.Sequential(
            ConvBNAct(4, out_ch // 2, k=3, p=1),
            ConvBNAct(out_ch // 2, out_ch, k=3, p=1),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        dx = F.conv2d(lidar, self.sobel_x, padding=1)
        dy = F.conv2d(lidar, self.sobel_y, padding=1)
        lap = F.conv2d(lidar, self.lap, padding=1)
        feat = torch.cat([lidar, dx, dy, lap], dim=1)
        return self.net(feat)


class UrbanStripMixer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.pre = ConvBNAct(in_ch, out_ch, k=1, p=0)
        self.h_local = ConvBNAct(out_ch, out_ch, k=(1, 5), p=(0, 2), groups=out_ch)
        self.v_local = ConvBNAct(out_ch, out_ch, k=(5, 1), p=(2, 0), groups=out_ch)
        self.mix = ConvBNAct(out_ch, out_ch, k=1, p=0, act=False)
        self.center_gate = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        b, c, h, w = x.shape
        center = x[:, :, h // 2, w // 2]
        gate = self.center_gate(center).view(b, c, 1, 1)
        h_local = self.h_local(x)
        v_local = self.v_local(x)
        h_strip = x.mean(dim=2, keepdim=True).expand(-1, -1, h, w)
        v_strip = x.mean(dim=3, keepdim=True).expand(-1, -1, h, w)
        y = self.mix((h_local + v_local + 0.5 * h_strip + 0.5 * v_strip) * gate)
        return self.act(self.norm(x + self.drop_path(y)))


class HSIStem(nn.Module):
    def __init__(self, in_hsi: int, in_pca: int, base_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.raw_proj = ConvBNAct(in_hsi, base_ch, k=1, p=0)
        self.pca_proj = ConvBNAct(in_pca, base_ch, k=1, p=0)
        self.raw_block = SpectralSelectionConv(base_ch)
        self.scan = CenterGuidedScanBlock(base_ch, dropout=drop_path)

    def forward(self, raw_hsi: torch.Tensor, pca_hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.raw_block(self.raw_proj(raw_hsi))
        pca = self.scan(self.pca_proj(pca_hsi), raw)
        return raw, pca



class PlainHSIStem(nn.Module):
    """Ablation stem: removes spectral-selection and center-guided scan.

    It keeps comparable output dimensions, but uses only local CNN refinement.
    This variant tests whether the center-guided spectral sequence modeling is useful.
    """
    def __init__(self, in_hsi: int, in_pca: int, base_ch: int):
        super().__init__()
        self.raw = nn.Sequential(
            ConvBNAct(in_hsi, base_ch, k=1, p=0),
            ConvBNAct(base_ch, base_ch, k=3, p=1),
        )
        self.pca = nn.Sequential(
            ConvBNAct(in_pca, base_ch, k=1, p=0),
            ConvBNAct(base_ch, base_ch, k=3, p=1),
        )

    def forward(self, raw_hsi: torch.Tensor, pca_hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.raw(raw_hsi), self.pca(pca_hsi)


class PlainLiDARStem(nn.Module):
    """Ablation stem: uses raw LiDAR only, without Sobel/Laplacian relief channels."""
    def __init__(self, out_ch: int):
        super().__init__()
        mid = max(8, out_ch // 2)
        self.net = nn.Sequential(
            ConvBNAct(1, mid, k=3, p=1),
            ConvBNAct(mid, out_ch, k=3, p=1),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        return self.net(lidar)


class ZeroLiDARStem(nn.Module):
    """HSI-only ablation: removes LiDAR information while preserving tensor dimensions."""
    def __init__(self, out_ch: int):
        super().__init__()
        self.out_ch = int(out_ch)

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        b, _, h, w = lidar.shape
        return torch.zeros((b, self.out_ch, h, w), device=lidar.device, dtype=lidar.dtype)


class PlainGraphFusion(nn.Module):
    """Ablation block: replaces graph-wavelet filtering with local CNN mixing.

    This keeps a trainable feature branch with the same output dimension, so the ablation
    measures the contribution of the graph wavelet operator rather than merely reducing capacity.
    """
    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.proj = ConvBNAct(in_ch, hidden_ch, k=1, p=0)
        self.local = nn.Sequential(
            ConvBNAct(hidden_ch, hidden_ch, k=3, p=1, groups=hidden_ch),
            ConvBNAct(hidden_ch, hidden_ch, k=1, p=0, act=False),
        )
        self.norm = nn.BatchNorm2d(hidden_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.local(x)
        return self.act(self.norm(x + y))


class PlainUrbanMixer(nn.Module):
    """Ablation block: removes horizontal/vertical strip priors and uses isotropic CNN mixing."""
    def __init__(self, in_ch: int, out_ch: int, drop_path: float = 0.0):
        super().__init__()
        self.pre = ConvBNAct(in_ch, out_ch, k=1, p=0)
        self.local = nn.Sequential(
            ConvBNAct(out_ch, out_ch, k=3, p=1, groups=out_ch),
            ConvBNAct(out_ch, out_ch, k=1, p=0, act=False),
        )
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        y = self.local(x)
        return self.act(self.norm(x + self.drop_path(y)))


class SlimWaveletHoustonNet(nn.Module):
    """USM-GWNet with switchable ablation variants.

    Supported ablation names:
        full              : full model
        no_center         : remove spectral selection + center-guided scan
        no_relief         : replace Sobel/Laplacian LiDAR relief stem with plain LiDAR CNN
        no_lidar          : remove LiDAR information completely
        no_graph          : replace graph-wavelet branch with local CNN branch
        no_urban          : replace strip mixer with isotropic CNN mixer
        no_urban_loss     : architecture unchanged, but the runner disables urban auxiliary/focus losses
        no_ema            : runner disables EMA; architecture unchanged
    """
    def __init__(
        self,
        hsi_channels: int,
        pca_channels: int,
        num_classes: int,
        patch_size: int,
        base_ch: int,
        lidar_ch: int,
        graph_ch: int,
        urban_ch: int,
        dropout: float,
        drop_path: float,
        urban_alpha: float,
        ablation: str = "full",
    ):
        super().__init__()
        self.ablation = str(ablation)
        if self.ablation == "no_center":
            self.hsi_stem = PlainHSIStem(hsi_channels, pca_channels, base_ch)
        else:
            self.hsi_stem = HSIStem(hsi_channels, pca_channels, base_ch, drop_path=drop_path)

        if self.ablation == "no_relief":
            self.lidar_stem = PlainLiDARStem(lidar_ch)
        elif self.ablation == "no_lidar":
            self.lidar_stem = ZeroLiDARStem(lidar_ch)
        else:
            self.lidar_stem = LiDARReliefStem(lidar_ch)

        graph_in_ch = base_ch * 2 + lidar_ch
        if self.ablation == "no_graph":
            self.graph = PlainGraphFusion(graph_in_ch, graph_ch)
        else:
            self.graph = GraphWaveletFusion(graph_in_ch, graph_ch, patch_size, scales=[0.7, 1.4, 2.8, 4.0])

        urban_in_ch = graph_in_ch + graph_ch
        if self.ablation == "no_urban":
            self.urban_strip = PlainUrbanMixer(urban_in_ch, urban_ch, drop_path=drop_path)
        else:
            self.urban_strip = UrbanStripMixer(urban_in_ch, urban_ch, drop_path=drop_path)

        self.center_mlp = nn.Sequential(
            nn.Linear(graph_in_ch, base_ch),
            nn.GELU(),
            nn.LayerNorm(base_ch),
        )
        fusion_dim = base_ch + base_ch + lidar_ch + graph_ch + urban_ch + base_ch
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        urban_vec_dim = urban_ch + graph_ch + base_ch
        self.urban_head = nn.Sequential(
            nn.Linear(urban_vec_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, len(URBAN_IDS)),
        )
        self.urban_alpha = float(urban_alpha)
        self.register_buffer("urban_class_ids", torch.tensor(URBAN_IDS, dtype=torch.long))

    def forward(self, raw_hsi: torch.Tensor, pca_hsi: torch.Tensor, lidar: torch.Tensor, return_aux: bool = False):
        raw_feat, scan_feat = self.hsi_stem(raw_hsi, pca_hsi)
        lidar_feat = self.lidar_stem(lidar)
        graph_in = torch.cat([raw_feat, scan_feat, lidar_feat], dim=1)
        graph_feat = self.graph(graph_in)
        urban_feat = self.urban_strip(torch.cat([graph_in, graph_feat], dim=1))

        b, _, h, w = graph_in.shape
        center_vec = graph_in[:, :, h // 2, w // 2]
        center_feat = self.center_mlp(center_vec)

        pooled = torch.cat(
            [
                raw_feat.mean(dim=(2, 3)),
                scan_feat.mean(dim=(2, 3)),
                lidar_feat.mean(dim=(2, 3)),
                graph_feat.mean(dim=(2, 3)),
                urban_feat.mean(dim=(2, 3)),
                center_feat,
            ],
            dim=1,
        )
        logits = self.head(pooled)

        urban_vec = torch.cat(
            [
                urban_feat[:, :, h // 2, w // 2],
                graph_feat[:, :, h // 2, w // 2],
                center_feat,
            ],
            dim=1,
        )
        urban_logits = self.urban_head(urban_vec)

        fused_logits = logits.clone()
        if self.urban_alpha != 0.0:
            fused_logits[:, self.urban_class_ids] = fused_logits[:, self.urban_class_ids] + self.urban_alpha * urban_logits
        if return_aux:
            return fused_logits, urban_logits
        return fused_logits


# =========================
# Metrics, EMA, evaluation
# =========================


@dataclass
class EvalResult:
    oa: float
    aa: float
    kappa: float
    per_class_acc: np.ndarray
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
    oa = (y_true == y_pred).mean()
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    aa = per_class.mean()
    kappa = cohen_kappa_score(y_true, y_pred, labels=np.arange(num_classes))
    return EvalResult(float(oa), float(aa), float(kappa), per_class, cm, y_true, y_pred)



def class_group_mean(per_class_acc: np.ndarray, class_ids: Sequence[int]) -> float:
    ids = [i for i in class_ids if 0 <= i < len(per_class_acc)]
    if not ids:
        return 0.0
    return float(np.mean(per_class_acc[ids]))


@torch.no_grad()
def infer_logits(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    for raw_hsi, pca_hsi, lidar, labels, _coords in loader:
        raw_hsi = raw_hsi.to(device)
        pca_hsi = pca_hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        if not tta:
            logits = model(raw_hsi, pca_hsi, lidar)
        else:
            transforms = [
                lambda a, b, c: (a, b, c),
                lambda a, b, c: (torch.flip(a, dims=[-1]), torch.flip(b, dims=[-1]), torch.flip(c, dims=[-1])),
                lambda a, b, c: (torch.flip(a, dims=[-2]), torch.flip(b, dims=[-2]), torch.flip(c, dims=[-2])),
                lambda a, b, c: (torch.rot90(a, 1, dims=[-2, -1]), torch.rot90(b, 1, dims=[-2, -1]), torch.rot90(c, 1, dims=[-2, -1])),
                lambda a, b, c: (torch.rot90(a, 2, dims=[-2, -1]), torch.rot90(b, 2, dims=[-2, -1]), torch.rot90(c, 2, dims=[-2, -1])),
                lambda a, b, c: (torch.rot90(a, 3, dims=[-2, -1]), torch.rot90(b, 3, dims=[-2, -1]), torch.rot90(c, 3, dims=[-2, -1])),
            ]
            logits_sum = 0.0
            for fn in transforms:
                a, b, c = fn(raw_hsi, pca_hsi, lidar)
                logits_sum = logits_sum + model(a, b, c)
            logits = logits_sum / float(len(transforms))

        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    return torch.cat(logits_all, dim=0).numpy(), torch.cat(labels_all, dim=0).numpy()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int, tta: bool = False) -> EvalResult:
    logits, labels = infer_logits(model, loader, device, tta=tta)
    preds = logits.argmax(axis=1)
    return compute_metrics(labels, preds, num_classes)


@torch.no_grad()
def evaluate_ensemble(
    state_list: Sequence[Dict[str, torch.Tensor]],
    model_builder,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    tta: bool,
) -> EvalResult:
    logits_accum = None
    labels_ref = None
    for state in state_list:
        model = model_builder().to(device)
        model.load_state_dict(state)
        logits, labels = infer_logits(model, loader, device, tta=tta)
        if logits_accum is None:
            logits_accum = logits.astype(np.float64)
            labels_ref = labels
        else:
            logits_accum += logits.astype(np.float64)
    logits_mean = logits_accum / float(len(state_list))
    preds = logits_mean.argmax(axis=1)
    return compute_metrics(labels_ref, preds, num_classes)


# =========================
# Training helpers
# =========================


@dataclass
class TrainArtifacts:
    best_epoch: int
    best_score: float
    best_val: EvalResult
    snapshot_states: List[Dict[str, torch.Tensor]]



def remap_subset_targets(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    lut = torch.full((int(max(class_ids)) + 1,), -1, dtype=torch.long, device=labels.device)
    for i, cls in enumerate(class_ids):
        lut[cls] = i
    return lut[labels]



def compute_train_loss(
    logits: torch.Tensor,
    urban_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    urban_class_weights: torch.Tensor,
    label_smoothing: float,
    urban_aux_w: float,
    focus_boost: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none", label_smoothing=label_smoothing)
    focus_mask = torch.zeros_like(labels, dtype=torch.bool)
    for cls in FOCUS_IDS:
        focus_mask = focus_mask | (labels == cls)
    sample_mult = torch.ones_like(ce)
    sample_mult[focus_mask] = sample_mult[focus_mask] * (1.0 + focus_boost)
    loss_main = (ce * sample_mult).mean()

    urban_mask = torch.zeros_like(labels, dtype=torch.bool)
    for cls in URBAN_IDS:
        urban_mask = urban_mask | (labels == cls)

    if urban_mask.any() and urban_aux_w > 0.0:
        urban_targets = remap_subset_targets(labels[urban_mask], URBAN_IDS)
        loss_urban = F.cross_entropy(
            urban_logits[urban_mask],
            urban_targets,
            weight=urban_class_weights,
            reduction="mean",
            label_smoothing=max(0.0, label_smoothing * 0.5),
        )
        return loss_main + urban_aux_w * loss_urban
    return loss_main



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: Optional[ModelEMA],
    class_weights: torch.Tensor,
    urban_class_weights: torch.Tensor,
    label_smoothing: float,
    urban_aux_w: float,
    focus_boost: float,
) -> float:
    model.train()
    running = 0.0
    total = 0
    for raw_hsi, pca_hsi, lidar, labels, _coords in loader:
        raw_hsi = raw_hsi.to(device)
        pca_hsi = pca_hsi.to(device)
        lidar = lidar.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, urban_logits = model(raw_hsi, pca_hsi, lidar, return_aux=True)
        loss = compute_train_loss(
            logits, urban_logits, labels,
            class_weights, urban_class_weights,
            label_smoothing=label_smoothing,
            urban_aux_w=urban_aux_w,
            focus_boost=focus_boost,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        bs = labels.size(0)
        running += loss.item() * bs
        total += bs
    return running / max(total, 1)



def get_loaders(
    hsi_norm: np.ndarray,
    pca_hsi: np.ndarray,
    lidar_norm: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, np.ndarray]:
    train_coords, train_labels = mask_to_coords_labels(train_mask)
    val_coords, val_labels = mask_to_coords_labels(val_mask)

    train_ds = HoustonPatchDataset(hsi_norm, pca_hsi, lidar_norm, train_coords, train_labels, patch_size, augment=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if len(val_labels) > 0:
        val_ds = HoustonPatchDataset(hsi_norm, pca_hsi, lidar_norm, val_coords, val_labels, patch_size, augment=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        val_loader = None
    return train_loader, val_loader, train_labels, val_labels



def make_builder(args, hsi_channels: int, pca_channels: int, num_classes: int):
    def _builder():
        return SlimWaveletHoustonNet(
            hsi_channels=hsi_channels,
            pca_channels=pca_channels,
            num_classes=num_classes,
            patch_size=args.patch_size,
            base_ch=args.base_ch,
            lidar_ch=args.lidar_ch,
            graph_ch=args.graph_ch,
            urban_ch=args.urban_ch,
            dropout=args.dropout,
            drop_path=args.drop_path,
            urban_alpha=args.urban_alpha,
            ablation=getattr(args, "ablation", "full"),
        )
    return _builder



def fit_split(
    args,
    hsi: np.ndarray,
    lidar: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    num_classes: int,
    device: torch.device,
    seed: int,
    epochs: int,
    save_snapshots: bool,
) -> TrainArtifacts:
    set_seed(seed)

    hsi_norm, _, _ = fit_standardize_train_only(hsi, train_mask)
    lidar_norm, _, _ = fit_standardize_train_only(lidar, train_mask)
    pca_hsi, pca_model = fit_pca_train_only(hsi_norm, train_mask, args.pca_dim)

    train_loader, val_loader, train_labels, _val_labels = get_loaders(
        hsi_norm, pca_hsi, lidar_norm,
        train_mask, val_mask,
        args.patch_size, args.batch_size, args.num_workers, device.type == "cuda"
    )

    builder = make_builder(args, hsi_norm.shape[-1], pca_hsi.shape[-1], num_classes)
    model = builder().to(device)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    class_weights = compute_class_weights(train_labels, num_classes=num_classes).to(device)
    urban_class_weights = compute_subset_class_weights(train_labels, URBAN_IDS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.05)

    snapshot_epochs = set()
    if save_snapshots:
        fracs = [0.78, 0.88, 0.94, 1.00]
        snapshot_epochs = {max(1, min(epochs, int(round(f * epochs)))) for f in fracs}

    best_score = -1.0
    best_epoch = -1
    best_val = EvalResult(0.0, 0.0, 0.0, np.zeros(num_classes), np.zeros((num_classes, num_classes), dtype=np.int64), np.array([]), np.array([]))
    snapshot_states: List[Dict[str, torch.Tensor]] = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, device, ema,
            class_weights=class_weights,
            urban_class_weights=urban_class_weights,
            label_smoothing=args.label_smoothing,
            urban_aux_w=args.urban_aux_w,
            focus_boost=args.focus_boost,
        )
        scheduler.step()

        if val_loader is not None:
            eval_model = ema.shadow if ema is not None else model
            val_res = evaluate(eval_model, val_loader, device, num_classes, tta=False)
            focus4 = class_group_mean(val_res.per_class_acc, FOCUS_IDS)
            urban7 = class_group_mean(val_res.per_class_acc, URBAN_IDS)
            score = val_res.oa + 0.55 * val_res.kappa + 0.22 * val_res.aa + 0.24 * focus4 + 0.10 * urban7
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val = val_res
        else:
            best_epoch = epochs
            best_score = 0.0
            focus4 = 0.0
            urban7 = 0.0

        if save_snapshots and epoch in snapshot_epochs:
            src = ema.shadow if ema is not None else model
            snapshot_states.append({k: v.detach().cpu().clone() for k, v in src.state_dict().items()})

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            if val_loader is not None:
                print(
                    f"epoch {epoch:03d}/{epochs} | loss={loss:.4f} | "
                    f"val_OA={best_val.oa if epoch == best_epoch else val_res.oa:.4f} | "
                    f"val_AA={best_val.aa if epoch == best_epoch else val_res.aa:.4f} | "
                    f"val_Kappa={best_val.kappa if epoch == best_epoch else val_res.kappa:.4f} | "
                    f"val_focus4={class_group_mean((best_val.per_class_acc if epoch == best_epoch else val_res.per_class_acc), FOCUS_IDS):.4f} | "
                    f"lr={lr_now:.2e}"
                )
            else:
                print(f"epoch {epoch:03d}/{epochs} | loss={loss:.4f} | lr={lr_now:.2e}")

    if save_snapshots and len(snapshot_states) == 0:
        src = ema.shadow if ema is not None else model
        snapshot_states.append({k: v.detach().cpu().clone() for k, v in src.state_dict().items()})

    return TrainArtifacts(best_epoch=best_epoch, best_score=best_score, best_val=best_val, snapshot_states=snapshot_states)


# =========================
# Main protocol
# =========================


def main() -> None:
    parser = argparse.ArgumentParser("SliMamba + multi-scale graph wavelet for Houston 2013, urban-focused CV-selected full-train ensemble")
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--cv-val-ratio", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-epochs", type=int, default=180)
    parser.add_argument("--final-epoch-scale", type=float, default=1.12)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lidar-ch", type=int, default=24)
    parser.add_argument("--graph-ch", type=int, default=36)
    parser.add_argument("--urban-ch", type=int, default=28)
    parser.add_argument("--dropout", type=float, default=0.16)
    parser.add_argument("--drop-path", type=float, default=0.03)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--block-size", type=int, default=18)
    parser.add_argument("--focus-boost", type=float, default=0.30)
    parser.add_argument("--urban-aux-w", type=float, default=0.35)
    parser.add_argument("--urban-alpha", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    data = load_houston_2013(args.data_root)
    hsi = data["hsi"]
    lidar = data["lidar"]
    tr_label = data["tr"]
    ts_label = data["ts"]
    gt = data["gt"]
    num_classes = int(max(gt.max(), tr_label.max(), ts_label.max()))

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    print(f"[INFO] data-root={args.data_root}")
    print(f"[INFO] HSI shape={hsi.shape}, LiDAR shape={lidar.shape}")
    print(f"[INFO] total train labels={int((tr_label > 0).sum())}, total test labels={int((ts_label > 0).sum())}")
    print(f"[INFO] device={device}")
    print(f"[INFO] protocol: spatial-CV={args.cv_folds} folds, then full-train ensemble={args.final_seeds} seeds")

    cv_best_epochs: List[int] = []
    cv_results: List[EvalResult] = []

    print("\n" + "=" * 80)
    print("[STAGE A] Spatial CV on official training labels only")
    for fold in range(args.cv_folds):
        fold_seed = args.seed + 100 * fold
        print("\n" + "-" * 80)
        print(f"[CV FOLD {fold + 1}/{args.cv_folds}] seed={fold_seed}")
        train_mask, val_mask = classwise_spatial_train_val_split(
            tr_label,
            val_ratio=args.cv_val_ratio,
            block_size=args.block_size,
            seed=fold_seed,
            min_train_per_class=16,
            min_val_per_class=8,
        )
        _tr_coords, tr_labels = mask_to_coords_labels(train_mask)
        _va_coords, va_labels = mask_to_coords_labels(val_mask)
        print(f"[INFO] split sizes -> train={len(tr_labels)} | val={len(va_labels)}")

        artifacts = fit_split(
            args=args,
            hsi=hsi,
            lidar=lidar,
            train_mask=train_mask,
            val_mask=val_mask,
            num_classes=num_classes,
            device=device,
            seed=fold_seed,
            epochs=args.cv_epochs,
            save_snapshots=False,
        )
        cv_best_epochs.append(artifacts.best_epoch)
        cv_results.append(artifacts.best_val)
        print(
            f"[CV BEST] epoch={artifacts.best_epoch} | OA={artifacts.best_val.oa:.4f} | "
            f"AA={artifacts.best_val.aa:.4f} | Kappa={artifacts.best_val.kappa:.4f} | "
            f"focus4={class_group_mean(artifacts.best_val.per_class_acc, FOCUS_IDS):.4f}"
        )

    cv_oa = np.array([m.oa for m in cv_results], dtype=np.float64)
    cv_aa = np.array([m.aa for m in cv_results], dtype=np.float64)
    cv_k = np.array([m.kappa for m in cv_results], dtype=np.float64)
    cv_focus = np.array([class_group_mean(m.per_class_acc, FOCUS_IDS) for m in cv_results], dtype=np.float64)
    selected_epoch = int(round(np.median(cv_best_epochs) * args.final_epoch_scale))
    selected_epoch = max(50, min(selected_epoch, int(round(args.cv_epochs * 1.40))))

    print("\n" + "-" * 80)
    print(f"[CV SUMMARY] best_epoch median={np.median(cv_best_epochs):.1f}, scaled final_epoch={selected_epoch}")
    print(f"[CV SUMMARY] OA={cv_oa.mean():.4f} ± {cv_oa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] AA={cv_aa.mean():.4f} ± {cv_aa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] Kappa={cv_k.mean():.4f} ± {cv_k.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] focus4={cv_focus.mean():.4f} ± {cv_focus.std(ddof=0):.4f}")

    print("\n" + "=" * 80)
    print("[STAGE B] Retrain on full official training set and ensemble snapshots")
    full_train_mask = tr_label.copy()
    empty_val = np.zeros_like(tr_label, dtype=np.int64)
    test_mask = ts_label.copy()

    hsi_full, _, _ = fit_standardize_train_only(hsi, full_train_mask)
    lidar_full, _, _ = fit_standardize_train_only(lidar, full_train_mask)
    pca_full, pca_model = fit_pca_train_only(hsi_full, full_train_mask, args.pca_dim)
    print(f"[INFO] full-train PCA dims={pca_model.n_components_}")

    test_coords, test_labels = mask_to_coords_labels(test_mask)
    test_ds = HoustonPatchDataset(hsi_full, pca_full, lidar_full, test_coords, test_labels, args.patch_size, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model_builder = make_builder(args, hsi_full.shape[-1], pca_full.shape[-1], num_classes)
    state_list: List[Dict[str, torch.Tensor]] = []

    for j in range(args.final_seeds):
        final_seed = args.seed + 1000 + j
        print("\n" + "-" * 80)
        print(f"[FINAL MODEL {j + 1}/{args.final_seeds}] seed={final_seed} | epochs={selected_epoch}")
        artifacts = fit_split(
            args=args,
            hsi=hsi,
            lidar=lidar,
            train_mask=full_train_mask,
            val_mask=empty_val,
            num_classes=num_classes,
            device=device,
            seed=final_seed,
            epochs=selected_epoch,
            save_snapshots=True,
        )
        state_list.extend(artifacts.snapshot_states)
        print(f"[INFO] saved snapshots={len(artifacts.snapshot_states)}")

    print("\n" + "=" * 80)
    print(f"[STAGE C] Ensemble inference with {len(state_list)} snapshot models")
    test_res = evaluate_ensemble(state_list, model_builder, test_loader, device, num_classes, tta=args.tta)
    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")
    print(f"[TEST] focus4={class_group_mean(test_res.per_class_acc, FOCUS_IDS):.4f} | urban7={class_group_mean(test_res.per_class_acc, URBAN_IDS):.4f}")
    print("[TEST] Per-class accuracy")
    for cls_idx in range(num_classes):
        name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"Class {cls_idx + 1}"
        print(f"  {cls_idx + 1:02d}. {name:<18s} : {test_res.per_class_acc[cls_idx]:.4f}")



# =========================
# Reporting, visualization, multi-run protocol
# =========================

import json
import shutil


def ensemble_predict(
    state_list: Sequence[Dict[str, torch.Tensor]],
    model_builder,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    logits_accum = None
    labels_ref = None
    for state in state_list:
        model = model_builder().to(device)
        model.load_state_dict(state, strict=True)
        logits, labels = infer_logits(model, loader, device, tta=tta)
        if logits_accum is None:
            logits_accum = logits.astype(np.float64)
            labels_ref = labels
        else:
            logits_accum += logits.astype(np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logits_mean = logits_accum / float(len(state_list))
    preds = logits_mean.argmax(axis=1).astype(np.int64)
    return preds, labels_ref.astype(np.int64)



def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert '#RRGGBB' into an RGB tuple."""
    h = hex_color.strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# Exact RGB palette used by every land-cover map.
# Index 0 is black background. Class 1..15 follows `colors` exactly.
PALETTE_RGB = np.array([[0, 0, 0]] + [list(hex_to_rgb(c)) for c in colors], dtype=np.uint8)


def label_map_to_rgb(label_img: np.ndarray) -> np.ndarray:
    """
    Convert an integer label map into a uint8 RGB image by direct table lookup.

    This function intentionally does NOT rely on matplotlib colormap interpolation.
    It guarantees that class 1 uses colors[0], class 2 uses colors[1], ..., class 15 uses colors[14].
    Background or unlabeled pixels must be 0 and are rendered black.
    """
    label_img = np.asarray(label_img)
    out = np.zeros(label_img.shape + (3,), dtype=np.uint8)
    valid = (label_img >= 0) & (label_img < len(PALETTE_RGB))
    out[valid] = PALETTE_RGB[label_img[valid].astype(np.int64)]
    return out


def save_rgb_map(
    label_img: np.ndarray,
    title: str,
    path: str,
    with_legend: bool = False,
    figsize: Tuple[float, float] = (18.0, 4.8),
    dpi: int = 300,
) -> None:
    """Save a land-cover map using direct RGB conversion from the exact user palette."""
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


def save_map_like_demo(
    img2d: np.ndarray,
    title: str,
    path: str,
    colors: Sequence[str] = colors,
    class_names: Sequence[str] = categories,
) -> None:
    # Keep the function name for compatibility, but force direct RGB mapping.
    save_rgb_map(img2d, title, path, with_legend=True, figsize=(18.0, 4.8), dpi=300)


def save_map_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_title: str,
    right_title: str,
    path: str,
    colors: Sequence[str] = colors,
    class_names: Sequence[str] = categories,
) -> None:
    """Save a two-panel GT/prediction map using the exact RGB palette."""
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 7.8))
    axes[0].imshow(label_map_to_rgb(left_img), interpolation="nearest")
    axes[0].set_title(left_title, fontsize=18)
    axes[0].axis("off")

    axes[1].imshow(label_map_to_rgb(right_img), interpolation="nearest")
    axes[1].set_title(right_title, fontsize=18)
    axes[1].axis("off")

    plt.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def save_paper_map_pair(
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    path: str,
    run_id: int,
    method_name: str = "MS-GWCN",
) -> None:
    """
    Save the paper-style land-cover classification map like the user's example.

    The output has two rows:
    1. Ground Truth
    2. Classification Result (Run k) by method_name

    There is no legend and no interpolated colormap. Pixels are drawn from PALETTE_RGB directly.
    """
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


def plot_per_class_bar(
    per_class: np.ndarray,
    class_names: Sequence[str],
    out_png: str,
    title: str,
    colors: Sequence[str] = colors,
) -> None:
    x = np.arange(1, len(per_class) + 1)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x, per_class, color=list(colors)[: len(per_class)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    for xi, yi in zip(x, per_class):
        ax.text(xi, min(0.985, yi + 0.015), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_mean_std(
    mean_pc: np.ndarray,
    std_pc: np.ndarray,
    class_names: Sequence[str],
    out_png: str,
    title: str,
    colors: Sequence[str] = colors,
) -> None:
    x = np.arange(1, len(mean_pc) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, mean_pc, color=list(colors)[: len(mean_pc)])
    ax.errorbar(x, mean_pc, yerr=std_pc, fmt="none", capsize=4, ecolor="black", elinewidth=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    for xi, m, s in zip(x, mean_pc, std_pc):
        ax.text(xi, min(0.985, m + s + 0.015), f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_result_image(shape_hw: Tuple[int, int], coords: np.ndarray, zero_based_pred: np.ndarray) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    if len(coords) > 0:
        out[coords[:, 0], coords[:, 1]] = zero_based_pred.astype(np.uint8) + 1
    return out


def save_run_artifacts(
    run_dir: str,
    gt: np.ndarray,
    ts_label: np.ndarray,
    full_coords: np.ndarray,
    full_pred: np.ndarray,
    test_coords: np.ndarray,
    test_pred: np.ndarray,
    per_class: np.ndarray,
    run_id: int,
    method_name: str,
) -> Dict[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    gt_path = os.path.join(run_dir, "ground_truth_all_labels.png")
    gt_test_path = os.path.join(run_dir, "ground_truth_test_labels.png")
    pred_full_path = os.path.join(run_dir, "prediction_all_labels.png")
    pred_test_path = os.path.join(run_dir, "prediction_test_labels.png")
    pair_full_path = os.path.join(run_dir, "map_pair_all_labels.png")
    pair_test_path = os.path.join(run_dir, "map_pair_test_labels.png")
    paper_pair_full_path = os.path.join(run_dir, "paper_landcover_map_all_labels.png")
    paper_pair_test_path = os.path.join(run_dir, "paper_landcover_map_test_labels.png")
    per_class_path = os.path.join(run_dir, "per_class_accuracy.png")

    pred_full_img = build_result_image(gt.shape, full_coords, full_pred)
    pred_test_img = build_result_image(gt.shape, test_coords, test_pred)

    save_rgb_map(gt.astype(np.uint8), "Ground Truth Land-cover Map (All Labeled Pixels)", gt_path, with_legend=True)
    save_rgb_map(ts_label.astype(np.uint8), "Ground Truth Land-cover Map (Official Test Pixels)", gt_test_path, with_legend=True)
    save_rgb_map(pred_full_img, "Predicted Land-cover Classification Map (All Labeled Pixels)", pred_full_path, with_legend=True)
    save_rgb_map(pred_test_img, "Predicted Land-cover Classification Map (Official Test Pixels)", pred_test_path, with_legend=True)

    save_map_pair(
        gt.astype(np.uint8),
        pred_full_img,
        "Ground Truth Land-cover Map (All Labeled Pixels)",
        "Predicted Land-cover Classification Map (All Labeled Pixels)",
        pair_full_path,
    )
    save_map_pair(
        ts_label.astype(np.uint8),
        pred_test_img,
        "Ground Truth Land-cover Map (Official Test Pixels)",
        "Predicted Land-cover Classification Map (Official Test Pixels)",
        pair_test_path,
    )

    # Paper-style two-row figures requested by the user.
    save_paper_map_pair(gt.astype(np.uint8), pred_full_img, paper_pair_full_path, run_id=run_id, method_name=method_name)
    save_paper_map_pair(ts_label.astype(np.uint8), pred_test_img, paper_pair_test_path, run_id=run_id, method_name=method_name)

    plot_per_class_bar(per_class, categories[: len(per_class)], per_class_path, "Land-cover Classification Accuracy", colors[: len(per_class)])

    per_class_csv = os.path.join(run_dir, "per_class_accuracy.csv")
    with open(per_class_csv, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,accuracy\n")
        for i, acc in enumerate(per_class):
            f.write(f"{i + 1},{categories[i]},{acc:.6f}\n")

    # Also save raw indexed maps. These are useful for verifying that labels 1..15 are mapped correctly.
    np.save(os.path.join(run_dir, "prediction_all_labels_indexed.npy"), pred_full_img)
    np.save(os.path.join(run_dir, "prediction_test_labels_indexed.npy"), pred_test_img)

    return {
        "gt_all": gt_path,
        "gt_test": gt_test_path,
        "pred_all": pred_full_path,
        "pred_test": pred_test_path,
        "pair_all": pair_full_path,
        "pair_test": pair_test_path,
        "paper_pair_all": paper_pair_full_path,
        "paper_pair_test": paper_pair_test_path,
        "per_class_bar": per_class_path,
        "per_class_csv": per_class_csv,
    }


def summarize_runs(run_results: List[Dict], summary_dir: str) -> Dict[str, np.ndarray]:
    os.makedirs(summary_dir, exist_ok=True)
    oa = np.array([r["oa"] for r in run_results], dtype=np.float64)
    aa = np.array([r["aa"] for r in run_results], dtype=np.float64)
    kappa = np.array([r["kappa"] for r in run_results], dtype=np.float64)
    per_class = np.stack([r["per_class"] for r in run_results], axis=0)

    mean_pc = per_class.mean(axis=0)
    std_pc = per_class.std(axis=0, ddof=0)

    summary = {
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
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(summary_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"OA    = {summary['OA_mean']:.4f} ± {summary['OA_std']:.4f}\n")
        f.write(f"AA    = {summary['AA_mean']:.4f} ± {summary['AA_std']:.4f}\n")
        f.write(f"Kappa = {summary['Kappa_mean']:.4f} ± {summary['Kappa_std']:.4f}\n")
        f.write("Per-class mean ± std\n")
        for i in range(len(mean_pc)):
            f.write(f"{i + 1:02d}. {CLASS_NAMES[i]:<18s} : {mean_pc[i]:.4f} ± {std_pc[i]:.4f}\n")

    with open(os.path.join(summary_dir, "per_class_overall.csv"), "w", encoding="utf-8") as f:
        f.write("class_id,class_name,mean,std\n")
        for i in range(len(mean_pc)):
            f.write(f"{i + 1},{CLASS_NAMES[i]},{mean_pc[i]:.6f},{std_pc[i]:.6f}\n")

    plot_per_class_mean_std(
        mean_pc,
        std_pc,
        CLASS_NAMES[: len(mean_pc)],
        os.path.join(summary_dir, "per_class_mean_std.png"),
        "Land-cover Classification Accuracy Mean ± Std over Runs",
        CLASS_COLORS[: len(mean_pc)],
    )

    best_idx = int(np.argmax(oa))
    best_run_dir = run_results[best_idx]["run_dir"]
    best_pair_test = os.path.join(best_run_dir, "map_pair_test_labels.png")
    if os.path.exists(best_pair_test):
        shutil.copyfile(best_pair_test, os.path.join(summary_dir, "best_run_map_pair_test_labels.png"))

    best_paper_pair_test = os.path.join(best_run_dir, "paper_landcover_map_test_labels.png")
    if os.path.exists(best_paper_pair_test):
        shutil.copyfile(best_paper_pair_test, os.path.join(summary_dir, "best_run_paper_landcover_map_test_labels.png"))

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "mean_pc": mean_pc,
        "std_pc": std_pc,
        "best_idx": best_idx,
    }


def run_single_protocol(
    args,
    data: Dict[str, np.ndarray],
    device: torch.device,
    run_id: int,
    base_seed: int,
) -> Dict:
    hsi = data["hsi"]
    lidar = data["lidar"]
    tr_label = data["tr"]
    ts_label = data["ts"]
    gt = data["gt"]
    num_classes = int(max(gt.max(), tr_label.max(), ts_label.max()))

    print("\n" + "#" * 88)
    print(f"[GLOBAL RUN {run_id}/{args.runs}] base_seed={base_seed}")

    cv_best_epochs: List[int] = []
    cv_results: List[EvalResult] = []

    print("\n" + "=" * 80)
    print("[STAGE A] Spatial CV on official training labels only")
    for fold in range(args.cv_folds):
        fold_seed = base_seed + 100 * fold
        print("\n" + "-" * 80)
        print(f"[CV FOLD {fold + 1}/{args.cv_folds}] seed={fold_seed}")
        train_mask, val_mask = classwise_spatial_train_val_split(
            tr_label,
            val_ratio=args.cv_val_ratio,
            block_size=args.block_size,
            seed=fold_seed,
            min_train_per_class=16,
            min_val_per_class=8,
        )
        _tr_coords, tr_labels = mask_to_coords_labels(train_mask)
        _va_coords, va_labels = mask_to_coords_labels(val_mask)
        print(f"[INFO] split sizes -> train={len(tr_labels)} | val={len(va_labels)}")

        artifacts = fit_split(
            args=args,
            hsi=hsi,
            lidar=lidar,
            train_mask=train_mask,
            val_mask=val_mask,
            num_classes=num_classes,
            device=device,
            seed=fold_seed,
            epochs=args.cv_epochs,
            save_snapshots=False,
        )
        cv_best_epochs.append(artifacts.best_epoch)
        cv_results.append(artifacts.best_val)
        print(
            f"[CV BEST] epoch={artifacts.best_epoch} | "
            f"OA={artifacts.best_val.oa:.4f} | "
            f"AA={artifacts.best_val.aa:.4f} | "
            f"Kappa={artifacts.best_val.kappa:.4f}"
        )

    cv_oa = np.array([m.oa for m in cv_results], dtype=np.float64)
    cv_aa = np.array([m.aa for m in cv_results], dtype=np.float64)
    cv_k = np.array([m.kappa for m in cv_results], dtype=np.float64)
    selected_epoch = int(round(np.median(cv_best_epochs) * args.final_epoch_scale))
    selected_epoch = max(50, min(selected_epoch, int(round(args.cv_epochs * 1.40))))

    print("\n" + "-" * 80)
    print(f"[CV SUMMARY] best_epoch median={np.median(cv_best_epochs):.1f}, scaled final_epoch={selected_epoch}")
    print(f"[CV SUMMARY] OA={cv_oa.mean():.4f} ± {cv_oa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] AA={cv_aa.mean():.4f} ± {cv_aa.std(ddof=0):.4f}")
    print(f"[CV SUMMARY] Kappa={cv_k.mean():.4f} ± {cv_k.std(ddof=0):.4f}")

    print("\n" + "=" * 80)
    print("[STAGE B] Retrain on full official training set and ensemble snapshots")
    full_train_mask = tr_label.copy()
    empty_val = np.zeros_like(tr_label, dtype=np.int64)
    test_mask = ts_label.copy()

    hsi_full, _, _ = fit_standardize_train_only(hsi, full_train_mask)
    lidar_full, _, _ = fit_standardize_train_only(lidar, full_train_mask)
    pca_full, pca_model = fit_pca_train_only(hsi_full, full_train_mask, args.pca_dim)
    print(f"[INFO] full-train PCA dims={pca_model.n_components_}")

    test_coords, test_labels = mask_to_coords_labels(test_mask)
    test_ds = HoustonPatchDataset(hsi_full, pca_full, lidar_full, test_coords, test_labels, args.patch_size, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model_builder = make_builder(args, hsi_full.shape[-1], pca_full.shape[-1], num_classes)
    state_list: List[Dict[str, torch.Tensor]] = []

    for j in range(args.final_seeds):
        final_seed = base_seed + 1000 + j
        print("\n" + "-" * 80)
        print(f"[FINAL MODEL {j + 1}/{args.final_seeds}] seed={final_seed} | epochs={selected_epoch}")
        artifacts = fit_split(
            args=args,
            hsi=hsi,
            lidar=lidar,
            train_mask=full_train_mask,
            val_mask=empty_val,
            num_classes=num_classes,
            device=device,
            seed=final_seed,
            epochs=selected_epoch,
            save_snapshots=True,
        )
        state_list.extend(artifacts.snapshot_states)
        print(f"[INFO] saved snapshots={len(artifacts.snapshot_states)}")

    print("\n" + "=" * 80)
    print(f"[STAGE C] Ensemble inference with {len(state_list)} snapshot models")
    test_res = evaluate_ensemble(state_list, model_builder, test_loader, device, num_classes, tta=args.tta)
    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")

    print("[TEST] Per-class accuracy")
    for cls_idx in range(num_classes):
        print(f"  {cls_idx + 1:02d}. {CLASS_NAMES[cls_idx]:<18s} : {test_res.per_class_acc[cls_idx]:.4f}")

    full_coords, full_labels = mask_to_coords_labels(gt)
    full_ds = HoustonPatchDataset(hsi_full, pca_full, lidar_full, full_coords, full_labels, args.patch_size, augment=False)
    full_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    full_pred, _ = ensemble_predict(state_list, model_builder, full_loader, device, tta=args.tta)
    test_pred = test_res.y_pred.astype(np.int64)

    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    paths = save_run_artifacts(
        run_dir=run_dir,
        gt=gt,
        ts_label=ts_label,
        full_coords=full_coords,
        full_pred=full_pred,
        test_coords=test_coords,
        test_pred=test_pred,
        per_class=test_res.per_class_acc,
        run_id=run_id,
        method_name=args.method_name,
    )

    return {
        "oa": test_res.oa,
        "aa": test_res.aa,
        "kappa": test_res.kappa,
        "per_class": test_res.per_class_acc.copy(),
        "run_dir": run_dir,
        "paths": paths,
        "selected_epoch": selected_epoch,
        "y_pred": test_pred.copy(),
    }


def normalize_ablation_name(name: str) -> str:
    name = str(name).strip().lower()
    aliases = {
        "full": "full",
        "baseline": "full",
        "ours": "full",
        "no_center": "no_center",
        "w_o_center": "no_center",
        "without_center": "no_center",
        "no_relief": "no_relief",
        "no_lidar_relief": "no_relief",
        "without_lidar_relief": "no_relief",
        "no_lidar": "no_lidar",
        "hsi_only": "no_lidar",
        "no_graph": "no_graph",
        "no_graph_wavelet": "no_graph",
        "without_graph_wavelet": "no_graph",
        "no_urban": "no_urban",
        "no_urban_strip": "no_urban",
        "without_urban_strip": "no_urban",
        "no_urban_loss": "no_urban_loss",
        "no_focus_loss": "no_urban_loss",
        "no_ema": "no_ema",
    }
    if name not in aliases:
        raise ValueError(f"Unknown ablation variant: {name}. Supported: {sorted(set(aliases.values()))}")
    return aliases[name]


def apply_ablation_variant(args, variant: str):
    """Return a deep-copied args object configured for one ablation variant."""
    out = copy.deepcopy(args)
    out.ablation = normalize_ablation_name(variant)

    # The architecture variants isolate one module. Loss-only variants change only training.
    if out.ablation == "no_urban_loss":
        out.urban_aux_w = 0.0
        out.focus_boost = 0.0
        out.urban_alpha = 0.0
    elif out.ablation == "no_urban":
        # Remove the strip prior. Keep the main branch fair, but remove the auxiliary urban logit injection.
        # Otherwise the ablation still receives a special urban classifier boost.
        out.urban_aux_w = 0.0
        out.focus_boost = 0.0
        out.urban_alpha = 0.0
    elif out.ablation == "no_lidar":
        # HSI-only setting should not receive a LiDAR-specific urban boost.
        out.urban_aux_w = 0.0
        out.urban_alpha = 0.0
    elif out.ablation == "no_ema":
        out.ema_decay = 0.0
    return out


def ablation_label(variant: str) -> str:
    labels = {
        "full": "Full USM-GWNet",
        "no_center": "w/o center-guided spectral scan",
        "no_relief": "w/o LiDAR relief operators",
        "no_lidar": "HSI only",
        "no_graph": "w/o graph wavelet fusion",
        "no_urban": "w/o urban strip mixer",
        "no_urban_loss": "w/o urban auxiliary/focus loss",
        "no_ema": "w/o EMA",
    }
    return labels.get(variant, variant)


def write_ablation_summary(all_summaries: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    txt_path = os.path.join(out_dir, "ablation_summary.txt")
    pc_path = os.path.join(out_dir, "ablation_per_class_mean_std.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,label,OA_mean,OA_std,AA_mean,AA_std,Kappa_mean,Kappa_std,focus4_mean,focus4_std,urban7_mean,urban7_std\n")
        for s in all_summaries:
            f.write(
                f"{s['variant']},{s['label']},{s['OA_mean']:.6f},{s['OA_std']:.6f},"
                f"{s['AA_mean']:.6f},{s['AA_std']:.6f},{s['Kappa_mean']:.6f},{s['Kappa_std']:.6f},"
                f"{s['focus4_mean']:.6f},{s['focus4_std']:.6f},{s['urban7_mean']:.6f},{s['urban7_std']:.6f}\n"
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        for s in all_summaries:
            f.write(f"[{s['variant']}] {s['label']}\n")
            f.write(f"  OA     = {s['OA_mean']:.4f} ± {s['OA_std']:.4f}\n")
            f.write(f"  AA     = {s['AA_mean']:.4f} ± {s['AA_std']:.4f}\n")
            f.write(f"  Kappa  = {s['Kappa_mean']:.4f} ± {s['Kappa_std']:.4f}\n")
            f.write(f"  focus4 = {s['focus4_mean']:.4f} ± {s['focus4_std']:.4f}\n")
            f.write(f"  urban7 = {s['urban7_mean']:.4f} ± {s['urban7_std']:.4f}\n\n")

    with open(pc_path, "w", encoding="utf-8") as f:
        f.write("variant,label,class_id,class_name,mean,std\n")
        for s in all_summaries:
            mean_pc = np.asarray(s["per_class_mean"], dtype=np.float64)
            std_pc = np.asarray(s["per_class_std"], dtype=np.float64)
            for i in range(len(mean_pc)):
                f.write(f"{s['variant']},{s['label']},{i+1},{CLASS_NAMES[i]},{mean_pc[i]:.6f},{std_pc[i]:.6f}\n")

    # Compact bar chart for paper tables.
    variants = [s["variant"] for s in all_summaries]
    x = np.arange(len(variants))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(variants)), 5.0))
    ax.bar(x - width, [s["OA_mean"] for s in all_summaries], width, yerr=[s["OA_std"] for s in all_summaries], capsize=3, label="OA")
    ax.bar(x, [s["AA_mean"] for s in all_summaries], width, yerr=[s["AA_std"] for s in all_summaries], capsize=3, label="AA")
    ax.bar(x + width, [s["Kappa_mean"] for s in all_summaries], width, yerr=[s["Kappa_std"] for s in all_summaries], capsize=3, label="Kappa")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study on Houston 2013")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_summary_bar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Ablation summary saved to: {out_dir}")


def run_ablation_variant(args, data: Dict[str, np.ndarray], device: torch.device, variant: str) -> Dict:
    v_args = apply_ablation_variant(args, variant)
    v_args.output_dir = os.path.join(args.output_dir, v_args.ablation)
    v_args.method_name = f"USM-GWNet-{v_args.ablation}"
    os.makedirs(v_args.output_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"[ABLATION] {v_args.ablation}: {ablation_label(v_args.ablation)}")
    print(f"[ABLATION] output_dir={v_args.output_dir}")
    print(f"[ABLATION] urban_aux_w={v_args.urban_aux_w}, focus_boost={v_args.focus_boost}, "
          f"urban_alpha={v_args.urban_alpha}, ema_decay={v_args.ema_decay}")

    run_results: List[Dict] = []
    for run_id in range(1, v_args.runs + 1):
        base_seed = v_args.seed + 10000 * (run_id - 1)
        set_seed(base_seed)
        run_res = run_single_protocol(v_args, data, device, run_id, base_seed)
        run_results.append(run_res)

    summary = summarize_runs(run_results, os.path.join(v_args.output_dir, "summary"))

    focus_vals = []
    urban_vals = []
    for r in run_results:
        pc = np.asarray(r["per_class"], dtype=np.float64)
        focus_vals.append(class_group_mean(pc, FOCUS_IDS))
        urban_vals.append(class_group_mean(pc, URBAN_IDS))

    variant_summary = {
        "variant": v_args.ablation,
        "label": ablation_label(v_args.ablation),
        "OA_mean": float(summary["oa"].mean()),
        "OA_std": float(summary["oa"].std(ddof=0)),
        "AA_mean": float(summary["aa"].mean()),
        "AA_std": float(summary["aa"].std(ddof=0)),
        "Kappa_mean": float(summary["kappa"].mean()),
        "Kappa_std": float(summary["kappa"].std(ddof=0)),
        "focus4_mean": float(np.mean(focus_vals)),
        "focus4_std": float(np.std(focus_vals, ddof=0)),
        "urban7_mean": float(np.mean(urban_vals)),
        "urban7_std": float(np.std(urban_vals, ddof=0)),
        "per_class_mean": np.asarray(summary["mean_pc"], dtype=np.float64).tolist(),
        "per_class_std": np.asarray(summary["std_pc"], dtype=np.float64).tolist(),
        "output_dir": v_args.output_dir,
    }

    with open(os.path.join(v_args.output_dir, "variant_summary.json"), "w", encoding="utf-8") as f:
        json.dump(variant_summary, f, ensure_ascii=False, indent=2)

    return variant_summary


def main() -> None:
    parser = argparse.ArgumentParser("Ablation study for USM-GWNet on Houston 2013")
    parser.add_argument("--data-root", type=str, default="E:/PythonProject1/Houston/")
    parser.add_argument("--output-dir", type=str, default="houston_usmgwnet_ablation_runs")
    parser.add_argument("--method-name", type=str, default="USM-GWNet")

    # Ablation control. Use comma-separated names.
    parser.add_argument(
        "--ablation-list",
        type=str,
        default="full,no_center,no_relief,no_graph,no_urban,no_urban_loss,no_lidar",
        help="Comma-separated variants: full,no_center,no_relief,no_graph,no_urban,no_urban_loss,no_lidar,no_ema",
    )
    parser.add_argument("--ablation", type=str, default="full", help="Internal use for one variant.")

    # Defaults are lighter than the full 5-run paper protocol because ablation suites are expensive.
    # For final paper tables, set --runs 5 --cv-folds 3 --cv-epochs 180 --final-seeds 3.
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=13)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--cv-val-ratio", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=2)
    parser.add_argument("--cv-epochs", type=int, default=120)
    parser.add_argument("--final-epoch-scale", type=float, default=1.12)
    parser.add_argument("--final-seeds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lidar-ch", type=int, default=24)
    parser.add_argument("--graph-ch", type=int, default=36)
    parser.add_argument("--urban-ch", type=int, default=28)
    parser.add_argument("--dropout", type=float, default=0.16)
    parser.add_argument("--drop-path", type=float, default=0.03)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--block-size", type=int, default=18)
    parser.add_argument("--focus-boost", type=float, default=0.30)
    parser.add_argument("--urban-aux-w", type=float, default=0.35)
    parser.add_argument("--urban-alpha", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    variants = [normalize_ablation_name(v) for v in args.ablation_list.split(",") if v.strip()]
    # Preserve order and remove duplicates.
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    data = load_houston_2013(args.data_root)
    hsi = data["hsi"]
    lidar = data["lidar"]
    tr_label = data["tr"]
    ts_label = data["ts"]

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] data-root={args.data_root}")
    print(f"[INFO] HSI shape={hsi.shape}, LiDAR shape={lidar.shape}")
    print(f"[INFO] total train labels={int((tr_label > 0).sum())}, total test labels={int((ts_label > 0).sum())}")
    print(f"[INFO] device={device}")
    print(f"[INFO] ablation variants={variants}")
    print(f"[INFO] protocol: runs={args.runs}, cv_folds={args.cv_folds}, cv_epochs={args.cv_epochs}, final_seeds={args.final_seeds}")

    all_summaries = []
    for variant in variants:
        s = run_ablation_variant(args, data, device, variant)
        all_summaries.append(s)

    write_ablation_summary(all_summaries, os.path.join(args.output_dir, "summary"))

    print("\n" + "#" * 100)
    print("Ablation summary")
    for s in all_summaries:
        print(
            f"{s['variant']:<15s} | OA={s['OA_mean']:.4f}±{s['OA_std']:.4f} | "
            f"AA={s['AA_mean']:.4f}±{s['AA_std']:.4f} | "
            f"Kappa={s['Kappa_mean']:.4f}±{s['Kappa_std']:.4f} | "
            f"focus4={s['focus4_mean']:.4f}±{s['focus4_std']:.4f}"
        )


if __name__ == "__main__":
    main()

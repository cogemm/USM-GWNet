# -*- coding: utf-8 -*-
"""
Fair baseline implementation for Houston AEFN on Houston 2013.

This script keeps the original local data path argument and the provided color list unchanged.
It rewrites the training and evaluation protocol to match the fair-comparison premise used in
houston_slimamba_msgw_v5_urbanfocus_5runs_maps_direct_rgb_palette.py.

Fair protocol:
    1. HSI standardization, LiDAR standardization, and PCA are fitted only on training pixels.
    2. The official TRLabel.mat is used for training and spatial validation only.
    3. The official TSLabel.mat is used only for final testing and final map generation.
    4. No test-aware model selection.
    5. No label propagation refinement over test/GT pixels.
    6. Final model is retrained on the full official training set after epoch selection.
    7. Five-run results are reported as mean ± std for OA, AA, Kappa, and per-class accuracy.
    8. Classification maps use the provided Houston 15-class color list through direct RGB lookup.
"""

import os
import json
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ==============================================================================
# 1) Configuration and global variables
# ==============================================================================

CATEGORIES = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway",
    "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"
]

COLORS = [
    "#006400", "#008000", "#00FF00", "#008080", "#8B4513",
    "#0000FF", "#FFFF00", "#FFD700", "#808080", "#A9A9A9",
    "#696969", "#FFA500", "#FF8C00", "#FF0000", "#FF1493"
]

cmap = ListedColormap(["#000000"] + COLORS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    p = argparse.ArgumentParser("Fair Houston AEFN baseline for Houston 2013")

    # data / io. Keep the original path argument and default path unchanged.
    p.add_argument("--data_path", type=str, default=r"E:\PythonProject1\Houston")
    p.add_argument("--results_dir", type=str, default="./results_AEFN_Houston_fair")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # preprocessing / model
    p.add_argument("--patch_size", type=int, default=11)
    p.add_argument("--pca_components", type=int, default=30)

    # leakage-aware model selection
    p.add_argument("--cv_val_ratio", type=float, default=0.20)
    p.add_argument("--block_size", type=int, default=18)
    p.add_argument("--cv_epochs", type=int, default=120)
    p.add_argument("--final_epoch_scale", type=float, default=1.00)
    p.add_argument("--epochs_min", type=int, default=30)

    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.08)
    p.add_argument("--mixup_alpha", type=float, default=1.0)
    p.add_argument("--use_sam", action="store_true", default=True)
    p.add_argument("--no_sam", action="store_false", dest="use_sam")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--tta", action="store_true", help="Optional final TTA. Default off for strict baseline.")

    # map output
    p.add_argument("--gt_mat", type=str, default="gt.mat")
    p.add_argument("--gt_key", type=str, default="gt")
    p.add_argument("--method_name", type=str, default="AEFN")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# 2) AEFN modules
# ==============================================================================

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = Hsigmoid()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_h = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels), nn.Sigmoid())
        self.gate_l = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels), nn.Sigmoid())
        self.out_conv = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x_h, x_l):
        combined = torch.cat([x_h, x_l], dim=1)
        feat_h = x_h * self.gate_h(combined) + x_h
        feat_l = x_l * self.gate_l(combined) + x_l
        return self.out_conv(torch.cat([feat_h, feat_l], dim=1))


class HighAcc_AEFN_Net(nn.Module):
    def __init__(self, hsi_bands, num_classes=15):
        super().__init__()
        self.conv_h1 = nn.Sequential(nn.Conv2d(hsi_bands, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU())
        self.ca_h1 = CoordAtt(64, 64)
        self.conv_h2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU())

        self.conv_l1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.SiLU())
        self.ca_l1 = CoordAtt(32, 32)
        self.conv_l2 = nn.Sequential(nn.Conv2d(32, 128, 3, padding=1), nn.BatchNorm2d(128), nn.SiLU())

        self.fusion = AdaptiveGatedFusion(128)
        self.ca_fuse = CoordAtt(128, 128)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_h, x_l):
        h = self.conv_h2(self.ca_h1(self.conv_h1(x_h)))
        l = self.conv_l2(self.ca_l1(self.conv_l1(x_l)))
        f = self.ca_fuse(self.fusion(h, l))
        return self.classifier(f)


# ==============================================================================
# 3) Optimizer, EMA, and losses
# ==============================================================================

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p_ in group["params"]:
                if p_.grad is None:
                    continue
                self.state[p_]["old_p"] = p_.data.clone()
                e_w = (torch.pow(p_, 2) if group["adaptive"] else 1.0) * p_.grad * scale.to(p_)
                p_.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p_ in group["params"]:
                if p_.grad is None:
                    continue
                p_.data = self.state[p_]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step() and second_step().")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p_ in group["params"]:
                if p_.grad is not None:
                    scale = torch.abs(p_) if group["adaptive"] else 1.0
                    norms.append((scale * p_.grad).norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def mixup_data(x1, x2, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    bs = x1.size(0)
    index = torch.randperm(bs, device=x1.device)
    return lam * x1 + (1.0 - lam) * x1[index], lam * x2 + (1.0 - lam) * x2[index], y, y[index], float(lam)


def weighted_label_smoothing_ce(logits, target, class_weights, smoothing):
    log_probs = F.log_softmax(logits, dim=-1)
    n_class = logits.size(1)
    with torch.no_grad():
        true_dist = torch.full_like(log_probs, smoothing / max(1, n_class - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    loss = (-true_dist * log_probs).sum(dim=1)
    if class_weights is not None:
        loss = loss * class_weights[target]
    return loss.mean()


def mixup_criterion(pred, y_a, y_b, lam, class_weights, smoothing):
    return lam * weighted_label_smoothing_ce(pred, y_a, class_weights, smoothing) + (1.0 - lam) * weighted_label_smoothing_ce(pred, y_b, class_weights, smoothing)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ==============================================================================
# 4) Data loading and leakage-controlled preprocessing
# ==============================================================================

def load_first_numeric_array(mat: Dict, preferred_key: Optional[str] = None) -> np.ndarray:
    if preferred_key is not None and preferred_key in mat:
        return np.asarray(mat[preferred_key])
    candidates = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            candidates.append((k, v))
    if not candidates:
        raise ValueError("No numeric array found in .mat file.")
    candidates.sort(key=lambda kv: (kv[1].ndim >= 2, kv[1].size), reverse=True)
    return np.asarray(candidates[0][1])


def ensure_hwc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape={x.shape}")
    if x.shape[0] < 20 and x.shape[-1] > 20:
        x = np.transpose(x, (1, 2, 0))
    return x


def load_data(path: str):
    hsi = ensure_hwc(load_first_numeric_array(sio.loadmat(os.path.join(path, "HSI.mat")), "HSI")).astype(np.float32)
    lidar = ensure_hwc(load_first_numeric_array(sio.loadmat(os.path.join(path, "LiDAR.mat")), "LiDAR")).astype(np.float32)
    tr_label = load_first_numeric_array(sio.loadmat(os.path.join(path, "TRLabel.mat")), "TRLabel").squeeze().astype(np.int64)
    ts_label = load_first_numeric_array(sio.loadmat(os.path.join(path, "TSLabel.mat")), "TSLabel").squeeze().astype(np.int64)
    return hsi, lidar, tr_label, ts_label


def load_gt_full(path: str, gt_mat: str = "gt.mat", gt_key: str = "gt", tr_label=None, ts_label=None):
    gt_path = os.path.join(path, gt_mat)
    if os.path.exists(gt_path):
        return load_first_numeric_array(sio.loadmat(gt_path), gt_key).squeeze().astype(np.int64)
    if tr_label is None or ts_label is None:
        raise FileNotFoundError(f"{gt_path} not found and TR/TSLabel not supplied.")
    return np.maximum(tr_label, ts_label).astype(np.int64)


def fit_preprocess_train_only(hsi_raw: np.ndarray, lidar_raw: np.ndarray, train_mask: np.ndarray, pca_components: int):
    """
    Fit all data-dependent preprocessing only on the current training mask.

    For CV: train_mask is the fit subset inside TRLabel.
    For final retraining: train_mask is the full official TRLabel.
    TSLabel is never used here.
    """
    H, W, B = hsi_raw.shape
    train_bool = train_mask > 0
    if train_bool.sum() == 0:
        raise ValueError("Empty train mask for preprocessing.")

    hsi_train = hsi_raw[train_bool].reshape(-1, B)
    hsi_mean = hsi_train.mean(axis=0, keepdims=True)
    hsi_std = hsi_train.std(axis=0, keepdims=True) + 1e-6
    hsi_scaled = ((hsi_raw.reshape(-1, B) - hsi_mean) / hsi_std).astype(np.float32)

    n_comp = min(int(pca_components), B, hsi_train.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="full", whiten=False)
    pca.fit(((hsi_train - hsi_mean) / hsi_std).astype(np.float32))
    hsi_pca = pca.transform(hsi_scaled).reshape(H, W, n_comp).astype(np.float32)

    lidar_raw = ensure_hwc(lidar_raw).astype(np.float32)
    C_l = lidar_raw.shape[-1]
    lidar_train = lidar_raw[train_bool].reshape(-1, C_l)
    lidar_mean = lidar_train.mean(axis=0, keepdims=True)
    lidar_std = lidar_train.std(axis=0, keepdims=True) + 1e-6
    lidar_scaled = ((lidar_raw.reshape(-1, C_l) - lidar_mean) / lidar_std).reshape(H, W, C_l).astype(np.float32)
    if lidar_scaled.shape[-1] != 1:
        lidar_scaled = lidar_scaled[..., :1]
    lidar_scaled = lidar_scaled[..., 0].astype(np.float32)

    return hsi_pca, lidar_scaled, pca


def label_to_positions(label_2d: np.ndarray):
    rows, cols = np.nonzero(label_2d)
    y = label_2d[rows, cols].astype(np.int64) - 1
    return rows.astype(np.int64), cols.astype(np.int64), y.astype(np.int64)


def classwise_spatial_train_val_split(tr_label: np.ndarray, val_ratio: float, block_size: int, seed: int,
                                      min_train_per_class: int = 16, min_val_per_class: int = 8):
    rows, cols, labels = label_to_positions(tr_label)
    rng = np.random.RandomState(seed)
    train_mask = np.zeros_like(tr_label, dtype=np.int64)
    val_mask = np.zeros_like(tr_label, dtype=np.int64)
    num_classes = int(labels.max()) + 1

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        n_cls = len(cls_idx)
        if n_cls == 0:
            continue

        target_val = int(round(n_cls * val_ratio))
        target_val = max(target_val, min_val_per_class)
        target_val = min(target_val, max(0, n_cls - min_train_per_class))

        if target_val <= 0:
            chosen_val = np.array([], dtype=np.int64)
            chosen_train = cls_idx
        else:
            block_map: Dict[Tuple[int, int], List[int]] = {}
            for local_idx in cls_idx:
                key = (int(rows[local_idx] // block_size), int(cols[local_idx] // block_size))
                block_map.setdefault(key, []).append(local_idx)

            keys = list(block_map.keys())
            rng.shuffle(keys)
            keys.sort(key=lambda k: len(block_map[k]))

            chosen = []
            cur = 0
            for key in keys:
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
                    chosen = rng.permutation(cls_idx)[:fallback_target].tolist()

            chosen_val = np.array(sorted(set(chosen)), dtype=np.int64)
            val_set = set(chosen_val.tolist())
            chosen_train = np.array([idx for idx in cls_idx if idx not in val_set], dtype=np.int64)

            if len(chosen_train) < min_train_per_class or len(chosen_val) < min_val_per_class:
                fallback_target = min(max(min_val_per_class, int(round(n_cls * val_ratio))), n_cls - min_train_per_class)
                perm = rng.permutation(cls_idx)
                chosen_val = perm[:fallback_target]
                chosen_train = perm[fallback_target:]

        train_mask[rows[chosen_train], cols[chosen_train]] = cls + 1
        if len(chosen_val) > 0:
            val_mask[rows[chosen_val], cols[chosen_val]] = cls + 1

    if (val_mask > 0).sum() == 0:
        raise RuntimeError("Spatial validation split failed and produced an empty validation set.")
    return train_mask, val_mask


def pad_hsi_lidar(hsi: np.ndarray, lidar: np.ndarray, patch_size: int):
    m = patch_size // 2
    h_pad = np.pad(hsi, ((m, m), (m, m), (0, 0)), mode="reflect")
    l_pad = np.pad(lidar, ((m, m), (m, m)), mode="reflect")
    return h_pad, l_pad


class HoustonPatchDataset(Dataset):
    def __init__(self, hsi_pad: np.ndarray, lidar_pad: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                 labels: Optional[np.ndarray], patch_size: int, augment: bool = False, return_index: bool = False):
        super().__init__()
        self.ps = int(patch_size)
        self.rows = rows.astype(np.int64)
        self.cols = cols.astype(np.int64)
        self.labels = None if labels is None else labels.astype(np.int64)
        self.augment = bool(augment)
        self.return_index = bool(return_index)
        self.h_pad = torch.from_numpy(hsi_pad).float()
        self.l_pad = torch.from_numpy(lidar_pad).float()

    def __len__(self):
        return int(self.rows.size)

    def _rand_aug(self, h: torch.Tensor, l: torch.Tensor):
        if torch.rand(1).item() < 0.5:
            h = torch.flip(h, dims=[1])
            l = torch.flip(l, dims=[1])
        if torch.rand(1).item() < 0.5:
            h = torch.flip(h, dims=[2])
            l = torch.flip(l, dims=[2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            h = torch.rot90(h, k, dims=[1, 2])
            l = torch.rot90(l, k, dims=[1, 2])
        return h, l

    def __getitem__(self, idx: int):
        r = int(self.rows[idx])
        c = int(self.cols[idx])
        h = self.h_pad[r:r + self.ps, c:c + self.ps, :].permute(2, 0, 1).contiguous()
        l = self.l_pad[r:r + self.ps, c:c + self.ps].unsqueeze(0).contiguous()

        if self.augment:
            h, l = self._rand_aug(h, l)

        if self.labels is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        if self.return_index:
            return h, l, y, torch.tensor(idx, dtype=torch.long)
        return h, l, y


def make_loader(hsi_pca: np.ndarray, lidar_norm: np.ndarray, mask: np.ndarray, patch_size: int, batch_size: int,
                augment: bool, shuffle: bool, num_workers: int, return_index: bool = False):
    h_pad, l_pad = pad_hsi_lidar(hsi_pca, lidar_norm, patch_size)
    rows, cols, labels = label_to_positions(mask)
    ds = HoustonPatchDataset(h_pad, l_pad, rows, cols, labels, patch_size, augment=augment, return_index=return_index)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                    pin_memory=(DEVICE.type == "cuda"), drop_last=False)
    return dl, rows, cols, labels


# ==============================================================================
# 5) Evaluation and visualization
# ==============================================================================

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
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    aa = float(per_class.mean())
    return EvalResult(oa, aa, kappa, per_class, cm, y_true, y_pred)


@torch.no_grad()
def forward_with_tta(model: nn.Module, h: torch.Tensor, l: torch.Tensor, tta: bool = False):
    if not tta:
        return model(h, l)
    logits = 0.0
    transforms = [
        lambda a, b: (a, b),
        lambda a, b: (torch.flip(a, dims=[-1]), torch.flip(b, dims=[-1])),
        lambda a, b: (torch.flip(a, dims=[-2]), torch.flip(b, dims=[-2])),
        lambda a, b: (torch.rot90(a, 1, dims=[-2, -1]), torch.rot90(b, 1, dims=[-2, -1])),
        lambda a, b: (torch.rot90(a, 2, dims=[-2, -1]), torch.rot90(b, 2, dims=[-2, -1])),
        lambda a, b: (torch.rot90(a, 3, dims=[-2, -1]), torch.rot90(b, 3, dims=[-2, -1])),
    ]
    for fn in transforms:
        h_aug, l_aug = fn(h, l)
        logits = logits + model(h_aug, l_aug)
    return logits / float(len(transforms))


@torch.no_grad()
def infer_loader(model: nn.Module, loader: DataLoader, tta: bool = False):
    model.eval()
    preds = []
    truths = []
    for h, l, y in loader:
        h = h.to(DEVICE, non_blocking=True)
        l = l.to(DEVICE, non_blocking=True)
        out = forward_with_tta(model, h, l, tta=tta)
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        truths.append(y.numpy())
    return np.concatenate(preds), np.concatenate(truths)


@torch.no_grad()
def infer_map(model: nn.Module, hsi_pca: np.ndarray, lidar_norm: np.ndarray, mask: np.ndarray,
              patch_size: int, batch_size: int, tta: bool):
    h_pad, l_pad = pad_hsi_lidar(hsi_pca, lidar_norm, patch_size)
    rows, cols = np.where(mask > 0)
    ds = HoustonPatchDataset(h_pad, l_pad, rows, cols, labels=None, patch_size=patch_size, augment=False, return_index=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    preds = np.zeros(len(ds), dtype=np.int64)
    model.eval()
    for h, l, _y, idx in dl:
        h = h.to(DEVICE, non_blocking=True)
        l = l.to(DEVICE, non_blocking=True)
        out = forward_with_tta(model, h, l, tta=tta)
        preds[idx.numpy()] = out.argmax(dim=1).detach().cpu().numpy()

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    out_img[rows, cols] = preds.astype(np.uint8) + 1
    return out_img, rows, cols, preds


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip()
    if h.startswith("#"):
        h = h[1:]
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PALETTE_RGB = np.array([[0, 0, 0]] + [list(hex_to_rgb(c)) for c in COLORS], dtype=np.uint8)


def label_map_to_rgb(label_img: np.ndarray) -> np.ndarray:
    label_img = np.asarray(label_img)
    out = np.zeros(label_img.shape + (3,), dtype=np.uint8)
    valid = (label_img >= 0) & (label_img < len(PALETTE_RGB))
    out[valid] = PALETTE_RGB[label_img[valid].astype(np.int64)]
    return out


def save_rgb_map(label_img: np.ndarray, title: str, out_path: str, with_legend: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.imshow(label_map_to_rgb(label_img), interpolation="nearest")
    ax.set_title(title, fontsize=18)
    ax.axis("off")
    if with_legend:
        handles = [Patch(facecolor=COLORS[i], edgecolor="k", label=CATEGORIES[i]) for i in range(len(CATEGORIES) - 1, -1, -1)]
        labels = [CATEGORIES[i] for i in range(len(CATEGORIES) - 1, -1, -1)]
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    plt.tight_layout(pad=0.25)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)


def save_paper_map_pair(gt_img: np.ndarray, pred_img: np.ndarray, out_path: str, run_id: int, method_name: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 7.2))
    axes[0].imshow(label_map_to_rgb(gt_img), interpolation="nearest")
    axes[0].set_title("Ground Truth", fontsize=18)
    axes[0].axis("off")
    axes[1].imshow(label_map_to_rgb(pred_img), interpolation="nearest")
    axes[1].set_title(f"Classification Result (Run {run_id}) by {method_name}", fontsize=18)
    axes[1].axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, hspace=0.18)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_per_class_bar(per_class: np.ndarray, out_path: str, title: str, std: Optional[np.ndarray] = None) -> None:
    x = np.arange(1, len(per_class) + 1)
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ax.bar(x, per_class, color=COLORS[:len(per_class)], edgecolor="black", alpha=0.85)
    if std is not None:
        ax.errorbar(x, per_class, yerr=std, fmt="none", capsize=4, ecolor="black", elinewidth=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES[:len(per_class)], rotation=45, ha="right")
    for xi, yi in zip(x, per_class):
        ax.text(xi, min(1.02, yi + 0.015), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 6) Training
# ==============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler,
                class_weights: torch.Tensor, args, ema: Optional[EMA]) -> float:
    model.train()
    running = 0.0
    total = 0

    for h, l, y in loader:
        h = h.to(DEVICE, non_blocking=True)
        l = l.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        if args.mixup_alpha and args.mixup_alpha > 0:
            h_in, l_in, y_a, y_b, lam = mixup_data(h, l, y, alpha=args.mixup_alpha)
            def loss_func(pred):
                return mixup_criterion(pred, y_a, y_b, lam, class_weights, args.label_smoothing)
        else:
            h_in, l_in = h, l
            def loss_func(pred):
                return weighted_label_smoothing_ce(pred, y, class_weights, args.label_smoothing)

        if args.use_sam:
            out = model(h_in, l_in)
            loss = loss_func(out)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            loss2 = loss_func(model(h_in, l_in))
            loss2.backward()
            optimizer.second_step(zero_grad=True)
            loss_value = float(loss.item())
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func(model(h_in, l_in))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.item())

        if ema is not None:
            ema.update()

        bs = y.size(0)
        running += loss_value * bs
        total += bs

    scheduler.step()
    return running / max(total, 1)


def evaluate_current(model: nn.Module, loader: DataLoader, num_classes: int, ema: Optional[EMA], tta: bool = False) -> EvalResult:
    if ema is not None:
        ema.apply_shadow()
    pred, true = infer_loader(model, loader, tta=tta)
    if ema is not None:
        ema.restore()
    return compute_metrics(true, pred, num_classes)


def build_model_and_optim(hsi_bands: int, num_classes: int, args):
    model = HighAcc_AEFN_Net(hsi_bands=hsi_bands, num_classes=num_classes).to(DEVICE)
    if args.use_sam:
        optimizer = SAM(model.parameters(), optim.AdamW, lr=args.lr, weight_decay=args.weight_decay, rho=0.05)
        sched_opt = optimizer.base_optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_opt = optimizer

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        sched_opt,
        T_0=max(10, args.cv_epochs // 6),
        T_mult=2,
        eta_min=1e-6,
    )
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    return model, optimizer, scheduler, ema


def fit_with_optional_validation(args, hsi_raw: np.ndarray, lidar_raw: np.ndarray, train_mask: np.ndarray,
                                 val_mask: Optional[np.ndarray], num_classes: int, seed: int, epochs: int):
    set_seed(seed)

    hsi_pca, lidar_norm, pca_model = fit_preprocess_train_only(hsi_raw, lidar_raw, train_mask, args.pca_components)
    train_loader, _, _, train_labels = make_loader(
        hsi_pca, lidar_norm, train_mask, args.patch_size, args.batch_size,
        augment=True, shuffle=True, num_workers=args.num_workers
    )

    val_loader = None
    if val_mask is not None and (val_mask > 0).sum() > 0:
        val_loader, _, _, _ = make_loader(
            hsi_pca, lidar_norm, val_mask, args.patch_size, args.test_batch_size,
            augment=False, shuffle=False, num_workers=args.num_workers
        )

    model, optimizer, scheduler, ema = build_model_and_optim(hsi_pca.shape[-1], num_classes, args)
    class_weights = compute_class_weights(train_labels, num_classes).to(DEVICE)

    best_state = None
    best_epoch = epochs
    best_score = -1.0
    best_val = None

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, class_weights, args, ema)

        if val_loader is not None and (epoch == 1 or epoch % args.eval_interval == 0 or epoch == epochs):
            val_res = evaluate_current(model, val_loader, num_classes, ema=ema, tta=False)
            score = val_res.oa + 0.35 * val_res.aa + 0.55 * val_res.kappa
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_val = val_res
                if ema is not None:
                    ema.apply_shadow()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if ema is not None:
                    ema.restore()

            print(
                f"epoch {epoch:03d}/{epochs} | loss={loss:.4f} | "
                f"val_OA={val_res.oa:.4f} | val_AA={val_res.aa:.4f} | "
                f"val_Kappa={val_res.kappa:.4f} | best_epoch={best_epoch}"
            )
        elif epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d}/{epochs} | loss={loss:.4f}")

    if best_state is None:
        if ema is not None:
            ema.apply_shadow()
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            ema.restore()

    model.load_state_dict(best_state, strict=True)
    return model, hsi_pca, lidar_norm, pca_model, best_epoch, best_val


def train_one_run(run_id: int, args, hsi_raw: np.ndarray, lidar_raw: np.ndarray,
                  tr_label: np.ndarray, ts_label: np.ndarray, gt_full: np.ndarray):
    run_seed = args.seed + run_id * 1000
    print("\n" + "=" * 88)
    print(f"[RUN {run_id}/{args.runs}] seed={run_seed}")

    # Stage A. Spatial validation inside official training set. No TSLabel is used.
    print("[Stage A] Spatial validation on official training labels only")
    cv_train_mask, cv_val_mask = classwise_spatial_train_val_split(
        tr_label,
        val_ratio=args.cv_val_ratio,
        block_size=args.block_size,
        seed=run_seed,
        min_train_per_class=16,
        min_val_per_class=8,
    )
    _, _, cv_train_y = label_to_positions(cv_train_mask)
    _, _, cv_val_y = label_to_positions(cv_val_mask)
    print(f"[INFO] spatial split -> train={len(cv_train_y)} | val={len(cv_val_y)}")

    cv_model, _cv_hsi_pca, _cv_lidar_norm, _cv_pca, best_epoch, best_val = fit_with_optional_validation(
        args=args,
        hsi_raw=hsi_raw,
        lidar_raw=lidar_raw,
        train_mask=cv_train_mask,
        val_mask=cv_val_mask,
        num_classes=len(CATEGORIES),
        seed=run_seed,
        epochs=args.cv_epochs,
    )

    selected_epoch = int(round(best_epoch * args.final_epoch_scale))
    selected_epoch = max(args.epochs_min, min(selected_epoch, int(round(args.cv_epochs * 1.35))))
    if best_val is not None:
        print(
            f"[Stage A best] epoch={best_epoch} | OA={best_val.oa:.4f} | "
            f"AA={best_val.aa:.4f} | Kappa={best_val.kappa:.4f}"
        )
    print(f"[INFO] selected final retraining epochs={selected_epoch}")

    del cv_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Stage B. Retrain on all official training labels using selected epoch count.
    print("[Stage B] Retrain on full official training labels")
    model, hsi_pca, lidar_norm, pca_model, _final_epoch, _ = fit_with_optional_validation(
        args=args,
        hsi_raw=hsi_raw,
        lidar_raw=lidar_raw,
        train_mask=tr_label,
        val_mask=None,
        num_classes=len(CATEGORIES),
        seed=run_seed + 777,
        epochs=selected_epoch,
    )
    print(f"[INFO] full-train PCA dims={pca_model.n_components_}")

    # Stage C. Final evaluation on official TSLabel only.
    print("[Stage C] Final official test evaluation")
    test_loader, test_rows, test_cols, test_y = make_loader(
        hsi_pca, lidar_norm, ts_label, args.patch_size, args.test_batch_size,
        augment=False, shuffle=False, num_workers=args.num_workers
    )
    test_pred, test_true = infer_loader(model, test_loader, tta=args.tta)
    test_res = compute_metrics(test_true, test_pred, len(CATEGORIES))

    print(f"[TEST] OA={test_res.oa:.4f} | AA={test_res.aa:.4f} | Kappa={test_res.kappa:.4f}")
    print("[TEST] Per-class accuracy")
    for i, acc in enumerate(test_res.per_class):
        print(f"  {i + 1:02d}. {CATEGORIES[i]:<18s}: {acc:.4f}")

    # Maps. Metrics are still official TSLabel only.
    run_dir = os.path.join(args.results_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    test_pred_img = np.zeros_like(ts_label, dtype=np.uint8)
    test_pred_img[test_rows, test_cols] = test_pred.astype(np.uint8) + 1

    all_pred_img, _all_rows, _all_cols, _all_pred = infer_map(
        model=model,
        hsi_pca=hsi_pca,
        lidar_norm=lidar_norm,
        mask=gt_full,
        patch_size=args.patch_size,
        batch_size=args.test_batch_size,
        tta=args.tta,
    )

    save_rgb_map(gt_full.astype(np.uint8), "Ground Truth Land-cover Map (All Labeled Pixels)",
                 os.path.join(run_dir, "ground_truth_all_labels.png"), with_legend=True)
    save_rgb_map(ts_label.astype(np.uint8), "Ground Truth Land-cover Map (Official Test Pixels)",
                 os.path.join(run_dir, "ground_truth_test_labels.png"), with_legend=True)
    save_rgb_map(all_pred_img, "Predicted Land-cover Classification Map (All Labeled Pixels)",
                 os.path.join(run_dir, "prediction_all_labels.png"), with_legend=True)
    save_rgb_map(test_pred_img, "Predicted Land-cover Classification Map (Official Test Pixels)",
                 os.path.join(run_dir, "prediction_test_labels.png"), with_legend=True)
    save_paper_map_pair(ts_label.astype(np.uint8), test_pred_img,
                        os.path.join(run_dir, "paper_landcover_map_test_labels.png"), run_id, args.method_name)
    plot_per_class_bar(test_res.per_class, os.path.join(run_dir, "per_class_accuracy.png"),
                       f"Per-class Accuracy (Run {run_id})")

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "OA": test_res.oa,
                "AA": test_res.aa,
                "Kappa": test_res.kappa,
                "selected_epoch": selected_epoch,
                "cv_best_epoch": best_epoch,
                "per_class": test_res.per_class.tolist(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(run_dir, "per_class_accuracy.csv"), "w", encoding="utf-8") as f:
        f.write("class_id,class_name,accuracy\n")
        for i, acc in enumerate(test_res.per_class):
            f.write(f"{i + 1},{CATEGORIES[i]},{acc:.6f}\n")

    return {
        "oa": test_res.oa,
        "aa": test_res.aa,
        "kappa": test_res.kappa,
        "per_class": test_res.per_class,
        "run_dir": run_dir,
        "selected_epoch": selected_epoch,
    }


# ==============================================================================
# 7) Summary
# ==============================================================================

def summarize_results(results: List[Dict], args) -> None:
    summary_dir = os.path.join(args.results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    oa = np.array([r["oa"] for r in results], dtype=np.float64)
    aa = np.array([r["aa"] for r in results], dtype=np.float64)
    kappa = np.array([r["kappa"] for r in results], dtype=np.float64)
    pcs = np.stack([r["per_class"] for r in results], axis=0)

    pc_mean = pcs.mean(axis=0)
    pc_std = pcs.std(axis=0, ddof=0)

    summary = {
        "runs": len(results),
        "OA_mean": float(oa.mean()),
        "OA_std": float(oa.std(ddof=0)),
        "AA_mean": float(aa.mean()),
        "AA_std": float(aa.std(ddof=0)),
        "Kappa_mean": float(kappa.mean()),
        "Kappa_std": float(kappa.std(ddof=0)),
        "per_class_mean": pc_mean.tolist(),
        "per_class_std": pc_std.tolist(),
    }

    with open(os.path.join(summary_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(summary_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"OA     : {summary['OA_mean']:.4f} ± {summary['OA_std']:.4f}\n")
        f.write(f"AA     : {summary['AA_mean']:.4f} ± {summary['AA_std']:.4f}\n")
        f.write(f"Kappa  : {summary['Kappa_mean']:.4f} ± {summary['Kappa_std']:.4f}\n")
        f.write("Per-class accuracy mean ± std\n")
        for i in range(len(pc_mean)):
            f.write(f"{i + 1:02d}. {CATEGORIES[i]:<18s}: {pc_mean[i]:.4f} ± {pc_std[i]:.4f}\n")

    rows = []
    for i in range(len(pc_mean)):
        rows.append({
            "Class ID": i + 1,
            "Class Name": CATEGORIES[i],
            "Mean": float(pc_mean[i]),
            "Std": float(pc_std[i]),
        })
    pd.DataFrame(rows).to_csv(os.path.join(summary_dir, "per_class_mean_std.csv"), index=False)

    pd.DataFrame([{
        "OA Mean": float(oa.mean()),
        "OA Std": float(oa.std(ddof=0)),
        "AA Mean": float(aa.mean()),
        "AA Std": float(aa.std(ddof=0)),
        "Kappa Mean": float(kappa.mean()),
        "Kappa Std": float(kappa.std(ddof=0)),
    }]).to_csv(os.path.join(summary_dir, "overall_metrics.csv"), index=False)

    plot_per_class_bar(pc_mean, os.path.join(summary_dir, "per_class_mean_std.png"),
                       "Per-class Accuracy Mean ± Std", std=pc_std)

    best_idx = int(np.argmax(oa))
    best_src = os.path.join(results[best_idx]["run_dir"], "paper_landcover_map_test_labels.png")
    if os.path.exists(best_src):
        import shutil
        shutil.copyfile(best_src, os.path.join(summary_dir, "best_run_paper_landcover_map_test_labels.png"))

    print("\n" + "#" * 88)
    print(f"Final summary across {len(results)} runs")
    print(f"OA     : {oa.mean():.4f} ± {oa.std(ddof=0):.4f}")
    print(f"AA     : {aa.mean():.4f} ± {aa.std(ddof=0):.4f}")
    print(f"Kappa  : {kappa.mean():.4f} ± {kappa.std(ddof=0):.4f}")
    print("Per-class accuracy mean ± std")
    for i in range(len(pc_mean)):
        print(f"  {i + 1:02d}. {CATEGORIES[i]:<18s}: {pc_mean[i]:.4f} ± {pc_std[i]:.4f}")
    print(f"[INFO] Summary saved to: {summary_dir}")


# ==============================================================================
# 8) Main
# ==============================================================================

def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    if DEVICE.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    print(f"[INFO] data_path={args.data_path}")
    print(f"[INFO] results_dir={args.results_dir}")
    print(f"[INFO] device={DEVICE}")
    print("[INFO] Fair protocol: train-only preprocessing, spatial validation, full official-train retraining, final TSLabel-only testing.")
    print("[INFO] No test-aware model selection and no label propagation refinement.")

    hsi_raw, lidar_raw, tr_label, ts_label = load_data(args.data_path)
    gt_full = load_gt_full(args.data_path, gt_mat=args.gt_mat, gt_key=args.gt_key, tr_label=tr_label, ts_label=ts_label)

    print(f"[INFO] HSI shape={hsi_raw.shape}, LiDAR shape={lidar_raw.shape}")
    print(f"[INFO] train labels={int((tr_label > 0).sum())}, test labels={int((ts_label > 0).sum())}")

    results = []
    for run_id in range(1, int(args.runs) + 1):
        result = train_one_run(run_id, args, hsi_raw, lidar_raw, tr_label, ts_label, gt_full)
        results.append(result)

    summarize_results(results, args)


if __name__ == "__main__":
    main()

import os
import math
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# Houston categories and colors (user-specified)
# ============================================================
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


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_first_array(mat_path: str):
    d = sio.loadmat(mat_path)
    for k, v in d.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray):
            return v
    raise ValueError(f'No ndarray found in {mat_path}')


def load_houston_5mats(data_root: str):
    root = Path(data_root)
    hsi = load_first_array(str(root / 'HSI.mat')).astype(np.float32)
    lidar = load_first_array(str(root / 'LiDAR.mat')).astype(np.float32)
    gt = load_first_array(str(root / 'gt.mat')).astype(np.int64)
    tr_label = load_first_array(str(root / 'TRLabel.mat')).astype(np.int64)
    ts_label = load_first_array(str(root / 'TSLabel.mat')).astype(np.int64)
    if lidar.ndim == 2:
        lidar = lidar[..., None]
    return hsi, lidar, gt, tr_label, ts_label


def mask_to_coords_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.where(mask > 0)
    coords = np.stack([yy, xx], axis=1)
    labels = mask[yy, xx].astype(np.int64) - 1
    return coords, labels


def stratified_train_val_split(coords: np.ndarray,
                               labels: np.ndarray,
                               val_ratio: float,
                               seed: int,
                               min_val_per_class: int = 1):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    num_classes = int(labels.max()) + 1
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = max(min_val_per_class, int(round(len(idx) * val_ratio)))
        if len(idx) >= 2:
            n_val = min(n_val, len(idx) - 1)
        else:
            n_val = 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return coords[np.array(train_idx)], labels[np.array(train_idx)], coords[np.array(val_idx)], labels[np.array(val_idx)]


def fit_preprocessors_on_train(hsi: np.ndarray, lidar: np.ndarray, train_coords: np.ndarray, pca_dim: int):
    train_pixels_hsi = hsi[train_coords[:, 0], train_coords[:, 1], :]
    train_pixels_lidar = lidar[train_coords[:, 0], train_coords[:, 1], :]

    hsi_scaler = StandardScaler().fit(train_pixels_hsi)
    lidar_scaler = StandardScaler().fit(train_pixels_lidar)

    H, W, B = hsi.shape
    hsi_flat = hsi.reshape(-1, B)
    hsi_scaled = hsi_scaler.transform(hsi_flat).reshape(H, W, B).astype(np.float32)

    Hl, Wl, Bl = lidar.shape
    lidar_flat = lidar.reshape(-1, Bl)
    lidar_scaled = lidar_scaler.transform(lidar_flat).reshape(Hl, Wl, Bl).astype(np.float32)

    pca = PCA(n_components=min(pca_dim, B), whiten=False, random_state=0)
    pca.fit(hsi_scaler.transform(train_pixels_hsi))
    hsi_pca = pca.transform(hsi_scaled.reshape(-1, B)).reshape(H, W, -1).astype(np.float32)

    return hsi_scaled, lidar_scaled, hsi_pca, hsi_scaler, lidar_scaler, pca


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    denom = np.maximum(cm.sum(axis=1), 1)
    per_class = cm.diagonal() / denom
    aa = float(np.mean(per_class))
    return oa, aa, kappa, per_class, cm


def hex_to_rgb255(hex_color: str):
    h = hex_color.lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def render_label_map(label_map_1based: np.ndarray, out_path: str, title: str = None):
    h, w = label_map_1based.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, hex_color in enumerate(colors, start=1):
        rgb[label_map_1based == cls_idx] = np.array(hex_to_rgb255(hex_color), dtype=np.uint8)
    plt.figure(figsize=(16, 4.5))
    ax = plt.gca()
    ax.imshow(rgb)
    if title:
        ax.set_title(title, fontsize=16)
    ax.axis('off')
    handles = [Patch(facecolor=colors[i], edgecolor='k', label=categories[i]) for i in range(len(categories)-1, -1, -1)]
    labels = [categories[i] for i in range(len(categories)-1, -1, -1)]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_mean_std(mean_pc: np.ndarray, std_pc: np.ndarray, out_path: str):
    x = np.arange(1, len(mean_pc) + 1)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x, mean_pc, color=colors)
    ax.errorbar(x, mean_pc, yerr=std_pc, fmt='none', ecolor='black', elinewidth=1.1, capsize=4)
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Houston 2013 Per-class Accuracy (mean ± std across 5 runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    for xi, m, s in zip(x, mean_pc, std_pc):
        ax.text(xi, min(0.985, m + s + 0.02), f'{m:.3f}±{s:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Dataset
# ============================================================
class HoustonPatchDataset(Dataset):
    def __init__(self,
                 hsi_pca: np.ndarray,
                 hsi: np.ndarray,
                 aux: np.ndarray,
                 coords: np.ndarray,
                 labels: np.ndarray = None,
                 patch_size: int = 11,
                 augment: bool = False):
        self.hsi_pca = hsi_pca
        self.hsi = hsi
        self.aux = aux
        self.coords = coords.astype(np.int64)
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.patch = patch_size
        self.pad = patch_size // 2
        self.augment = augment

        pad_mode = 'reflect' if patch_size % 2 == 1 else 'symmetric'
        self.hsi_pca_pad = np.pad(self.hsi_pca, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=pad_mode)
        self.hsi_pad = np.pad(self.hsi, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=pad_mode)
        self.aux_pad = np.pad(self.aux, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode=pad_mode)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        y0, x0 = y, x
        y1, x1 = y0 + self.patch, x0 + self.patch
        pca = self.hsi_pca_pad[y0:y1, x0:x1, :]
        hsi = self.hsi_pad[y0:y1, x0:x1, :]
        aux = self.aux_pad[y0:y1, x0:x1, :]

        if self.augment:
            if random.random() < 0.5:
                pca = np.flip(pca, axis=0).copy(); hsi = np.flip(hsi, axis=0).copy(); aux = np.flip(aux, axis=0).copy()
            if random.random() < 0.5:
                pca = np.flip(pca, axis=1).copy(); hsi = np.flip(hsi, axis=1).copy(); aux = np.flip(aux, axis=1).copy()

        pca_t = torch.from_numpy(np.transpose(pca, (2, 0, 1))).float()
        hsi_t = torch.from_numpy(np.transpose(hsi, (2, 0, 1))).float()
        aux_t = torch.from_numpy(np.transpose(aux, (2, 0, 1))).float()
        if self.labels is None:
            return pca_t, hsi_t, aux_t, torch.tensor(y, dtype=torch.long), torch.tensor(x, dtype=torch.long)
        return pca_t, hsi_t, aux_t, torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# RSCNet
# ============================================================
class CrossModalFusion(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = max(8, channels // r)
        self.conv_3d = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.conv_2d = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        self.agg_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2 * channels, 1)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 2 * channels, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, fh, fx):
        b, c, h, w = fh.shape
        fh_prime = self.conv_3d(fh.unsqueeze(1)).squeeze(1)
        fx_prime = self.conv_2d(fx)
        gates = self.cross_gate(torch.cat([fh_prime, fx_prime], dim=1))
        w_h, w_x = torch.split(gates, 1, dim=1)
        fh_s1 = fh_prime + fx_prime * w_x
        fx_s1 = fx_prime + fh_prime * w_h
        u = self.agg_conv(torch.cat([fh_s1, fx_s1], dim=1))
        logits = self.local_att(u) * self.global_att(u)
        attn = self.softmax(logits.view(b, 2, c, h, w))
        return fh_s1 * attn[:, 0] + fx_s1 * attn[:, 1]


class KeyBandSelectionBlock(nn.Module):
    def __init__(self, hsi_channels, fused_dim, topk_ratio=0.2):
        super().__init__()
        self.k = max(1, int(hsi_channels * topk_ratio))
        self.linear = nn.Linear(fused_dim, 1)
        hidden = max(4, hsi_channels // 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hsi_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hsi_channels), nn.Sigmoid()
        )

    def forward(self, hsi_feats, fused_feats):
        b, c, h, w = hsi_feats.shape
        _, k, _, _ = fused_feats.shape
        hsi_flat = hsi_feats.view(b, c, -1)
        fused_flat = fused_feats.view(b, k, -1).permute(0, 2, 1)
        attn_map = torch.bmm(hsi_flat, fused_flat)
        attn_vec = self.linear(attn_map).squeeze(-1)
        scores = self.mlp(hsi_feats) * attn_vec
        _, topk_idx = torch.topk(scores, k=self.k, dim=1)
        idx_exp = topk_idx.view(b, self.k, 1, 1).expand(-1, -1, h, w)
        return torch.gather(hsi_feats, 1, idx_exp), topk_idx


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class DepthwiseOnlyEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class FFN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return x + self.net(x)


class RSCM(nn.Module):
    def __init__(self, hsi_dim, fused_dim, topk_ratio=0.2):
        super().__init__()
        self.kbsb = KeyBandSelectionBlock(hsi_dim, fused_dim, topk_ratio=topk_ratio)
        actual_k = self.kbsb.k
        self.align_conv = nn.Conv2d(actual_k, fused_dim, kernel_size=1)
        self.cmaf = CrossModalFusion(channels=fused_dim)
        self.ffn = FFN(fused_dim)

    def forward(self, hsi_feats, fused_feats):
        selected_bands, topk_idx = self.kbsb(hsi_feats, fused_feats)
        aligned_bands = self.align_conv(selected_bands)
        fused_out = self.cmaf(fh=aligned_bands, fx=fused_feats)
        out = self.ffn(fused_out)
        return out, selected_bands, topk_idx


class RSCNet(nn.Module):
    def __init__(self, hsi_channels, pca_channels, aux_channels, num_classes, embed_dim=144, topk_ratio=0.2, num_rscm_layers=2):
        super().__init__()
        self.hsi_encoder = DepthwiseOnlyEncoder(hsi_channels)
        self.pca_encoder = SimpleEncoder(pca_channels, embed_dim)
        self.aux_encoder = SimpleEncoder(aux_channels, embed_dim)
        self.initial_fusion = CrossModalFusion(channels=embed_dim)
        self.initial_ffn = FFN(embed_dim)
        self.rscm_layers = nn.ModuleList([
            RSCM(hsi_dim=hsi_channels, fused_dim=embed_dim, topk_ratio=topk_ratio)
            for _ in range(num_rscm_layers)
        ])
        final_k = max(1, int(hsi_channels * topk_ratio))
        self.final_align = nn.Conv2d(final_k, embed_dim, 1)
        self.final_fusion = CrossModalFusion(channels=embed_dim)
        self.refinement = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x_hsi, x_pca, x_aux):
        feat_hsi = self.hsi_encoder(x_hsi)
        feat_pca = self.pca_encoder(x_pca)
        feat_aux = self.aux_encoder(x_aux)
        curr_fused = self.initial_fusion(fh=feat_pca, fx=feat_aux)
        curr_fused = self.initial_ffn(curr_fused)
        last_selected_bands = None
        last_topk_idx = None
        for rscm in self.rscm_layers:
            curr_fused, last_selected_bands, last_topk_idx = rscm(feat_hsi, curr_fused)
        feat_bands_aligned = self.final_align(last_selected_bands)
        final_feat = self.final_fusion(fh=feat_bands_aligned, fx=curr_fused)
        logits = self.refinement(final_feat)
        return logits, last_topk_idx


# ============================================================
# Training / evaluation
# ============================================================
@torch.no_grad()
def infer_loader(model, loader, device):
    model.eval()
    preds, gts = [], []
    for pca, hsi, aux, labels in loader:
        pca = pca.to(device)
        hsi = hsi.to(device)
        aux = aux.to(device)
        logits, _ = model(hsi, pca, aux)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
        gts.append(labels.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)
    return y_true, y_pred


@torch.no_grad()
def predict_map(model, loader, h: int, w: int, device):
    model.eval()
    pred_map = np.zeros((h, w), dtype=np.int64)
    for pca, hsi, aux, yy, xx in loader:
        pca = pca.to(device)
        hsi = hsi.to(device)
        aux = aux.to(device)
        logits, _ = model(hsi, pca, aux)
        pred = logits.argmax(dim=1).cpu().numpy() + 1
        yy = yy.numpy(); xx = xx.numpy()
        pred_map[yy, xx] = pred
    return pred_map


def train_one_run(args, run_id: int, seed: int, hsi: np.ndarray, lidar: np.ndarray, gt: np.ndarray, tr_label: np.ndarray, ts_label: np.ndarray, device: torch.device):
    set_seed(seed)
    run_dir = os.path.join(args.out_dir, f'run_{run_id:02d}')
    ensure_dir(run_dir)

    tr_coords_all, tr_labels_all = mask_to_coords_labels(tr_label)
    ts_coords, ts_labels = mask_to_coords_labels(ts_label)
    train_coords, train_labels, val_coords, val_labels = stratified_train_val_split(
        tr_coords_all, tr_labels_all, val_ratio=args.val_ratio, seed=seed
    )

    hsi_scaled, lidar_scaled, hsi_pca, _, _, _ = fit_preprocessors_on_train(hsi, lidar, train_coords, args.pca_channels)

    train_ds = HoustonPatchDataset(hsi_pca, hsi_scaled, lidar_scaled, train_coords, train_labels, args.patch_size, augment=True)
    val_ds = HoustonPatchDataset(hsi_pca, hsi_scaled, lidar_scaled, val_coords, val_labels, args.patch_size, augment=False)
    test_ds = HoustonPatchDataset(hsi_pca, hsi_scaled, lidar_scaled, ts_coords, ts_labels, args.patch_size, augment=False)

    full_coords = np.stack(np.meshgrid(np.arange(hsi.shape[0]), np.arange(hsi.shape[1]), indexing='ij'), axis=-1).reshape(-1, 2)
    full_ds = HoustonPatchDataset(hsi_pca, hsi_scaled, lidar_scaled, full_coords, labels=None, patch_size=args.patch_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False)

    model = RSCNet(
        hsi_channels=hsi.shape[2],
        pca_channels=hsi_pca.shape[2],
        aux_channels=lidar.shape[2],
        num_classes=len(categories),
        embed_dim=args.embed_dim,
        topk_ratio=args.topk_ratio,
        num_rscm_layers=args.num_rscm_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_state = None
    best_score = -1.0
    best_epoch = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for pca, hsi_patch, aux, labels in train_loader:
            pca = pca.to(device)
            hsi_patch = hsi_patch.to(device)
            aux = aux.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(hsi_patch, pca, aux)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        y_val_true, y_val_pred = infer_loader(model, val_loader, device)
        val_oa, val_aa, val_kappa, _, _ = compute_metrics(y_val_true, y_val_pred, len(categories))
        score = val_oa
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'val_oa': val_oa, 'val_aa': val_aa, 'val_kappa': val_kappa})
        print(f'[RUN {run_id}/5][Epoch {epoch:03d}/{args.epochs}] loss={np.mean(losses):.4f} | val_OA={val_oa:.4f} | val_AA={val_aa:.4f} | val_Kappa={val_kappa:.4f}')
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_test_true, y_test_pred = infer_loader(model, test_loader, device)
    test_oa, test_aa, test_kappa, per_class, cm = compute_metrics(y_test_true, y_test_pred, len(categories))

    pred_full = predict_map(model, full_loader, hsi.shape[0], hsi.shape[1], device)
    pred_test_only = np.zeros_like(ts_label, dtype=np.int64)
    pred_test_only[ts_label > 0] = pred_full[ts_label > 0]

    render_label_map(pred_full, os.path.join(run_dir, 'pred_map_full.png'), title=f'RSCNet Houston2013 Run {run_id} Full Prediction')
    render_label_map(pred_test_only, os.path.join(run_dir, 'pred_map_test_only.png'), title=f'RSCNet Houston2013 Run {run_id} Test Prediction')
    render_label_map(ts_label.astype(np.int64), os.path.join(run_dir, 'gt_test_map.png'), title='Houston2013 Official Test Labels')

    with open(os.path.join(run_dir, 'run_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'best_epoch: {best_epoch}\n')
        f.write(f'OA: {test_oa:.6f}\n')
        f.write(f'AA: {test_aa:.6f}\n')
        f.write(f'Kappa: {test_kappa:.6f}\n')
        f.write('Per-class accuracy\n')
        for i, v in enumerate(per_class):
            f.write(f'{i+1:02d}. {categories[i]}: {v:.6f}\n')

    np.savetxt(os.path.join(run_dir, 'confusion_matrix.csv'), cm, fmt='%d', delimiter=',')
    with open(os.path.join(run_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'oa': test_oa,
        'aa': test_aa,
        'kappa': test_kappa,
        'per_class': per_class,
        'pred_full': pred_full,
        'run_dir': run_dir,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='RSCNet Houston2013 5-run evaluation with classification maps')
    parser.add_argument('--data-root', type=str, default='E:/PythonProject1/Houston/')
    parser.add_argument('--out-dir', type=str, default='./rscnet_houston2013_5runs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='*', default=[1, 2, 3, 4, 5])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--patch-size', type=int, default=11)
    parser.add_argument('--pca-channels', type=int, default=30)
    parser.add_argument('--embed-dim', type=int, default=144)
    parser.add_argument('--topk-ratio', type=float, default=0.2)
    parser.add_argument('--num-rscm-layers', type=int, default=2)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    print(f'[INFO] data-root={args.data_root}')
    print(f'[INFO] out-dir={args.out_dir}')
    print(f'[INFO] device={device}')
    print(f'[INFO] num-runs={args.num_runs}, seeds={args.seeds[:args.num_runs]}')

    hsi, lidar, gt, tr_label, ts_label = load_houston_5mats(args.data_root)
    print(f'[INFO] HSI shape={hsi.shape}, LiDAR shape={lidar.shape}')
    print(f'[INFO] train labels={int((tr_label>0).sum())}, test labels={int((ts_label>0).sum())}')

    results = []
    for i in range(args.num_runs):
        seed = args.seeds[i] if i < len(args.seeds) else (i + 1)
        print('\n' + '=' * 90)
        print(f'[RUN {i+1}/{args.num_runs}] seed={seed}')
        res = train_one_run(args, i + 1, seed, hsi, lidar, gt, tr_label, ts_label, device)
        results.append(res)

    oa_arr = np.array([r['oa'] for r in results], dtype=np.float64)
    aa_arr = np.array([r['aa'] for r in results], dtype=np.float64)
    kappa_arr = np.array([r['kappa'] for r in results], dtype=np.float64)
    per_class_arr = np.stack([r['per_class'] for r in results], axis=0)

    oa_mean, oa_std = oa_arr.mean(), oa_arr.std(ddof=1) if len(oa_arr) > 1 else 0.0
    aa_mean, aa_std = aa_arr.mean(), aa_arr.std(ddof=1) if len(aa_arr) > 1 else 0.0
    kappa_mean, kappa_std = kappa_arr.mean(), kappa_arr.std(ddof=1) if len(kappa_arr) > 1 else 0.0
    per_class_mean = per_class_arr.mean(axis=0)
    per_class_std = per_class_arr.std(axis=0, ddof=1) if per_class_arr.shape[0] > 1 else np.zeros(per_class_arr.shape[1])

    summary_dir = os.path.join(args.out_dir, 'summary')
    ensure_dir(summary_dir)

    plot_per_class_mean_std(per_class_mean, per_class_std, os.path.join(summary_dir, 'per_class_mean_std.png'))

    best_idx = int(np.argmax(oa_arr))
    best_full_map = results[best_idx]['pred_full']
    best_test_map = np.zeros_like(ts_label, dtype=np.int64)
    best_test_map[ts_label > 0] = best_full_map[ts_label > 0]
    render_label_map(best_full_map, os.path.join(summary_dir, 'best_run_pred_full.png'), title=f'Best Run Full Prediction (Run {best_idx+1})')
    render_label_map(best_test_map, os.path.join(summary_dir, 'best_run_pred_test_only.png'), title=f'Best Run Test Prediction (Run {best_idx+1})')
    render_label_map(ts_label.astype(np.int64), os.path.join(summary_dir, 'gt_test_map.png'), title='Houston2013 Official Test Labels')

    csv_path = os.path.join(summary_dir, 'per_class_mean_std.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('class_id,class_name,mean,std\n')
        for i, (m, s) in enumerate(zip(per_class_mean, per_class_std), start=1):
            f.write(f'{i},{categories[i-1]},{m:.6f},{s:.6f}\n')

    txt_path = os.path.join(summary_dir, 'summary_metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('RSCNet Houston2013 5-run summary\n')
        f.write(f'OA: {oa_mean:.4f} ± {oa_std:.4f}\n')
        f.write(f'AA: {aa_mean:.4f} ± {aa_std:.4f}\n')
        f.write(f'Kappa: {kappa_mean:.4f} ± {kappa_std:.4f}\n')
        f.write('Per-class accuracy mean ± std\n')
        for i, (m, s) in enumerate(zip(per_class_mean, per_class_std), start=1):
            f.write(f'{i:02d}. {categories[i-1]}: {m:.4f} ± {s:.4f}\n')

    print('\n' + '#' * 90)
    print('Final 5-run summary')
    print(f'OA    : {oa_mean:.4f} ± {oa_std:.4f}')
    print(f'AA    : {aa_mean:.4f} ± {aa_std:.4f}')
    print(f'Kappa : {kappa_mean:.4f} ± {kappa_std:.4f}')
    print('Per-class accuracy mean ± std')
    for i, (m, s) in enumerate(zip(per_class_mean, per_class_std), start=1):
        print(f'  {i:02d}. {categories[i-1]:<18s}: {m:.4f} ± {s:.4f}')
    print(f'[INFO] summary saved to: {summary_dir}')


if __name__ == '__main__':
    main()

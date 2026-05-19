#!/usr/bin/env python3
"""
ImageNet-100 sequential trainer + corruption evaluator for all baselines.

Methods (in run order — fastest/most-cited first):
    baseline, eca, se, cbam, srm, ela, fca, scsa

LACE is excluded by default since you already have weights for it.
Pass --include-lace to also train LACE.

Usage:
    pip install setuptools<81 numpy<2.0 scikit-image<0.19 imagecorruptions
    export IMAGENET100_ROOT=/workspace/imagenet100
    python all_baselines_imagenet100.py --seed 42
    # later:
    python all_baselines_imagenet100.py --seed 7

Per-method outputs:
    /workspace/baselines/<method>_seed<S>/best.pt
    /workspace/baselines/<method>_seed<S>/log.csv
    /workspace/baselines/<method>_seed<S>/per_corruption.csv  (15x5)
    /workspace/baselines/<method>_seed<S>/summary.json

Estimated time per method on RTX 4090: ~6h training + ~30min eval = ~6.5h.
Eight methods × 6.5h = ~52 hours per seed.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import Bottleneck, ResNet
from PIL import Image


# ───── compat shims for imagecorruptions on modern stacks ─────────────
import numpy as _np
if not hasattr(_np, 'float_'):
    _np.float_ = _np.float64
if not hasattr(_np, 'bool_'):
    _np.bool_ = bool
if not hasattr(_np, 'int_'):
    _np.int_ = _np.int64
try:
    import skimage.filters
    _orig_gaussian = skimage.filters.gaussian
    def _gaussian_compat(image, *args, **kwargs):
        if "multichannel" in kwargs:
            mc = kwargs.pop("multichannel")
            if mc and "channel_axis" not in kwargs:
                kwargs["channel_axis"] = -1
        return _orig_gaussian(image, *args, **kwargs)
    skimage.filters.gaussian = _gaussian_compat
except ImportError:
    pass


# ───── config ─────────────────────────────────────────────────────────
IMAGENET100_ROOT = os.environ.get("IMAGENET100_ROOT", "/workspace/imagenet100")
OUTPUT_DIR = os.environ.get("BASELINES_OUTPUT", "/workspace/baselines")
EPOCHS = 90
BATCH_SIZE = 256
LR = 0.1
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 8


# =====================================================================
# Attention modules — all 8 implementations
# =====================================================================

# ─── SE (Hu et al., CVPR 2018) ────────────────────────────────────────
class SE(nn.Module):
    """GAP → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid. r=16, min hidden=8."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ─── CBAM (Woo et al., ECCV 2018) ─────────────────────────────────────
class CBAM(nn.Module):
    """
    Channel attention (GAP + GMP through shared MLP) followed by
    spatial attention (7×7 conv on [avg; max] along channel axis).
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        pad = spatial_kernel // 2
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=pad, bias=False)

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        cha = torch.sigmoid(avg + mx)
        x = x * cha
        sa_in = torch.cat([x.mean(dim=1, keepdim=True),
                           x.amax(dim=1, keepdim=True)], dim=1)
        sa = torch.sigmoid(self.spatial(sa_in))
        return x * sa


# ─── ECA (Wang et al., CVPR 2020) ─────────────────────────────────────
class ECA(nn.Module):
    """GAP → 1D conv with adaptive kernel size k=|log2(C)/γ + b/γ|_odd."""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        k = max(k, 3)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * torch.sigmoid(y)


# ─── SRM (Lee et al., ICCV 2019) ──────────────────────────────────────
class SRM(nn.Module):
    """
    Style pool (μ, σ per channel) → channel-wise FC (depthwise Conv1d k=2)
    → BN1d → sigmoid.
    """
    def __init__(self, channels):
        super().__init__()
        self.cfc = nn.Conv1d(channels, channels, kernel_size=2,
                             groups=channels, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        mu = x.mean(dim=[2, 3])
        sigma = x.std(dim=[2, 3], unbiased=False)
        u = torch.stack([mu, sigma], dim=2)            # (B, C, 2)
        z = self.cfc(u).squeeze(-1)                    # (B, C)
        return x * torch.sigmoid(self.bn(z)).view(b, c, 1, 1)


# ─── FCA-Net (Qin et al., ICCV 2021) ──────────────────────────────────
def _dct_basis(pos, freq, n):
    if freq == 0:
        return 1.0 / math.sqrt(n)
    return math.sqrt(2.0 / n) * math.cos(math.pi * (pos + 0.5) * freq / n)


def _build_dct_filter(h, w, mapper_x, mapper_y, channels):
    """
    2D DCT basis as a (channels, h, w) tensor. Channels are split into
    len(mapper_x) groups; each group uses a different (u, v) frequency
    component.
    """
    filt = torch.zeros(channels, h, w)
    n_groups = len(mapper_x)
    c_part = channels // n_groups
    for i, (ux, vy) in enumerate(zip(mapper_x, mapper_y)):
        for tx in range(h):
            for ty in range(w):
                val = _dct_basis(tx, ux, h) * _dct_basis(ty, vy, w)
                filt[i * c_part:(i + 1) * c_part, tx, ty] = val
    return filt


# Top-16 frequencies from the official FCA-Net paper/repo, ordered by
# importance (validated on ImageNet ResNet-50).
FCA_TOP16_X = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2]
FCA_TOP16_Y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2]


class FCA(nn.Module):
    """
    Frequency Channel Attention. C channels split into 16 groups, each
    dotted with a different 2D DCT basis. Result goes through SE-style
    FC bottleneck.

    For feature maps smaller than the largest frequency index, frequencies
    are clamped to (min(H,W)-1). For h=1 or w=1 (degenerate), falls back
    to GAP — which is exactly the (0,0) DCT component.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.mapper_x = FCA_TOP16_X
        self.mapper_y = FCA_TOP16_Y
        # SE-style bottleneck on top of the 16-frequency descriptor
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )
        self._cached = {}

    def _get_filter(self, h, w, device, dtype):
        key = (h, w, str(device), str(dtype))
        if key not in self._cached:
            # Clamp frequency indices to be < min(h, w)
            cap = max(min(h, w) - 1, 0)
            mx = [min(f, cap) for f in self.mapper_x]
            my = [min(f, cap) for f in self.mapper_y]
            f = _build_dct_filter(h, w, mx, my, self.channels).to(
                device=device, dtype=dtype)
            self._cached[key] = f
        return self._cached[key]

    def forward(self, x):
        b, c, h, w = x.shape
        if h < 2 or w < 2:
            y = x.mean(dim=[2, 3])                     # (B, C)
        else:
            f = self._get_filter(h, w, x.device, x.dtype)
            y = (x * f.unsqueeze(0)).sum(dim=[2, 3])   # (B, C)
        s = self.fc(y).view(b, c, 1, 1)
        return x * s


# ─── ELA (Xu & Wan, 2024) ─────────────────────────────────────────────
class ELA(nn.Module):
    """
    Efficient Local Attention. Horizontal and vertical strip pooling,
    each branch goes through a depthwise 1D conv (k=7, groups=C) and
    GroupNorm → sigmoid. The two direction maps multiply with x.
    """
    def __init__(self, channels, kernel_size=7, gn_groups=16):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        # Pick a group count that divides channels
        g = gn_groups
        while channels % g != 0 and g > 1:
            g //= 2
        g = max(g, 1)
        self.conv_h = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, groups=channels, bias=False)
        self.conv_w = nn.Conv1d(channels, channels, kernel_size,
                                padding=pad, groups=channels, bias=False)
        self.gn_h = nn.GroupNorm(g, channels)
        self.gn_w = nn.GroupNorm(g, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        zh = x.mean(dim=3)                              # (B, C, H)
        zw = x.mean(dim=2)                              # (B, C, W)
        ah = torch.sigmoid(self.gn_h(self.conv_h(zh)))   # (B, C, H)
        aw = torch.sigmoid(self.gn_w(self.conv_w(zw)))   # (B, C, W)
        return x * ah.unsqueeze(-1) * aw.unsqueeze(-2)


# ─── SCSA (Si et al., 2025) ───────────────────────────────────────────
class SCSA(nn.Module):
    """
    Spatial and Channel Synergistic Attention. SMSA → PCSA pipeline.

    SMSA (Shareable Multi-Semantic Spatial Attention):
      - Split C channels into K=4 sub-features
      - Strip pool each sub-feature along H and W
      - Apply depth-shared 1D convs at kernels {3,5,7,9} to each sub-feature
      - GroupNorm, sigmoid, broadcast back, multiply with x

    PCSA (Progressive Channel-wise Self-Attention):
      - Pool x to a small spatial size
      - Compute channel-wise Q/K/V via 1D conv along channel axis
      - Single-head scaled dot-product attention over channels
      - Residual gating with sigmoid
    """
    def __init__(self, channels, K=4, pcsa_pool_size=7):
        super().__init__()
        assert channels % K == 0, \
            f"SCSA: C={channels} must be divisible by K={K}"
        self.K = K
        self.Cs = channels // K
        kernels = (3, 5, 7, 9)
        assert len(kernels) == K

        # SMSA: one 1D conv per (direction, sub-feature). Each acts on
        # its own Cs-channel slice as a depthwise (groups=Cs) conv.
        self.smsa_convs_h = nn.ModuleList([
            nn.Conv1d(self.Cs, self.Cs, k, padding=k // 2,
                      groups=self.Cs, bias=False)
            for k in kernels])
        self.smsa_convs_w = nn.ModuleList([
            nn.Conv1d(self.Cs, self.Cs, k, padding=k // 2,
                      groups=self.Cs, bias=False)
            for k in kernels])
        self.smsa_gn_h = nn.GroupNorm(K, channels)
        self.smsa_gn_w = nn.GroupNorm(K, channels)

        # PCSA: pool to (pcsa_pool_size × pcsa_pool_size), then channel
        # self-attention. Use channel compression (groups) to match the
        # paper's parameter budget — full Conv2d Q/K/V at C=2048 would
        # explode the model size. The paper uses depthwise-style channel
        # projection where Q/K/V are produced through grouped 1×1 convs.
        self.pcsa_pool_size = pcsa_pool_size
        # Use groups=channels (depthwise) for Q/K/V to keep PCSA cheap.
        # This is the key compression in the official SCSA implementation:
        # per-channel projections rather than full cross-channel ones.
        self.pcsa_q = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.pcsa_k = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.pcsa_v = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.pcsa_proj = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)

    def _smsa(self, x):
        b, c, h, w = x.shape
        # Strip pool along W and H
        zh = x.mean(dim=3)                              # (B, C, H)
        zw = x.mean(dim=2)                              # (B, C, W)
        # Split into K sub-features along channel axis
        zh_parts = zh.chunk(self.K, dim=1)              # list of (B, Cs, H)
        zw_parts = zw.chunk(self.K, dim=1)
        # Each sub-feature uses its own kernel
        out_h = torch.cat(
            [conv(p) for conv, p in zip(self.smsa_convs_h, zh_parts)],
            dim=1)                                      # (B, C, H)
        out_w = torch.cat(
            [conv(p) for conv, p in zip(self.smsa_convs_w, zw_parts)],
            dim=1)                                      # (B, C, W)
        out_h = torch.sigmoid(self.smsa_gn_h(out_h))    # (B, C, H)
        out_w = torch.sigmoid(self.smsa_gn_w(out_w))    # (B, C, W)
        return x * out_h.unsqueeze(-1) * out_w.unsqueeze(-2)

    def _pcsa(self, x):
        b, c, h, w = x.shape
        # Pool to a small fixed spatial size to make channel-wise attention
        # tractable. Use min(H, W, pool_size) to be safe on small features.
        ps = min(self.pcsa_pool_size, h, w)
        ps = max(ps, 1)
        xp = F.adaptive_avg_pool2d(x, ps)               # (B, C, ps, ps)
        q = self.pcsa_q(xp).flatten(2)                   # (B, C, ps*ps)
        k = self.pcsa_k(xp).flatten(2)
        v = self.pcsa_v(xp).flatten(2)
        # Channel-wise: each channel is a "token", spatial dims are the
        # feature vector. Q (B,C,L) · K^T (B,L,C) → (B,C,C) attention.
        scale = q.size(-1) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, C, C)
        out = attn @ v                                   # (B, C, ps*ps)
        out = out.view(b, c, ps, ps)
        out = self.pcsa_proj(out)
        # Broadcast back to original spatial size as a gating multiplier
        out = F.interpolate(out, size=(h, w), mode='bilinear',
                            align_corners=False)
        return x * torch.sigmoid(out)

    def forward(self, x):
        x = self._smsa(x)
        x = self._pcsa(x)
        return x


# ─── LACE (proposed) ──────────────────────────────────────────────────
class LACE(nn.Module):
    """Lag-1 Autocorrelation-based Channel Excitation. Reference V0."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.empty(1, channels, 3))
        nn.init.normal_(self.w, std=0.01)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xt = x - mu
        if h > 1 and w > 1:
            r_h = (xt[:, :, :, :-1] * xt[:, :, :, 1:]).mean(dim=[2, 3], keepdim=True)
            r_v = (xt[:, :, :-1, :] * xt[:, :, 1:, :]).mean(dim=[2, 3], keepdim=True)
            rho = (r_h + r_v) / (2.0 * (var + self.eps))
        else:
            rho = torch.zeros_like(mu)
        phi = (sigma * (1.0 - rho)).detach()
        stats = torch.cat([mu.view(b, c, 1),
                           sigma.view(b, c, 1),
                           phi.view(b, c, 1)], dim=2)
        a = (stats * self.w).sum(dim=2)
        s = self.bn(a).sigmoid().view(b, c, 1, 1)
        return x * s


# ─── Factory ──────────────────────────────────────────────────────────
def make_attention(method, channels):
    method = method.lower()
    table = {
        'baseline': lambda c: nn.Identity(),
        'se':       lambda c: SE(c),
        'cbam':     lambda c: CBAM(c),
        'eca':      lambda c: ECA(c),
        'srm':      lambda c: SRM(c),
        'fca':      lambda c: FCA(c),
        'ela':      lambda c: ELA(c),
        'scsa':     lambda c: SCSA(c),
        'lace':     lambda c: LACE(c),
    }
    if method not in table:
        raise ValueError(f"Unknown method '{method}'. Options: {list(table)}")
    return table[method](channels)


# =====================================================================
# ResNet-50 backbone with attention placement
# =====================================================================
class AttnBottleneck(Bottleneck):
    """torchvision Bottleneck with attention after bn3, before residual add."""
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 attention='baseline'):
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)
        self.attn = make_attention(attention, planes * self.expansion)

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


def build_resnet50(num_classes, attention):
    class _B(AttnBottleneck):
        def __init__(self, *a, **kw):
            super().__init__(*a, attention=attention, **kw)
    model = ResNet(_B, [3, 4, 6, 3])
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =====================================================================
# Data
# =====================================================================
def build_loaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(), normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])
    train_set = ImageFolder(os.path.join(IMAGENET100_ROOT, "train"), transform=train_tf)
    val_set = ImageFolder(os.path.join(IMAGENET100_ROOT, "val"), transform=val_tf)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    return train_loader, val_loader


class CorruptedVal(Dataset):
    """Apply a single (corruption, severity) on PIL val images."""
    def __init__(self, corruption, severity):
        from imagecorruptions import corrupt
        self.corrupt_fn = corrupt
        self.corruption = corruption
        self.severity = severity
        self.pre = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224)])
        self.post = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.dataset = ImageFolder(os.path.join(IMAGENET100_ROOT, "val"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.pre(img)
        arr = np.array(img.convert("RGB"))
        arr = self.corrupt_fn(arr, corruption_name=self.corruption,
                              severity=self.severity)
        return self.post(Image.fromarray(arr)), label


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


# =====================================================================
# Train / eval
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for img, tgt in loader:
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(img)
        correct += (logits.argmax(1) == tgt).sum().item()
        total += img.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, opt, scaler, criterion, device, epoch):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    t0 = time.time()
    for i, (img, tgt) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = model(img)
            loss = criterion(logits, tgt)
        scaler.scale(loss).backward()
        # Gradient clipping helps with the few methods that have
        # bigger attention modules (SCSA, FCA).
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * img.size(0)
        correct += (logits.argmax(1) == tgt).sum().item()
        total += img.size(0)
        if (i + 1) % 50 == 0:
            print(f"  ep{epoch} step {i+1}/{len(loader)} "
                  f"loss {loss_sum/total:.4f} top1 {100*correct/total:.2f}% "
                  f"({time.time()-t0:.0f}s)")
    return loss_sum / total, 100 * correct / total


def train_method(method, seed, train_loader, val_loader):
    set_seed(seed)
    device = torch.device("cuda")
    run_dir = Path(OUTPUT_DIR) / f"{method}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already trained
    best_ckpt = run_dir / "best.pt"
    if best_ckpt.exists():
        print(f"=== {method}_seed{seed}: best.pt already exists, "
              f"skipping training (delete it to retrain) ===")
        return best_ckpt

    model = build_resnet50(num_classes=100, attention=method).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_attn = sum(p.numel() for n, p in model.named_parameters()
                 if 'attn' in n)
    print(f"=== TRAIN {method}_seed{seed}: "
          f"params={n_params/1e6:.2f}M (attn={n_attn/1e6:.3f}M) ===")

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    def lrlam(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        p = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrlam)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_top1", "val_top1"])

    best = 0.0
    for ep in range(EPOCHS):
        cur_lr = opt.param_groups[0]["lr"]
        print(f"\n--- {method}_seed{seed} epoch {ep+1}/{EPOCHS} lr={cur_lr:.4f} ---")
        tl, tt = train_one_epoch(model, train_loader, opt, scaler, criterion, device, ep+1)
        v1 = evaluate(model, val_loader, device)
        sched.step()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep+1, cur_lr, tl, tt, v1])
        if v1 > best:
            best = v1
            torch.save({"model": model.state_dict(), "method": method,
                        "seed": seed, "epoch": ep, "top1": v1},
                       best_ckpt)
        print(f"{method}_seed{seed} ep{ep+1}: train {tt:.2f}% val {v1:.2f}% best {best:.2f}%")

    print(f"\n=== {method}_seed{seed} training done: best top1 = {best:.2f}% ===")
    return best_ckpt


def eval_method(method, seed, ckpt_path):
    """Full 15x5 corruption eval on the best checkpoint."""
    device = torch.device("cuda")
    run_dir = Path(OUTPUT_DIR) / f"{method}_seed{seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        print(f"=== {method}_seed{seed}: summary.json exists, skipping eval ===")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n=== EVAL {method}_seed{seed} ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_resnet50(num_classes=100, attention=method).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Sanity: clean accuracy
    _, val_loader = build_loaders()
    clean = evaluate(model, val_loader, device)
    print(f"  clean top1: {clean:.2f}% (ckpt recorded: {ckpt.get('top1', '?')})")

    csv_path = run_dir / "per_corruption.csv"
    raw = {}
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corruption", "severity", "top1"])
        t_start = time.time()
        done = 0
        for corruption in CORRUPTIONS:
            raw[corruption] = {}
            for severity in [1, 2, 3, 4, 5]:
                t0 = time.time()
                ds = CorruptedVal(corruption, severity)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True)
                acc = evaluate(model, loader, device)
                raw[corruption][severity] = acc
                writer.writerow([corruption, severity, acc])
                f.flush()
                done += 1
                elapsed = time.time() - t0
                total_e = time.time() - t_start
                print(f"  [{done:2d}/75] {corruption:22s} sev{severity}  "
                      f"top1={acc:6.2f}%  ({elapsed:.0f}s, total {total_e/60:.1f}min)")

    per_corruption_mean = {
        c: float(np.mean([raw[c][s] for s in range(1, 6)]))
        for c in CORRUPTIONS
    }
    per_severity_mean = {
        s: float(np.mean([raw[c][s] for c in CORRUPTIONS]))
        for s in range(1, 6)
    }
    category_mean = {
        cat: float(np.mean([raw[c][s] for c in members for s in range(1, 6)]))
        for cat, members in CATEGORIES.items()
    }
    overall_mca = float(np.mean(
        [raw[c][s] for c in CORRUPTIONS for s in range(1, 6)]))

    summary = {
        "method": method,
        "seed": seed,
        "clean_top1": clean,
        "ckpt_recorded_top1": ckpt.get("top1"),
        "overall_mCA": overall_mca,
        "category_mCA": category_mean,
        "per_severity_mean": per_severity_mean,
        "per_corruption_mean": per_corruption_mean,
        "raw": raw,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {method}_seed{seed} eval done ===")
    print(f"  clean      {clean:.2f}%")
    print(f"  overall mCA {overall_mca:.2f}%")
    for cat, val in category_mean.items():
        print(f"  {cat:8s}   {val:.2f}%")
    return summary


# =====================================================================
# Main: sequential runner
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", type=str, default=None,
                    help="Comma-separated method list. Default: baseline,eca,"
                         "se,cbam,srm,ela,fca,scsa")
    ap.add_argument("--include-lace", action="store_true",
                    help="Also train LACE (by default skipped — assumes you "
                         "have LACE weights already)")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Train only, don't run corruption eval")
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training, only run eval on existing checkpoints")
    args = ap.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        # Order: fastest/simplest first so user gets early signal if
        # anything is broken. ECA is the cheapest, baseline is the
        # reference. Then progressively more complex modules.
        methods = ['baseline', 'eca', 'se', 'cbam', 'srm', 'ela', 'fca', 'scsa']
        if args.include_lace:
            methods.append('lace')

    print(f"\n{'='*60}")
    print(f"SEQUENTIAL RUN: seed={args.seed}")
    print(f"methods: {methods}")
    print(f"{'='*60}\n")

    # Build loaders once and reuse — saves a few minutes per method.
    train_loader, val_loader = build_loaders()

    overall_t0 = time.time()
    summaries = []
    for i, method in enumerate(methods, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(methods)}] {method.upper()} (seed {args.seed})")
        print(f"# total elapsed: {(time.time()-overall_t0)/3600:.1f} hours")
        print(f"{'#'*60}")

        if args.eval_only:
            ckpt = Path(OUTPUT_DIR) / f"{method}_seed{args.seed}" / "best.pt"
            if not ckpt.exists():
                print(f"  no checkpoint at {ckpt}, skipping")
                continue
        else:
            ckpt = train_method(method, args.seed, train_loader, val_loader)

        if not args.skip_eval:
            summary = eval_method(method, args.seed, ckpt)
            summaries.append(summary)

    # Build seed-level summary table
    if summaries:
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY: seed {args.seed}")
        print(f"{'='*60}")
        print(f"{'method':<10} {'clean':>7} {'mCA':>7} {'noise':>7} {'blur':>7} {'weather':>8} {'digital':>8}")
        for s in summaries:
            cm = s['category_mCA']
            print(f"{s['method']:<10} "
                  f"{s['clean_top1']:>7.2f} "
                  f"{s['overall_mCA']:>7.2f} "
                  f"{cm['noise']:>7.2f} "
                  f"{cm['blur']:>7.2f} "
                  f"{cm['weather']:>8.2f} "
                  f"{cm['digital']:>8.2f}")

        # Cross-method summary CSV for paper-table generation
        out_csv = Path(OUTPUT_DIR) / f"summary_seed{args.seed}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "clean", "mCA", "noise", "blur", "weather", "digital"])
            for s in summaries:
                cm = s['category_mCA']
                w.writerow([s['method'], s['clean_top1'], s['overall_mCA'],
                            cm['noise'], cm['blur'], cm['weather'], cm['digital']])
        print(f"\nsummary written to {out_csv}")

    print(f"\nTOTAL elapsed: {(time.time()-overall_t0)/3600:.1f} hours")


if __name__ == "__main__":
    main()

"""Differentiable attack layers used between encoder and decoder during training.

Every attack has the signature fn(encoded, cover) -> attacked image, values in [0,1].
Gradients flow through all of them, so the encoder learns to survive them.
"""
import math
import random

import torch
import torch.nn.functional as F


def _identity(enc, cover):
    return enc


def _noise(enc, cover):
    std = random.uniform(0.005, 0.04)
    return (enc + torch.randn_like(enc) * std).clamp(0, 1)


def _blur(enc, cover):
    sigma, k = random.uniform(0.4, 1.6), 7
    ax = torch.arange(k, device=enc.device, dtype=enc.dtype) - (k - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    ker = (g[:, None] * g[None, :]).expand(3, 1, k, k).contiguous()
    return F.conv2d(F.pad(enc, (k // 2,) * 4, mode="reflect"), ker, groups=3)


def _resize(enc, cover):
    h, w = enc.shape[-2:]
    s = random.uniform(0.5, 0.9)
    small = F.interpolate(enc, scale_factor=s, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def _crop(enc, cover):
    h, w = enc.shape[-2:]
    frac = random.uniform(0.5, 1.0)  # kept area fraction
    ch = max(16, int(h * math.sqrt(frac)))
    cw = max(16, int(w * math.sqrt(frac)))
    y, x = random.randint(0, h - ch), random.randint(0, w - cw)
    return enc[:, :, y:y + ch, x:x + cw]


def _dropout(enc, cover):
    p = random.uniform(0.2, 0.6)
    keep = (torch.rand(enc.shape[0], 1, *enc.shape[-2:], device=enc.device) > p).to(enc.dtype)
    return enc * keep + cover * (1 - keep)


# ---- differentiable JPEG approximation (8x8 block DCT + quantisation, straight-through rounding)
_Q_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=torch.float32)


def _dct_matrix(n=8):
    m = torch.zeros(n, n)
    for k in range(n):
        a = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            m[k, i] = a * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return m


def _round_ste(x):
    return x + (x.round() - x).detach()


def _jpeg(enc, cover):
    q = random.randint(30, 90)
    scale = 5000 / q if q < 50 else 200 - 2 * q
    Q = (_Q_LUMA.to(enc) * scale / 100).clamp(min=1)
    D = _dct_matrix().to(enc)
    b, c, h, w = enc.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x = F.pad(enc, (0, pw, 0, ph), mode="replicate") * 255 - 128
    blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)            # B,C,nh,nw,8,8
    coef = D @ blocks @ D.t()
    rec = D.t() @ (_round_ste(coef / Q) * Q) @ D
    rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h + ph, w + pw)
    return ((rec + 128) / 255).clamp(0, 1)[:, :, :h, :w]


ATTACKS = {
    "identity": _identity, "noise": _noise, "blur": _blur, "resize": _resize,
    "crop": _crop, "dropout": _dropout, "jpeg": _jpeg,
}


def apply_random(enc, cover, names=None):
    name = random.choice(names or list(ATTACKS))
    return name, ATTACKS[name](enc, cover)

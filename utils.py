"""Helpers: message <-> bits (with optional secret key), image IO, tiled embed/decode, statistics."""
import glob
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import Decoder, Encoder

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ---------------- message ----------------
def id_to_bits(value, bits):
    if value < 0 or value >= 2 ** bits:
        raise ValueError(f"id must fit in {bits} bits")
    return np.array([(value >> (bits - 1 - i)) & 1 for i in range(bits)], dtype=np.float32)


def keystream(key, bits):
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).integers(0, 2, bits).astype(np.float32)


def make_message(value, bits, key=""):
    """ID -> bit vector; with a key the bits are XOR-masked by a key-derived stream."""
    m = id_to_bits(value, bits)
    return np.abs(m - keystream(key, bits)) if key else m


def bits_to_id(arr, key=""):
    b = np.array(arr, dtype=np.float32)
    if key:
        b = np.abs(b - keystream(key, len(b)))
    return int("".join(str(int(v)) for v in b), 2)


def false_positive_prob(matches, bits):
    """P(at least `matches` of `bits` bits agree by pure chance)."""
    return sum(math.comb(bits, k) for k in range(matches, bits + 1)) / 2 ** bits


def min_matches(bits, alpha=1e-6):
    for k in range(bits + 1):
        if false_positive_prob(k, bits) < alpha:
            return k
    return bits


# ---------------- images ----------------
def list_images(path):
    if os.path.isfile(path):
        return [path]
    files = glob.glob(os.path.join(path, "**", "*"), recursive=True)
    return sorted(p for p in files if p.lower().endswith(EXT))


def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_image(path):
    return pil_to_tensor(Image.open(path))


def save_image(t, path):
    arr = (t.squeeze(0).clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def quantize8(x):
    return (x.clamp(0, 1) * 255).round() / 255


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else 10 * math.log10(1 / mse)


# ---------------- model io ----------------
def load_model(path, device):
    ck = torch.load(path, map_location=device)
    enc, dec = Encoder(ck["bits"], ck["ch"]).to(device), Decoder(ck["bits"], ck["ch"]).to(device)
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])
    return enc.eval(), dec.eval(), ck["bits"]


@torch.no_grad()
def embed_image(enc, img, msg, tile=512, pad=16, alpha=1.0):
    """Tiled embedding so large photos fit in memory. alpha scales the watermark strength."""
    _, _, H, W = img.shape
    out = torch.empty_like(img)
    padded = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            patch = padded[:, :, y:y1 + 2 * pad, x:x1 + 2 * pad]
            e = patch + alpha * (enc(patch, msg) - patch)
            out[:, :, y:y1, x:x1] = e[:, :, pad:pad + (y1 - y), pad:pad + (x1 - x)]
    return out


@torch.no_grad()
def decode_image(dec, img, tile=512):
    """Average the logits of all tiles (weighted by area)."""
    _, _, H, W = img.shape
    total, n = 0, 0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            patch = img[:, :, y:y + tile, x:x + tile]
            if min(patch.shape[-2:]) < 16:
                continue
            area = patch.shape[-1] * patch.shape[-2]
            total = total + dec(patch) * area
            n += area
    return total / n if n else dec(img)

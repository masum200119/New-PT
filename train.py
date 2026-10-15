"""Train the watermark encoder/decoder.

Example (GPU):  python train.py --data path/to/images --val path/to/val_images --epochs 40
Example (quick CPU test): python train.py --data imgs --size 64 --ch 32 --epochs 5
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from distortions import ATTACKS, apply_random
from model import Decoder, Encoder
from utils import list_images, psnr, quantize8, save_image


class Crops(Dataset):
    """Random size x size crops from a folder of images (small folders are repeated)."""

    def __init__(self, folder, size, repeat):
        self.files = list_images(folder)
        if not self.files:
            raise SystemExit(f"no images found in {folder}")
        self.size, self.repeat = size, repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, i):
        img = Image.open(self.files[i % len(self.files)]).convert("RGB")
        w, h = img.size
        if min(w, h) < self.size:
            s = self.size / min(w, h)
            img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.BICUBIC)
            w, h = img.size
        x, y = random.randint(0, w - self.size), random.randint(0, h - self.size)
        img = img.crop((x, y, x + self.size, y + self.size))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)


def ste_quant(x):
    """8-bit rounding in the forward pass, identity gradient (the image will be saved as PNG)."""
    return x + (quantize8(x) - x).detach()


def evaluate(enc, dec, batch, bits, dev):
    enc.eval(), dec.eval()
    state = random.getstate()
    res = {}
    with torch.no_grad():
        msg = torch.randint(0, 2, (batch.shape[0], bits), device=dev).float()
        encoded = quantize8(enc(batch, msg))
        for name, fn in ATTACKS.items():
            random.seed(0)
            res[name] = ((dec(fn(encoded, batch)) > 0).float() == msg).float().mean().item()
    random.setstate(state)
    enc.train(), dec.train()
    return res, psnr(encoded, batch), encoded


def make_grid(cover, encoded, n=4):
    resid = ((encoded - cover) * 10 + 0.5).clamp(0, 1)   # residual amplified x10
    rows = [torch.cat([cover[i], encoded[i], resid[i]], dim=2) for i in range(min(n, cover.shape[0]))]
    return torch.cat(rows, dim=1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="", help="optional held-out image folder")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_msg", type=float, default=1.0)
    ap.add_argument("--w_img", type=float, default=10.0, help="higher = more invisible, less robust")
    ap.add_argument("--warmup", type=int, default=1000, help="steps to ramp image loss from 0")
    ap.add_argument("--attacks", default="", help="comma list, default: all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    names = [n for n in args.attacks.split(",") if n] or None
    ds = Crops(args.data, args.size, args.repeat)
    val_ds = Crops(args.val, args.size, 1) if args.val else ds
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers)
    val_batch = torch.stack([val_ds[i % len(val_ds)] for i in range(args.batch)]).to(dev)

    enc, dec = Encoder(args.bits, args.ch).to(dev), Decoder(args.bits, args.ch).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "loss_msg", "loss_img", "psnr"] + list(ATTACKS))

    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        for cover in loader:
            cover = cover.to(dev)
            msg = torch.randint(0, 2, (cover.shape[0], args.bits), device=dev).float()
            encoded = ste_quant(enc(cover, msg))
            _, attacked = apply_random(encoded, cover, names)
            l_msg = F.binary_cross_entropy_with_logits(dec(attacked), msg)
            l_img = F.mse_loss(encoded, cover)
            w_img = args.w_img * min(1.0, step / max(1, args.warmup))
            loss = args.w_msg * l_msg + w_img * l_img
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step:6d}  msg {l_msg.item():.4f}  img_mse {l_img.item():.5f}  {time.time() - t0:.0f}s", flush=True)

        res, p, encoded = evaluate(enc, dec, val_batch, args.bits, dev)
        print(f"== epoch {epoch}  PSNR {p:.2f} dB  " + "  ".join(f"{k} {v * 100:.0f}%" for k, v in res.items()), flush=True)
        writer.writerow([epoch, step, f"{l_msg.item():.4f}", f"{l_img.item():.6f}", f"{p:.2f}"] + [f"{v:.4f}" for v in res.values()])
        log.flush()
        save_image(make_grid(val_batch, encoded), os.path.join(args.out, "sample.png"))
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "bits": args.bits, "ch": args.ch},
                   os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()

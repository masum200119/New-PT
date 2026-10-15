"""Robustness test with REAL attacks (PIL JPEG, resize, crop, blur, noise, rotation).

python test_robust.py --ckpt runs/exp1/ckpt.pt --input marked --id 123456 --key mysecret --limit 20
"""
import argparse
import csv
import io

import numpy as np
import torch
from PIL import Image, ImageFilter

from utils import (decode_image, list_images, load_model, make_message, min_matches,
                   pil_to_tensor)


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize(img, s):
    w, h = img.size
    return img.resize((int(w * s), int(h * s)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def crop(img, area):
    w, h = img.size
    cw, ch = int(w * area ** 0.5), int(h * area ** 0.5)
    x, y = (w - cw) // 2, (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch))


def noise(img, std):
    a = np.asarray(img, dtype=np.float32) + np.random.randn(img.size[1], img.size[0], 3) * std * 255
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


ATTACKS = {
    "none": lambda im: im,
    "jpeg q=90": lambda im: jpeg(im, 90),
    "jpeg q=70": lambda im: jpeg(im, 70),
    "jpeg q=50": lambda im: jpeg(im, 50),
    "resize 70%": lambda im: resize(im, 0.7),
    "resize 50%": lambda im: resize(im, 0.5),
    "crop 80% area": lambda im: crop(im, 0.8),
    "crop 50% area": lambda im: crop(im, 0.5),
    "blur r=1": lambda im: im.filter(ImageFilter.GaussianBlur(1)),
    "noise 2%": lambda im: noise(im, 0.02),
    "rotate 3 deg": lambda im: im.rotate(3, resample=Image.BILINEAR),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--csv", default="robustness.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    _, dec, bits = load_model(args.ckpt, dev)
    expect = make_message(args.id, bits, args.key)
    need = min_matches(bits, 1e-6)
    paths = list_images(args.input)[:args.limit]
    rows = []
    print(f"{'attack':16s} mean bit-acc   detected(p<1e-6)")
    for name, fn in ATTACKS.items():
        accs, hits = [], 0
        for p in paths:
            att = pil_to_tensor(fn(Image.open(p).convert("RGB"))).to(dev)
            got = (decode_image(dec, att) > 0).float().squeeze(0).cpu().numpy()
            m = int((got == expect).sum())
            accs.append(m / bits)
            hits += m >= need
        rows.append([name, f"{np.mean(accs):.4f}", f"{hits}/{len(paths)}"])
        print(f"{name:16s} {np.mean(accs) * 100:6.1f}%        {hits}/{len(paths)}")
    with open(args.csv, "w", newline="") as f:
        csv.writer(f).writerows([["attack", "mean_bit_acc", "detected"]] + rows)


if __name__ == "__main__":
    main()

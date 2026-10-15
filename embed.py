"""Embed an ID into every image of a folder (output is always lossless PNG).

python embed.py --ckpt runs/exp1/ckpt.pt --input images --output marked --id 123456 --key mysecret
Add --per_image to give each image its own ID (base id + index); the mapping is saved in ids.csv.
"""
import argparse
import csv
import os

import torch

from utils import (embed_image, list_images, load_image, load_model, make_message, psnr,
                   quantize8, save_image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="marked")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--alpha", type=float, default=1.0, help="watermark strength multiplier")
    ap.add_argument("--per_image", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    enc, _, bits = load_model(args.ckpt, dev)
    os.makedirs(args.output, exist_ok=True)
    rows = []
    for i, path in enumerate(list_images(args.input)):
        ident = args.id + i if args.per_image else args.id
        msg = torch.tensor(make_message(ident, bits, args.key), device=dev).view(1, -1)
        img = load_image(path).to(dev)
        out = quantize8(embed_image(enc, img, msg, alpha=args.alpha))
        name = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_image(out, os.path.join(args.output, name))
        rows.append([name, ident])
        print(f"{name}: id={ident}  PSNR={psnr(out, img):.2f} dB")
    with open(os.path.join(args.output, "ids.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["file", "id"]] + rows)


if __name__ == "__main__":
    main()

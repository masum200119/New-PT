"""Read the watermark from images and (optionally) test it against an expected ID.

python extract.py --ckpt runs/exp1/ckpt.pt --input marked --id 123456 --key mysecret
"""
import argparse

import torch

from utils import (bits_to_id, decode_image, false_positive_prob, list_images, load_image,
                   load_model, make_message, min_matches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--id", type=int, default=None, help="expected ID (enables detection verdict)")
    ap.add_argument("--key", default="")
    ap.add_argument("--alpha", type=float, default=1e-6, help="false-positive rate for the verdict")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    _, dec, bits = load_model(args.ckpt, dev)
    need = min_matches(bits, args.alpha)
    for path in list_images(args.input):
        got = (decode_image(dec, load_image(path).to(dev)) > 0).float().squeeze(0).cpu().numpy()
        line = f"{path}: decoded id={bits_to_id(got, args.key)}"
        if args.id is not None:
            expect = make_message(args.id, bits, args.key)
            m = int((got == expect).sum())
            p = false_positive_prob(m, bits)
            line += f"  matches {m}/{bits}  p={p:.1e}  " + ("DETECTED" if m >= need else "not detected")
        print(line)


if __name__ == "__main__":
    main()

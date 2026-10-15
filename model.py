"""Encoder / Decoder for a HiDDeN-style invisible watermark prototype.

Both networks are fully convolutional, so they accept any image size.
The decoder ends with global average pooling, so it also works on cropped images.
"""
import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """cover image (B,3,H,W in [0,1]) + message bits (B,bits in {0,1}) -> watermarked image."""

    def __init__(self, bits=32, ch=64, blocks=4, max_amp=0.1):
        super().__init__()
        self.bits, self.max_amp = bits, max_amp
        layers = [conv_block(3, ch)] + [conv_block(ch, ch) for _ in range(blocks - 1)]
        self.features = nn.Sequential(*layers)
        # features + message map + original image -> 3-channel residual
        self.fuse = nn.Sequential(conv_block(ch + bits + 3, ch), nn.Conv2d(ch, 3, 1))

    def forward(self, img, msg):
        b, _, h, w = img.shape
        x = img * 2 - 1
        f = self.features(x)
        m = (msg * 2 - 1).view(b, self.bits, 1, 1).expand(b, self.bits, h, w)
        res = torch.tanh(self.fuse(torch.cat([f, m, x], dim=1))) * self.max_amp
        return (img + res).clamp(0, 1)


class Decoder(nn.Module):
    """(possibly attacked) image -> logits for each message bit."""

    def __init__(self, bits=32, ch=64, blocks=6):
        super().__init__()
        layers = [conv_block(3, ch)] + [conv_block(ch, ch) for _ in range(blocks - 1)]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, bits)

    def forward(self, x):
        h = self.net(x * 2 - 1)
        return self.fc(self.pool(h).flatten(1))

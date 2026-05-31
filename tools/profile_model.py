import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from arb_liteunet import ARBLiteUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    model = ARBLiteUNet().eval()
    x = torch.randn(1, 3, args.height, args.width)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params / 1e6:.4f} M")

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        print(f"GFLOPs: {macs / 1e9:.4f}")
    except Exception as exc:
        print(f"GFLOPs: unavailable ({exc})")

    with torch.no_grad():
        out = model(x)
    print(f"Output type: {type(out)}")


if __name__ == "__main__":
    main()

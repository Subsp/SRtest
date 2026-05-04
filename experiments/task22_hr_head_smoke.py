"""
Smoke test for Phase 2.2 HRGeometricPriorHead (shape + param count).

  python task22_hr_head_smoke.py [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from configs import LR_SIZE, SR_SIZE, SR_SCALE
from models.hr_head import HRGeometricPriorHead, count_parameters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = args.device

    B, V = 1, 4
    H, W = LR_SIZE, LR_SIZE
    model = HRGeometricPriorHead(
        use_rgb=True,
        use_sr_prior=True,
        base_channels=96,
        sr_scale=SR_SCALE,
    ).to(dev)
    n_params = count_parameters(model)
    print(f"HRGeometricPriorHead trainable params: {n_params / 1e6:.2f}M")

    depth_lr = torch.rand(B, V, 1, H, W, device=dev) * 5.0 + 0.5
    rgb_lr = torch.rand(B, V, 3, H, W, device=dev)
    sr_hr = torch.rand(B, V, 3, SR_SIZE, SR_SIZE, device=dev)

    model.eval()
    with torch.no_grad():
        out = model(depth_lr=depth_lr, rgb_lr=rgb_lr, sr_prior_hr=sr_hr)

    for k, t in out.items():
        print(f"  {k}: {tuple(t.shape)} dtype={t.dtype}")
    assert out["depth_hr"].shape == (B, V, 1, SR_SIZE, SR_SIZE)
    assert out["normal_hr"].shape == (B, V, 3, SR_SIZE, SR_SIZE)
    assert out["confidence_hr"].shape == (B, V, 1, SR_SIZE, SR_SIZE)
    print("OK — shapes match SR_SIZE.")


if __name__ == "__main__":
    main()

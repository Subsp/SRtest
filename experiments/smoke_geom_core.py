#!/usr/bin/env python3
"""
Smoke-test experiments/geom_prior.py only (no SOF branch, no full VGGT pipeline).

Runs:
  1) Synthetic autograd: loss decreases when pred → prior under fixed noise prior.
  2) Optional: load flat-directory oracle .npy (e.g. kitchen task02 outputs) and
     treat noisy depth as faux "render" prediction.

Examples:
  python smoke_geom_core.py

  python smoke_geom_core.py \\
      --oracle_npy_dir /root/autodl-tmp/SRtest/experiments/results/task02/oracle/kitchen \\
      --max_npy_files 8
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geom_prior import (  # noqa: E402
    PriorPackDepth,
    geom_depth_loss_l1,
    median_scale_align_factor,
)


def _pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "cuda" and not torch.cuda.is_available():
        print("[smoke_geom_core] CUDA requested but unavailable → cpu")
        return torch.device("cpu")
    raise ValueError(pref)


def test_synthetic(device: torch.device) -> None:
    torch.manual_seed(0)
    h, w = 96, 128
    prior = torch.rand(h, w, device=device, dtype=torch.float32) * 3.0 + 0.5
    noise = 0.12 * torch.randn(h, w, device=device)
    pred = (prior * (1.0 + noise)).clone().detach().requires_grad_(True)

    pack = PriorPackDepth(depth=prior, confidence=torch.ones_like(prior))
    loss0, diag0 = geom_depth_loss_l1(pred, pack)
    loss0.backward()
    assert pred.grad is not None
    assert torch.isfinite(loss0)

    scale = diag0["scale"].item()
    print(f"[synthetic] loss={loss0.item():.6f} scale_factor={scale:.6f} valid={diag0['valid_ratio'].item():.4f}")

    # Second step toward prior should reduce symmetric error (optimization toy)
    with torch.no_grad():
        pred.data = pred.data - 0.3 * pred.grad
        pred.grad = None

    pred.requires_grad_(True)
    loss1, _ = geom_depth_loss_l1(pred, pack)
    assert loss1.item() <= loss0.item() + 1e-5, "expected descent step on L_geom proxy"
    print(f"[synthetic] after shallow grad step loss={loss1.item():.6f} (<= {loss0.item():.6f})")


def test_scale_identity(device: torch.device) -> None:
    torch.manual_seed(1)
    d = torch.linspace(1.0, 2.0, 50, device=device)
    dd = torch.stack([d, d * 0.98], dim=-1).reshape(10, 10)
    m = torch.ones_like(dd, dtype=torch.bool)
    s = median_scale_align_factor(dd, dd, m)
    assert (s - 1.0).abs() < 1e-5
    print("[median_scale_align_factor] identity check ok")


def test_oracle_noise_loop(device: torch.device, oracle_dir: str, max_files: int) -> None:
    pattern = os.path.join(oracle_dir, "*.npy")
    paths = sorted(glob.glob(pattern))[: max(1, max_files)]
    if not paths:
        raise FileNotFoundError(f"No .npy under {oracle_dir}")

    for p in paths:
        arr = np.load(p).astype(np.float32)
        if arr.ndim != 2:
            print(f"  skip {Path(p).name}: shape {arr.shape}")
            continue
        ref = torch.from_numpy(arr).to(device=device)

        rng = torch.Generator(device=device)
        rng.manual_seed(42 + paths.index(p))
        pred = ref * (1.0 + 0.06 * torch.randn(ref.shape, device=device, generator=rng))
        pred.requires_grad_(True)

        conf = torch.sigmoid(torch.randn_like(ref) * 0.1).clamp(0.1, 1.0)

        loss, diag = geom_depth_loss_l1(pred, PriorPackDepth(depth=ref, confidence=conf))
        loss.backward()

        msg = (
            f"  {Path(p).name}: L={loss.item():.6f} scale={diag['scale'].item():.6f} "
            f"|grad_mean|={pred.grad.abs().mean().item():.2e}"
        )
        print(msg)
        pred.grad = None


def parse_args():
    ap = argparse.ArgumentParser(description="Smoke geom_prior core module")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    ap.add_argument(
        "--oracle_npy_dir",
        default=None,
        help="Flat directory of depth .npys (kitchen oracle layout)",
    )
    ap.add_argument("--max_npy_files", type=int, default=10)
    return ap.parse_args()


def main():
    args = parse_args()
    device = _pick_device(args.device)
    print(f"[smoke_geom_core] device={device}")

    test_scale_identity(device)
    test_synthetic(device)

    if args.oracle_npy_dir:
        print(f"[oracle_npy] scanning {args.oracle_npy_dir}")
        test_oracle_noise_loop(device, args.oracle_npy_dir, args.max_npy_files)
    else:
        print("[oracle_npy] skip (pass --oracle_npy_dir for kitchen npy sweep)")

    print("[smoke_geom_core] ALL OK")


if __name__ == "__main__":
    main()

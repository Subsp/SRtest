"""
HR Head forward on disk data (kitchen / SOFSR LR mip-splatting scene).

Loads LR RGB + COLMAP sparse depth (+ optional VGGT depth) + optional StableSR HR
priors from a separate folder, runs ``HRGeometricPriorHead``, saves ``.npy`` maps.

Typical layout:
  scene_root/images/ …  (``images_8`` / ``images`` / … — auto-detected or ``--image_subdir``)
  scene_root/sparse/0/*.bin

Example (autodl):
  cd experiments && python task22_hr_head_realdata.py \\
    --scene_root /root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_lr_ablation_v1/mipsplatting_x8to2_baseline_directsrc_v1 \\
    --priors_dir /root/autodl-tmp/kitchen/priors \\
    --auto_images \\
    --depth_source colmap \\
    --output_dir ./results/task22_kitchen_lr \\
    --device cuda

If COLMAP binaries are not under ``scene_root``: pass ``--sparse_dir`` (folder that
contains ``sparse/0`` or the ``sparse/0`` path itself).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs import LR_SIZE, SR_SIZE, VGGT_ROOT
from models.hr_head import HRGeometricPriorHead
from utils.dataset import frames_to_tensors, load_scene_frames, pick_image_subdir


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scene_root",
        required=True,
        help="Scene directory with LR images + sparse/ (SOFSR output or COLMAP-style scene).",
    )
    p.add_argument(
        "--image_subdir",
        default=None,
        help="LR image subfolder under scene_root. If omitted with --auto_images, scan defaults.",
    )
    p.add_argument(
        "--auto_images",
        action="store_true",
        help="Auto-pick first existing folder among images_8, images, images_2, …",
    )
    p.add_argument(
        "--sparse_dir",
        default=None,
        help="Override COLMAP dir: folder containing sparse/0 or sparse/0 itself.",
    )
    p.add_argument(
        "--priors_dir",
        default=None,
        help="Folder of StableSR caches (<stem>.png). Omit to run without SR conditioning.",
    )
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--target_lr_size", type=int, default=LR_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--depth_source",
        choices=("colmap", "vggt"),
        default="colmap",
        help="Sparse COLMAP LR depth vs frozen VGGT depth (needs VGGT_ROOT).",
    )
    p.add_argument("--vggt_root", default=VGGT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="./results/task22_realdata")
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument(
        "--force_no_sr_prior",
        action="store_true",
        help="Do not load / use StableSR conditioning even if priors_dir is set.",
    )
    return p.parse_args()


def _resolve_img_subdir(scene_root: str, args) -> str:
    if args.image_subdir is not None:
        return pick_image_subdir(scene_root, preferred=args.image_subdir)
    if args.auto_images:
        return pick_image_subdir(scene_root, preferred=None)
    # default legacy: mipnerf-like
    try:
        return pick_image_subdir(scene_root, preferred="images_8")
    except FileNotFoundError:
        return pick_image_subdir(scene_root, preferred=None)


def _stack_views(frames_t: List[Dict[str, Any]], device: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    depth = (
        torch.stack([f["depth_lr"] for f in frames_t], dim=0)
        .unsqueeze(1)
        .to(device)
        .clamp_min(1e-3)
    )
    rgb = torch.stack([f["image_lr"] for f in frames_t], dim=0).to(device)
    depth_b = depth.unsqueeze(0)
    rgb_b = rgb.unsqueeze(0)
    have = ["prior_sr_hr" in f for f in frames_t]
    if all(have):
        sr_b = torch.stack([f["prior_sr_hr"] for f in frames_t], dim=0).unsqueeze(0).to(device)
        return depth_b, rgb_b, sr_b
    if any(have):
        print("[WARN] Partial priors → SR branch disabled for this batch.")
    return depth_b, rgb_b, None


def main():
    args = _parse_args()
    scene_root = str(Path(args.scene_root).expanduser().resolve())
    priors_dir = str(Path(args.priors_dir).expanduser().resolve()) if args.priors_dir else None
    sparse_override = str(Path(args.sparse_dir).expanduser().resolve()) if args.sparse_dir else None

    img_dir = _resolve_img_subdir(scene_root, args)

    prior_dir_kw = priors_dir if (priors_dir and not args.force_no_sr_prior) else None

    print(f"[data] scene_root  = {scene_root}")
    print(f"[data] image_subdir= {img_dir}")
    print(f"[data] sparse_ovr  = {sparse_override or '(scene_root)'}\n[data] priors      = {prior_dir_kw or '(off)'}")

    frames = load_scene_frames(
        scene_root,
        image_subdir=img_dir,
        prior_dir=prior_dir_kw,
        prior_subdir=None,
        sparse_dir=sparse_override,
        n_frames=args.n_frames,
        target_lr_size=args.target_lr_size,
        seed=args.seed,
    )
    frames_t = frames_to_tensors(frames, device=args.device)

    if args.depth_source == "vggt":
        import task02_vggt_geometry as t2

        model_vggt, pose_fn = t2._setup_vggt(args.vggt_root, args.device)
        vggt_out = t2.run_vggt_on_frames(model_vggt, pose_fn, frames_t, args.device)
        for i, f in enumerate(frames_t):
            d = torch.from_numpy(vggt_out[i]["depth_vggt"]).float().to(args.device).clamp_min(1e-3)
            f["depth_lr"] = d.unsqueeze(0)

    depth_b, rgb_b, sr_b = _stack_views(frames_t, args.device)
    use_sr = sr_b is not None and not args.force_no_sr_prior
    sr_scale = max(1, int(round(SR_SIZE / float(args.target_lr_size))))

    model = HRGeometricPriorHead(
        use_rgb=True,
        use_sr_prior=use_sr,
        base_channels=args.base_channels,
        sr_scale=sr_scale,
    ).to(args.device)
    model.eval()

    fwd_kw: Dict[str, Any] = {"depth_lr": depth_b, "rgb_lr": rgb_b}
    if use_sr:
        fwd_kw["sr_prior_hr"] = sr_b
    with torch.no_grad():
        out = model(**fwd_kw)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    v = depth_b.shape[1]
    for vi in range(v):
        name = frames_t[vi]["name"]
        np.save(out_dir / f"{name}_depth_hr.npy", out["depth_hr"][0, vi, 0].float().cpu().numpy())
        np.save(out_dir / f"{name}_normal_hr.npy", out["normal_hr"][0, vi].float().cpu().numpy())
        np.save(out_dir / f"{name}_confidence_hr.npy", out["confidence_hr"][0, vi, 0].float().cpu().numpy())

    print(f"[ok] Saved {v} views under {out_dir}")
    dh = tuple(out["depth_hr"].shape)
    print(f"     shapes: depth_hr {dh}")


if __name__ == "__main__":
    main()

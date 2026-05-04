"""
Task 0.2 – Oracle Depth Renderer
==================================
从已训练好的 mip-splatting 模型 render 深度图，作为 VGGT 几何 fidelity 的对照。

输入：训练好的 model_path（含 chkpnt*.pth 或 point_cloud/iteration_*）
输出：每个训练视角的深度 .npy 文件

Usage:
  python task02_render_oracle_depth.py \
      --mip_root   /root/autodl-tmp/mip-splatting \
      --model_path /root/autodl-tmp/SOFSR/output/kitchen_mipsplatting_prior_repro/stablesr_mipsplatting_hrprior_finetune_stronger_34k_v1 \
      --source_path /root/autodl-tmp/kitchen \
      --output_dir ./results/task02/oracle/kitchen \
      --images images_2 \
      --iteration 34000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mip_root",    required=True, help="Path to mip-splatting repo")
    p.add_argument("--model_path",  required=True, help="Trained model directory")
    p.add_argument("--source_path", required=True, help="Scene COLMAP directory")
    p.add_argument("--output_dir",  required=True, help="Where to save .npy depth maps")
    p.add_argument("--images",      default="images_2", help="Which image folder for camera resolution")
    p.add_argument("--iteration",   type=int, default=None,
                   help="Iteration to load (auto-detect if None)")
    p.add_argument("--resolution",  type=int, default=-1,
                   help="-1 = use scene_resolution, otherwise downsample factor")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all relative paths BEFORE chdir
    args.model_path  = str(Path(args.model_path).resolve())
    args.source_path = str(Path(args.source_path).resolve())
    args.mip_root    = str(Path(args.mip_root).resolve())

    # ── isolate from experiments/utils/, switch to mip-splatting context ─────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]
    os.chdir(args.mip_root)
    sys.path.insert(0, args.mip_root)

    # ── auto-detect iteration ─────────────────────────────────────────────────
    if args.iteration is None:
        pc_dir = Path(args.model_path) / "point_cloud"
        if pc_dir.exists():
            iters = [int(d.name.split("_")[1]) for d in pc_dir.iterdir()
                     if d.name.startswith("iteration_")]
            if iters:
                args.iteration = max(iters)
        if args.iteration is None:
            ckpts = list(Path(args.model_path).glob("chkpnt*.pth"))
            if ckpts:
                args.iteration = max(int(p.stem.replace("chkpnt", "")) for p in ckpts)
        if args.iteration is None:
            raise RuntimeError("Cannot auto-detect iteration; pass --iteration")
    print(f"[oracle] Loading iteration {args.iteration}")

    # ── import mip-splatting modules ──────────────────────────────────────────
    from argparse import Namespace
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from gaussian_renderer import GaussianModel, render
    from utils.general_utils import safe_state

    safe_state(silent=True)

    # ── build a fake parser with all the args mip-splatting expects ──────────
    fake_parser = argparse.ArgumentParser()
    model_params = ModelParams(fake_parser, sentinel=False)
    pipe_params  = PipelineParams(fake_parser)

    args_list = [
        "-s", args.source_path,
        "-m", args.model_path,
        "--images", args.images,
        "--resolution", str(args.resolution),
    ]
    fake_args = fake_parser.parse_args(args_list)

    # Read cfg_args from saved model to recover kernel_size etc.
    cfg_args_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_args_path):
        with open(cfg_args_path) as f:
            saved = eval(f.read())
        for k, v in vars(saved).items():
            if hasattr(fake_args, k) and not k.startswith("_"):
                setattr(fake_args, k, v)
    # Override critical paths in case cfg_args has stale paths
    fake_args.source_path = args.source_path
    fake_args.model_path  = args.model_path
    fake_args.images      = args.images
    fake_args.resolution  = args.resolution

    # ── instantiate scene + Gaussians, load trained weights ──────────────────
    sh_degree = getattr(fake_args, "sh_degree", 3)
    gaussians = GaussianModel(sh_degree)
    scene     = Scene(fake_args, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color   = [1, 1, 1] if getattr(fake_args, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = getattr(fake_args, "kernel_size", 0.1)

    # ── render depth on TRAIN cameras ─────────────────────────────────────────
    train_cams = scene.getTrainCameras()
    print(f"[oracle] Rendering depth for {len(train_cams)} train views …")

    pipe = pipe_params.extract(fake_args)

    n_saved = 0
    for cam in tqdm(train_cams, desc="Render depth"):
        with torch.no_grad():
            out = render(
                cam, gaussians, pipe, background,
                kernel_size=kernel_size,
                scale_factor=1.0,
            )
        # depth lives in 'depth' key for merged renderer; fall back to 'render_full'
        depth = out.get("depth", None)
        if depth is None:
            full = out.get("render_full", None)
            if full is not None and full.shape[0] >= 7:
                depth = full[6:7]
            else:
                continue
        depth_np = depth.detach().squeeze().cpu().numpy().astype(np.float32)

        # Use cam.image_name as filename (matches MipNeRF360 frame stem)
        save_path = out_dir / f"{cam.image_name}.npy"
        np.save(str(save_path), depth_np)
        n_saved += 1

    print(f"[oracle] Saved {n_saved} depth maps to {out_dir}")
    print(f"[oracle] Sample depth shape: {depth_np.shape}, "
          f"min={depth_np[depth_np>0].min() if (depth_np>0).any() else 0:.3f}, "
          f"max={depth_np.max():.3f}, "
          f"median={np.median(depth_np[depth_np>0]) if (depth_np>0).any() else 0:.3f}")


if __name__ == "__main__":
    main()

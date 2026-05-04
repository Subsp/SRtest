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

    # ── compute 3D filter (mip-splatting specific) ────────────────────────────
    if hasattr(gaussians, "compute_3D_filter"):
        print("[oracle] Computing 3D filter …")
        gaussians.compute_3D_filter(train_cams.copy())

    # ── try to construct splat_args (ExtendedSettings) ────────────────────────
    splat_args = None
    try:
        from diff_gaussian_rasterization import ExtendedSettings
        splat_args = ExtendedSettings.from_defaults() if hasattr(ExtendedSettings, "from_defaults") else ExtendedSettings()
        print(f"[oracle] splat_args constructed: {type(splat_args).__name__}")
    except Exception as e:
        print(f"[oracle] [WARN] cannot build ExtendedSettings: {e}")

    # ── prepare render kwargs by introspection ────────────────────────────────
    import inspect
    render_sig = inspect.signature(render)
    base_kwargs = {}
    if "kernel_size" in render_sig.parameters:
        base_kwargs["kernel_size"] = kernel_size
    if "splat_args" in render_sig.parameters and splat_args is not None:
        base_kwargs["splat_args"] = splat_args
    print(f"[oracle] render() kwargs: {list(base_kwargs.keys())}")

    # ── DEPTH-AS-COLOR TRICK ──────────────────────────────────────────────────
    # Render with override_color = depth value of each gaussian in camera space.
    # The resulting "RGB" image is then the depth map.
    n_saved = 0
    xyz_world = gaussians.get_xyz                      # (N, 3)

    for cam in tqdm(train_cams, desc="Render depth"):
        with torch.no_grad():
            # Compute per-gaussian depth in this camera's frame
            # world_view_transform: (4,4) world→camera; depth = z after transform
            wvt   = cam.world_view_transform            # (4,4) row-major
            xyz_h = torch.cat([xyz_world, torch.ones_like(xyz_world[:, :1])], dim=1)  # (N,4)
            xyz_cam = xyz_h @ wvt                        # (N,4)
            depth_per_gauss = xyz_cam[:, 2:3]            # (N,1)
            override_color  = depth_per_gauss.repeat(1, 3)   # (N,3) – broadcast as RGB

            out = render(
                cam, gaussians, pipe, background,
                override_color=override_color,
                **base_kwargs,
            )
        rendered = out["render"]                         # (3, H, W) – all 3 channels = depth
        depth    = rendered[0]                           # take any channel
        depth_np = depth.detach().cpu().numpy().astype(np.float32)

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

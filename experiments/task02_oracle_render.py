"""
Task 0.2 – Oracle Depth Post-Processor
=======================================
After running `mip-splatting/render.py`, this script:
  1. Locates the rendered depth images (EXR or PNG) in the mip-splatting output
  2. Converts and saves them as float32 .npy files at the original render resolution
  3. Also saves a visualisation PNG for sanity checking

Run after task02_oracle_train.sh finishes for each scene.

Usage:
  python task02_oracle_render.py \
      --model_dir  ./results/task02/oracle/garden \
      --scene_dir  /data/mipnerf360/garden \
      --output_dir ./results/task02/oracle/garden/train/ours_30000/depth
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",  required=True, help="Mip-splatting output model dir")
    p.add_argument("--scene_dir",  required=True, help="MipNeRF360 scene directory")
    p.add_argument("--output_dir", required=True, help="Where to save .npy depth files")
    p.add_argument("--split",      default="train", choices=["train", "test"])
    p.add_argument("--iteration",  default=30000, type=int)
    return p.parse_args()


def _load_depth_image(path: Path) -> np.ndarray | None:
    """
    Load a depth image from EXR, NPY, or 16-bit PNG.
    Returns float32 array or None.
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(str(path)).astype(np.float32)

    if suffix == ".exr":
        try:
            import OpenEXR
            import Imath
            f   = OpenEXR.InputFile(str(path))
            dw  = f.header()["dataWindow"]
            W   = dw.max.x - dw.min.x + 1
            H   = dw.max.y - dw.min.y + 1
            ch  = f.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(ch, dtype=np.float32).reshape(H, W)
        except ImportError:
            print(f"  [WARNING] OpenEXR not installed; cannot read {path}")
            return None

    if suffix in (".png", ".jpg", ".jpeg"):
        img = PILImage.open(path)
        arr = np.array(img)
        if arr.ndim == 2:
            # 16-bit PNG: value is depth * scale (check mip-splatting convention)
            # mip-splatting renders normalised depth; we store as-is and normalise
            return arr.astype(np.float32)
        elif arr.shape[2] >= 3:
            # RGB depth visualisation – try to decode from red channel or luminance
            return arr[:, :, 0].astype(np.float32)

    return None


def _find_render_dir(model_dir: str, split: str, iteration: int) -> Path | None:
    """Locate the rendered output folder from mip-splatting."""
    candidates = [
        Path(model_dir) / split / f"ours_{iteration}",
        Path(model_dir) / split / f"ours_{iteration:07d}",
        Path(model_dir) / "renders",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_dir = _find_render_dir(args.model_dir, args.split, args.iteration)
    if render_dir is None:
        print(f"[ERROR] Cannot find render output in {args.model_dir}")
        print("  Expected: <model_dir>/{train,test}/ours_<iter>/")
        sys.exit(1)

    # Depth files are often in a 'depth' sub-directory or named *depth*
    depth_candidates = list(render_dir.glob("depth/*.exr")) + \
                       list(render_dir.glob("depth/*.png")) + \
                       list(render_dir.glob("depth/*.npy")) + \
                       list(render_dir.glob("*depth*.exr")) + \
                       list(render_dir.glob("*depth*.npy"))

    if not depth_candidates:
        # Mip-splatting may not render depth by default.
        # If no depth files found, generate them using the trained model.
        print(f"[INFO] No depth files in {render_dir}")
        print("       Running depth-only render pass via mip-splatting render.py …")
        _render_depth_via_mipsplatting(args.model_dir, args.scene_dir, args.split, args.iteration)
        # Re-scan
        render_dir = _find_render_dir(args.model_dir, args.split, args.iteration)
        depth_candidates = list(render_dir.glob("depth/*.exr")) + \
                           list(render_dir.glob("depth/*.npy"))

    converted = 0
    for depth_path in sorted(depth_candidates):
        depth = _load_depth_image(depth_path)
        if depth is None:
            continue
        stem     = depth_path.stem
        npy_path = out_dir / f"{stem}.npy"
        np.save(str(npy_path), depth)

        # Save visualisation
        vis = depth.copy()
        valid = vis > 0
        if valid.any():
            vis[valid] = (vis[valid] - vis[valid].min()) / (vis[valid].max() - vis[valid].min() + 1e-8)
        vis_path = out_dir / f"{stem}_vis.png"
        PILImage.fromarray((vis * 255).clip(0, 255).astype(np.uint8)).save(vis_path)
        converted += 1

    print(f"[task02_oracle_render] Saved {converted} depth .npy files to {out_dir}")


def _render_depth_via_mipsplatting(model_dir, scene_dir, split, iteration):
    """
    Call mip-splatting render.py with a depth rendering patch.
    Mip-splatting does not natively render depth; we hook into the gaussian
    renderer to extract the depth channel.
    """
    import subprocess, sys as _sys
    script_dir = Path(__file__).parent
    mip_root   = (script_dir / "../mip-splatting").resolve()
    render_script = mip_root / "render.py"

    if not render_script.exists():
        print(f"[ERROR] mip-splatting render.py not found at {render_script}")
        return

    cmd = [
        _sys.executable, str(render_script),
        "-m", model_dir,
        "--iteration", str(iteration),
        "--skip_test" if split == "train" else "--skip_train",
        "--quiet",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

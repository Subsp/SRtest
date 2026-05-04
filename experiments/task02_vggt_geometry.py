"""
Task 0.2 – VGGT Geometry Fidelity at 200×200 Input
====================================================
Pipeline (per scene):
  1. Load images_8, resize to 200×200
  2. Run frozen VGGT  → predicted depth maps (at VGGT's internal 518×518, then
     resized back to 200×200 for fair comparison)
  3. Load oracle depth maps produced by vanilla mip-splatting trained on HR
     images (run task02_oracle_train.sh first, then task02_oracle_render.py)
  4. Compute per-scene AbsRel, Scale-Invariant L1, RMSE

Decision thresholds (AbsRel):
  < 0.10  → VGGT directly usable
  0.10–0.20 → HR Head strengthening needed
  > 0.20  → LoRA fine-tune on VGGT required

Usage:
  # Step 1: produce oracle depth (run once per scene, ~30 min GPU):
  bash task02_oracle_train.sh /path/to/mipnerf360 ./results/task02/oracle

  # Step 2: run VGGT inference + comparison:
  python task02_vggt_geometry.py \
      --data_root   /path/to/mipnerf360 \
      --oracle_root ./results/task02/oracle \
      --output_dir  ./results/task02 \
      --vggt_root   /root/autodl-tmp/vggt \
      [--device cuda] [--seed 42]

  # Or: export VGGT_ROOT=/root/autodl-tmp/vggt  (see experiments/configs.py)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from configs import (
    SCENES_PHASE0, FRAMES_PER_SCENE, LR_SIZE, LR_IMAGE_SUBDIR,
    VGGT_ROOT, ABSREL_OK, ABSREL_FINETUNE,
)
from utils.dataset import load_scene_frames, frames_to_tensors
from utils.metrics import compute_all_depth_metrics


# ── VGGT inference ────────────────────────────────────────────────────────────

def _setup_vggt(vggt_root: str, device: str):
    """Import VGGT and load the pretrained 1-B model."""
    vggt_root = os.path.abspath(vggt_root)
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)

    from vggt.models.vggt import VGGT                        # type: ignore
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

    model = VGGT()
    _URL  = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

    ckpt_cache = Path(vggt_root) / "model.pt"
    if ckpt_cache.exists():
        state = torch.load(ckpt_cache, map_location="cpu")
    else:
        print("[VGGT] Downloading model weights (≈4 GB) …")
        import urllib.request
        urllib.request.urlretrieve(_URL, ckpt_cache)
        state = torch.load(ckpt_cache, map_location="cpu")

    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"[VGGT] Model loaded on {device}")
    return model, pose_encoding_to_extri_intri


@torch.no_grad()
def run_vggt_on_frames(
    model,
    pose_encoding_to_extri_intri,
    frames_t: list,
    device: str,
    vggt_resolution: int = 518,
) -> list:
    """
    Run VGGT on a list of LR frame tensors.

    Returns a list of dicts:
        depth_vggt  : np.float32 (200, 200) – predicted depth at LR resolution
        depth_conf  : np.float32 (200, 200) – confidence map
        extrinsic   : np.float64 (3, 4)     – predicted camera (OpenCV)
        intrinsic   : np.float64 (3, 3)     – predicted intrinsic
    """
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if device == "cpu":
        dtype = torch.float32

    # Stack all LR images into a sequence: (S, 3, H, W)
    images = torch.stack([f["image_lr"] for f in frames_t], dim=0).to(device)  # (S,3,200,200)

    # VGGT internally resizes to vggt_resolution (518)
    images_vggt = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False,
    )

    with torch.cuda.amp.autocast(dtype=dtype, enabled=(device != "cpu")):
        images_in = images_vggt[None]                    # (1, S, 3, 518, 518)
        aggregated_tokens_list, ps_idx = model.aggregator(images_in)

        pose_enc   = model.camera_head(aggregated_tokens_list)[-1]
        depth_raw, depth_conf_raw = model.depth_head(
            aggregated_tokens_list, images_in, ps_idx
        )

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, images_vggt.shape[-2:]
    )

    # Squeeze batch dim, move to CPU numpy
    extrinsic  = extrinsic.squeeze(0).float().cpu().numpy()    # (S, 3, 4)
    intrinsic  = intrinsic.squeeze(0).float().cpu().numpy()    # (S, 3, 3)
    depth_raw  = depth_raw.squeeze(0).float().cpu()            # (S, 518, 518) or (S,518,518,1)
    depth_conf_raw = depth_conf_raw.squeeze(0).float().cpu()

    if depth_raw.ndim == 4:
        depth_raw = depth_raw.squeeze(-1)
    if depth_conf_raw.ndim == 4:
        depth_conf_raw = depth_conf_raw.squeeze(-1)

    # Resize depth + conf from 518×518 back to LR_SIZE×LR_SIZE
    depth_lr   = F.interpolate(
        depth_raw.unsqueeze(1), size=(LR_SIZE, LR_SIZE),
        mode="bilinear", align_corners=False,
    ).squeeze(1).numpy()                                       # (S, 200, 200)

    conf_lr = F.interpolate(
        depth_conf_raw.unsqueeze(1), size=(LR_SIZE, LR_SIZE),
        mode="bilinear", align_corners=False,
    ).squeeze(1).numpy()

    outputs = []
    for i, f in enumerate(frames_t):
        outputs.append(dict(
            name        = f["name"],
            depth_vggt  = depth_lr[i].astype(np.float32),
            depth_conf  = conf_lr[i].astype(np.float32),
            extrinsic   = extrinsic[i],
            intrinsic   = intrinsic[i],
        ))
    return outputs


# ── oracle depth loading ──────────────────────────────────────────────────────

def load_oracle_depth(oracle_scene_dir: str, frame_name: str) -> np.ndarray | None:
    """
    Load oracle depth rendered by mip-splatting.

    Expected layout (produced by task02_oracle_render.py):
      oracle_scene_dir/
        train/
          ours_30000/
            depth/
              <frame_name>.npy   (or .png with depth encoding)

    Returns float32 (H, W) array or None if not found.
    """
    # Try multiple layouts (flat and nested)
    candidates = [
        Path(oracle_scene_dir) / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.npy",
        Path(oracle_scene_dir) / "depth" / f"{frame_name}.npy",
    ]
    for npy_path in candidates:
        if npy_path.exists():
            return np.load(str(npy_path)).astype(np.float32)

    # Try EXR (output from mip-splatting render)
    try:
        import OpenEXR
        import Imath
        exr_path = Path(oracle_scene_dir) / "train" / "ours_30000" / "depth" / f"{frame_name}.exr"
        if exr_path.exists():
            f = OpenEXR.InputFile(str(exr_path))
            dw   = f.header()["dataWindow"]
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            ch   = f.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(ch, dtype=np.float32).reshape(size[1], size[0])
    except ImportError:
        pass

    return None


# ── per-scene processing ──────────────────────────────────────────────────────

def process_scene(
    scene: str,
    scene_root: str,
    oracle_root: str,
    model,
    pose_encoding_to_extri_intri,
    device: str,
    n_frames: int,
    seed: int,
) -> dict | None:
    oracle_scene_dir = os.path.join(oracle_root, scene)

    print(f"\n[Scene: {scene}]")

    # ── load frames ───────────────────────────────────────────────────────────
    try:
        frames = load_scene_frames(
            scene_root,
            image_subdir   = LR_IMAGE_SUBDIR,
            n_frames       = n_frames,
            target_lr_size = LR_SIZE,
            seed           = seed,
        )
    except Exception as e:
        print(f"  [ERROR] loading frames: {e}")
        return None

    frames_t = frames_to_tensors(frames, device=device)
    print(f"  Loaded {len(frames_t)} frames")

    # ── run VGGT ─────────────────────────────────────────────────────────────
    print("  Running frozen VGGT …")
    t0 = time.time()
    vggt_outputs = run_vggt_on_frames(
        model, pose_encoding_to_extri_intri, frames_t, device
    )
    print(f"  VGGT done in {time.time()-t0:.1f}s")

    # ── compare vs oracle ─────────────────────────────────────────────────────
    per_frame = []
    missing_oracle = 0

    for vo in vggt_outputs:
        oracle_depth = load_oracle_depth(oracle_scene_dir, vo["name"])
        if oracle_depth is None:
            missing_oracle += 1
            continue

        # Resize oracle depth to LR_SIZE for apple-to-apple comparison
        oracle_lr = F.interpolate(
            torch.from_numpy(oracle_depth).float().unsqueeze(0).unsqueeze(0),
            size=(LR_SIZE, LR_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        pred  = vo["depth_vggt"]
        valid = (oracle_lr > 0) & np.isfinite(oracle_lr) & (pred > 0) & np.isfinite(pred)

        if valid.sum() < 100:
            continue

        m = compute_all_depth_metrics(pred, oracle_lr, mask=valid)
        m["frame"] = vo["name"]
        per_frame.append(m)

    if missing_oracle > 0:
        print(f"  [WARNING] Oracle depth missing for {missing_oracle}/{len(vggt_outputs)} frames")
        print(f"            → Run task02_oracle_train.sh first, then task02_oracle_render.py")

    if not per_frame:
        print(f"  [SKIP] No valid frame comparisons for {scene}")
        return None

    df = pd.DataFrame(per_frame)
    scene_metrics = {
        "abs_rel"      : round(df["abs_rel"].mean(), 4),
        "scale_inv_l1" : round(df["scale_inv_l1"].mean(), 4),
        "rmse"         : round(df["rmse"].mean(), 4),
        "n_frames"     : len(per_frame),
    }
    print(f"  AbsRel={scene_metrics['abs_rel']:.4f}  "
          f"ScaleInvL1={scene_metrics['scale_inv_l1']:.4f}  "
          f"RMSE={scene_metrics['rmse']:.4f}")
    return scene_metrics, df


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 0.2: VGGT geometry fidelity test")
    p.add_argument("--data_root",   required=True, help="MipNeRF360 root directory")
    p.add_argument("--oracle_root", required=True,
                   help="Root of oracle depth dirs (produced by task02_oracle_train.sh)")
    p.add_argument("--output_dir",  default="./results/task02")
    p.add_argument("--scenes",      nargs="+", default=SCENES_PHASE0)
    p.add_argument("--n_frames",    type=int, default=FRAMES_PER_SCENE)
    p.add_argument("--vggt_root",   default=VGGT_ROOT)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def verdict(absrel: float) -> str:
    if absrel < ABSREL_OK:
        return "🟢 DIRECTLY USABLE  (AbsRel < 0.10)"
    elif absrel < ABSREL_FINETUNE:
        return "🟡 HR HEAD NEEDED   (0.10 ≤ AbsRel < 0.20)"
    else:
        return "🔴 LoRA FINE-TUNE   (AbsRel ≥ 0.20)"


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"\n{'='*60}")
    print(" Task 0.2 – VGGT Geometry Fidelity Test")
    print(f"{'='*60}")
    print(f" Scenes   : {args.scenes}")
    print(f" Frames   : {args.n_frames} per scene")
    print(f" LR size  : {LR_SIZE}×{LR_SIZE}")
    print(f" Device   : {device}")
    print()

    # ── load VGGT model ───────────────────────────────────────────────────────
    print("[1/2] Loading VGGT model …")
    model, pose_enc_fn = _setup_vggt(args.vggt_root, device)

    all_rows      = []
    scene_summary = {}

    for scene in args.scenes:
        scene_root = os.path.join(args.data_root, scene)
        if not os.path.isdir(scene_root):
            print(f"  [SKIP] {scene}: not found at {scene_root}")
            continue

        result = process_scene(
            scene, scene_root, args.oracle_root,
            model, pose_enc_fn, device,
            n_frames=args.n_frames, seed=args.seed,
        )
        if result is None:
            continue

        scene_metrics, df = result
        scene_summary[scene] = scene_metrics
        df["scene"] = scene
        all_rows.append(df)
        df.to_csv(out_dir / f"{scene}_frames.csv", index=False)

    # ── overall summary ───────────────────────────────────────────────────────
    if not scene_summary:
        print("\n[ERROR] No scenes processed. Check data_root and oracle_root.")
        return

    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    summary_rows = []
    for scene, s in scene_summary.items():
        print(f"  {scene:<12}  AbsRel={s['abs_rel']:.4f}  ScaleInvL1={s['scale_inv_l1']:.4f}  {verdict(s['abs_rel'])}")
        summary_rows.append({"scene": scene, **s})

    all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not all_df.empty:
        overall_absrel     = all_df["abs_rel"].mean()
        overall_scaleinvl1 = all_df["scale_inv_l1"].mean()
        overall_rmse       = all_df["rmse"].mean()
        print(f"\n  {'OVERALL':<12}  AbsRel={overall_absrel:.4f}  ScaleInvL1={overall_scaleinvl1:.4f}")
        print(f"  {verdict(overall_absrel)}")

    # ── save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(summary_rows).to_csv(out_dir / "scene_summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(
            {"scenes": scene_summary,
             "overall": {
                 "abs_rel"     : round(overall_absrel, 4)     if not all_df.empty else None,
                 "scale_inv_l1": round(overall_scaleinvl1, 4) if not all_df.empty else None,
                 "rmse"        : round(overall_rmse, 4)        if not all_df.empty else None,
             },
             "verdict": verdict(overall_absrel) if not all_df.empty else "n/a"},
            fp, indent=2,
        )
    print(f"\n Results saved to: {out_dir}")
    _print_decision_matrix(overall_absrel if not all_df.empty else None)


def _print_decision_matrix(overall_absrel):
    print(f"\n{'='*60}")
    print(" DECISION MATRIX")
    print(f"{'='*60}")
    print(f"  Threshold reference:")
    print(f"    AbsRel < {ABSREL_OK}  → VGGT directly usable")
    print(f"    {ABSREL_OK} ≤ AbsRel < {ABSREL_FINETUNE} → HR Head / fine-tune needed")
    print(f"    AbsRel ≥ {ABSREL_FINETUNE} → LoRA fine-tune on VGGT required")
    print()

    if overall_absrel is None:
        return

    if overall_absrel < ABSREL_OK:
        print("  ➤ ACTION: Frozen VGGT is sufficient as geometry prior")
        print("           → Proceed directly to HR Head training (Phase 2.2)")
    elif overall_absrel < ABSREL_FINETUNE:
        print("  ➤ ACTION: Add HR Head with strong depth supervision")
        print("           → Consider HR-aware feature projection in HR Head")
        print("           → Confidence C_HR should mask low-quality VGGT regions")
    else:
        print("  ➤ ACTION: Fine-tune VGGT with LoRA at 200×200 resolution")
        print("           → Use Mip-NeRF360 oracle depths as supervision")
        print("           → Only update LoRA weights, keep backbone frozen")


if __name__ == "__main__":
    main()

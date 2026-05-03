"""
Task 0.1 (alternative) – 2DSR View-Inconsistency via SIFT Patch Matching
=========================================================================
无需深度图 / 相机参数，直接用 SIFT 特征匹配找跨视角对应点，
在对应点周围提取小 patch 比较 PSNR / SSIM。

原理：
  SIFT 匹配点 = 同一 3D 表面点在两个视角的投影
  → 比较两个视角 SR 结果在该点的局部外观
  → 测量 2DSR 跨视角一致性

Usage:
  python task01_sift_consistency.py \
      --sr_dir  /path/to/sr_images \
      --scenes  kitchen \
      --output_dir ./results/task01_sift
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from configs import SCENES_PHASE0, FRAMES_PER_SCENE, PSNR_SEVERE, PSNR_MODERATE


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sr_images(sr_scene_dir: str, n_frames: int = None, seed: int = 42):
    """Load SR image paths from a scene directory (skip .ipynb_checkpoints)."""
    paths = sorted([
        p for p in Path(sr_scene_dir).glob("*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        and ".ipynb_checkpoints" not in str(p)
    ])
    if n_frames and len(paths) > n_frames:
        import random; random.seed(seed)
        paths = random.sample(paths, n_frames)
        paths = sorted(paths)
    return paths


def compare_pair_sift(
    path_i: Path,
    path_j: Path,
    patch_half: int = 32,
    ratio_thresh: float = 0.70,
    n_features: int = 2000,
):
    """
    SIFT-match two SR images and compare patch PSNR / SSIM at match locations.

    Returns dict with psnr, ssim, n_matches, n_patches, or None if too few matches.
    """
    img_i = np.array(Image.open(path_i).convert("RGB"))
    img_j = np.array(Image.open(path_j).convert("RGB"))
    H, W  = img_i.shape[:2]

    g_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY)
    g_j = cv2.cvtColor(img_j, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_i, d_i = sift.detectAndCompute(g_i, None)
    kp_j, d_j = sift.detectAndCompute(g_j, None)

    if d_i is None or d_j is None or len(kp_i) < 5 or len(kp_j) < 5:
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d_i, d_j, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < 5:
        return None

    psnrs, ssims = [], []
    p = patch_half
    for m in good:
        xi, yi = map(int, kp_i[m.queryIdx].pt)
        xj, yj = map(int, kp_j[m.trainIdx].pt)
        # Bounds check
        if (xi-p < 0 or yi-p < 0 or xi+p >= W or yi+p >= H or
                xj-p < 0 or yj-p < 0 or xj+p >= W or yj+p >= H):
            continue
        pi = img_i[yi-p:yi+p, xi-p:xi+p].astype(np.float32) / 255.0
        pj = img_j[yj-p:yj+p, xj-p:xj+p].astype(np.float32) / 255.0
        psnrs.append(peak_signal_noise_ratio(pi, pj, data_range=1.0))
        ssims.append(structural_similarity(pi, pj, channel_axis=2, data_range=1.0))

    if not psnrs:
        return None

    return dict(
        pair      = f"{path_i.stem}>{path_j.stem}",
        n_matches = len(good),
        n_patches = len(psnrs),
        psnr      = float(np.mean(psnrs)),
        ssim      = float(np.mean(ssims)),
        psnr_p10  = float(np.percentile(psnrs, 10)),   # worst 10%
        psnr_p90  = float(np.percentile(psnrs, 90)),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def verdict(psnr_val: float) -> str:
    if psnr_val < PSNR_SEVERE:
        return "🔴 SEVERE  → view-consistent SR module mandatory"
    elif psnr_val < PSNR_MODERATE:
        return "🟡 MODERATE → confidence weighting sufficient"
    else:
        return "🟢 NEGLIGIBLE → simplify narrative"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sr_dir",     required=True,
                   help="SR 图根目录，结构：<sr_dir>/<scene>/<frame>.png")
    p.add_argument("--scenes",     nargs="+", default=SCENES_PHASE0)
    p.add_argument("--n_frames",   type=int, default=FRAMES_PER_SCENE)
    p.add_argument("--patch_size", type=int, default=32,
                   help="patch 半径（patch 边长 = 2×patch_size）")
    p.add_argument("--output_dir", default="./results/task01_sift")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(" Task 0.1 – 2DSR Consistency (SIFT Patch Method)")
    print(f"{'='*60}")
    print(f" Scenes  : {args.scenes}")
    print(f" Frames  : {args.n_frames} per scene")
    print(f" Patch   : {args.patch_size*2}×{args.patch_size*2} px")
    print()

    all_rows     = []
    scene_summary = {}

    for scene in args.scenes:
        sr_scene_dir = os.path.join(args.sr_dir, scene)
        if not os.path.isdir(sr_scene_dir):
            # Try flat layout: sr_dir itself IS the scene dir
            if scene in os.path.basename(args.sr_dir.rstrip("/")):
                sr_scene_dir = args.sr_dir
            else:
                print(f"  [SKIP] {scene}: not found at {sr_scene_dir}")
                continue

        paths = load_sr_images(sr_scene_dir, n_frames=args.n_frames, seed=args.seed)
        if len(paths) < 2:
            print(f"  [SKIP] {scene}: need ≥2 images, found {len(paths)}")
            continue

        print(f"[Scene: {scene}]  {len(paths)} frames")
        rows = []

        import itertools
        pairs = list(itertools.combinations(range(len(paths)), 2))
        for idx_i, idx_j in tqdm(pairs, desc=f"  {scene} pairs", leave=False):
            result = compare_pair_sift(
                paths[idx_i], paths[idx_j],
                patch_half=args.patch_size,
            )
            if result:
                result["scene"] = scene
                rows.append(result)

        if not rows:
            print(f"  [WARNING] No valid pairs for {scene}")
            continue

        df = pd.DataFrame(rows)
        mean_psnr = df["psnr"].mean()
        mean_ssim = df["ssim"].mean()
        p10_psnr  = df["psnr_p10"].mean()

        scene_summary[scene] = dict(
            psnr       = round(mean_psnr, 2),
            ssim       = round(mean_ssim, 4),
            psnr_p10   = round(p10_psnr, 2),
            n_pairs    = len(rows),
            avg_matches= round(df["n_matches"].mean(), 1),
        )
        all_rows.append(df)

        print(f"  PSNR(mean)={mean_psnr:.2f} dB  "
              f"PSNR(P10)={p10_psnr:.2f} dB  "
              f"SSIM={mean_ssim:.4f}  "
              f"avg_matches={df['n_matches'].mean():.0f}")
        print(f"  {verdict(mean_psnr)}")
        df.to_csv(out_dir / f"{scene}_pairs.csv", index=False)

    if not scene_summary:
        print("\n[ERROR] No scenes processed.")
        return

    # ── overall ───────────────────────────────────────────────────────────────
    all_df = pd.concat(all_rows, ignore_index=True)
    overall_psnr = all_df["psnr"].mean()
    overall_ssim = all_df["ssim"].mean()

    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    for scene, s in scene_summary.items():
        print(f"  {scene:<12}  PSNR={s['psnr']:.2f} dB  SSIM={s['ssim']:.4f}  "
              f"matches={s['avg_matches']:.0f}")
    print(f"\n  {'OVERALL':<12}  PSNR={overall_psnr:.2f} dB  SSIM={overall_ssim:.4f}")
    print(f"  {verdict(overall_psnr)}")

    # ── save ──────────────────────────────────────────────────────────────────
    pd.DataFrame([{"scene": k, **v} for k, v in scene_summary.items()]).to_csv(
        out_dir / "scene_summary.csv", index=False
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(dict(
            scenes       = scene_summary,
            overall_psnr = round(overall_psnr, 2),
            overall_ssim = round(overall_ssim, 4),
            verdict      = verdict(overall_psnr),
        ), f, indent=2)
    print(f"\n Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

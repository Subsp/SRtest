"""
Task 0.1 – 2DSR View-Inconsistency Severity Test
=================================================
支持两种 SR 来源模式：

  模式 A  --sr_dir <path>  （推荐）
    直接读取预先生成好的 800×800 SR 图，跳过 SwinIR 推理。
    目录结构：
      sr_dir/
        <scene>/
          <frame_name>.png   (或 .jpg)
    帧名必须与 COLMAP images.bin 中的文件名 stem 一致。

  模式 B  （不传 --sr_dir）
    由脚本实时运行 SwinIR ×4，首次运行自动下载模型权重。

Pipeline (per scene, 8 frames):
  1. 加载 SR 图（模式 A：从磁盘；模式 B：SwinIR 推理）
  2. 从 images_8 加载 COLMAP 相机参数和稀疏深度（仅用于 warp，不做 SR）
  3. 对每对帧 (i, j)：
       - 把深度图从 200×200 上采到 800×800
       - Backward-warp I_sr^{v_j} 到 v_i 视角
       - 计算 PSNR/SSIM（全局 + Sobel 边缘区域）
  4. 汇总统计，输出决策结论

Decision thresholds (PSNR):
  ≥ 28 dB  → negligible inconsistency
  22–28 dB → moderate (confidence weighting sufficient)
  < 22 dB  → severe (view-consistent SR module mandatory)

Usage:
  # 模式 A（传预计算 SR 图）：
  python task01_2dsr_consistency.py \
      --data_root /path/to/mipnerf360 \
      --sr_dir    /path/to/sr_images \
      --output_dir ./results/task01

  # 模式 B（实时 SwinIR）：
  python task01_2dsr_consistency.py \
      --data_root /path/to/mipnerf360 \
      --output_dir ./results/task01
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from configs import (
    SCENES_PHASE0, FRAMES_PER_SCENE, LR_SIZE, SR_SCALE, SR_SIZE,
    LR_IMAGE_SUBDIR, PSNR_SEVERE, PSNR_MODERATE,
)
from utils.dataset import load_scene_frames, frames_to_tensors
from utils.metrics import compute_all_image_metrics, sobel_edge_mask
from utils.warp import backward_warp, upsample_depth


# ── SR loading helpers ────────────────────────────────────────────────────────

_SR_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _build_sr_index(sr_scene_dir: str) -> dict:
    """
    Scan sr_scene_dir and return a dict: stem → absolute_path.
    Supports flat directory or any single level of subdirectory.
    """
    index = {}
    p = Path(sr_scene_dir)
    if not p.is_dir():
        return index
    for ext in _SR_EXTENSIONS:
        for f in p.glob(f"*{ext}"):
            index[f.stem] = str(f)
    return index


def load_sr_from_dir(
    frames_t: list,
    sr_scene_dir: str,
    device: str,
    sr_size: int = None,   # None = keep native resolution
) -> list:
    """
    Load pre-computed SR images from disk at their native resolution.

    Matches by frame name stem. If a frame is missing, raises FileNotFoundError.
    If sr_size is given AND the image is square, resize to (sr_size, sr_size).
    Otherwise the native (possibly non-square) resolution is preserved.

    Returns list of (3, H, W) float tensors on `device`.
    """
    from PIL import Image as PILImage

    index = _build_sr_index(sr_scene_dir)
    if not index:
        raise FileNotFoundError(
            f"No SR images found in: {sr_scene_dir}\n"
            f"Expected files like: {sr_scene_dir}/<frame_name>.png"
        )

    sr_images = []
    for f in frames_t:
        name = f["name"]
        if name not in index:
            raise FileNotFoundError(
                f"SR image not found for frame '{name}' in {sr_scene_dir}\n"
                f"Available: {sorted(index.keys())[:5]} …"
            )
        img = PILImage.open(index[name]).convert("RGB")
        W_img, H_img = img.size   # PIL: (W, H)

        # Only resize if explicitly requested AND sizes differ
        if sr_size is not None and (W_img, H_img) != (sr_size, sr_size):
            print(f"  [INFO] SR image {name}: {W_img}×{H_img} → loaded as-is "
                  f"(non-square native resolution, K will be recomputed)")

        t = torch.from_numpy(np.array(img, dtype=np.uint8)).float().div(255.0)
        t = t.permute(2, 0, 1).to(device)          # (3, H, W)
        sr_images.append(t)
    return sr_images


def sr_with_swinir(frames_t: list, device: str) -> list:
    """Lazy-import SwinIR and run ×4 SR on all frames."""
    from utils.swinir_wrapper import SwinIRSuperResolver
    resolver = SwinIRSuperResolver(device=device)
    sr_images = []
    for f in tqdm(frames_t, desc="  SwinIR SR", leave=False):
        sr = resolver.upscale_tensor(f["image_lr"])
        sr_images.append(sr)
    return sr_images


def warp_and_compare(
    sr_images: list,
    frames_t: list,
    device: str,
    sr_size: int = SR_SIZE,
) -> list:
    """
    For all ordered pairs (i, j):
      - Upsample depth of view i to SR resolution
      - Backward-warp sr_images[j] into view i
      - Compute metrics vs sr_images[i]

    Returns a list of per-pair metric dicts.
    """
    n = len(frames_t)
    results = []

    # ── Detect actual SR dimensions from first image ──────────────────────────
    _, H_sr, W_sr = sr_images[0].shape
    print(f"  SR image size detected: {W_sr}×{H_sr}")

    # ── Precompute K_sr for each frame at actual SR resolution ────────────────
    # K_sr = scale_K(K_lr, (LR_SIZE, LR_SIZE) → (W_sr, H_sr))
    from utils.colmap_reader import scale_K as _scale_K
    K_sr_list = []
    for fi in frames_t:
        K_lr_np = fi["K_lr"].cpu().numpy()
        K_sr_np = _scale_K(K_lr_np, (LR_SIZE, LR_SIZE), (W_sr, H_sr))
        K_sr_list.append(torch.from_numpy(K_sr_np).float().to(device))

    for idx_i, idx_j in tqdm(
        list(itertools.permutations(range(n), 2)),
        desc="  Warp pairs", leave=False,
    ):
        fi  = frames_t[idx_i]
        fj  = frames_t[idx_j]
        sri = sr_images[idx_i]   # (3, H_sr, W_sr)
        srj = sr_images[idx_j]
        K_sr_i = K_sr_list[idx_i]
        K_sr_j = K_sr_list[idx_j]

        # Upsample COLMAP sparse depth of view i from LR → actual SR resolution
        depth_sr_i = upsample_depth(
            fi["depth_lr"], target_hw=(H_sr, W_sr)
        ).to(device)

        # Backward-warp srj into view i's perspective
        # Camera params are defined at LR resolution; we need them at SR resolution
        # K_sr already corresponds to 800×800 (built in dataset.py)
        warped, valid_mask = backward_warp(
            src_image = srj,
            depth_tgt = depth_sr_i,
            K_src     = K_sr_j,
            R_src     = fj["R"],
            t_src     = fj["t"],
            K_tgt     = K_sr_i,
            R_tgt     = fi["R"],
            t_tgt     = fi["t"],
        )

        # Convert to numpy float [0,1] for metric computation
        ref_np    = sri.cpu().permute(1, 2, 0).numpy()            # (H,W,3)
        warped_np = warped.cpu().permute(1, 2, 0).numpy()         # (H,W,3)
        valid_np  = valid_mask.squeeze(0).cpu().numpy().astype(bool)  # (H,W)

        # Only compare pixels where the warp is valid
        if valid_np.sum() < 1000:
            continue   # too little overlap; skip pair

        metrics = compute_all_image_metrics(ref_np, warped_np)

        # Edge mask derived from ref SR image
        edge_mask = sobel_edge_mask(ref_np, threshold=0.1)

        # Edge-weighted metrics restricted to valid warp pixels
        combined_mask = valid_np & edge_mask

        from utils.metrics import psnr as _psnr, ssim as _ssim
        metrics["psnr_edge_valid"] = _psnr(ref_np, warped_np, mask=combined_mask) if combined_mask.sum() > 100 else float("nan")
        metrics["ssim_edge_valid"] = _ssim(ref_np, warped_np, mask=combined_mask) if combined_mask.sum() > 100 else float("nan")
        metrics["valid_ratio"]     = float(valid_np.mean())
        metrics["pair"]            = f"{fi['name']}→{fj['name']}"
        metrics["view_i"]          = fi["name"]
        metrics["view_j"]          = fj["name"]

        results.append(metrics)

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Task 0.1: 2DSR view-inconsistency test")
    p.add_argument("--data_root",   required=True,
                   help="MipNeRF360 根目录（提供 COLMAP 相机 + 稀疏深度）")
    p.add_argument("--sr_dir",      default=None,
                   help="预计算 SR 图根目录（模式 A）。\n"
                        "结构：<sr_dir>/<scene>/<frame>.png\n"
                        "不传则实时运行 SwinIR（模式 B）。")
    p.add_argument("--output_dir",  default="./results/task01")
    p.add_argument("--scenes",      nargs="+", default=SCENES_PHASE0)
    p.add_argument("--n_frames",    type=int, default=FRAMES_PER_SCENE)
    p.add_argument("--image_subdir",default=LR_IMAGE_SUBDIR,
                   help="用于读取 COLMAP 相机/深度时对应的子目录（默认 images_8）")
    p.add_argument("--sr_size",     type=int, default=SR_SIZE,
                   help="SR 图分辨率（默认 800）")
    p.add_argument("--depth_mode",  default="midas",
                   choices=["colmap", "midas"],
                   help="深度来源：midas（默认，稠密单目）或 colmap（COLMAP 稀疏插值）")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--save_visuals",action="store_true",
                   help="保存 [ref SR | warped SR | ×5 diff] 对比图")
    return p.parse_args()


def verdict(psnr_val: float) -> str:
    if psnr_val < PSNR_SEVERE:
        return "🔴 SEVERE  → view-consistent SR module mandatory"
    elif psnr_val < PSNR_MODERATE:
        return "🟡 MODERATE → confidence weighting sufficient"
    else:
        return "🟢 NEGLIGIBLE → simplify narrative"


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"\n{'='*60}")
    print(" Task 0.1 – 2DSR View-Inconsistency Test")
    print(f"{'='*60}")
    print(f" Scenes   : {args.scenes}")
    print(f" Frames   : {args.n_frames} per scene")
    print(f" SR size  : {args.sr_size}×{args.sr_size}")
    print(f" SR 来源  : {'预计算目录  ' + args.sr_dir if args.sr_dir else '实时 SwinIR ×4'}")
    print(f" 深度来源 : {args.depth_mode}")
    print(f" Device   : {device}")
    print()

    all_scene_rows = []
    scene_summary  = {}

    for scene in args.scenes:
        scene_root = os.path.join(args.data_root, scene)
        if not os.path.isdir(scene_root):
            print(f"  [SKIP] {scene}: directory not found at {scene_root}")
            continue

        print(f"\n[Scene: {scene}]")
        t0 = time.time()

        # ── load frames ───────────────────────────────────────────────────────
        print("  Loading frames & sparse depth …")
        try:
            frames = load_scene_frames(
                scene_root,
                image_subdir  = args.image_subdir,
                n_frames      = args.n_frames,
                target_lr_size= LR_SIZE,
                seed          = args.seed,
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        frames_t = frames_to_tensors(frames, device=device)
        print(f"  Loaded {len(frames_t)} frames in {time.time()-t0:.1f}s")

        # ── load / generate SR images ─────────────────────────────────────────
        t1 = time.time()
        if args.sr_dir:
            sr_scene_dir = os.path.join(args.sr_dir, scene)
            print(f"  Loading SR images from {sr_scene_dir} …")
            try:
                sr_images = load_sr_from_dir(
                    frames_t, sr_scene_dir, device, sr_size=args.sr_size
                )
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")
                continue
        else:
            print("  Running SwinIR ×4 …")
            sr_images = sr_with_swinir(frames_t, device)
        print(f"  SR ready in {time.time()-t1:.1f}s")

        # ── warp + metrics ────────────────────────────────────────────────────
        # ── compute / override depth maps ────────────────────────────────────────
        if args.depth_mode == "midas":
            print("  Estimating MiDaS depth on LR images …")
            from utils.depth_midas import depth_batch
            t_midas = time.time()
            midas_depths = depth_batch([f["image_lr"] for f in frames_t], device)
            for f, d in zip(frames_t, midas_depths):
                f["depth_lr"] = d
            print(f"  MiDaS done in {time.time()-t_midas:.1f}s")

        # ── warp + metrics ─────────────────────────────────────────────────────
        print(f"  Warping {len(frames_t)*(len(frames_t)-1)} pairs …")
        t2 = time.time()
        pair_results = warp_and_compare(sr_images, frames_t, device, sr_size=args.sr_size)
        print(f"  Warp+metrics done in {time.time()-t2:.1f}s  ({len(pair_results)} valid pairs)")

        if not pair_results:
            print(f"  [WARNING] No valid pairs for {scene}")
            continue

        # ── aggregate ─────────────────────────────────────────────────────────
        df = pd.DataFrame(pair_results)
        df["scene"] = scene

        psnr_full_mean   = df["psnr_full"].mean()
        ssim_full_mean   = df["ssim_full"].mean()
        psnr_edge_mean   = df["psnr_edge"].mean()
        ssim_edge_mean   = df["ssim_edge"].mean()
        psnr_edgevalid_m = df["psnr_edge_valid"].dropna().mean()

        scene_summary[scene] = {
            "psnr_full" : round(psnr_full_mean, 2),
            "ssim_full" : round(ssim_full_mean, 4),
            "psnr_edge" : round(psnr_edge_mean, 2),
            "ssim_edge" : round(ssim_edge_mean, 4),
            "psnr_edge_valid": round(psnr_edgevalid_m, 2),
            "n_pairs"   : len(pair_results),
            "verdict"   : verdict(psnr_full_mean),
        }
        all_scene_rows.append(df)

        print(f"  PSNR(full)={psnr_full_mean:.2f} dB  SSIM(full)={ssim_full_mean:.4f}")
        print(f"  PSNR(edge)={psnr_edge_mean:.2f} dB  SSIM(edge)={ssim_edge_mean:.4f}")
        print(f"  {verdict(psnr_full_mean)}")

        # ── save per-scene CSV ─────────────────────────────────────────────────
        df.to_csv(out_dir / f"{scene}_pairs.csv", index=False)

        # ── optionally save visuals ───────────────────────────────────────────
        if args.save_visuals and pair_results:
            _save_visual_comparison(
                scene, pair_results[:3],
                frames_t, sr_images,
                out_dir / "visuals" / scene,
                device,
                sr_size=args.sr_size,
            )

    # ── overall summary ───────────────────────────────────────────────────────
    if not scene_summary:
        print("\n[ERROR] No scenes processed.")
        return

    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    summary_rows = []
    for scene, s in scene_summary.items():
        print(f"  {scene:<12}  PSNR={s['psnr_full']:.2f} dB  SSIM={s['ssim_full']:.4f}  {s['verdict']}")
        summary_rows.append({"scene": scene, **{k: v for k, v in s.items() if k != "verdict"}})

    # Compute overall mean across scenes
    all_df = pd.concat(all_scene_rows, ignore_index=True) if all_scene_rows else pd.DataFrame()
    if not all_df.empty:
        overall_psnr = all_df["psnr_full"].mean()
        overall_ssim = all_df["ssim_full"].mean()
        print(f"\n  {'OVERALL':<12}  PSNR={overall_psnr:.2f} dB  SSIM={overall_ssim:.4f}")
        print(f"  {verdict(overall_psnr)}")

    # ── save outputs ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "scene_summary.csv", index=False)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"scenes": scene_summary,
             "overall_psnr": round(overall_psnr, 2) if not all_df.empty else None,
             "overall_ssim": round(overall_ssim, 4) if not all_df.empty else None,
             "verdict": verdict(overall_psnr) if not all_df.empty else "n/a"},
            f, indent=2,
        )

    print(f"\n Results saved to: {out_dir}")
    _print_decision_matrix(scene_summary, overall_psnr if not all_df.empty else None)


def _print_decision_matrix(scene_summary: dict, overall_psnr):
    print(f"\n{'='*60}")
    print(" DECISION MATRIX")
    print(f"{'='*60}")
    print(f"  Threshold reference:")
    print(f"    PSNR ≥ {PSNR_MODERATE} dB → negligible")
    print(f"    {PSNR_SEVERE} ≤ PSNR < {PSNR_MODERATE} dB → moderate")
    print(f"    PSNR < {PSNR_SEVERE} dB → severe (VC-SR module required)")
    print()

    if overall_psnr is not None:
        if overall_psnr < PSNR_SEVERE:
            print("  ➤ ACTION: Implement View-Consistent SR (F_VCSR) module")
            print("           → Use VGGT cameras + epipolar attention for cross-view features")
        elif overall_psnr < PSNR_MODERATE:
            print("  ➤ ACTION: Add Confidence-Weighted loss (C_HR from HR Head)")
            print("           → Down-weight 2DSR supervision in edge regions")
        else:
            print("  ➤ ACTION: None – 2DSR inconsistency negligible")
            print("           → Narrative: simplify VC-SR description")


def _save_visual_comparison(scene, pair_results, frames_t, sr_images, vis_dir, device, sr_size=SR_SIZE):
    """Save side-by-side: [ref SR | warped SR | diff×5] for first few pairs."""
    from PIL import Image as PILImage
    import torch

    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(pair_results):
        view_i = row["view_i"]
        view_j = row["view_j"]
        fi_idx = next(k for k, f in enumerate(frames_t) if f["name"] == view_i)
        fj_idx = next(k for k, f in enumerate(frames_t) if f["name"] == view_j)

        sri = sr_images[fi_idx].cpu().permute(1, 2, 0).numpy()
        srj = sr_images[fj_idx]

        from utils.warp import upsample_depth, backward_warp
        from utils.colmap_reader import scale_K as _scale_K
        _, H_sr_v, W_sr_v = sr_images[fi_idx].shape
        depth_sr_i = upsample_depth(
            frames_t[fi_idx]["depth_lr"], target_hw=(H_sr_v, W_sr_v)
        ).to(device)
        K_lr_i = frames_t[fi_idx]["K_lr"].cpu().numpy()
        K_lr_j = frames_t[fj_idx]["K_lr"].cpu().numpy()
        K_sr_vi = torch.from_numpy(_scale_K(K_lr_i, (LR_SIZE, LR_SIZE), (W_sr_v, H_sr_v))).float().to(device)
        K_sr_vj = torch.from_numpy(_scale_K(K_lr_j, (LR_SIZE, LR_SIZE), (W_sr_v, H_sr_v))).float().to(device)
        warped, _ = backward_warp(
            srj, depth_sr_i,
            K_sr_vj, frames_t[fj_idx]["R"], frames_t[fj_idx]["t"],
            K_sr_vi, frames_t[fi_idx]["R"], frames_t[fi_idx]["t"],
        )
        warped_np = warped.cpu().permute(1, 2, 0).numpy()
        diff_np   = np.abs(sri - warped_np) * 5.0

        H, W = sri.shape[:2]
        canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
        canvas[:, :W]      = (sri * 255).clip(0, 255).astype(np.uint8)
        canvas[:, W:2*W]   = (warped_np * 255).clip(0, 255).astype(np.uint8)
        canvas[:, 2*W:3*W] = (diff_np * 255).clip(0, 255).astype(np.uint8)

        out_path = vis_dir / f"{view_i}_from_{view_j}.png"
        PILImage.fromarray(canvas).save(out_path)


if __name__ == "__main__":
    main()

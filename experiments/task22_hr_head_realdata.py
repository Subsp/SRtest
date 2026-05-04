"""
HR Head forward on real LR RGB + COLMAP (+ optional VGGT depth + StableSR priors).

**Important:** mip-splatting ``output/…/experiment_name`` is **training output**
(checkpoints etc.), normally **without** JPEG training views. LR images and
matching ``sparse/`` still live under the **source scene** (e.g. ``…/kitchen``).

This script separates:
  * ``scene_root``: can be mip-splat output **or** ignored for images when you set paths below
  * ``sparse_dir``: COLMAP tree (often ``…/kitchen``)
  * ``image_root``: folder that contains ``images`` / ``images_8`` / … (often **same as sparse_dir parent** → auto-derived from ``sparse_dir`` when possible)

Example (your layout):
  python task22_hr_head_realdata.py \\
    --scene_root /root/autodl-tmp/SOFSR/output/.../mipsplatting_x8to2_baseline_directsrc_v1 \\
    --sparse_dir /root/autodl-tmp/kitchen \\
    --auto_images \\
    --priors_dir /root/autodl-tmp/kitchen/priors \\
    --depth_source colmap \\
    --output_dir ./results/task22_kitchen_mipsplat_lr \\
    --device cuda

可加 ``--eval_vggt_upsampled_baseline``（需 ``--oracle_dir``）：与同 oracle 度量 **VGGT LR 深度双线性上采样到 HR**，与 HR Head 输出对齐可比。
with ``sparse/0/cameras.bin``, or equals ``kitchen`` when ``sparse_dir`` is ``kitchen/sparse/0``.
Otherwise pass ``--image_root`` explicitly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from configs import LR_SIZE, SR_SIZE, VGGT_ROOT
from models.hr_head import HRGeometricPriorHead
from models.hr_head_hd_vggt_style import HDVGGTStyleGeomHead
from utils.dataset import frames_to_tensors, load_scene_frames, pick_image_subdir


def _infer_image_root_from_sparse(sparse_path: Path) -> Optional[str]:
    """
    sparse_path is either the scene root (…/kitchen with sparse/0/) or …/kitchen/sparse/0.
    Returns the scene root that should contain LR image folders.
    """
    p = sparse_path.resolve()
    if (p / "sparse" / "0" / "cameras.bin").is_file():
        return str(p)
    cam = p / "cameras.bin"
    if cam.is_file() and p.name == "0" and p.parent.name == "sparse":
        return str(p.parent.parent)
    return None


def _resolve_image_base(args, scene_root: str, sparse_override: Optional[str]) -> str:
    if args.image_root:
        return str(Path(args.image_root).expanduser().resolve())
    if sparse_override:
        inferred = _infer_image_root_from_sparse(Path(sparse_override))
        if inferred:
            return inferred
    return scene_root


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scene_root",
        required=True,
        help="Mip-splatting train output dir **or** full COLMAP scene; LR files use --image_root / sparse inference.",
    )
    p.add_argument(
        "--image_root",
        default=None,
        help="Parent of LR image folder (images / images_8 / …). "
        "Default: infer from --sparse_dir when it points at kitchen-style tree.",
    )
    p.add_argument(
        "--image_subdir",
        default=None,
        help="Subfolder under image_root. With --auto_images, scan common names.",
    )
    p.add_argument(
        "--auto_images",
        action="store_true",
        help="Pick first existing images_8 / images / images_2 / … under image_root.",
    )
    p.add_argument(
        "--sparse_dir",
        default=None,
        help="COLMAP source: directory containing sparse/0 or path to sparse/0 itself.",
    )
    p.add_argument(
        "--priors_dir",
        default=None,
        help="StableSR cache folder (<stem>.png). Optional.",
    )
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--target_lr_size", type=int, default=LR_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--depth_source",
        choices=("colmap", "vggt"),
        default="colmap",
    )
    p.add_argument("--vggt_root", default=VGGT_ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="./results/task22_realdata")
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument(
        "--head_variant",
        choices=("unet", "hd_vggt_style"),
        default="unet",
        help="unet=原卷积 U-Net；hd_vggt_style=LR ViT + HD-VGGT 式引导上采样(HR) + 卷积 refiner（非官方权重）。",
    )
    p.add_argument("--force_no_sr_prior", action="store_true")
    p.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint .pt（纯 state_dict 或 train_hr_head 保存的 {\"model\": ...}）。不传则为随机初始化。",
    )
    p.add_argument(
        "--oracle_dir",
        default=None,
        help="可选：oracle 深度目录，导出后对每帧算 AbsRel / scale-inv L1 / RMSE（对齐尺度）。",
    )
    p.add_argument(
        "--eval_vggt_upsampled_baseline",
        action="store_true",
        help="需同时提供 --oracle_dir：对冻结 VGGT 的 LR 深度双线性上到 HR(vs SR_SIZE)，"
        "用与 HR Head **相同**oracle 度量；便于回答「有无强过仅用 VGGT+上采样」。",
    )
    return p.parse_args()


def _resolve_img_subdir(image_base: str, args) -> str:
    if args.image_subdir is not None:
        return pick_image_subdir(image_base, preferred=args.image_subdir)
    if args.auto_images:
        return pick_image_subdir(image_base, preferred=None)
    try:
        return pick_image_subdir(image_base, preferred="images_8")
    except FileNotFoundError:
        return pick_image_subdir(image_base, preferred=None)


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
    sparse_override = str(Path(args.sparse_dir).expanduser().resolve()) if args.sparse_dir else None
    priors_dir = str(Path(args.priors_dir).expanduser().resolve()) if args.priors_dir else None

    image_base = _resolve_image_base(args, scene_root, sparse_override)
    img_dir = _resolve_img_subdir(image_base, args)

    prior_dir_kw = priors_dir if (priors_dir and not args.force_no_sr_prior) else None

    print(f"[data] mip/output tag   scene_root = {scene_root}")
    print(f"[data] image_root (LR)= {image_base}")
    print(f"[data] image_subdir    = {img_dir}")
    print(f"[data] sparse_dir      = {sparse_override or '(derived from scene_root)'}")
    print(f"[data] priors_dir      = {prior_dir_kw or '(off)'}")

    scene_abs = os.path.abspath(scene_root)
    img_abs = os.path.abspath(image_base)
    image_root_kw = None
    if args.image_root or img_abs != scene_abs:
        image_root_kw = image_base

    frames = load_scene_frames(
        scene_root,
        image_subdir=img_dir,
        image_root=image_root_kw,
        prior_dir=prior_dir_kw,
        prior_subdir=None,
        sparse_dir=sparse_override,
        n_frames=args.n_frames,
        target_lr_size=args.target_lr_size,
        seed=args.seed,
    )
    frames_t = frames_to_tensors(frames, device=args.device)

    vggt_vo_cache = None
    need_vggt = args.depth_source == "vggt" or args.eval_vggt_upsampled_baseline
    if need_vggt:
        import task02_vggt_geometry as t2

        model_vggt, pose_fn = t2._setup_vggt(args.vggt_root, args.device)
        vggt_vo_cache = t2.run_vggt_on_frames(model_vggt, pose_fn, frames_t, args.device)
        if args.depth_source == "vggt":
            for i, f in enumerate(frames_t):
                d = torch.from_numpy(vggt_vo_cache[i]["depth_vggt"]).float().to(args.device).clamp_min(1e-3)
                f["depth_lr"] = d.unsqueeze(0)

    depth_b, rgb_b, sr_b = _stack_views(frames_t, args.device)
    use_sr = sr_b is not None and not args.force_no_sr_prior
    sr_scale = max(1, int(round(SR_SIZE / float(args.target_lr_size))))

    model = (
        HDVGGTStyleGeomHead(use_rgb=True, use_sr_prior=use_sr, sr_scale=sr_scale)
        if args.head_variant == "hd_vggt_style"
        else HRGeometricPriorHead(
            use_rgb=True,
            use_sr_prior=use_sr,
            base_channels=args.base_channels,
            sr_scale=sr_scale,
        )
    ).to(args.device)

    if args.ckpt:
        ck_path = Path(args.ckpt).expanduser().resolve()
        try:
            blob = torch.load(str(ck_path), map_location=args.device, weights_only=False)
        except TypeError:
            blob = torch.load(str(ck_path), map_location=args.device)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        bad = model.load_state_dict(sd, strict=False)
        if bad.missing_keys or bad.unexpected_keys:
            print(f"[ckpt] load strict=False missing={bad.missing_keys[:3]}… unexpected={bad.unexpected_keys[:3]}…")
        print(f"[ckpt] {ck_path}")

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
    print(f"     depth_hr shape: {tuple(out['depth_hr'].shape)}")

    summary_hr: Optional[Tuple[float, float, int]] = None
    summary_vggt_bl: Optional[Tuple[float, float, int]] = None

    if args.oracle_dir:
        import csv

        import torch.nn.functional as F

        from task02_vggt_geometry import load_oracle_depth
        from utils.metrics import compute_all_depth_metrics

        ora = str(Path(args.oracle_dir).expanduser().resolve())
        rows: List[dict] = []
        for vi in range(v):
            name = frames_t[vi]["name"]
            pred = out["depth_hr"][0, vi, 0].detach().float().cpu().numpy()
            gd = load_oracle_depth(ora, name)
            if gd is None:
                print(f"[oracle] skip {name} (missing)")
                continue
            if gd.ndim != 2:
                gd = np.squeeze(gd)
            gt_t = torch.from_numpy(gd.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            gt = (
                F.interpolate(gt_t, size=(SR_SIZE, SR_SIZE), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )
            m = gt > 1e-6
            metric = compute_all_depth_metrics(pred, gt, mask=m)
            metric["frame"] = name
            rows.append(metric)
        if rows:
            csv_path = out_dir / "depth_metrics_vs_oracle.csv"
            keys = sorted(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            import statistics as stats

            ar = stats.mean(float(r["abs_rel"]) for r in rows if not np.isnan(r["abs_rel"]))
            si = stats.mean(float(r["scale_inv_l1"]) for r in rows if not np.isnan(r["scale_inv_l1"]))
            summary_hr = (ar, si, len(rows))
            print(f"[oracle] HR Head vs oracle: mean AbsRel={ar:.4f}  ScaleInvL1={si:.4f} → {csv_path}")
        else:
            print("[oracle] no overlaps with oracle depth; check oracle_dir stems")

    if (
        args.eval_vggt_upsampled_baseline
        and args.oracle_dir
        and vggt_vo_cache is not None
    ):
        import csv

        import torch.nn.functional as F

        from task02_vggt_geometry import load_oracle_depth
        from utils.metrics import compute_all_depth_metrics

        import statistics as stats

        ora = str(Path(args.oracle_dir).expanduser().resolve())
        bros: List[dict] = []
        for vi in range(v):
            name = frames_t[vi]["name"]
            vd = torch.from_numpy(vggt_vo_cache[vi]["depth_vggt"].astype(np.float32)).clamp_min(1e-6)
            pred = (
                F.interpolate(
                    vd.unsqueeze(0).unsqueeze(0),
                    size=(SR_SIZE, SR_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            gd = load_oracle_depth(ora, name)
            if gd is None:
                continue
            if gd.ndim != 2:
                gd = np.squeeze(gd)
            gt_t = torch.from_numpy(gd.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            gt = (
                F.interpolate(gt_t, size=(SR_SIZE, SR_SIZE), mode="bilinear", align_corners=False)
                .squeeze()
                .numpy()
            )
            m = gt > 1e-6
            metric = compute_all_depth_metrics(pred, gt, mask=m)
            metric["frame"] = name
            bros.append(metric)
        if bros:
            bp = out_dir / "depth_metrics_vggt_upsampled_vs_oracle.csv"
            keys = sorted(bros[0].keys())
            with open(bp, "w", newline="", encoding="utf-8") as fp:
                w = csv.DictWriter(fp, fieldnames=keys)
                w.writeheader()
                for r in bros:
                    w.writerow(r)
            ar_b = stats.mean(float(r["abs_rel"]) for r in bros if not np.isnan(r["abs_rel"]))
            si_b = stats.mean(float(r["scale_inv_l1"]) for r in bros if not np.isnan(r["scale_inv_l1"]))
            summary_vggt_bl = (ar_b, si_b, len(bros))
            print(
                f"[baseline] VGGT LR depth → HR bilinear vs oracle (same protocol): "
                f"mean AbsRel={ar_b:.4f}  ScaleInvL1={si_b:.4f} → {bp}"
            )
        else:
            print("[baseline] no VGGT-vs-oracle rows (oracle depth missing for sampled frames?).")
    elif args.eval_vggt_upsampled_baseline and args.oracle_dir:
        print(
            "[baseline][ERROR] --eval_vggt_upsampled_baseline 需要 VGGT 输出，"
            "但 vggt_vo_cache 为空（前端 VGGT 是否加载/跑失败）。不会生成 VGGT baseline CSV。"
        )

    if summary_hr is not None and summary_vggt_bl is not None:
        ar_h, si_h, n_h = summary_hr
        ar_v, si_v, n_v = summary_vggt_bl
        c_hr = out_dir / "depth_metrics_vs_oracle.csv"
        c_bl = out_dir / "depth_metrics_vggt_upsampled_vs_oracle.csv"
        lines = [
            "HR Head vs VGGT（LR 深度双线性上采样到 HR）— 与同一 oracle、同一评测协议",
            f"  HR Head:               mean AbsRel={ar_h:.4f}  ScaleInvL1={si_h:.4f}  (n={n_h} frames)",
            f"  VGGT LR→HR bilinear:  mean AbsRel={ar_v:.4f}  ScaleInvL1={si_v:.4f}  (n={n_v} frames)",
            f"  Δ (HR − VGGT baseline): ΔAbsRel={ar_h - ar_v:+.4f}  ΔScaleInvL1={si_h - si_v:+.4f}",
            "    （AbsRel / Scale-inv L1：数值越小越好；Δ 为负表示 HR Head 优于双线性 VGGT）",
            "",
            f"  {c_hr}",
            f"  {c_bl}",
        ]
        msg = "\n".join(lines)
        print("\n" + "=" * 72 + "\n" + msg + "\n" + "=" * 72)
        comp = out_dir / "compare_hrhead_vs_vggt_lr_bilinear.txt"
        comp.write_text(msg + "\n", encoding="utf-8")
        print(f"[compare] 并排对比摘要: {comp}")
    elif (
        summary_hr is not None
        and args.eval_vggt_upsampled_baseline
        and summary_vggt_bl is None
        and vggt_vo_cache is not None
    ):
        print(
            "[compare][WARN] VGGT 已跑通但 baseline 无有效行（与 oracle 无重叠帧？）。"
            " 不会生成 compare 摘要；请查 depth_metrics_vggt_upsampled_vs_oracle.csv 是否为空。"
        )


if __name__ == "__main__":
    main()
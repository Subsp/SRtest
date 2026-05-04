"""
Phase 2.3 — HR Head 训练（oracle GS 蒸馏深度，v0）

- 数据来源与 ``task22_hr_head_realdata.py`` 一致：LR 图在 ``image_root`` / ``image_subdir``，
  COLMAP 在 ``sparse_dir``（或 scene_root）。
- **监督**：mip-splatting 渲染的 oracle HR 深度（``task02_oracle_render.py``），
  resize 至 ``SR_SIZE`` 后与预测 ``depth_hr`` 对齐，使用 ``geom_prior.geom_depth_loss_l1``
  （median-scale + 置信度可选）。

不含：法向监督、分布式、完整数据管线扩展（后续可加）。

示例：
  cd experiments && python train_hr_head.py \\
    --scene_root /root/autodl-tmp/SOFSR/output/.../mipsplatting_x8to2_baseline_directsrc_v1 \\
    --sparse_dir /root/autodl-tmp/kitchen \\
    --oracle_dir ./results/task02/oracle/kitchen \\
    --priors_dir /root/autodl-tmp/kitchen/priors \\
    --auto_images \\
    --epochs 400 --lr 3e-4 \\
    --output_dir ./checkpoints/hr_head_kitchen_v0 \\
    --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import task22_hr_head_realdata as pathcfg
from configs import LR_SIZE, SR_SIZE, VGGT_ROOT
from geom_prior import PriorPackDepth, geom_depth_loss_l1
from models.hr_head import HRGeometricPriorHead
from task02_vggt_geometry import load_oracle_depth, run_vggt_on_frames, _setup_vggt
from utils.dataset import frames_to_tensors, load_scene_frames

# ── oracle batch ─────────────────────────────────────────────────────────────


def oracle_stack_for_frames(
    frames_np: List[Dict[str, Any]],
    oracle_scene_dir: str,
    target_hw: tuple[int, int],
    device: str = "cpu",
    dtype=torch.float32,
) -> tuple[Optional[torch.Tensor], List[Dict[str, Any]]]:
    """
    Builds (V,target_h,target_w) oracle depth tensor aligned with filtered ``frames``.
    Drops views with missing oracle depth.
    """
    Ht, Wt = target_hw
    kept_frames: List[Dict[str, Any]] = []
    chunks: List[torch.Tensor] = []
    for f in frames_np:
        name = f["name"]
        depth_np = load_oracle_depth(oracle_scene_dir, name)
        if depth_np is None:
            continue
        if depth_np.ndim != 2:
            depth_np = np.squeeze(depth_np)
            if depth_np.ndim != 2:
                raise ValueError(f"Oracle depth for {name} must be H×W")
        dt = torch.from_numpy(depth_np).to(dtype=dtype, device=device)
        dt = F.interpolate(
            dt.unsqueeze(0).unsqueeze(0),
            size=(Ht, Wt),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        chunks.append(dt)
        kept_frames.append(f)

    if not chunks:
        return None, []
    stack = torch.stack(chunks, dim=0)
    return stack, kept_frames


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_root", required=True)
    p.add_argument("--sparse_dir", default=None)
    p.add_argument("--image_root", default=None)
    p.add_argument("--image_subdir", default=None)
    p.add_argument("--auto_images", action="store_true")
    p.add_argument("--priors_dir", default=None)
    p.add_argument("--oracle_dir", required=True,
                   help="Per-scene oracle root (directory containing *.npy or train/…/depth/)")
    p.add_argument("--n_frames", type=int, default=32,
                   help="Max training views sampled per epoch (filtered by oracle availability).")
    p.add_argument(
        "--views_per_forward",
        type=int,
        default=1,
        help="Views per HR-Head forward (default 1 saves VRAM; V>1 needs more GPU memory).",
    )
    p.add_argument("--target_lr_size", type=int, default=LR_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth_source", choices=("colmap", "vggt"), default="colmap")
    p.add_argument("--vggt_root", default=VGGT_ROOT)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--base_channels", type=int, default=96)

    p.add_argument("--lambda_normal", type=float, default=0.05,
                   help="Encourage normals to vary smoothly (L2 on Laplacian, cheap prior).")
    p.add_argument("--lambda_conf_entropy", type=float, default=0.001,
                   help="Keep confidence from collapsing (penalise |c-0.5|).")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="./checkpoints/hr_head_train")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--force_no_sr_prior", action="store_true")
    return p.parse_args()


def _laplacian_var(x: torch.Tensor) -> torch.Tensor:
    """x (B,V,3,H,W) — crude smoothness."""
    xv = x.reshape(-1, *x.shape[2:])
    return (xv[..., 2:, 1:-1] + xv[..., :-2, 1:-1] + xv[..., 1:-1, 2:] + xv[..., 1:-1, :-2] - 4 * xv[..., 1:-1, 1:-1]).pow(2).mean()


def main():
    args = _parse_args()
    rng = Path(args.scene_root).expanduser().resolve()
    scene_root = str(rng)
    sparse_ov = str(Path(args.sparse_dir).expanduser().resolve()) if args.sparse_dir else None
    priors = str(Path(args.priors_dir).expanduser().resolve()) if args.priors_dir else None
    oracle_scene = str(Path(args.oracle_dir).expanduser().resolve())

    ns = argparse.Namespace(
        scene_root=args.scene_root,
        image_root=args.image_root,
        image_subdir=args.image_subdir,
        auto_images=args.auto_images,
        sparse_dir=args.sparse_dir,
    )
    image_base = pathcfg._resolve_image_base(ns, scene_root, sparse_ov)
    img_dir = pathcfg._resolve_img_subdir(image_base, ns)

    scene_abs = os.path.abspath(scene_root)
    img_abs = os.path.abspath(image_base)
    image_root_kw = None
    if args.image_root or img_abs != scene_abs:
        image_root_kw = image_base

    prior_kw = priors if (priors and not args.force_no_sr_prior) else None

    frames = load_scene_frames(
        scene_root,
        image_subdir=img_dir,
        image_root=image_root_kw,
        prior_dir=prior_kw,
        prior_subdir=None,
        sparse_dir=sparse_ov,
        n_frames=args.n_frames,
        target_lr_size=args.target_lr_size,
        seed=args.seed,
    )
    oracle_tensor, frames_f = oracle_stack_for_frames(
        frames, oracle_scene, (SR_SIZE, SR_SIZE), device="cpu"
    )
    if oracle_tensor is None or len(frames_f) == 0:
        raise RuntimeError(
            f"No oracle depth matched under {oracle_scene}. "
            f"Produce .npy with task02_oracle_render.py and check stem names vs images."
        )

    frames_t = frames_to_tensors(frames_f, device="cpu")
    print(f"[train] {len(frames_f)} / {len(frames)} views with oracle supervision")

    if args.depth_source == "vggt":
        frames_t_gpu = frames_to_tensors(frames_f, device=args.device)
        mv, pfn = _setup_vggt(args.vggt_root, args.device)
        vo = run_vggt_on_frames(mv, pfn, frames_t_gpu, args.device)
        for i, f in enumerate(frames_t):
            d = torch.from_numpy(vo[i]["depth_vggt"]).float().clamp_min(1e-3)
            f["depth_lr"] = d

    sr_scale = max(1, int(round(SR_SIZE / float(args.target_lr_size))))
    use_sr = priors is not None and not args.force_no_sr_prior

    # Keep training batches on CPU; move only ``views_per_forward`` slices to GPU (OOM fix).
    depth_all = torch.stack([f["depth_lr"] for f in frames_t], dim=0).unsqueeze(1).clamp_min(1e-3)  # V,1,h,w
    rgb_all = torch.stack([f["image_lr"] for f in frames_t], dim=0)  # V,3,h,w
    sr_all: Optional[torch.Tensor] = None
    if use_sr:
        hv = []
        missing = []
        for f in frames_t:
            if "prior_sr_hr" in f:
                hv.append(f["prior_sr_hr"])
            else:
                missing.append(f["name"])
        if missing:
            print(f"[WARN] Missing priors for {missing}: train without StableSR conditioning.")
            use_sr = False
        else:
            sr_all = torch.stack(hv, dim=0)  # V,3,SR,SR

    vp = max(1, int(args.views_per_forward))
    if vp > 1:
        print(f"[mem] views_per_forward={vp} — reduce to 1 if you still hit OOM.")

    model = HRGeometricPriorHead(
        use_rgb=True,
        use_sr_prior=use_sr,
        base_channels=args.base_channels,
        sr_scale=sr_scale,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    oracle_cpu = oracle_tensor  # V,H,W on CPU
    V = depth_all.shape[0]

    outp = Path(args.output_dir).resolve()
    outp.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    pbar = tqdm(range(args.epochs), desc="hr_head train")
    for ep in pbar:
        model.train()
        opt.zero_grad(set_to_none=True)

        depth_loss_epoch = 0.0
        diag_scales: List[torch.Tensor] = []
        n_loss_epoch = 0.0
        conf_epoch = 0.0

        for v0 in range(0, V, vp):
            v1 = min(V, v0 + vp)
            sl = slice(v0, v1)
            nb = v1 - v0

            depth_b = depth_all[sl].unsqueeze(0).to(args.device, non_blocking=True)  # 1,nb,1,h,w
            rgb_b = rgb_all[sl].unsqueeze(0).to(args.device, non_blocking=True)
            fwd_kw: Dict[str, Any] = {"depth_lr": depth_b, "rgb_lr": rgb_b}
            if use_sr and sr_all is not None:
                fwd_kw["sr_prior_hr"] = sr_all[sl].unsqueeze(0).to(args.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                out = model(**fwd_kw)

            depth_hr = out["depth_hr"]
            normals = out["normal_hr"]
            conf_hr = out["confidence_hr"]

            chunk_depth_loss = depth_hr.new_zeros(())
            for j in range(nb):
                vi = v0 + j
                o_oracle = oracle_cpu[vi].to(args.device, non_blocking=True)
                pack = PriorPackDepth(
                    depth=o_oracle,
                    confidence=torch.ones_like(o_oracle),
                    normal_world=None,
                )
                oracle_valid = o_oracle > 1e-6
                lv, diag = geom_depth_loss_l1(
                    depth_hr[0, j, 0],
                    pack,
                    extra_mask=oracle_valid,
                )
                chunk_depth_loss = chunk_depth_loss + lv
                diag_scales.append(diag["scale"].detach())

            depth_loss_epoch += float(chunk_depth_loss.detach())

            n_loss_v = normals.new_tensor(0.0)
            if args.lambda_normal > 0:
                n_loss_v = _laplacian_var(normals)
            conf_v = (conf_hr - 0.5).pow(2).mean() * args.lambda_conf_entropy

            n_loss_epoch += float(n_loss_v.detach()) * nb
            conf_epoch += float(conf_v.detach()) * nb

            reg_w = float(nb) / float(V)
            loss_chunk = chunk_depth_loss / float(V) + (
                args.lambda_normal * n_loss_v + conf_v
            ) * reg_w

            scaler.scale(loss_chunk).backward()

        vcount = max(1, V)
        depth_loss_accum = depth_loss_epoch / float(vcount)
        mean_scale = torch.stack(diag_scales).mean().item() if diag_scales else 0.0
        n_loss_mean = n_loss_epoch / float(vcount)
        conf_mean = conf_epoch / float(vcount)

        scaler.unscale_(opt)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        pbar.set_postfix(
            Ld=float(depth_loss_accum),
            Ln=float(n_loss_mean) if args.lambda_normal > 0 else 0.0,
            sc=float(mean_scale),
        )

        if (ep + 1) % args.save_every == 0 or ep == 0:
            ck = outp / f"hr_head_ep{ep+1:05d}.pt"
            torch.save(
                {
                    "epoch": ep + 1,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "n_supervised_views": len(frames_f),
                },
                ck,
            )
            print(f"  saved {ck}")

    torch.save(model.state_dict(), outp / "hr_head_last.pt")
    print(f"[done] last weights → {outp / 'hr_head_last.pt'}")


if __name__ == "__main__":
    main()

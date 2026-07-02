#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# SP-IE-SRGS v0 additions are research scaffolding around the original trainer.
#

"""SP-IE-SRGS v0 routed trainer.

This entrypoint is intentionally smaller than ``hybrid_sdfgs/train.py``.  It
keeps the original Gaussian schema and optimizer, but performs two forward /
backward routes per iteration:

* geometry route: LR anchor supervision updates xyz/scaling/rotation/opacity
  and is the only route used for densification statistics.
* appearance route: prepared SR prior supervision updates SH features only.

Use ``scripts/run_sp_ie_srgs_v0_kitchen.sh`` for the current kitchen server
layout.
"""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint
from typing import Dict, Iterable, Optional

from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from hybrid_sdfgs.blocks import ScaffoldGeometryBlock, ScaffoldGeometryConfig
from hybrid_sdfgs.geometry import ScaffoldLoadConfig, load_scaffold_data
from routing.param_groups import build_routed_param_groups
from routing.train_step import routed_one_optimizer_step
from scene import GaussianModel, Scene
from surface.losses import SPV0SurfaceConfig, SPV0SurfaceLoss
from surface.metrics import render_proxy_metrics
from utils.general_utils import PILtoTorch, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")


def _image_stem(path: Path) -> str:
    return path.stem


def _iter_image_files(root: Path, exts: Iterable[str] = IMAGE_EXTS):
    exts_lower = {f".{ext.lower().lstrip('.')}" for ext in exts}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in exts_lower:
            yield path


class ImageTensorBank:
    """Stem-indexed image loader for prior/anchor folders."""

    def __init__(self, root: str | os.PathLike[str], *, label: str, grayscale: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.label = label
        self.grayscale = bool(grayscale)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {self.root}")
        self.index: Dict[str, Path] = {}
        for path in _iter_image_files(self.root):
            self.index.setdefault(_image_stem(path), path)
        if not self.index:
            raise RuntimeError(f"{label} directory has no image files: {self.root}")

    def get(
        self,
        image_name: str,
        *,
        device: torch.device | str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        path = self.index.get(image_name)
        if path is None:
            return None
        mode = "L" if self.grayscale else "RGB"
        with Image.open(path) as handle:
            image = handle.convert(mode)
        if width is None or height is None:
            resolution = image.size
        else:
            resolution = (int(width), int(height))
        tensor = PILtoTorch(image, resolution).to(device=device, dtype=torch.float32)
        if self.grayscale:
            tensor = tensor[:1]
        else:
            tensor = tensor[:3]
        return tensor.clamp(0.0, 1.0)

    def coverage(self, image_names: Iterable[str]) -> Dict[str, object]:
        names = list(image_names)
        missing = [name for name in names if name not in self.index]
        return {
            "label": self.label,
            "root": str(self.root),
            "available": len(self.index),
            "requested": len(names),
            "matched": len(names) - len(missing),
            "missing": missing[:32],
            "missing_total": len(missing),
        }


def _resize_to_match(image: torch.Tensor, target: torch.Tensor, *, mode: str = "area") -> torch.Tensor:
    if image.shape[-2:] == target.shape[-2:]:
        return image
    if mode == "area":
        return F.interpolate(image.unsqueeze(0), size=target.shape[-2:], mode="area").squeeze(0)
    return F.interpolate(
        image.unsqueeze(0),
        size=target.shape[-2:],
        mode=mode,
        align_corners=False,
    ).squeeze(0)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return l1_loss(pred, target)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if mask.shape[0] == 1 and pred.shape[0] != 1:
        mask = mask.expand(pred.shape[0], -1, -1)
    denom = torch.clamp(mask.sum(), min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def _dssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return 1.0 - ssim(pred, target)
    mask = mask.to(device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    if mask.shape[0] == 1 and pred.shape[0] != 1:
        mask = mask.expand(pred.shape[0], -1, -1)
    pred_masked = pred * mask + target.detach() * (1.0 - mask)
    return 1.0 - ssim(pred_masked, target)


def _weighted_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_dssim: float,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = _masked_l1(pred, target, mask)
    dssim = _dssim_loss(pred, target, mask) if float(lambda_dssim) > 0.0 else pred.new_tensor(0.0)
    total = (1.0 - float(lambda_dssim)) * l1 + float(lambda_dssim) * dssim
    return total, l1, dssim


def _prepare_output_and_logger(args: Namespace):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def _write_jsonl(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


MappingLike = Dict[str, object]


def _audit_payload(iteration: int, result, point_count: int) -> MappingLike:
    return {
        "iteration": int(iteration),
        "points": int(point_count),
        "metrics": result.metrics,
        "grad_audit": result.grad_audit,
    }


def training(
    dataset,
    opt,
    pipe,
    sp_args,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = _prepare_output_and_logger(sp_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    if sp_args.sp_init_ply:
        init_ply = Path(sp_args.sp_init_ply).expanduser().resolve()
        if not init_ply.is_file():
            raise FileNotFoundError(f"--sp_init_ply not found: {init_ply}")
        print(f"[SP-IE-SRGS] initializing Gaussians from PLY: {init_ply}")
        gaussians.load_ply(str(init_ply))
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    gaussians.training_setup(opt)
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras = scene.getTrainCameras().copy()
    if not train_cameras:
        raise RuntimeError("Scene has no training cameras")
    test_cameras = scene.getTestCameras().copy()

    highresolution_index = [
        index for index, camera in enumerate(train_cameras) if camera.image_width >= 800
    ]

    gaussians.compute_3D_filter(cameras=train_cameras)

    lr_anchor_bank = ImageTensorBank(sp_args.sp_lr_anchor_dir, label="LR-ANCHOR", grayscale=False)
    prior_root = Path(sp_args.prepared_sr_prior_root).expanduser().resolve()
    prior_dir = prior_root / sp_args.sr_prior_subdir
    prior_bank = ImageTensorBank(prior_dir, label="SR-PRIOR", grayscale=False)
    prior_mask_bank = None
    if sp_args.sr_prior_mask_subdir:
        mask_dir = prior_root / sp_args.sr_prior_mask_subdir
        if mask_dir.is_dir():
            prior_mask_bank = ImageTensorBank(mask_dir, label="SR-PRIOR-MASK", grayscale=True)
        elif sp_args.sp_require_prior_mask:
            raise FileNotFoundError(f"SR prior mask directory not found: {mask_dir}")
        else:
            print(f"[SP-IE-SRGS] prior mask directory missing, continuing unmasked: {mask_dir}")

    camera_names = [camera.image_name for camera in train_cameras]
    coverage_report = {
        "lr_anchor": lr_anchor_bank.coverage(camera_names),
        "prior": prior_bank.coverage(camera_names),
        "prior_mask": prior_mask_bank.coverage(camera_names) if prior_mask_bank is not None else None,
    }
    print("[SP-IE-SRGS] asset coverage:")
    print(json.dumps(coverage_report, indent=2, sort_keys=True))
    if coverage_report["lr_anchor"]["missing_total"] and not sp_args.sp_allow_missing_lr_anchor:
        raise RuntimeError("LR anchor coverage is incomplete; rerun preflight or set --sp_allow_missing_lr_anchor")
    if coverage_report["prior"]["missing_total"] and not sp_args.sp_allow_missing_prior:
        raise RuntimeError("SR prior coverage is incomplete; rerun preflight or set --sp_allow_missing_prior")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    audit_path = Path(sp_args.model_path) / "sp_route_audit.jsonl"
    surface_loss_block = None
    if bool(sp_args.sp_surface_enable) and float(sp_args.sp_lambda_surface) > 0.0:
        surface_loss_block = SPV0SurfaceLoss(
            SPV0SurfaceConfig(
                lambda_surface=float(sp_args.sp_lambda_surface),
                lambda_distortion=float(sp_args.sp_lambda_distortion),
                lambda_depth_normal=float(sp_args.sp_lambda_depth_normal),
                lambda_smoothness=float(sp_args.sp_lambda_normal_smooth),
                ramp_start_iter=int(sp_args.sp_surface_ramp_start_iter),
                ramp_end_iter=int(sp_args.sp_surface_ramp_end_iter),
            )
        )
        print(
            "[SP-IE-SRGS] surface route enabled: "
            f"lambda={sp_args.sp_lambda_surface} "
            f"dist={sp_args.sp_lambda_distortion} "
            f"dn={sp_args.sp_lambda_depth_normal} "
            f"smooth={sp_args.sp_lambda_normal_smooth} "
            f"ramp={sp_args.sp_surface_ramp_start_iter}->{sp_args.sp_surface_ramp_end_iter}"
        )
    else:
        print("[SP-IE-SRGS] surface route disabled: running sp_routing_only.")

    scaffold_block = None
    scaffold_chamfer_weight = float(sp_args.sp_scaffold_chamfer_weight)
    scaffold_normal_weight = float(sp_args.sp_scaffold_normal_weight)
    if bool(sp_args.sp_scaffold_enable):
        if not sp_args.sp_scaffold_path:
            raise ValueError("--sp_scaffold_enable requires --sp_scaffold_path")
        scaffold_data = load_scaffold_data(ScaffoldLoadConfig(path=sp_args.sp_scaffold_path))
        scaffold_block = ScaffoldGeometryBlock(
            ScaffoldGeometryConfig(
                sample_size=int(sp_args.sp_scaffold_sample_size),
                interval=int(sp_args.sp_scaffold_interval),
                axis=sp_args.sp_scaffold_axis,
            ),
            scaffold_points_cpu=scaffold_data.points,
            scaffold_normals_cpu=scaffold_data.normals,
        )
        print(
            "[SP-IE-SRGS] scaffold geometry enabled: "
            f"path={scaffold_data.source_path} "
            f"points={scaffold_data.num_points} normals={int(scaffold_data.has_normals)} "
            f"weights=({scaffold_chamfer_weight}, {scaffold_normal_weight}) "
            f"sample={sp_args.sp_scaffold_sample_size} interval={sp_args.sp_scaffold_interval}"
        )
    else:
        print("[SP-IE-SRGS] scaffold geometry disabled.")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    max_points_notice_shown = False
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="SP-IE-SRGS training")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (
            random.random() < 0.3
            and dataset.sample_more_highres
            and len(highresolution_index) > 0
        ):
            viewpoint_cam = train_cameras[highresolution_index[randint(0, len(highresolution_index) - 1)]]

        if (iteration - 1) == debug_from:
            pipe.debug = True

        geo_holder: Dict[str, object] = {}

        def geometry_closure():
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                kernel_size=dataset.kernel_size,
                subpixel_offset=None,
            )
            geo_holder["render_pkg"] = render_pkg
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()
            lr_anchor = lr_anchor_bank.get(viewpoint_cam.image_name, device=image.device)
            if lr_anchor is None:
                if not sp_args.sp_allow_missing_lr_anchor:
                    raise RuntimeError(f"Missing LR anchor for view {viewpoint_cam.image_name}")
                lr_anchor = F.interpolate(
                    viewpoint_cam.original_image.cuda().unsqueeze(0),
                    scale_factor=1.0 / max(float(sp_args.sp_lr_fallback_downscale), 1.0),
                    mode="area",
                ).squeeze(0)
            pred_lr = _resize_to_match(image, lr_anchor, mode="area")
            loss_geo, geo_l1, geo_dssim = _weighted_recon_loss(
                pred_lr,
                lr_anchor,
                lambda_dssim=sp_args.sp_geo_lambda_dssim,
            )
            metrics = {
                "geo_l1": float(geo_l1.detach().item()),
                "geo_dssim": float(geo_dssim.detach().item()),
                "geo_lr_h": float(lr_anchor.shape[-2]),
                "geo_lr_w": float(lr_anchor.shape[-1]),
            }
            metrics.update(render_proxy_metrics(render_pkg))
            if surface_loss_block is not None:
                surface_loss, surface_metrics = surface_loss_block.compute(
                    gaussians=gaussians,
                    iteration=iteration,
                    render_ctx={
                        "viewpoint_camera": viewpoint_cam,
                        "gt_image": gt_image,
                        "render_pkg": render_pkg,
                        "render_fn": render,
                        "pipe": pipe,
                        "background": background,
                        "kernel_size": dataset.kernel_size,
                        "subpixel_offset": None,
                    },
                )
                metrics.update(surface_metrics)
                if surface_loss is not None:
                    loss_geo = loss_geo + surface_loss
            if scaffold_block is not None and (scaffold_chamfer_weight > 0.0 or scaffold_normal_weight > 0.0):
                scaffold_chamfer, scaffold_normal, scaffold_info = scaffold_block.compute(
                    xyz_all=gaussians.get_xyz,
                    rotations_raw_all=gaussians._rotation,
                    scales_all=gaussians.get_scaling,
                    iteration=iteration,
                )
                scaffold_loss = scaffold_chamfer_weight * scaffold_chamfer + scaffold_normal_weight * scaffold_normal
                loss_geo = loss_geo + scaffold_loss
                metrics.update(
                    {
                        "scaffold_total": float(scaffold_loss.detach().item()),
                        "scaffold_chamfer": float(scaffold_chamfer.detach().item()),
                        "scaffold_normal": float(scaffold_normal.detach().item()),
                        "scaffold_selected_gs": float(scaffold_info.get("selected_gs", 0.0)),
                        "scaffold_selected": float(scaffold_info.get("selected_scaffold", 0.0)),
                        "scaffold_has_normals": float(scaffold_info.get("has_normals", 0.0)),
                    }
                )
            return {
                "loss": loss_geo,
                "render_pkg": render_pkg,
                "metrics": metrics,
            }

        def appearance_closure():
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                background,
                kernel_size=dataset.kernel_size,
                subpixel_offset=None,
            )
            image = render_pkg["render"]
            prior = prior_bank.get(
                viewpoint_cam.image_name,
                device=image.device,
                width=int(viewpoint_cam.image_width),
                height=int(viewpoint_cam.image_height),
            )
            if prior is None:
                if not sp_args.sp_allow_missing_prior:
                    raise RuntimeError(f"Missing SR prior for view {viewpoint_cam.image_name}")
                zero = image.sum() * 0.0
                return {"loss": zero, "render_pkg": render_pkg, "metrics": {"app_missing_prior": 1.0}}

            mask = None
            if prior_mask_bank is not None:
                mask = prior_mask_bank.get(
                    viewpoint_cam.image_name,
                    device=image.device,
                    width=int(viewpoint_cam.image_width),
                    height=int(viewpoint_cam.image_height),
                )

            app_recon, app_l1, app_dssim = _weighted_recon_loss(
                image,
                prior,
                lambda_dssim=sp_args.sp_app_lambda_dssim,
                mask=mask,
            )
            app_loss = float(sp_args.sp_app_l1_weight) * app_recon
            app_lr = image.new_tensor(0.0)
            if float(sp_args.sp_app_lr_weight) > 0.0:
                lr_anchor = lr_anchor_bank.get(viewpoint_cam.image_name, device=image.device)
                if lr_anchor is not None:
                    app_lr = l1_loss(_resize_to_match(image, lr_anchor, mode="area"), lr_anchor)
                    app_loss = app_loss + float(sp_args.sp_app_lr_weight) * app_lr
            return {
                "loss": app_loss,
                "render_pkg": render_pkg,
                "metrics": {
                    "app_l1": float(app_l1.detach().item()),
                    "app_dssim": float(app_dssim.detach().item()),
                    "app_lr": float(app_lr.detach().item()),
                    "app_mask_mean": float(mask.mean().detach().item()) if mask is not None else 1.0,
                },
            }

        groups = build_routed_param_groups(gaussians)
        routed_result = routed_one_optimizer_step(
            gaussians=gaussians,
            geometry_closure=geometry_closure,
            appearance_closure=appearance_closure,
            groups=groups,
            collect_densification=not bool(sp_args.sp_disable_densification_route),
            fail_on_app_geometry_grad=bool(sp_args.sp_fail_on_app_geometry_grad),
        )
        iter_end.record()

        with torch.no_grad():
            total_loss_for_log = routed_result.geo_loss + routed_result.app_loss
            ema_loss_for_log = 0.4 * total_loss_for_log + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Geo": f"{routed_result.geo_loss:.5f}",
                        "App": f"{routed_result.app_loss:.5f}",
                        "Pts": f"{gaussians.get_xyz.shape[0]}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer,
                iteration,
                routed_result,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, dataset.kernel_size),
                test_cameras,
            )

            if iteration % max(1, int(sp_args.sp_audit_interval)) == 0:
                _write_jsonl(audit_path, _audit_payload(iteration, routed_result, gaussians.get_xyz.shape[0]))

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            current_points = int(gaussians.get_xyz.shape[0])
            max_points = int(sp_args.sp_max_points)
            densification_enabled = iteration < opt.densify_until_iter and not bool(
                sp_args.sp_disable_densification_route
            )
            if densification_enabled and max_points > 0 and current_points >= max_points:
                densification_enabled = False
                if not max_points_notice_shown:
                    print(
                        f"\n[SP-IE-SRGS] point cap reached: {current_points} >= {max_points}; "
                        "stopping densification."
                    )
                    max_points_notice_shown = True

            if densification_enabled:
                render_pkg = geo_holder.get("render_pkg", {})
                visibility_filter = render_pkg.get("visibility_filter") if isinstance(render_pkg, dict) else None
                radii = render_pkg.get("radii") if isinstance(render_pkg, dict) else None
                if visibility_filter is not None and radii is not None and torch.any(visibility_filter):
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    gaussians.compute_3D_filter(cameras=train_cameras)

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    gaussians.compute_3D_filter(cameras=train_cameras)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"),
                )


def training_report(
    tb_writer,
    iteration,
    routed_result,
    elapsed,
    testing_iterations,
    scene: Scene,
    render_func,
    render_args,
    test_cameras,
):
    if tb_writer:
        tb_writer.add_scalar("sp_route/loss_geo", routed_result.geo_loss, iteration)
        tb_writer.add_scalar("sp_route/loss_app", routed_result.app_loss, iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        for key, value in routed_result.metrics.items():
            if isinstance(value, (int, float)):
                tb_writer.add_scalar(f"sp_route/{key}", float(value), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": test_cameras},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(render_func(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 5:
                        tb_writer.add_images(
                            config["name"] + f"_view_{viewpoint.image_name}/render",
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def build_parser():
    parser = ArgumentParser(description="SP-IE-SRGS v0 routed training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--prepared_sr_prior_root", type=str, required=True)
    parser.add_argument("--sr_prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--sp_lr_anchor_dir", type=str, required=True)
    parser.add_argument("--sp_require_prior_mask", action="store_true")
    parser.add_argument("--sp_allow_missing_prior", action="store_true")
    parser.add_argument("--sp_allow_missing_lr_anchor", action="store_true")
    parser.add_argument("--sp_lr_fallback_downscale", type=float, default=4.0)
    parser.add_argument(
        "--sp_init_ply",
        type=str,
        default="",
        help="Optional 3DGS-format PLY used to initialize Gaussians before optimizer setup.",
    )

    parser.add_argument("--sp_geo_lambda_dssim", type=float, default=0.2)
    parser.add_argument("--sp_app_l1_weight", type=float, default=0.05)
    parser.add_argument("--sp_app_lambda_dssim", type=float, default=0.0)
    parser.add_argument("--sp_app_lr_weight", type=float, default=0.01)
    parser.add_argument("--sp_fail_on_app_geometry_grad", action="store_true", default=True)
    parser.add_argument("--sp_audit_interval", type=int, default=100)
    parser.add_argument("--sp_disable_densification_route", action="store_true")
    parser.add_argument("--sp_max_points", type=int, default=0)
    parser.add_argument("--sp_surface_enable", action="store_true")
    parser.add_argument("--sp_lambda_surface", type=float, default=0.0)
    parser.add_argument("--sp_lambda_distortion", type=float, default=1000.0)
    parser.add_argument("--sp_lambda_depth_normal", type=float, default=0.05)
    parser.add_argument("--sp_lambda_normal_smooth", type=float, default=0.01)
    parser.add_argument("--sp_surface_ramp_start_iter", type=int, default=1000)
    parser.add_argument("--sp_surface_ramp_end_iter", type=int, default=5000)
    parser.add_argument("--sp_scaffold_enable", action="store_true")
    parser.add_argument("--sp_scaffold_path", type=str, default="")
    parser.add_argument("--sp_scaffold_sample_size", type=int, default=2048)
    parser.add_argument("--sp_scaffold_interval", type=int, default=1)
    parser.add_argument(
        "--sp_scaffold_axis",
        type=str,
        default="min_scale",
        choices=["min_scale", "max_scale", "x", "y", "z"],
    )
    parser.add_argument("--sp_scaffold_chamfer_weight", type=float, default=0.0)
    parser.add_argument("--sp_scaffold_normal_weight", type=float, default=0.0)
    return parser, lp, op, pp


def main() -> None:
    parser, lp, op, pp = build_parser()
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
    print("\nSP-IE-SRGS training complete.")


if __name__ == "__main__":
    main()

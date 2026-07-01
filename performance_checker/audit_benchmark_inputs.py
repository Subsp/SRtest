#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one benchmark run's image inputs, render outputs, and geometry "
            "references. The camera JSON maps numbered render outputs to COLMAP "
            "camera-name assets."
        )
    )
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True, help="Numbered render GT dir, e.g. gt_2.")
    parser.add_argument("--render-dir", type=Path, default=None, help="Numbered final render dir.")
    parser.add_argument("--mip-render-dir", type=Path, default=None, help="Numbered mip render anchor dir.")
    parser.add_argument("--lr-anchor-dir", type=Path, default=None, help="Camera-name LR anchor dir.")
    parser.add_argument("--prior-dir", type=Path, default=None, help="Camera-name prepared prior dir.")
    parser.add_argument("--raw-prior-dir", type=Path, default=None, help="Camera-name raw prior dir.")
    parser.add_argument("--ie-run-dir", type=Path, default=None, help="IE-SRGS model output dir with cfg_args.")
    parser.add_argument("--pred-geometry", type=Path, default=None)
    parser.add_argument("--gt-geometry", type=Path, default=None)
    parser.add_argument(
        "--pred-transform",
        choices=("none", "dtu-colmap-to-world", "dtu-normalized-to-world"),
        default="dtu-colmap-to-world",
    )
    parser.add_argument("--dtu-cameras", type=Path, default=None)
    parser.add_argument("--geometry-sample-points", type=int, default=200000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def load_camera_mapping(cameras_json: Path) -> list[dict[str, Any]]:
    data = json.loads(cameras_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected cameras JSON list: {cameras_json}")
    rows = []
    for row in sorted(data, key=lambda item: int(item["id"])):
        cam_stem = Path(str(row.get("img_name") or row.get("image_name"))).stem
        idx = int(row["id"])
        rows.append({"id": idx, "numbered": f"{idx:05d}", "camera": cam_stem})
    if not rows:
        raise ValueError(f"no camera rows in: {cameras_json}")
    return rows


def index_images(folder: Path | None) -> dict[str, Path]:
    if folder is None:
        return {}
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def coverage(name: str, folder: Path | None, expected: list[str]) -> dict[str, Any]:
    indexed = index_images(folder)
    expected_set = set(expected)
    available_set = set(indexed)
    missing = sorted(expected_set - available_set)
    extra = sorted(available_set - expected_set)
    return {
        "name": name,
        "folder": None if folder is None else str(folder.expanduser().resolve()),
        "exists": bool(folder is not None and folder.expanduser().resolve().is_dir()),
        "available": len(indexed),
        "expected": len(expected),
        "matched": len(expected_set & available_set),
        "missing_total": len(missing),
        "missing_examples": missing[:10],
        "extra_total": len(extra),
        "extra_examples": extra[:10],
    }


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, resample=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(3):
        x = a[..., channel]
        y = b[..., channel]
        ux = float(x.mean())
        uy = float(y.mean())
        vx = float(((x - ux) ** 2).mean())
        vy = float(((y - uy) ** 2).mean())
        cxy = float(((x - ux) * (y - uy)).mean())
        num = (2.0 * ux * uy + c1) * (2.0 * cxy + c2)
        den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        values.append(num / den if den else 1.0)
    return float(mean(values))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        if min(a.shape[:2]) >= 7:
            return float(structural_similarity(b, a, data_range=1.0, channel_axis=2, win_size=7))
    except Exception:
        pass
    return global_ssim(a, b)


def image_size(path: Path | None) -> list[int] | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as handle:
        width, height = handle.size
    return [int(width), int(height)]


def pair_metrics(
    name: str,
    left: dict[str, Path],
    right: dict[str, Path],
    pairs: list[tuple[str, str]],
    *,
    examples: int,
) -> dict[str, Any]:
    rows = []
    missing_left = []
    missing_right = []
    resized = 0
    for left_key, right_key in pairs:
        left_path = left.get(left_key)
        right_path = right.get(right_key)
        if left_path is None:
            missing_left.append(left_key)
            continue
        if right_path is None:
            missing_right.append(right_key)
            continue
        right_rgb = read_rgb(right_path)
        target_size = (right_rgb.shape[1], right_rgb.shape[0])
        left_size = image_size(left_path)
        if left_size != [target_size[0], target_size[1]]:
            resized += 1
        left_rgb = read_rgb(left_path, size=target_size)
        rows.append(
            {
                "left": left_path.name,
                "right": right_path.name,
                "psnr": psnr(left_rgb, right_rgb),
                "ssim": ssim(left_rgb, right_rgb),
                "l1": float(np.abs(left_rgb - right_rgb).mean()),
            }
        )
    finite_psnr = [row["psnr"] for row in rows if math.isfinite(row["psnr"])]
    return {
        "name": name,
        "count": len(rows),
        "psnr": None if not finite_psnr else float(mean(finite_psnr)),
        "ssim": None if not rows else float(mean(row["ssim"] for row in rows)),
        "l1": None if not rows else float(mean(row["l1"] for row in rows)),
        "resized_left_to_right_count": resized,
        "missing_left_total": len(missing_left),
        "missing_left_examples": missing_left[:examples],
        "missing_right_total": len(missing_right),
        "missing_right_examples": missing_right[:examples],
        "worst_psnr_examples": sorted(rows, key=lambda row: row["psnr"])[:examples],
    }


def parse_cfg_args(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    cfg_path = run_dir.expanduser().resolve() / "cfg_args"
    if not cfg_path.is_file():
        return {"cfg_args_path": str(cfg_path), "exists": False}
    text = cfg_path.read_text(encoding="utf-8")
    out: dict[str, Any] = {"cfg_args_path": str(cfg_path), "exists": True}
    keys = [
        "source_path",
        "model_path",
        "images",
        "resolution",
        "prepared_sr_prior_root",
        "sr_prior_subdir",
        "sr_prior_mask_subdir",
        "sp_lr_anchor_dir",
        "sp_geo_lambda_dssim",
        "sp_app_l1_weight",
        "sp_app_lambda_dssim",
        "sp_app_lr_weight",
        "sp_max_points",
        "sp_surface_enable",
    ]
    try:
        from argparse import Namespace

        namespace = eval(text, {"Namespace": Namespace})  # noqa: S307
        for key in keys:
            if hasattr(namespace, key):
                value = getattr(namespace, key)
                out[key] = value
    except Exception as exc:  # noqa: BLE001
        out["parse_error"] = repr(exc)
        out["raw_prefix"] = text[:500]
    return out


def geometry_summary(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.pred_geometry is None and args.gt_geometry is None and args.dtu_cameras is None:
        return None
    payload: dict[str, Any] = {
        "pred_geometry": None if args.pred_geometry is None else str(args.pred_geometry.expanduser().resolve()),
        "gt_geometry": None if args.gt_geometry is None else str(args.gt_geometry.expanduser().resolve()),
        "dtu_cameras": None if args.dtu_cameras is None else str(args.dtu_cameras.expanduser().resolve()),
        "pred_transform": args.pred_transform,
    }
    try:
        from performance_checker.geometry_metrics import bbox_stats, load_geometry, transform_pred_points

        if args.pred_geometry is not None and args.pred_geometry.expanduser().resolve().is_file():
            pred = load_geometry(args.pred_geometry, args.geometry_sample_points)
            payload["pred_points_raw"] = int(pred.shape[0])
            payload["pred_bbox_raw"] = bbox_stats(pred)
            pred_t, transform_info = transform_pred_points(pred, args.pred_transform, args.dtu_cameras)
            payload["pred_transform_info"] = transform_info
            payload["pred_bbox_after_transform"] = bbox_stats(pred_t)
        if args.gt_geometry is not None and args.gt_geometry.expanduser().resolve().is_file():
            gt = load_geometry(args.gt_geometry, args.geometry_sample_points)
            payload["gt_points"] = int(gt.shape[0])
            payload["gt_bbox"] = bbox_stats(gt)
        if args.dtu_cameras is not None and args.dtu_cameras.expanduser().resolve().is_file():
            data = np.load(args.dtu_cameras.expanduser().resolve())
            payload["dtu_cameras_keys"] = sorted(data.files)[:20]
            if "scale_mat_0" in data:
                scale_mat = np.asarray(data["scale_mat_0"], dtype=np.float64)
                payload["scale_mat_0_diag"] = np.diag(scale_mat[:3, :3]).tolist()
                payload["scale_mat_0_translation"] = scale_mat[:3, 3].tolist()
    except Exception as exc:  # noqa: BLE001
        payload["error"] = repr(exc)
    return payload


def main() -> int:
    args = parse_args()
    mapping = load_camera_mapping(args.cameras_json.expanduser().resolve())
    numbered = [row["numbered"] for row in mapping]
    camera = [row["camera"] for row in mapping]
    number_to_camera = [(row["numbered"], row["camera"]) for row in mapping]
    camera_to_number = [(row["camera"], row["numbered"]) for row in mapping]

    dirs = {
        "gt": index_images(args.gt_dir),
        "render": index_images(args.render_dir),
        "mip_render": index_images(args.mip_render_dir),
        "lr_anchor": index_images(args.lr_anchor_dir),
        "prior": index_images(args.prior_dir),
        "raw_prior": index_images(args.raw_prior_dir),
    }

    payload: dict[str, Any] = {
        "mode": "benchmark_input_audit",
        "cameras_json": str(args.cameras_json.expanduser().resolve()),
        "num_cameras": len(mapping),
        "camera_examples": mapping[: args.examples],
        "cfg_args": parse_cfg_args(args.ie_run_dir),
        "coverage": {
            "gt_numbered": coverage("gt_numbered", args.gt_dir, numbered),
            "render_numbered": coverage("render_numbered", args.render_dir, numbered),
            "mip_render_numbered": coverage("mip_render_numbered", args.mip_render_dir, numbered),
            "lr_anchor_camera": coverage("lr_anchor_camera", args.lr_anchor_dir, camera),
            "prior_camera": coverage("prior_camera", args.prior_dir, camera),
            "raw_prior_camera": coverage("raw_prior_camera", args.raw_prior_dir, camera),
        },
        "image_sizes": {},
        "image_pairs": [],
        "geometry": geometry_summary(args),
    }

    size_sources = {
        "gt_first": (dirs["gt"], numbered[0]),
        "render_first": (dirs["render"], numbered[0]),
        "mip_render_first": (dirs["mip_render"], numbered[0]),
        "lr_anchor_first": (dirs["lr_anchor"], camera[0]),
        "prior_first": (dirs["prior"], camera[0]),
        "raw_prior_first": (dirs["raw_prior"], camera[0]),
    }
    for label, (indexed, key) in size_sources.items():
        payload["image_sizes"][label] = image_size(indexed.get(key))

    if dirs["mip_render"] and dirs["gt"]:
        payload["image_pairs"].append(
            pair_metrics("mip_render_vs_gt", dirs["mip_render"], dirs["gt"], [(n, n) for n in numbered], examples=args.examples)
        )
    if dirs["lr_anchor"] and dirs["gt"]:
        payload["image_pairs"].append(
            pair_metrics("lr_anchor_vs_gt", dirs["lr_anchor"], dirs["gt"], camera_to_number, examples=args.examples)
        )
    if dirs["prior"] and dirs["gt"]:
        payload["image_pairs"].append(
            pair_metrics("prior_vs_gt", dirs["prior"], dirs["gt"], camera_to_number, examples=args.examples)
        )
    if dirs["raw_prior"] and dirs["gt"]:
        payload["image_pairs"].append(
            pair_metrics("raw_prior_vs_gt", dirs["raw_prior"], dirs["gt"], camera_to_number, examples=args.examples)
        )
    if dirs["prior"] and dirs["lr_anchor"]:
        payload["image_pairs"].append(
            pair_metrics("prior_vs_lr_anchor", dirs["prior"], dirs["lr_anchor"], [(c, c) for c in camera], examples=args.examples)
        )
    if dirs["render"] and dirs["gt"]:
        payload["image_pairs"].append(
            pair_metrics("render_vs_gt", dirs["render"], dirs["gt"], [(n, n) for n in numbered], examples=args.examples)
        )
    if dirs["render"] and dirs["lr_anchor"]:
        payload["image_pairs"].append(
            pair_metrics("render_vs_lr_anchor", dirs["render"], dirs["lr_anchor"], number_to_camera, examples=args.examples)
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[audit-benchmark-inputs] {exc}", file=sys.stderr)
        raise SystemExit(1)

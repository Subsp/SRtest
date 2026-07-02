#!/usr/bin/env python3
"""Run DTU official-style point-cloud Chamfer for 3DGS-style PLY exports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from geometry_metrics import load_geometry, transform_pred_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Predicted mesh or point cloud.")
    parser.add_argument("--scan-id", type=int, required=True, help="DTU scan id, e.g. 24.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Official DTU root with Points/ and ObsMask/.")
    parser.add_argument("--output", type=Path, required=True, help="Normalized metric JSON path.")
    parser.add_argument(
        "--eval-code-root",
        type=Path,
        default=None,
        help="Directory containing DTU eval.py. Defaults to gs2mesh/evaluation/DTU/eval_code under the workspace.",
    )
    parser.add_argument(
        "--vis-out-dir",
        type=Path,
        default=None,
        help="Directory for official eval visualizations and intermediate pred_world.ply.",
    )
    parser.add_argument("--sample-points", type=int, default=200000, help="Mesh sample count before DTU eval.")
    parser.add_argument(
        "--pred-transform",
        choices=("none", "dtu-colmap-to-world", "dtu-normalized-to-world"),
        default="dtu-colmap-to-world",
        help="Transform applied before official DTU pcd evaluation.",
    )
    parser.add_argument("--dtu-cameras", type=Path, default=None, help="cameras.npz for DTU prediction transforms.")
    parser.add_argument("--downsample-density", type=float, default=0.2)
    parser.add_argument("--patch-size", type=float, default=60.0)
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument("--visualize-threshold", type=float, default=10.0)
    return parser.parse_args()


def default_eval_code_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    return workspace_root / "gs2mesh" / "evaluation" / "DTU" / "eval_code"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def require_dtu_assets(dataset_dir: Path, scan_id: int) -> None:
    require_file(dataset_dir / "Points" / "stl" / f"stl{scan_id:03d}_total.ply", "DTU STL")
    require_file(dataset_dir / "ObsMask" / f"ObsMask{scan_id}_10.mat", "DTU ObsMask")
    require_file(dataset_dir / "ObsMask" / f"Plane{scan_id}.mat", "DTU Plane")


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("open3d is required for DTU official pcd metrics") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"failed to write point cloud: {path}")


def run_eval(args: argparse.Namespace, eval_code_root: Path, pred_world: Path, vis_out_dir: Path) -> dict[str, Any]:
    raw_path = vis_out_dir / "results.json"
    cmd = [
        sys.executable,
        str(eval_code_root / "eval.py"),
        "--data",
        str(pred_world),
        "--scan",
        str(args.scan_id),
        "--mode",
        "pcd",
        "--dataset_dir",
        str(args.dataset_dir),
        "--vis_out_dir",
        str(vis_out_dir),
        "--downsample_density",
        str(args.downsample_density),
        "--patch_size",
        str(args.patch_size),
        "--max_dist",
        str(args.max_dist),
        "--visualize_threshold",
        str(args.visualize_threshold),
    ]
    print("[dtu-official-pcd] run:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(eval_code_root), check=True)
    with raw_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    eval_code_root = (args.eval_code_root or default_eval_code_root()).expanduser().resolve()
    vis_out_dir = (args.vis_out_dir or args.output.parent / "dtu_official_pcd_eval").expanduser().resolve()

    require_file(eval_code_root / "eval.py", "DTU eval.py")
    require_dtu_assets(args.dataset_dir, args.scan_id)

    pred_raw = load_geometry(args.pred, args.sample_points)
    pred_world, transform_meta = transform_pred_points(pred_raw, args.pred_transform, args.dtu_cameras)
    pred_world_path = vis_out_dir / "pred_world.ply"
    write_point_cloud(pred_world_path, pred_world)

    official = run_eval(args, eval_code_root, pred_world_path, vis_out_dir)
    accuracy = float(official["mean_d2s"])
    completion = float(official["mean_s2d"])
    chamfer_l1 = float(official["overall"])
    normalized = {
        "metric_version": "dtu-official-pcd-v1",
        "geometry_protocol": "dtu_official_pcd",
        "scan_id": int(args.scan_id),
        "pred": str(args.pred.expanduser().resolve()),
        "pred_world_ply": str(pred_world_path),
        "pred_points_raw": int(len(pred_raw)),
        "pred_transform": transform_meta,
        "dataset_dir": str(args.dataset_dir),
        "eval_code_root": str(eval_code_root),
        "downsample_density": float(args.downsample_density),
        "patch_size": float(args.patch_size),
        "max_dist": float(args.max_dist),
        "accuracy": accuracy,
        "completion": completion,
        "chamfer_l1": chamfer_l1,
        "mean_d2s": accuracy,
        "mean_s2d": completion,
        "overall": chamfer_l1,
        "official_results_path": str(vis_out_dir / "results.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: normalized[k] for k in ("accuracy", "completion", "chamfer_l1")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

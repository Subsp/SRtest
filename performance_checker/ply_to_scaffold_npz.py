#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a mesh or point cloud PLY into an IE-SRGS scaffold npz."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input mesh or point cloud PLY.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz with points and optional normals.")
    parser.add_argument("--sample-points", type=int, default=200000, help="Uniform samples for triangle meshes.")
    parser.add_argument("--max-points", type=int, default=200000, help="Random cap after loading/sampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-transform",
        choices=("none", "dtu-world-to-colmap", "dtu-colmap-to-world"),
        default="none",
        help="Coordinate transform applied before saving the scaffold.",
    )
    parser.add_argument("--dtu-cameras", type=Path, default=None, help="DTU cameras.npz for dtu-* transforms.")
    parser.add_argument("--estimate-normals", action="store_true", help="Estimate normals when the input has none.")
    parser.add_argument("--normal-radius", type=float, default=0.03)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    return parser.parse_args()


def load_dtu_scale_transform(cameras_path: Path | None) -> tuple[float, np.ndarray]:
    if cameras_path is None:
        raise ValueError("--dtu-cameras is required for DTU transforms")
    data = np.load(cameras_path.expanduser().resolve())
    if "scale_mat_0" not in data:
        raise ValueError(f"DTU cameras file has no scale_mat_0: {cameras_path}")
    scale_mat = np.asarray(data["scale_mat_0"], dtype=np.float64)
    diagonal = np.diag(scale_mat[:3, :3])
    if not np.allclose(diagonal, diagonal[0], rtol=1e-5, atol=1e-7):
        raise ValueError(f"DTU scale_mat_0 is not isotropic: {diagonal.tolist()}")
    return float(diagonal[0]), scale_mat[:3, 3].astype(np.float64)


def transform_points(points: np.ndarray, mode: str, dtu_cameras: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return points, {"type": "none"}
    scale, translation = load_dtu_scale_transform(dtu_cameras)
    if mode == "dtu-colmap-to-world":
        transformed = points * scale + translation[None, :]
    elif mode == "dtu-world-to-colmap":
        transformed = (points - translation[None, :]) / scale
    else:
        raise ValueError(f"unsupported transform: {mode}")
    return transformed, {
        "type": mode,
        "scale": scale,
        "translation": translation.tolist(),
        "dtu_cameras": str(dtu_cameras.expanduser().resolve()) if dtu_cameras is not None else None,
    }


def cap_points(
    points: np.ndarray,
    normals: np.ndarray | None,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, normals
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], None if normals is None else normals[idx]


def load_ply_points(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("open3d is required for ply_to_scaffold_npz.py") from exc

    path = args.input.expanduser().resolve()
    mesh = o3d.io.read_triangle_mesh(str(path))
    meta: dict[str, Any] = {
        "input": str(path),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
    }

    if len(mesh.vertices) and len(mesh.triangles):
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(args.sample_points))
        source_kind = "triangle_mesh_uniform_samples"
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        source_kind = "point_cloud"

    if not len(pcd.points):
        raise ValueError(f"input produced no points: {path}")
    if args.estimate_normals and not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(args.normal_radius),
                max_nn=int(args.normal_max_nn),
            )
        )
        pcd.normalize_normals()

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
    meta.update(
        {
            "source_kind": source_kind,
            "loaded_points": int(points.shape[0]),
            "has_normals": bool(normals is not None),
        }
    )
    return points, normals, meta


def bbox(points: np.ndarray) -> dict[str, Any]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    return {
        "min": minimum.astype(float).tolist(),
        "max": maximum.astype(float).tolist(),
        "center": ((minimum + maximum) * 0.5).astype(float).tolist(),
        "extent": extent.astype(float).tolist(),
    }


def main() -> int:
    args = parse_args()
    points, normals, meta = load_ply_points(args)
    points = points[np.isfinite(points).all(axis=1)]
    if normals is not None and normals.shape[0] != points.shape[0]:
        normals = None
    points, transform_info = transform_points(points.astype(np.float64), args.input_transform, args.dtu_cameras)
    points = points.astype(np.float32)
    points, normals = cap_points(points, normals, int(args.max_points), int(args.seed))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if normals is None:
        np.savez_compressed(output, points=points)
    else:
        np.savez_compressed(output, points=points, normals=normals.astype(np.float32))

    manifest = {
        **meta,
        "output": str(output),
        "saved_points": int(points.shape[0]),
        "saved_normals": bool(normals is not None),
        "sample_points": int(args.sample_points),
        "max_points": int(args.max_points),
        "transform": transform_info,
        "bbox": bbox(points),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

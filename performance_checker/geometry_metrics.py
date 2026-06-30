#!/usr/bin/env python3
"""Compute lightweight point-cloud geometry metrics for one reconstructed scene."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="Predicted mesh or point cloud.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth point cloud.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--sample-points", type=int, default=200000, help="Mesh sample count when Open3D is available.")
    parser.add_argument("--max-points", type=int, default=200000, help="Randomly cap each cloud to this many points.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional F-score distance threshold.")
    parser.add_argument(
        "--pred-transform",
        choices=("none", "dtu-colmap-to-world", "dtu-normalized-to-world"),
        default="none",
        help="Optional transform applied to predicted points before evaluation.",
    )
    parser.add_argument(
        "--dtu-cameras",
        type=Path,
        default=None,
        help="DTU cameras.npz used by dtu-* prediction transforms.",
    )
    parser.add_argument(
        "--crop-pred-to-gt-bbox-margin",
        type=float,
        default=None,
        help="Optional diagnostic crop: keep predicted points inside GT bbox plus this margin.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_with_open3d(path: Path, sample_points: int) -> np.ndarray | None:
    try:
        import open3d as o3d
    except ModuleNotFoundError:
        return None

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) and len(mesh.triangles):
        sampled = mesh.sample_points_uniformly(number_of_points=sample_points)
        return np.asarray(sampled.points, dtype=np.float64)

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points):
        return np.asarray(pcd.points, dtype=np.float64)
    return None


PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def parse_ply_header(handle: Any) -> tuple[str, int, list[tuple[str, str]]]:
    fmt = ""
    vertex_count = 0
    in_vertex = False
    vertex_properties: list[tuple[str, str]] = []
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        text = line.decode("ascii", errors="replace").strip()
        if text.startswith("format "):
            fmt = text.split()[1]
        elif text.startswith("element vertex "):
            vertex_count = int(text.split()[2])
            in_vertex = True
        elif text.startswith("element "):
            in_vertex = False
        elif in_vertex and text.startswith("property "):
            fields = text.split()
            if fields[1] == "list":
                raise ValueError("List property inside vertex block is unsupported")
            vertex_properties.append((fields[1], fields[2]))
        elif text == "end_header":
            break
    if not fmt or vertex_count <= 0:
        raise ValueError("PLY file has no supported vertex header")
    names = {name for _, name in vertex_properties}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex block must contain x, y, z properties")
    return fmt, vertex_count, vertex_properties


def load_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        first = handle.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"Unsupported geometry file, expected PLY: {path}")
        fmt, vertex_count, vertex_properties = parse_ply_header(handle)
        if fmt == "ascii":
            points = []
            names = [name for _, name in vertex_properties]
            xyz_idx = [names.index(axis) for axis in ("x", "y", "z")]
            for _ in range(vertex_count):
                fields = handle.readline().decode("ascii", errors="replace").split()
                points.append([float(fields[idx]) for idx in xyz_idx])
            return np.asarray(points, dtype=np.float64)
        if fmt == "binary_little_endian":
            offsets: dict[str, tuple[int, str]] = {}
            stride = 0
            for prop_type, name in vertex_properties:
                if prop_type not in PLY_SCALAR_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {prop_type}")
                fmt_code, size = PLY_SCALAR_TYPES[prop_type]
                offsets[name] = (stride, fmt_code)
                stride += size
            raw = handle.read(vertex_count * stride)
            if len(raw) < vertex_count * stride:
                raise ValueError("Binary PLY ended before vertex block was complete")
            points = np.empty((vertex_count, 3), dtype=np.float64)
            for row in range(vertex_count):
                base = row * stride
                for col, axis in enumerate(("x", "y", "z")):
                    offset, fmt_code = offsets[axis]
                    points[row, col] = struct.unpack_from("<" + fmt_code, raw, base + offset)[0]
            return points
    raise ValueError(f"Unsupported PLY format: {fmt}")


def load_geometry(path: Path, sample_points: int) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    points = load_with_open3d(path, sample_points)
    if points is None:
        points = load_ply_vertices(path)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Geometry did not produce Nx3 points: {path}")
    return points[np.isfinite(points).all(axis=1)]


def cap_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def bbox_stats(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "min": None,
            "max": None,
            "center": None,
            "extent": None,
            "diag": None,
        }
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "center": ((minimum + maximum) / 2.0).tolist(),
        "extent": extent.tolist(),
        "diag": float(np.linalg.norm(extent)),
    }


def load_dtu_scale_transform(cameras_path: Path | None) -> tuple[float, np.ndarray]:
    if cameras_path is None:
        raise ValueError("--dtu-cameras is required for DTU prediction transforms")
    data = np.load(cameras_path.expanduser().resolve())
    if "scale_mat_0" not in data:
        raise ValueError(f"DTU cameras file has no scale_mat_0: {cameras_path}")
    scale_mat = np.asarray(data["scale_mat_0"], dtype=np.float64)
    diagonal = np.diag(scale_mat[:3, :3])
    if not np.allclose(diagonal, diagonal[0], rtol=1e-5, atol=1e-7):
        raise ValueError(f"DTU scale_mat_0 is not isotropic: {diagonal.tolist()}")
    return float(diagonal[0]), scale_mat[:3, 3].astype(np.float64)


def transform_pred_points(
    points: np.ndarray,
    transform: str,
    dtu_cameras: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if transform == "none":
        return points, {"type": "none"}
    if transform in {"dtu-colmap-to-world", "dtu-normalized-to-world"}:
        scale, translation = load_dtu_scale_transform(dtu_cameras)
        return points * scale + translation[None, :], {
            "type": transform,
            "scale": scale,
            "translation": translation.tolist(),
            "dtu_cameras": str(dtu_cameras.expanduser().resolve()) if dtu_cameras is not None else None,
        }
    raise ValueError(f"Unsupported prediction transform: {transform}")


def crop_pred_to_gt_bbox(
    pred: np.ndarray,
    gt: np.ndarray,
    margin: float | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if margin is None:
        return pred, None
    minimum = gt.min(axis=0) - float(margin)
    maximum = gt.max(axis=0) + float(margin)
    keep = np.logical_and(pred >= minimum[None, :], pred <= maximum[None, :]).all(axis=1)
    cropped = pred[keep]
    return cropped, {
        "type": "gt_bbox",
        "margin": float(margin),
        "kept_points": int(len(cropped)),
        "dropped_points": int(len(pred) - len(cropped)),
    }


def nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(dst)
        distances, _ = tree.query(src, k=1, workers=-1)
        return np.asarray(distances, dtype=np.float64)
    except Exception:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
        nn.fit(dst)
        distances, _ = nn.kneighbors(src, return_distance=True)
        return distances[:, 0].astype(np.float64)
    except Exception:
        pass

    chunk = 4096
    out = np.empty(len(src), dtype=np.float64)
    for start in range(0, len(src), chunk):
        block = src[start : start + chunk]
        squared = ((block[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = np.sqrt(squared.min(axis=1))
    return out


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pred_raw = load_geometry(args.pred, args.sample_points)
    gt_raw = load_geometry(args.gt, args.sample_points)
    pred_transformed, pred_transform = transform_pred_points(pred_raw, args.pred_transform, args.dtu_cameras)
    pred_cropped, crop = crop_pred_to_gt_bbox(pred_transformed, gt_raw, args.crop_pred_to_gt_bbox_margin)
    pred = cap_points(pred_cropped, args.max_points, rng)
    gt = cap_points(gt_raw, args.max_points, rng)

    if len(pred) == 0:
        raise ValueError("Prediction has zero points after transform/crop")

    pred_to_gt = nearest_distances(pred, gt)
    gt_to_pred = nearest_distances(gt, pred)
    accuracy = safe_mean(pred_to_gt)
    completion = safe_mean(gt_to_pred)
    chamfer_l1 = (accuracy + completion) / 2.0

    result: dict[str, Any] = {
        "metric_version": "geometry-lightweight-v1",
        "pred": str(args.pred.expanduser().resolve()),
        "gt": str(args.gt.expanduser().resolve()),
        "pred_points": int(len(pred)),
        "gt_points": int(len(gt)),
        "pred_points_raw": int(len(pred_raw)),
        "gt_points_raw": int(len(gt_raw)),
        "pred_transform": pred_transform,
        "pred_crop": crop,
        "bbox": {
            "pred_raw": bbox_stats(pred_raw),
            "pred_transformed": bbox_stats(pred_transformed),
            "pred_evaluated": bbox_stats(pred),
            "gt_raw": bbox_stats(gt_raw),
            "gt_evaluated": bbox_stats(gt),
        },
        "accuracy": accuracy,
        "completion": completion,
        "chamfer_l1": chamfer_l1,
    }
    if args.threshold is not None:
        precision = float((pred_to_gt < args.threshold).mean())
        recall = float((gt_to_pred < args.threshold).mean())
        denom = precision + recall
        result.update(
            {
                "threshold": float(args.threshold),
                "precision": precision,
                "recall": recall,
                "fscore": 2.0 * precision * recall / denom if denom > 0.0 else 0.0,
            }
        )

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("accuracy", "completion", "chamfer_l1")}, indent=2))
    if math.isnan(chamfer_l1):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

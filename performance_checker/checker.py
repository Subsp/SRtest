#!/usr/bin/env python3
"""Benchmark layout checker, render metric calculator, and result collector."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("performance_checker/benchmark_config.example.json"),
        help="Path to benchmark config JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_subcommand_config(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            type=Path,
            default=argparse.SUPPRESS,
            help="Path to benchmark config JSON.",
        )

    def add_filters(p: argparse.ArgumentParser) -> None:
        add_subcommand_config(p)
        p.add_argument("--scene-set", default="single_scene", help="Scene set from config.")
        p.add_argument("--method", action="append", help="Method id. Can be repeated.")
        p.add_argument("--dataset", action="append", help="Dataset id. Can be repeated.")
        p.add_argument("--scene", action="append", help="Scene id. Can be repeated.")
        p.add_argument("--include-disabled", action="store_true", help="Include disabled methods.")

    plan = subparsers.add_parser("plan", help="Print the benchmark matrix and commands.")
    add_filters(plan)
    plan.add_argument("--commands", action="store_true", help="Also print filled command templates.")

    check = subparsers.add_parser("check-layout", help="Check repositories, data roots, and artifacts.")
    add_filters(check)
    check.add_argument("--strict", action="store_true", help="Exit non-zero if required paths are missing.")
    check.add_argument("--require-data", action="store_true", help="Require dataset roots and scene roots to exist.")

    render = subparsers.add_parser("render-metrics", help="Compute PSNR/SSIM for one method/dataset/scene.")
    add_subcommand_config(render)
    render.add_argument("--method", required=True)
    render.add_argument("--dataset", required=True)
    render.add_argument("--scene", required=True)
    render.add_argument("--renders-dir", type=Path, help="Override renders directory.")
    render.add_argument("--gt-dir", type=Path, help="Override ground-truth directory.")
    render.add_argument("--output", type=Path, help="Override output render_metrics.json path.")

    collect = subparsers.add_parser("collect", help="Collect render and geometry metric JSON files.")
    add_filters(collect)
    collect.add_argument("--output-dir", type=Path, help="Override output directory.")

    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config, config_path


def format_template(value: Any, context: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    return expanded.format_map(SafeDict({k: str(v) for k, v in context.items()}))


def resolve_path(value: str | Path, base: Path, context: dict[str, Any] | None = None) -> Path:
    raw = str(value)
    if context:
        raw = format_template(raw, context)
    raw = os.path.expandvars(raw)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def base_context(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    config_dir = config_path.parent
    context: dict[str, Any] = {"config_dir": config_dir}
    workspace_root = resolve_path(config.get("workspace_root", ".."), config_dir, context)
    context["workspace_root"] = workspace_root
    run_root = resolve_path(config.get("run_root", "../benchmark_runs"), config_dir, context)
    output_root = resolve_path(config.get("output_root", "results"), config_dir, context)
    context.update({"run_root": run_root, "output_root": output_root})

    source_roots = {}
    for key, value in config.get("source_roots", {}).items():
        source_roots[key] = resolve_path(value, config_dir, context)
        context[f"repo_{key}"] = source_roots[key]
    context["source_roots"] = source_roots
    return context


@dataclass(frozen=True)
class SceneRef:
    dataset: dict[str, Any]
    scene: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        return str(self.dataset["id"])

    @property
    def scene_id(self) -> str:
        return str(self.scene["id"])


def enabled_methods(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = set(args.method or [])
    methods = []
    for method in config.get("methods", []):
        if selected and method.get("id") not in selected:
            continue
        if not args.include_disabled and not method.get("enabled", False):
            continue
        methods.append(method)
    return methods


def selected_scenes(config: dict[str, Any], args: argparse.Namespace) -> list[SceneRef]:
    dataset_filter = set(args.dataset or [])
    scene_filter = set(args.scene or [])
    scene_set = config.get("scene_sets", {}).get(args.scene_set)
    if scene_set is None:
        raise SystemExit(f"Unknown scene set: {args.scene_set}")

    refs: list[SceneRef] = []
    for dataset in config.get("datasets", []):
        dataset_id = dataset["id"]
        if dataset_filter and dataset_id not in dataset_filter:
            continue
        allowed_scenes = set(scene_set.get(dataset_id, []))
        if not allowed_scenes:
            continue
        for scene in dataset.get("scenes", []):
            scene_id = scene["id"]
            if scene_id not in allowed_scenes:
                continue
            if scene_filter and scene_id not in scene_filter:
                continue
            refs.append(SceneRef(dataset=dataset, scene=scene))
    return refs


def dataset_scene_context(base: dict[str, Any], ref: SceneRef) -> dict[str, Any]:
    ctx = dict(base)
    dataset_root = format_template(ref.dataset.get("root", ""), ctx)
    official_eval_root = format_template(ref.dataset.get("official_eval_root", ""), ctx)
    scene_relative = ref.scene.get("relative_path", ref.scene_id)
    scene_root = str(Path(dataset_root) / str(scene_relative)) if dataset_root else ""
    scan_id = ref.scene.get("scan_id", ref.scene_id)
    try:
        scan_id_padded = f"{int(scan_id):03d}"
    except (TypeError, ValueError):
        scan_id_padded = str(scan_id)
    ctx.update(
        {
            "dataset_id": ref.dataset_id,
            "dataset_label": ref.dataset.get("label", ref.dataset_id),
            "dataset_role": ref.dataset.get("role", ""),
            "dataset_root": dataset_root,
            "official_eval_root": official_eval_root,
            "scene_id": ref.scene_id,
            "scene_root": scene_root,
            "scan_id": scan_id,
            "scan_id_padded": scan_id_padded,
            "geometry_protocol": ref.dataset.get("geometry_protocol", "none"),
            "render_protocol": ref.dataset.get("render_protocol", "none"),
        }
    )
    return ctx


def method_context(base: dict[str, Any], method: dict[str, Any]) -> dict[str, Any]:
    ctx = dict(base)
    repo_key = method.get("repo_key", method["id"])
    repo = base.get("source_roots", {}).get(repo_key, "")
    ctx.update(
        {
            "method_id": method["id"],
            "method_label": method.get("label", method["id"]),
            "method_family": method.get("family", ""),
            "repo_key": repo_key,
            "repo": repo,
        }
    )
    return ctx


def artifact_context(
    config: dict[str, Any],
    base: dict[str, Any],
    method: dict[str, Any],
    ref: SceneRef,
) -> dict[str, Any]:
    ctx = dataset_scene_context(method_context(base, method), ref)
    artifact_templates = dict(config.get("default_artifacts", {}))
    artifact_templates.update(method.get("artifacts", {}))

    resolved: dict[str, str] = {}
    for _ in range(4):
        progress = False
        for key, template in artifact_templates.items():
            value = format_template(template, {**ctx, **resolved})
            if resolved.get(key) != value:
                progress = True
            resolved[key] = value
        ctx.update(resolved)
        geometry_metrics_path = resolved.get("geometry_metrics_path", "")
        if geometry_metrics_path:
            ctx["geometry_metrics_dir"] = str(Path(geometry_metrics_path).parent)
        if not progress:
            break

    for key, value in list(resolved.items()):
        ctx[key] = value
    return ctx


def fill_command(template: str, context: dict[str, Any]) -> str:
    return format_template(template, context)


def wants_render(method: dict[str, Any], dataset: dict[str, Any]) -> bool:
    return bool(method.get("eval", {}).get("render")) and dataset.get("role") in {
        "geometry_render",
        "render_only",
    }


def wants_geometry(method: dict[str, Any], dataset: dict[str, Any]) -> bool:
    return bool(method.get("eval", {}).get("geometry")) and dataset.get("geometry_protocol") != "none"


def wants_any_metric(method: dict[str, Any], dataset: dict[str, Any]) -> bool:
    return wants_render(method, dataset) or wants_geometry(method, dataset)


def cmd_plan(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> int:
    base = base_context(config, config_path)
    methods = enabled_methods(config, args)
    scenes = selected_scenes(config, args)
    print(f"# Benchmark plan: scene_set={args.scene_set}")
    print(f"workspace_root: {base['workspace_root']}")
    print(f"run_root      : {base['run_root']}")
    print("")
    print("| dataset | scene | method | render | geometry | role |")
    print("| --- | --- | --- | --- | --- | --- |")
    for ref in scenes:
        for method in methods:
            if not wants_any_metric(method, ref.dataset):
                continue
            print(
                "| {dataset} | {scene} | {method} | {render} | {geometry} | {role} |".format(
                    dataset=ref.dataset_id,
                    scene=ref.scene_id,
                    method=method["id"],
                    render="yes" if wants_render(method, ref.dataset) else "no",
                    geometry="yes" if wants_geometry(method, ref.dataset) else "no",
                    role=ref.dataset.get("role", ""),
                )
            )
            if args.commands:
                ctx = artifact_context(config, base, method, ref)
                for name, template in method.get("commands", {}).items():
                    print(f"  - {method['id']}:{name}: {fill_command(template, ctx)}")
                geom_template = config.get("geometry_eval_commands", {}).get(ctx.get("geometry_protocol"))
                if geom_template and wants_geometry(method, ref.dataset):
                    print(f"  - {method['id']}:geometry_eval: {fill_command(geom_template, ctx)}")
    return 0


def path_state(path_text: str) -> tuple[str, bool]:
    if not path_text:
        return "empty", False
    if "${" in path_text or "{" in path_text:
        return "unresolved", False
    path = Path(path_text)
    return ("ok" if path.exists() else "missing"), path.exists()


def cmd_check_layout(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> int:
    base = base_context(config, config_path)
    methods = enabled_methods(config, args)
    scenes = selected_scenes(config, args)
    failures = 0

    print("# Repository roots")
    for method in methods:
        ctx = method_context(base, method)
        state, ok = path_state(str(ctx["repo"]))
        if not ok:
            failures += 1
        print(f"{method['id']:24s} {state:10s} {ctx['repo']}")

    if args.require_data:
        print("\n# Dataset scene roots")
        for ref in scenes:
            ctx = dataset_scene_context(base, ref)
            state, ok = path_state(ctx["scene_root"])
            if not ok:
                failures += 1
            print(f"{ref.dataset_id}/{ref.scene_id:18s} {state:10s} {ctx['scene_root']}")

    print("\n# Expected artifacts")
    for ref in scenes:
        for method in methods:
            if not wants_any_metric(method, ref.dataset):
                continue
            ctx = artifact_context(config, base, method, ref)
            checks = []
            if wants_render(method, ref.dataset):
                checks.extend(["renders_dir", "gt_dir", "render_metrics_path"])
            if wants_geometry(method, ref.dataset):
                checks.extend(["mesh_path", "geometry_metrics_path"])
            states = []
            for key in checks:
                state, ok = path_state(str(ctx.get(key, "")))
                states.append(f"{key}={state}")
            print(f"{method['id']}/{ref.dataset_id}/{ref.scene_id}: " + ", ".join(states))

    if args.strict and failures:
        return 2
    return 0


def iter_images(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {path}")
    images: dict[str, Path] = {}
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            images.setdefault(child.stem, child)
    return images


def require_image_deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "render-metrics requires numpy and Pillow. In this workspace, use the "
            "`python` shim/environment rather than the system `python3`."
        ) from exc
    return np, Image


def read_rgb(path: Path) -> Any:
    np, Image = require_image_deps()
    with Image.open(path) as handle:
        arr = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0
    return arr


def psnr(pred: Any, gt: Any) -> float:
    mse = float(((pred - gt) ** 2).mean())
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def global_ssim(pred: Any, gt: Any) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(3):
        x = pred[..., channel]
        y = gt[..., channel]
        ux = float(x.mean())
        uy = float(y.mean())
        vx = float(((x - ux) ** 2).mean())
        vy = float(((y - uy) ** 2).mean())
        cxy = float(((x - ux) * (y - uy)).mean())
        numerator = (2.0 * ux * uy + c1) * (2.0 * cxy + c2)
        denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        values.append(numerator / denominator if denominator else 1.0)
    return float(mean(values))


def ssim(pred: Any, gt: Any) -> float:
    try:
        from skimage.metrics import structural_similarity

        size = min(pred.shape[0], pred.shape[1])
        if size >= 7:
            win_size = 7 if size >= 7 else size | 1
            return float(
                structural_similarity(
                    gt,
                    pred,
                    data_range=1.0,
                    channel_axis=2,
                    win_size=win_size,
                )
            )
    except Exception:
        pass
    return global_ssim(pred, gt)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def compute_render_metrics(
    renders_dir: Path,
    gt_dir: Path,
    *,
    method_id: str,
    dataset_id: str,
    scene_id: str,
) -> dict[str, Any]:
    renders = iter_images(renders_dir)
    gts = iter_images(gt_dir)
    common = sorted(set(renders) & set(gts))
    if not common:
        raise RuntimeError(f"No paired images found: renders={renders_dir}, gt={gt_dir}")

    per_image = []
    for stem in common:
        pred = read_rgb(renders[stem])
        gt = read_rgb(gts[stem])
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {stem}: render={pred.shape}, gt={gt.shape}")
        per_image.append(
            {
                "name": renders[stem].name,
                "psnr": psnr(pred, gt),
                "ssim": ssim(pred, gt),
            }
        )

    psnr_values = [row["psnr"] for row in per_image if math.isfinite(row["psnr"])]
    payload = {
        "metric_version": "render-v0",
        "method": method_id,
        "dataset": dataset_id,
        "scene": scene_id,
        "num_images": len(per_image),
        "psnr": mean(psnr_values) if psnr_values else float("inf"),
        "ssim": mean(row["ssim"] for row in per_image),
        "per_image": per_image,
        "missing_gt": sorted(set(renders) - set(gts)),
        "missing_render": sorted(set(gts) - set(renders)),
    }
    return payload


def find_one(config: dict[str, Any], kind: str, value: str) -> dict[str, Any]:
    for item in config.get(kind, []):
        if item.get("id") == value:
            return item
    raise SystemExit(f"Unknown {kind[:-1]}: {value}")


def find_scene(dataset: dict[str, Any], scene_id: str) -> dict[str, Any]:
    for scene in dataset.get("scenes", []):
        if scene.get("id") == scene_id:
            return scene
    raise SystemExit(f"Unknown scene for dataset {dataset['id']}: {scene_id}")


def cmd_render_metrics(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> int:
    base = base_context(config, config_path)
    method = find_one(config, "methods", args.method)
    dataset = find_one(config, "datasets", args.dataset)
    scene = find_scene(dataset, args.scene)
    ref = SceneRef(dataset=dataset, scene=scene)
    ctx = artifact_context(config, base, method, ref)

    renders_dir = args.renders_dir or Path(str(ctx["renders_dir"]))
    gt_dir = args.gt_dir or Path(str(ctx["gt_dir"]))
    output = args.output or Path(str(ctx["render_metrics_path"]))

    metrics = compute_render_metrics(
        renders_dir.expanduser().resolve(),
        gt_dir.expanduser().resolve(),
        method_id=args.method,
        dataset_id=args.dataset,
        scene_id=args.scene,
    )
    write_json(output.expanduser().resolve(), metrics)
    print(f"[render-metrics] wrote {output}")
    print(f"  images: {metrics['num_images']}")
    print(f"  PSNR  : {metrics['psnr']:.4f}")
    print(f"  SSIM  : {metrics['ssim']:.4f}")
    return 0


def read_metric_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_number(data: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])
        upper = key.upper()
        if upper in data and isinstance(data[upper], (int, float)):
            return float(data[upper])
    return None


def normalize_render(data: dict[str, Any] | None) -> dict[str, float | None]:
    if not data:
        return {"psnr": None, "ssim": None, "lpips": None}
    if "PSNR" in data or "SSIM" in data:
        return {
            "psnr": first_number(data, ["PSNR", "psnr"]),
            "ssim": first_number(data, ["SSIM", "ssim"]),
            "lpips": first_number(data, ["LPIPS", "lpips"]),
        }
    if "overall" in data and isinstance(data["overall"], dict):
        return normalize_render(data["overall"])
    return {
        "psnr": first_number(data, ["psnr", "mean_psnr"]),
        "ssim": first_number(data, ["ssim", "mean_ssim"]),
        "lpips": first_number(data, ["lpips", "mean_lpips"]),
    }


def normalize_geometry(data: dict[str, Any] | None) -> dict[str, float | None]:
    if not data:
        return {
            "chamfer_l1": None,
            "accuracy": None,
            "completion": None,
            "precision": None,
            "recall": None,
            "fscore": None,
        }
    chamfer = first_number(data, ["chamfer_l1", "chamfer", "overall"])
    accuracy = first_number(data, ["accuracy", "mean_d2s"])
    completion = first_number(data, ["completion", "mean_s2d"])
    return {
        "chamfer_l1": chamfer,
        "accuracy": accuracy,
        "completion": completion,
        "precision": first_number(data, ["precision"]),
        "recall": first_number(data, ["recall"]),
        "fscore": first_number(data, ["fscore", "f_score", "F-score"]),
    }


def status_for(row: dict[str, Any], method: dict[str, Any], dataset: dict[str, Any]) -> str:
    missing = []
    if wants_render(method, dataset):
        if row.get("psnr") is None or row.get("ssim") is None:
            missing.append("render")
    if wants_geometry(method, dataset):
        has_geometry = any(row.get(key) is not None for key in ("chamfer_l1", "precision", "recall", "fscore"))
        if not has_geometry:
            missing.append("geometry")
    return "ok" if not missing else "missing_" + "_".join(missing)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "dataset",
        "scene",
        "method",
        "status",
        "psnr",
        "ssim",
        "lpips",
        "chamfer_l1",
        "accuracy",
        "completion",
        "precision",
        "recall",
        "fscore",
        "render_metrics_path",
        "geometry_metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{precision}f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "scene", "method", "status", "psnr", "ssim", "chamfer_l1", "fscore"]
    lines = ["# Benchmark Summary", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(key)) for key in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        group = grouped.setdefault(row["method"], {})
        for key in ("psnr", "ssim", "lpips", "chamfer_l1", "accuracy", "completion", "precision", "recall", "fscore"):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                group.setdefault(key, []).append(float(value))

    output: dict[str, Any] = {}
    for method, metrics in grouped.items():
        output[method] = {
            "scene_count": len([row for row in rows if row["method"] == method]),
        }
        for key, values in metrics.items():
            if values:
                output[method][f"mean_{key}"] = mean(values)
    return output


def cmd_collect(config: dict[str, Any], config_path: Path, args: argparse.Namespace) -> int:
    base = base_context(config, config_path)
    methods = enabled_methods(config, args)
    scenes = selected_scenes(config, args)
    rows: list[dict[str, Any]] = []

    for ref in scenes:
        for method in methods:
            if not wants_any_metric(method, ref.dataset):
                continue
            ctx = artifact_context(config, base, method, ref)
            render_path = Path(str(ctx["render_metrics_path"]))
            geom_path = Path(str(ctx["geometry_metrics_path"]))
            render_metrics = normalize_render(read_metric_json(render_path))
            geometry_metrics = normalize_geometry(read_metric_json(geom_path))
            row: dict[str, Any] = {
                "dataset": ref.dataset_id,
                "scene": ref.scene_id,
                "method": method["id"],
                "render_metrics_path": str(render_path),
                "geometry_metrics_path": str(geom_path),
            }
            row.update(render_metrics)
            row.update(geometry_metrics)
            row["status"] = status_for(row, method, ref.dataset)
            rows.append(row)

    output_dir = args.output_dir or Path(str(base["output_root"]))
    output_dir = output_dir.expanduser().resolve()
    summary = {
        "schema_version": config.get("schema_version"),
        "scene_set": args.scene_set,
        "rows": rows,
        "means_by_method": aggregate_means(rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", rows)
    print(f"[collect] wrote {output_dir}")
    print(json.dumps(summary["means_by_method"], indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    config, config_path = load_config(args.config)
    if args.command == "plan":
        return cmd_plan(config, config_path, args)
    if args.command == "check-layout":
        return cmd_check_layout(config, config_path, args)
    if args.command == "render-metrics":
        return cmd_render_metrics(config, config_path, args)
    if args.command == "collect":
        return cmd_collect(config, config_path, args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())

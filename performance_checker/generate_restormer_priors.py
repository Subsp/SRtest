#!/usr/bin/env python3
"""Generate flat Restormer prior images for a benchmark scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Flat source image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Flat output prior directory.")
    parser.add_argument(
        "--restormer-root",
        type=Path,
        default=Path(os.environ.get("RESTORMER_ROOT", "")) if os.environ.get("RESTORMER_ROOT") else None,
        help="External Restormer repo root containing demo.py. Defaults to RESTORMER_ROOT.",
    )
    parser.add_argument("--external-python", default=sys.executable, help="Python used to run Restormer demo.py.")
    parser.add_argument("--task", default=os.environ.get("RESTORMER_TASK", "Single_Image_Defocus_Deblurring"))
    parser.add_argument("--tile", type=int, default=int(os.environ.get("RESTORMER_TILE", "0")))
    parser.add_argument("--tile-overlap", type=int, default=int(os.environ.get("RESTORMER_TILE_OVERLAP", "32")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_images(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")
    suffixes = {ext.lower() for ext in exts}
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no source images found under: {root}")
    seen: dict[str, Path] = {}
    for path in paths:
        if path.stem in seen:
            raise ValueError(f"duplicate image stem: {seen[path.stem]} and {path}")
        seen[path.stem] = path
    return paths


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_restormer_output(result_dir: Path, task: str, stem: str) -> Path:
    task_dir = result_dir / task
    candidates = [
        task_dir / f"{stem}.png",
        task_dir / f"{stem}.jpg",
        task_dir / f"{stem}.jpeg",
        result_dir / f"{stem}.png",
        result_dir / f"{stem}.jpg",
        result_dir / f"{stem}.jpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches: list[Path] = []
    if task_dir.is_dir():
        matches.extend(sorted(task_dir.glob(f"{stem}.*")))
    matches.extend(sorted(result_dir.rglob(f"{stem}.png")))
    matches.extend(sorted(result_dir.rglob(f"{stem}.jpg")))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Restormer produced no output for stem '{stem}' under {result_dir}")


def run_restormer(
    *,
    demo_py: Path,
    external_python: str,
    task: str,
    input_dir: Path,
    result_dir: Path,
    tile: int,
    tile_overlap: int,
    cwd: Path,
) -> None:
    cmd = [
        external_python,
        str(demo_py),
        "--task",
        task,
        "--input_dir",
        str(input_dir),
        "--result_dir",
        str(result_dir),
    ]
    if tile > 0:
        cmd.extend(["--tile", str(tile), "--tile_overlap", str(tile_overlap)])
    print("[restormer-priors] run:", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.restormer_root is None:
        raise ValueError("--restormer-root or RESTORMER_ROOT is required")
    restormer_root = args.restormer_root.expanduser().resolve()
    demo_py = restormer_root / "demo.py"
    if not demo_py.is_file():
        raise FileNotFoundError(f"Restormer demo.py not found: {demo_py}")

    image_paths = collect_images(input_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    skipped = 0
    for path in image_paths:
        dst = output_dir / f"{path.stem}.png"
        if dst.is_file() and not args.overwrite:
            skipped += 1
        else:
            selected.append(path)

    if selected:
        with tempfile.TemporaryDirectory(prefix="restormer_priors_") as tmp:
            tmp_root = Path(tmp)
            batch_input = tmp_root / "input"
            result_dir = tmp_root / "restored"
            batch_input.mkdir(parents=True, exist_ok=True)
            for path in selected:
                link_or_copy(path, batch_input / path.name)
            run_restormer(
                demo_py=demo_py,
                external_python=args.external_python,
                task=args.task,
                input_dir=batch_input,
                result_dir=result_dir,
                tile=int(args.tile),
                tile_overlap=int(args.tile_overlap),
                cwd=restormer_root,
            )
            for index, path in enumerate(selected, start=1):
                produced = find_restormer_output(result_dir, args.task, path.stem)
                dst = output_dir / f"{path.stem}.png"
                shutil.copy2(produced, dst)
                print(f"[restormer-priors] {index}/{len(selected)} {path.name} -> {dst.name}")

    manifest = {
        "mode": "restormer_prior_generation",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "restormer_root": str(restormer_root),
        "task": args.task,
        "tile": int(args.tile),
        "tile_overlap": int(args.tile_overlap),
        "num_inputs": len(image_paths),
        "num_written": len(selected),
        "num_skipped_existing": skipped,
        "overwrite": bool(args.overwrite),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

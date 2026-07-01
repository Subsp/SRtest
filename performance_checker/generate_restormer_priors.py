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
import zipfile
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
RESTORMER_DEFAULT_CHECKPOINTS = {
    "Single_Image_Defocus_Deblurring": "Defocus_Deblurring/pretrained_models/single_image_defocus_deblurring.pth",
}


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
    parser.add_argument("--device", default=os.environ.get("RESTORMER_DEVICE", "cuda"))
    parser.add_argument(
        "--checkpoint-mode",
        choices=("auto", "demo", "torchscript"),
        default=os.environ.get("RESTORMER_CHECKPOINT_MODE", "auto"),
        help="auto uses torch.jit.load when the checkpoint is a TorchScript archive.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(os.environ.get("RESTORMER_CHECKPOINT_PATH", ""))
        if os.environ.get("RESTORMER_CHECKPOINT_PATH")
        else None,
        help="Optional explicit Restormer checkpoint path.",
    )
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


def resolve_checkpoint(root: Path, task: str, checkpoint_path: Path | None) -> Path | None:
    path = checkpoint_path
    if path is None:
        default_rel = RESTORMER_DEFAULT_CHECKPOINTS.get(task, "")
        if not default_rel:
            return None
        path = Path(default_rel)
    path = path.expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def is_torchscript_archive(path: Path | None) -> bool:
    if path is None or not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return False
    return any(name.startswith("code/") or "/code/" in name for name in names)


def tile_starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, max(size - tile, 0) + 1, stride))
    last = size - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def forward_torchscript_padded(model, tensor, torch_module):
    import torch.nn.functional as F

    _, _, height, width = tensor.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    output = model(tensor)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output[..., :height, :width].clamp(0.0, 1.0)


def forward_torchscript(model, tensor, torch_module, tile: int, tile_overlap: int):
    if tile <= 0:
        return forward_torchscript_padded(model, tensor, torch_module)
    _, channels, height, width = tensor.shape
    if height <= tile and width <= tile:
        return forward_torchscript_padded(model, tensor, torch_module)
    overlap = max(0, min(int(tile_overlap), int(tile) - 1))
    stride = max(1, int(tile) - overlap)
    y_starts = tile_starts(height, int(tile), stride)
    x_starts = tile_starts(width, int(tile), stride)
    output = torch_module.zeros((1, channels, height, width), device=tensor.device)
    weight = torch_module.zeros((1, 1, height, width), device=tensor.device)
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + int(tile), height)
            x1 = min(x0 + int(tile), width)
            restored = forward_torchscript_padded(model, tensor[..., y0:y1, x0:x1], torch_module)
            output[..., y0:y1, x0:x1] += restored[..., : y1 - y0, : x1 - x0]
            weight[..., y0:y1, x0:x1] += 1.0
    return (output / weight.clamp_min(1.0)).clamp(0.0, 1.0)


def load_image_tensor(path: Path, torch_module, device: str):
    import numpy as np
    from PIL import Image

    with Image.open(path) as handle:
        rgb = handle.convert("RGB")
        arr = np.asarray(rgb)
    tensor = (
        torch_module.from_numpy(arr)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    return tensor, tuple(int(v) for v in tensor.shape[-2:])


def save_tensor_image(path: Path, tensor, original_hw: tuple[int, int]) -> None:
    import numpy as np
    from PIL import Image

    height, width = original_hw
    tensor = tensor[..., :height, :width].squeeze(0).detach().cpu()
    arr = (
        tensor.permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .byte()
        .numpy()
    )
    Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path)


def run_torchscript_restormer(
    *,
    image_paths: list[Path],
    output_dir: Path,
    checkpoint: Path,
    device: str,
    tile: int,
    tile_overlap: int,
) -> None:
    import torch

    actual_device = device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    print(f"[restormer-priors] use TorchScript checkpoint: {checkpoint}", flush=True)
    print(f"[restormer-priors] device: {actual_device}", flush=True)
    model = torch.jit.load(str(checkpoint), map_location=actual_device).eval().to(actual_device)
    for index, path in enumerate(image_paths, start=1):
        dst = output_dir / f"{path.stem}.png"
        tensor, original_hw = load_image_tensor(path, torch, actual_device)
        with torch.no_grad():
            restored = forward_torchscript(model, tensor, torch, int(tile), int(tile_overlap))
        save_tensor_image(dst, restored, original_hw)
        print(f"[restormer-priors] {index}/{len(image_paths)} {path.name} -> {dst.name}")


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
    checkpoint = resolve_checkpoint(restormer_root, str(args.task), args.checkpoint_path)
    use_torchscript = bool(
        args.checkpoint_mode == "torchscript"
        or (args.checkpoint_mode == "auto" and is_torchscript_archive(checkpoint))
    )
    if args.checkpoint_mode == "torchscript" and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(f"Restormer TorchScript checkpoint not found: {checkpoint}")

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

    if selected and use_torchscript:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Restormer TorchScript checkpoint not found: {checkpoint}")
        run_torchscript_restormer(
            image_paths=selected,
            output_dir=output_dir,
            checkpoint=checkpoint,
            device=str(args.device),
            tile=int(args.tile),
            tile_overlap=int(args.tile_overlap),
        )
    elif selected:
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
        "device": str(args.device),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "checkpoint_mode": str(args.checkpoint_mode),
        "used_torchscript": bool(use_torchscript),
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

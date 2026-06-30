#!/usr/bin/env python3
"""Prepare proxy LR anchors and SR priors for an SP-IE-SRGS smoke benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--source-subdir", default="images")
    parser.add_argument("--output-image-subdir", default="images_r2")
    parser.add_argument("--prepared-sr-prior-root", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    return parser.parse_args()


def iter_images(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    suffixes = {ext.lower() for ext in exts}
    images = [path for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in suffixes]
    if not images:
        raise FileNotFoundError(f"No source images found: {root}")
    return images


def prepare_dir(path: Path, *, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not force and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory without --force: {path}")
    if force:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                raise IsADirectoryError(f"Refusing to remove nested directory: {child}")


def resized_size(width: int, height: int, scale: float) -> tuple[int, int]:
    if scale <= 0.0:
        raise ValueError("--scale must be positive")
    return max(1, round(width / scale)), max(1, round(height / scale))


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    source_dir = scene_root / args.source_subdir
    lr_anchor_dir = scene_root / args.output_image_subdir
    prior_root = (
        args.prepared_sr_prior_root.expanduser().resolve()
        if args.prepared_sr_prior_root is not None
        else scene_root / f"_ie_srgs_proxy_prior_{args.output_image_subdir}"
    )
    fused_dir = prior_root / "fused_priors"
    mask_dir = prior_root / "usable_masks"

    sources = iter_images(source_dir)
    prepare_dir(lr_anchor_dir, force=bool(args.force))
    prepare_dir(fused_dir, force=bool(args.force))
    if not args.no_mask:
        prepare_dir(mask_dir, force=bool(args.force))

    records = []
    for idx, src in enumerate(sources, start=1):
        with Image.open(src) as handle:
            rgb = handle.convert("RGB")
        size = resized_size(*rgb.size, float(args.scale))
        resized = rgb.resize(size, resample=Image.Resampling.BICUBIC)
        out_name = f"{src.stem}.png"
        resized.save(lr_anchor_dir / out_name)
        resized.save(fused_dir / out_name)
        if not args.no_mask:
            Image.new("L", size, color=255).save(mask_dir / out_name)
        records.append(
            {
                "source": src.name,
                "output": out_name,
                "source_size": list(rgb.size),
                "output_size": list(size),
            }
        )
        print(f"[ie-srgs-proxy] {idx}/{len(sources)} {src.name} -> {out_name} {size[0]}x{size[1]}")

    manifest = {
        "asset": "ie_srgs_proxy_prior",
        "scene_root": str(scene_root),
        "source_dir": str(source_dir),
        "lr_anchor_dir": str(lr_anchor_dir),
        "prepared_sr_prior_root": str(prior_root),
        "sr_prior_subdir": "fused_priors",
        "sr_prior_mask_subdir": "" if args.no_mask else "usable_masks",
        "scale": float(args.scale),
        "count": len(records),
        "records": records,
        "notes": "Proxy prior for fast SP-IE-SRGS smoke benchmarking; not an external SR prior.",
    }
    prior_root.mkdir(parents=True, exist_ok=True)
    (prior_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ie-srgs-proxy] lr_anchor_dir={lr_anchor_dir}")
    print(f"[ie-srgs-proxy] prepared_sr_prior_root={prior_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

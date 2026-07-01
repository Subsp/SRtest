#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy numbered render outputs such as 00000.png to COLMAP camera-image "
            "stems using a gaussian-splatting cameras.json file."
        )
    )
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_camera_names(cameras_json: Path) -> list[str]:
    data = json.loads(cameras_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a list in cameras json: {cameras_json}")

    def sort_key(item: dict) -> int:
        try:
            return int(item["id"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"camera entry missing integer id: {item}") from exc

    names: list[str] = []
    for item in sorted(data, key=sort_key):
        name = item.get("img_name") or item.get("image_name")
        if not name:
            raise ValueError(f"camera entry missing img_name/image_name: {item}")
        names.append(Path(str(name)).stem)
    if not names:
        raise ValueError(f"no camera names found in: {cameras_json}")
    return names


def _find_numbered_image(input_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = input_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing numbered image {stem}.* under: {input_dir}")


def main() -> int:
    args = _parse_args()
    cameras_json = args.cameras_json.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not cameras_json.is_file():
        raise FileNotFoundError(f"missing cameras json: {cameras_json}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"missing input dir: {input_dir}")

    names = _load_camera_names(cameras_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    skipped: list[str] = []
    for out_idx, camera_stem in enumerate(names):
        numbered = f"{args.start_index + out_idx:0{args.digits}d}"
        src = _find_numbered_image(input_dir, numbered)
        dst = output_dir / f"{camera_stem}{src.suffix.lower()}"
        if dst.exists() and not args.overwrite:
            skipped.append(dst.name)
            continue
        copied.append(
            {
                "numbered_stem": numbered,
                "camera_stem": camera_stem,
                "src": str(src),
                "dst": str(dst),
            }
        )
        if not args.dry_run:
            shutil.copy2(src, dst)

    manifest = {
        "mode": "remap_sequence_to_camera_names",
        "cameras_json": str(cameras_json),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_cameras": len(names),
        "num_copied": len(copied),
        "num_skipped_existing": len(skipped),
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
        "frames": copied,
        "skipped_existing": skipped[:32],
    }
    if not args.dry_run:
        (output_dir / "remap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[remap-sequence] {exc}", file=sys.stderr)
        raise SystemExit(1)

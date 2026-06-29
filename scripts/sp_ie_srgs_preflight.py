#!/usr/bin/env python3
"""Preflight server assets for SP-IE-SRGS v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = ("png", "jpg", "jpeg", "webp")


def image_index(root: Path, exts: Iterable[str] = IMAGE_EXTS) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    suffixes = {f".{ext.lower().lstrip('.')}" for ext in exts}
    index: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.stem, path)
    return index


def check_dir(path: Path, label: str, *, required: bool = True) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        message = f"{label} not found: {path}"
        if required:
            errors.append(message)
        else:
            print(f"[preflight] optional {message}")
    elif not path.is_dir():
        errors.append(f"{label} is not a directory: {path}")
    return errors


def coverage(label: str, reference: dict[str, Path], candidate: dict[str, Path]) -> dict[str, object]:
    ref_stems = set(reference)
    cand_stems = set(candidate)
    missing = sorted(ref_stems - cand_stems)
    extra = sorted(cand_stems - ref_stems)
    return {
        "label": label,
        "reference": len(ref_stems),
        "candidate": len(cand_stems),
        "matched": len(ref_stems & cand_stems),
        "missing_total": len(missing),
        "missing_sample": missing[:32],
        "extra_total": len(extra),
        "extra_sample": extra[:16],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SP-IE-SRGS v0 server assets.")
    parser.add_argument("--scene_root", type=Path, required=True)
    parser.add_argument("--hr_images_subdir", type=str, default="images_2")
    parser.add_argument("--lr_anchor_dir", type=Path, required=True)
    parser.add_argument("--prepared_sr_prior_root", type=Path, required=True)
    parser.add_argument("--sr_prior_subdir", type=str, default="fused_priors")
    parser.add_argument("--sr_prior_mask_subdir", type=str, default="usable_masks")
    parser.add_argument("--require_prior_mask", action="store_true")
    parser.add_argument("--npse_cache_root", type=Path, default=None)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--no_eval_split", action="store_false", dest="eval_split")
    parser.set_defaults(eval_split=True)
    parser.add_argument("--min_match_ratio", type=float, default=0.95)
    parser.add_argument("--json_out", type=Path, default=Path(""))
    args = parser.parse_args()

    scene_root = args.scene_root.expanduser().resolve()
    hr_dir = scene_root / args.hr_images_subdir
    lr_dir = args.lr_anchor_dir.expanduser().resolve()
    prior_root = args.prepared_sr_prior_root.expanduser().resolve()
    prior_dir = prior_root / args.sr_prior_subdir
    mask_dir = prior_root / args.sr_prior_mask_subdir if args.sr_prior_mask_subdir else Path("")

    errors: list[str] = []
    errors.extend(check_dir(scene_root, "scene_root"))
    errors.extend(check_dir(hr_dir, "hr image directory"))
    errors.extend(check_dir(lr_dir, "lr anchor directory"))
    errors.extend(check_dir(scene_root / "sparse" / "0", "COLMAP sparse/0"))
    errors.extend(check_dir(prior_root, "prepared SR prior root"))
    errors.extend(check_dir(prior_dir, "SR prior subdir"))
    if args.sr_prior_mask_subdir:
        errors.extend(check_dir(mask_dir, "SR prior mask subdir", required=bool(args.require_prior_mask)))

    npse_report = None
    if args.npse_cache_root is not None:
        npse_root = args.npse_cache_root.expanduser().resolve()
        npse_report = {
            "root": str(npse_root),
            "edge_target": (npse_root / "edge_target").is_dir(),
            "trust_edge": (npse_root / "trust_edge").is_dir(),
            "continuous_target": (npse_root / "continuous_target").is_dir(),
            "trust_continuous": (npse_root / "trust_continuous").is_dir(),
        }

    hr_index = image_index(hr_dir)
    lr_index = image_index(lr_dir)
    prior_index = image_index(prior_dir)
    mask_index = image_index(mask_dir) if args.sr_prior_mask_subdir and mask_dir.is_dir() else {}
    reference_index = hr_index
    if args.eval_split and int(args.llffhold) > 0:
        reference_index = {
            stem: path
            for idx, (stem, path) in enumerate(sorted(hr_index.items()))
            if idx % int(args.llffhold) != 0
        }

    if not hr_index:
        errors.append(f"no HR/reference images found: {hr_dir}")
    if not lr_index:
        errors.append(f"no LR anchor images found: {lr_dir}")
    if not prior_index:
        errors.append(f"no SR prior images found: {prior_dir}")
    if args.require_prior_mask and not mask_index:
        errors.append(f"no SR prior mask images found: {mask_dir}")

    reports = {
        "scene_root": str(scene_root),
        "hr_dir": str(hr_dir),
        "lr_anchor_dir": str(lr_dir),
        "prepared_sr_prior_root": str(prior_root),
        "sr_prior_dir": str(prior_dir),
        "sr_mask_dir": str(mask_dir) if args.sr_prior_mask_subdir else "",
        "reference_split": {
            "eval_split": bool(args.eval_split),
            "llffhold": int(args.llffhold),
            "hr_total": len(hr_index),
            "reference_total": len(reference_index),
        },
        "coverage": {
            "lr_anchor_vs_hr": coverage("lr_anchor_vs_hr", reference_index, lr_index),
            "prior_vs_hr": coverage("prior_vs_hr", reference_index, prior_index),
            "mask_vs_hr": coverage("mask_vs_hr", reference_index, mask_index) if mask_index else None,
        },
        "npse": npse_report,
    }

    min_ratio = max(0.0, min(1.0, float(args.min_match_ratio)))
    reference_count = max(len(reference_index), 1)
    for key in ("lr_anchor_vs_hr", "prior_vs_hr"):
        matched = int(reports["coverage"][key]["matched"])
        ratio = matched / reference_count
        if ratio < min_ratio:
            errors.append(f"{key} match ratio {ratio:.3f} is below {min_ratio:.3f}")
    if args.require_prior_mask and reports["coverage"]["mask_vs_hr"] is not None:
        matched = int(reports["coverage"]["mask_vs_hr"]["matched"])
        ratio = matched / reference_count
        if ratio < min_ratio:
            errors.append(f"mask_vs_hr match ratio {ratio:.3f} is below {min_ratio:.3f}")

    reports["ok"] = not errors
    reports["errors"] = errors
    print(json.dumps(reports, indent=2, sort_keys=True))
    if str(args.json_out):
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(reports, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

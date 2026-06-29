#!/usr/bin/env python3
"""Collect ablation metrics into one JSON table.

Each input is passed as ``name=/path/to/results.json``. The script preserves all
top-level numeric metrics and common nested ``results`` / ``metrics`` fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


def _numeric_items(payload: Mapping) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    for nested_key in ("results", "metrics"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    metrics[str(key)] = float(value)
    return metrics


def _load_metrics(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return _numeric_items(payload)


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be name=/path/to/results.json")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("variant name cannot be empty")
    return name, Path(path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", type=parse_variant)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    table = {}
    for name, path in args.variants:
        if not path.exists():
            raise FileNotFoundError(path)
        table[name] = _load_metrics(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

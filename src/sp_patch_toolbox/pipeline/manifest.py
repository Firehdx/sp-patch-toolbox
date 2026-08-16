"""Portable JSONL manifest utilities.

The manifest is the contract between dataset discovery, the agent harness and
the segmentation run.  It intentionally carries no absolute output paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..models import ImageSpec


def load_manifest(path: str | Path) -> list[ImageSpec]:
    """Load non-comment JSONL rows, retaining unknown fields under ``extras``."""
    specs: list[ImageSpec] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if "error" in data:
            continue
        specs.append(ImageSpec.from_dict(data))
    return specs


def write_manifest(specs: Iterable[ImageSpec], path: str | Path) -> Path:
    """Write a deterministic UTF-8 JSONL manifest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True) for spec in specs]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target

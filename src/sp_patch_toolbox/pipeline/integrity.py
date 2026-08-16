"""Strict preflight validation for image bytes.

Segmentation must never conceal a failed marker page.  In particular, QPTIFF
files are validated by fully decoding every page at the exact pyramid level
used for foreground thumbnail generation.  A single failure marks the slide as
invalid for patch generation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import tifffile

from ..io.readers import open_image_reader
from ..models import ImageSpec, IntegrityIssue, IntegrityReport


def _page_channel_name(page: tifffile.TiffPage, index: int) -> str:
    match = re.search(r"<Name>([^<]+)</Name>", str(page.description or ""))
    return match.group(1).strip() if match else f"page_{index}"


def _qptiff_thumbnail_pages(path: Path, expected_channels: int, thumbnail_max_size: int):
    """Mirror the QPTIFF thumbnail pyramid selection without pixel fusion."""
    handle = tifffile.TiffFile(path)
    groups: dict[tuple[int, int], list[tuple[int, tifffile.TiffPage]]] = {}
    for index, page in enumerate(handle.pages):
        shape = (int(page.imagelength), int(page.imagewidth))
        if min(shape) > 0:
            groups.setdefault(shape, []).append((index, page))
    eligible = [(shape, pages) for shape, pages in groups.items() if len(pages) == expected_channels]
    if not eligible:
        handle.close()
        available = ", ".join(f"{width}x{height}:{len(pages)}" for (height, width), pages in groups.items())
        raise ValueError(
            f"No QPTIFF pyramid level has exactly {expected_channels} channel pages in {path}; "
            f"found {available or 'none'}."
        )
    within_target = [item for item in eligible if max(item[0]) <= thumbnail_max_size]
    selected = max(within_target, key=lambda item: max(item[0])) if within_target else min(
        eligible, key=lambda item: max(item[0])
    )
    return handle, selected


def scan_qptiff_strict(path: str | Path, *, thumbnail_max_size: int = 1600) -> list[IntegrityIssue]:
    """Return every QPTIFF channel-page decoding failure for one image.

    The page is decoded as a whole with ``TiffPage.asarray``.  This is
    deliberately stricter than region reads: it catches malformed LZW data in
    any strip/tile, including background regions that a current contour might
    not otherwise visit.
    """
    image_path = Path(path)
    reader = open_image_reader(image_path, reader_type="qptiff")
    try:
        expected_channels = int(reader.channel_count)
    finally:
        reader.close()
    handle, (_, pages) = _qptiff_thumbnail_pages(image_path, expected_channels, thumbnail_max_size)
    issues: list[IntegrityIssue] = []
    try:
        for page_index, page in pages:
            try:
                page.asarray()
            except Exception as error:  # decoder exceptions are backend-specific
                issues.append(
                    IntegrityIssue(
                        path=str(image_path),
                        stage="qptiff_thumbnail_page_full_decode",
                        channel=_page_channel_name(page, page_index),
                        page_index=int(page_index),
                        error_type=type(error).__name__,
                        message=str(error),
                    )
                )
    finally:
        handle.close()
    return issues


def scan_manifest_strict(
    specs: Iterable[ImageSpec],
    *,
    data_root: str | Path | None = None,
    thumbnail_max_size: int = 1600,
) -> IntegrityReport:
    """Strictly inspect QPTIFF rows and verify that other reader types open."""
    root = Path(data_root).expanduser().resolve() if data_root else None
    report = IntegrityReport()
    for spec in specs:
        report.total += 1
        image_path = Path(spec.path)
        if not image_path.is_absolute() and root is not None:
            image_path = root / image_path
        try:
            if spec.reader_type == "qptiff" or image_path.suffix.lower() == ".qptiff":
                issues = scan_qptiff_strict(image_path, thumbnail_max_size=thumbnail_max_size)
            else:
                reader = open_image_reader(image_path, reader_type=spec.reader_type)
                try:
                    _ = reader.spatial_shape, reader.channel_count, reader.channel_names()
                finally:
                    reader.close()
                issues = []
        except Exception as error:
            issues = [
                IntegrityIssue(
                    path=str(image_path),
                    stage="reader_open_or_metadata",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            ]
        if issues:
            report.issues.extend(issues)
        else:
            report.passed += 1
    return report

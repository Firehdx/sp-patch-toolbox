import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..io.readers import open_image_reader, read_tiff_page_region

def _as_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=h5py.string_dtype(encoding="utf-8"))


def _decode_hdf5_attr_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_hdf5_attr_text(value.item())
        return "".join(_decode_hdf5_attr_text(item) for item in value.flatten())
    if isinstance(value, (list, tuple)):
        return "".join(_decode_hdf5_attr_text(item) for item in value)
    return str(value)


def _hdf5_attr_float(attrs, key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in attrs:
        return default
    text = _decode_hdf5_attr_text(attrs[key]).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        return float(match.group(0)) if match else default


def _infer_mpp_from_ims_metadata(image_path: Path) -> Optional[float]:
    """Infer mpp from Imaris IMS extents.

    Some IMS files do not expose an explicit microscope objective or mpp, but
    they store physical extents and pixel dimensions under DataSetInfo/Image:
    mpp_x = (ExtMax0 - ExtMin0) / X, mpp_y = (ExtMax1 - ExtMin1) / Y.
    """
    if image_path.suffix.lower() != ".ims":
        return None
    try:
        with h5py.File(str(image_path), "r") as handle:
            if "DataSetInfo/Image" not in handle:
                return None
            attrs = handle["DataSetInfo/Image"].attrs
            x_pixels = _hdf5_attr_float(attrs, "X")
            y_pixels = _hdf5_attr_float(attrs, "Y")
            ext_min_x = _hdf5_attr_float(attrs, "ExtMin0", 0.0)
            ext_max_x = _hdf5_attr_float(attrs, "ExtMax0")
            ext_min_y = _hdf5_attr_float(attrs, "ExtMin1", 0.0)
            ext_max_y = _hdf5_attr_float(attrs, "ExtMax1")
            if None in [x_pixels, y_pixels, ext_min_x, ext_max_x, ext_min_y, ext_max_y]:
                return None
            if x_pixels <= 0 or y_pixels <= 0:
                return None
            mpp_x = (ext_max_x - ext_min_x) / x_pixels
            mpp_y = (ext_max_y - ext_min_y) / y_pixels
            if mpp_x <= 0 or mpp_y <= 0:
                return None
            if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) > 0.05:
                return None
            return float((mpp_x + mpp_y) / 2.0)
    except Exception:
        return None


def infer_mpp_from_image_metadata(image_path: str | Path) -> Optional[float]:
    """Infer microns-per-pixel from common WSI/TIFF metadata fields."""
    image_path = Path(image_path)
    ims_mpp = _infer_mpp_from_ims_metadata(image_path)
    if ims_mpp is not None:
        return ims_mpp

    try:
        import openslide

        slide = openslide.OpenSlide(str(image_path))
        try:
            for key in ["openslide.mpp-x", "aperio.MPP", "hamamatsu.XResolution"]:
                if key in slide.properties:
                    value = float(slide.properties[key])
                    if value > 0:
                        return value
            comment = slide.properties.get("openslide.comment", "")
            match = __import__("re").search(r"<PixelSizeMicrons>([0-9.]+)</PixelSizeMicrons>", comment)
            if match:
                return float(match.group(1))
        finally:
            slide.close()
    except Exception:
        pass

    try:
        import re
        import tifffile

        with tifffile.TiffFile(str(image_path)) as tf:
            texts = []
            if getattr(tf, "ome_metadata", None):
                texts.append(tf.ome_metadata)
            for page in tf.pages[: min(len(tf.pages), 8)]:
                if page.description:
                    texts.append(page.description)
            joined = "\n".join(texts)
            for pattern in [
                r"<PixelSizeMicrons>([0-9.]+)</PixelSizeMicrons>",
                r'PhysicalSizeX="([0-9.]+)"',
            ]:
                match = re.search(pattern, joined)
                if match:
                    value = float(match.group(1))
                    if value > 0:
                        return value

            # Some ImageJ TCYX files lack OME PhysicalSizeX but retain an
            # X/YResolution ratio and declare ``unit=mm`` in the description.
            # Prefer that explicit unit over a raw ResolutionUnit of NONE.
            page = tf.pages[0] if tf.pages else None
            if page is not None:
                x_resolution = page.tags.get("XResolution")
                y_resolution = page.tags.get("YResolution")
                resolution_unit = page.tags.get("ResolutionUnit")
                if x_resolution and y_resolution:
                    try:
                        x_ppu = float(x_resolution.value[0]) / float(x_resolution.value[1])
                        y_ppu = float(y_resolution.value[0]) / float(y_resolution.value[1])
                        unit_text = ""
                        unit_match = re.search(r"(?:^|\n)unit=([^\n\r]+)", joined, flags=re.IGNORECASE)
                        if unit_match:
                            unit_text = unit_match.group(1).strip().lower()
                        tag_unit = str(resolution_unit.value).lower() if resolution_unit else ""
                        if unit_text in {"mm", "millimeter", "millimetre"}:
                            microns_per_unit = 1000.0
                        elif unit_text in {"um", "µm", "micron", "microns", "micrometer", "micrometre"}:
                            microns_per_unit = 1.0
                        elif "centimeter" in tag_unit or "centimetre" in tag_unit:
                            microns_per_unit = 10000.0
                        elif "inch" in tag_unit:
                            microns_per_unit = 25400.0
                        else:
                            microns_per_unit = None
                        if microns_per_unit is not None and x_ppu > 0 and y_ppu > 0:
                            mpp_x, mpp_y = microns_per_unit / x_ppu, microns_per_unit / y_ppu
                            if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) <= 0.05:
                                return float((mpp_x + mpp_y) / 2.0)
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
    except Exception:
        pass
    return None


def infer_mag_and_mpp(
    image_path: str | Path,
    *,
    mag: Optional[int] = None,
    mpp: Optional[float] = None,
) -> Tuple[Optional[float], int, float]:
    """Infer TRIDENT target magnification and normalized mpp for WSI loading.

    Some fluorescence QPTIFF files expose a raw mpp such as 0.49886. TRIDENT's
    coordinate extraction is cleaner if we snap that to a common magnification:
    mag = round(10 / raw_mpp), then normalized_mpp = 10 / mag.
    """
    raw_mpp = float(mpp) if mpp is not None else infer_mpp_from_image_metadata(image_path)
    if raw_mpp is not None and raw_mpp <= 0:
        raise ValueError(f"Invalid mpp for {image_path}: {raw_mpp}")

    if mag is None:
        if raw_mpp is None:
            raise ValueError(
                "Could not infer microns-per-pixel from image metadata. "
                "Pass --mpp with the raw scan mpp, or pass --mag explicitly."
            )
        mag = int(round(10.0 / raw_mpp))
    mag = int(mag)
    if mag <= 0:
        raise ValueError(f"Invalid target magnification for {image_path}: {mag}")
    normalized_mpp = 10.0 / float(mag)
    return raw_mpp, mag, normalized_mpp


def write_sp_coords_h5(
    save_path: str | Path,
    coords_xy: np.ndarray,
    *,
    source_path: str,
    reader_type: str,
    dataset: str,
    spatial_shape_yx: Tuple[int, int],
    patch_size: int,
    overlap: int,
    channel_names: Sequence[str],
    marker_names: Sequence[str],
    foreground_method: str,
    foreground_attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    coords_xy = np.asarray(coords_xy, dtype=np.int64)
    if coords_xy.size == 0:
        coords_xy = coords_xy.reshape(0, 2)
    height, width = int(spatial_shape_yx[0]), int(spatial_shape_yx[1])
    attrs = {
        "format": "sp_trident_coords_v1",
        "patch_size": int(patch_size),
        "patch_size_level0": int(patch_size),
        "patch_level": 0,
        "level0_magnification": -1,
        "target_magnification": -1,
        "overlap": int(overlap),
        "name": Path(source_path).stem,
        "source_path": str(source_path),
        "reader_type": str(reader_type),
        "dataset": str(dataset),
        "level0_width": width,
        "level0_height": height,
        "foreground_method": foreground_method,
        "foreground_attrs_json": json.dumps(foreground_attrs or {}, ensure_ascii=True, default=_as_jsonable),
        "channel_count": len(channel_names),
    }
    with h5py.File(save_path, "w") as handle:
        dset = handle.create_dataset("coords", data=coords_xy, maxshape=(None, 2), chunks=(min(max(len(coords_xy), 1), 4096), 2))
        for key, value in attrs.items():
            dset.attrs[key] = value
        handle.create_dataset("channel_names", data=_string_array(channel_names))
        handle.create_dataset("marker_names", data=_string_array(marker_names))
    return save_path


def append_channel_metadata_to_h5(
    coords_h5: str | Path,
    *,
    source_path: str,
    reader_type: str,
    dataset: str,
    channel_names: Sequence[str],
    marker_names: Sequence[str],
) -> None:
    with h5py.File(coords_h5, "a") as handle:
        for key in ["channel_names", "raw_channel_names", "marker_ids", "marker_names"]:
            if key in handle:
                del handle[key]
        handle.create_dataset("channel_names", data=_string_array(channel_names))
        handle.create_dataset("marker_names", data=_string_array(marker_names))
        coords = handle["coords"]
        coords.attrs["format"] = "sp_trident_coords_v1"
        coords.attrs["source_path"] = str(source_path)
        coords.attrs["reader_type"] = str(reader_type)
        coords.attrs["dataset"] = str(dataset)
        coords.attrs["channel_count"] = len(channel_names)


def load_sp_coords_h5(path: str | Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as handle:
        coords = handle["coords"][:]
        attrs = dict(handle["coords"].attrs)
        out = {"coords": coords, "attrs": attrs}
        for key in ["channel_names", "raw_channel_names", "marker_ids", "marker_names"]:
            if key in handle:
                values = handle[key][:]
                if values.dtype.kind in {"S", "O"}:
                    values = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v) for v in values]
                out[key] = values
    return out


def _ensure_trident_import(trident_root: str | Path) -> None:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    trident_root = Path(trident_root).resolve()
    if str(trident_root) not in sys.path:
        sys.path.insert(0, str(trident_root))


def _read_channel_metadata(
    image_path: str | Path,
    *,
    data_root: str | Path | None,
    reader_type: str,
    marker_registry_path: str | Path | None = None,
    channel_names_override: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    sp_reader = open_image_reader(image_path, data_root=data_root, reader_type=reader_type)
    try:
        channel_names = list(channel_names_override) if channel_names_override is not None else sp_reader.channel_names()
    finally:
        sp_reader.close()
    # Raw reader names are preserved.  Mapping a dye or filter label to a
    # biological marker is external curation, not preprocessing.
    return channel_names, list(channel_names)


def _odd_kernel(radius: int) -> Optional[np.ndarray]:
    radius = int(radius)
    if radius <= 0:
        return None
    size = radius * 2 + 1
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_binary_holes(
    mask: np.ndarray,
    *,
    max_hole_area_fraction: Optional[float] = None,
) -> np.ndarray:
    """Fill enclosed background components, optionally only small holes."""
    import cv2

    mask = (mask > 0).astype(np.uint8)
    if max_hole_area_fraction is None:
        flood = mask.copy()
        h, w = flood.shape
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), dtype=np.uint8), (0, 0), 1)
        holes = (flood == 0).astype(np.uint8)
        return ((mask | holes) > 0).astype(np.uint8)

    inverted = (mask == 0).astype(np.uint8)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    max_hole_area = max(1, int(round(mask.size * float(max_hole_area_fraction))))
    holes = np.zeros_like(mask)
    height, width = mask.shape
    for label in range(1, label_count):
        x, y, component_width, component_height, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + component_width == width or y + component_height == height
        if not touches_edge and int(area) <= max_hole_area:
            holes[labels == label] = 1
    return ((mask | holes) > 0).astype(np.uint8)


def _filter_connected_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, int]:
    import cv2

    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask * 0, 0
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_area)
    filtered = keep[labels].astype(np.uint8)
    return filtered, int(keep[1:].sum())


def _remove_border_frame_components(
    mask: np.ndarray,
    *,
    max_component_density: float = 0.30,
    min_component_span_fraction: float = 0.65,
    border_margin_px: int = 2,
) -> Tuple[np.ndarray, int]:
    """Remove sparse, image-spanning components attached to a thumbnail edge.

    QPTIFF illumination halos and tile seams form long, thin components at the
    image boundary. Real tissue can also meet an edge, so it is retained unless
    the component is sparse inside a very large bounding box.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels_count <= 1:
        return binary, 0

    keep = np.ones(labels_count, dtype=bool)
    keep[0] = False
    removed = 0
    margin = max(0, int(border_margin_px))
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = stats[label]
        spans_width = float(component_width) / max(float(width), 1.0)
        spans_height = float(component_height) / max(float(height), 1.0)
        spans_image = max(spans_width, spans_height) >= float(min_component_span_fraction)
        if not spans_image:
            continue
        touches_edge = (
            x <= margin
            or y <= margin
            or x + component_width >= width - margin
            or y + component_height >= height - margin
        )
        if not touches_edge:
            continue
        density = float(area) / max(float(component_width * component_height), 1.0)
        if density <= float(max_component_density):
            keep[label] = False
            removed += 1
    return keep[labels].astype(np.uint8), removed


def _remove_border_strip_components(
    mask: np.ndarray,
    *,
    border_margin_fraction: float = 0.15,
    max_strip_thickness_fraction: float = 0.16,
    min_strip_span_fraction: float = 0.25,
) -> Tuple[np.ndarray, int]:
    """Remove thin horizontal/vertical scanner-frame components near an edge.

    This is intentionally applied after the light closing/opening pass.  At
    that point a scanner frame is represented by a coherent, thin strip while
    a tissue section remains a two-dimensional component.  Unlike removing
    every component touching an edge, this retains a real section which was
    cropped by the scan boundary.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels_count <= 1:
        return binary, 0

    margin_y = max(1, int(round(height * float(border_margin_fraction))))
    margin_x = max(1, int(round(width * float(border_margin_fraction))))
    keep = np.ones(labels_count, dtype=bool)
    keep[0] = False
    removed = 0
    for label in range(1, labels_count):
        x, y, component_width, component_height, _ = stats[label]
        near_top_or_bottom = y <= margin_y or y + component_height >= height - margin_y
        near_left_or_right = x <= margin_x or x + component_width >= width - margin_x
        horizontal_strip = (
            near_top_or_bottom
            and float(component_height) / max(float(height), 1.0) <= float(max_strip_thickness_fraction)
            and float(component_width) / max(float(width), 1.0) >= float(min_strip_span_fraction)
        )
        vertical_strip = (
            near_left_or_right
            and float(component_width) / max(float(width), 1.0) <= float(max_strip_thickness_fraction)
            and float(component_height) / max(float(height), 1.0) >= float(min_strip_span_fraction)
        )
        if horizontal_strip or vertical_strip:
            keep[label] = False
            removed += 1
    return keep[labels].astype(np.uint8), removed


def _remove_small_edge_components(
    mask: np.ndarray,
    *,
    max_area_fraction: float = 0.005,
    edge_margin_fraction: float = 0.15,
) -> Tuple[np.ndarray, int]:
    """Remove small residual components only when they lie near a scanner edge.

    Small central components can be genuine tissue fragments.  In contrast,
    small components near the thumbnail boundary are typically remnants of a
    frame ring after the thin-strip removal.  Keeping this distinction avoids
    losing small biopsies while preventing peripheral ring patches.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if float(max_area_fraction) <= 0:
        return binary, 0
    height, width = binary.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels_count <= 1:
        return binary, 0

    max_small_area = max(1, int(round(binary.size * float(max_area_fraction))))
    margin_y = max(1, int(round(height * float(edge_margin_fraction))))
    margin_x = max(1, int(round(width * float(edge_margin_fraction))))
    keep = np.ones(labels_count, dtype=bool)
    keep[0] = False
    removed = 0
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = stats[label]
        is_small = int(area) < max_small_area
        touches_margin = (
            x < margin_x
            or y < margin_y
            or x + component_width > width - margin_x
            or y + component_height > height - margin_y
        )
        if is_small and touches_margin:
            keep[label] = False
            removed += 1
    return keep[labels].astype(np.uint8), removed


def _remove_sparse_peripheral_components(
    mask: np.ndarray,
    *,
    max_area_fraction: float = 0.15,
    max_density: float = 0.35,
    edge_margin_fraction: float = 0.15,
) -> Tuple[np.ndarray, int]:
    """Remove medium-sized, sparse scanner arcs near a thumbnail boundary."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels_count <= 1:
        return binary, 0
    max_area = max(1, int(round(binary.size * float(max_area_fraction))))
    margin_y = max(1, int(round(height * float(edge_margin_fraction))))
    margin_x = max(1, int(round(width * float(edge_margin_fraction))))
    keep = np.ones(labels_count, dtype=bool)
    keep[0] = False
    removed = 0
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = stats[label]
        near_edge = (
            x < margin_x
            or y < margin_y
            or x + component_width > width - margin_x
            or y + component_height > height - margin_y
        )
        density = float(area) / max(float(component_width * component_height), 1.0)
        if near_edge and int(area) <= max_area and density <= float(max_density):
            keep[label] = False
            removed += 1
    return keep[labels].astype(np.uint8), removed


def _remove_thin_elongated_components(
    mask: np.ndarray,
    *,
    min_aspect_ratio: float = 5.0,
    max_thickness_fraction: float = 0.025,
    max_area_fraction: float = 0.008,
) -> Tuple[np.ndarray, int]:
    """Remove isolated thin grid/seam components without cropping tissue.

    This targets the short, high-intensity horizontal/vertical bars left by
    some CODEX tile reconstructions.  It is component based rather than a
    positional exclusion, so tissue elsewhere in the same row or column is
    unaffected.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels_count <= 1:
        return binary, 0
    max_area = max(1, int(round(binary.size * float(max_area_fraction))))
    max_thickness = max(1.0, float(min(height, width)) * float(max_thickness_fraction))
    keep = np.ones(labels_count, dtype=bool)
    keep[0] = False
    removed = 0
    for label in range(1, labels_count):
        _, _, component_width, component_height, area = stats[label]
        short_axis = float(min(component_width, component_height))
        long_axis = float(max(component_width, component_height))
        aspect_ratio = long_axis / max(short_axis, 1.0)
        if (
            short_axis <= max_thickness
            and aspect_ratio >= float(min_aspect_ratio)
            and int(area) <= max_area
        ):
            keep[label] = False
            removed += 1
    return keep[labels].astype(np.uint8), removed


def make_sp_fluorescence_foreground(
    thumbnail_rgb: np.ndarray,
    *,
    threshold_percentile: float = 50.0,
    min_signal: float = 8.0,
    blur_sigma: float = 4.0,
    close_radius: int = 20,
    open_radius: int = 2,
    dilate_radius: int = 3,
    min_component_area_fraction: float = 0.001,
    remove_border_frame_artifacts: bool = False,
    border_frame_max_density: float = 0.30,
    border_frame_min_span_fraction: float = 0.65,
    remove_border_strip_artifacts: bool = False,
    remove_small_edge_artifacts: bool = False,
    small_edge_max_area_fraction: float = 0.005,
    small_edge_margin_fraction: float = 0.15,
    remove_sparse_peripheral_artifacts: bool = False,
    sparse_peripheral_max_area_fraction: float = 0.15,
    sparse_peripheral_max_density: float = 0.35,
    sparse_peripheral_edge_margin_fraction: float = 0.15,
    max_hole_area_fraction: Optional[float] = None,
    excluded_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    forced_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a coarse foreground mask for black-background fluorescence/SP images."""
    import cv2

    rgb = np.asarray(thumbnail_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected RGB thumbnail shaped [H, W, 3], got {rgb.shape}")

    gray = rgb[..., :3].max(axis=2).astype(np.float32)
    if blur_sigma > 0:
        score = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    else:
        score = gray

    positive = score[score > 0]
    if positive.size == 0:
        threshold = float(min_signal)
    else:
        threshold = max(float(min_signal), float(np.percentile(positive, float(threshold_percentile))))

    mask = (score > threshold).astype(np.uint8)
    border_components_removed = 0
    if remove_border_frame_artifacts:
        mask, border_components_removed = _remove_border_frame_components(
            mask,
            max_component_density=border_frame_max_density,
            min_component_span_fraction=border_frame_min_span_fraction,
        )
    kernel = _odd_kernel(close_radius)
    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = _odd_kernel(open_radius)
    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    border_strip_components_removed = 0
    if remove_border_strip_artifacts:
        mask, border_strip_components_removed = _remove_border_strip_components(mask)
    sparse_peripheral_components_removed = 0
    if remove_sparse_peripheral_artifacts:
        mask, sparse_peripheral_components_removed = _remove_sparse_peripheral_components(
            mask,
            max_area_fraction=sparse_peripheral_max_area_fraction,
            max_density=sparse_peripheral_max_density,
            edge_margin_fraction=sparse_peripheral_edge_margin_fraction,
        )
    mask = _fill_binary_holes(mask, max_hole_area_fraction=max_hole_area_fraction)
    kernel = _odd_kernel(dilate_radius)
    if kernel is not None:
        mask = cv2.dilate(mask, kernel)
    small_edge_components_removed = 0
    if remove_small_edge_artifacts:
        mask, small_edge_components_removed = _remove_small_edge_components(
            mask,
            max_area_fraction=small_edge_max_area_fraction,
            edge_margin_fraction=small_edge_margin_fraction,
        )
    applied_excluded_regions = []
    for region in excluded_thumbnail_regions or ():
        x0, y0, x1, y1 = (float(value) for value in region)
        x0_px = max(0, min(mask.shape[1], int(round(x0 * mask.shape[1]))))
        x1_px = max(0, min(mask.shape[1], int(round(x1 * mask.shape[1]))))
        y0_px = max(0, min(mask.shape[0], int(round(y0 * mask.shape[0]))))
        y1_px = max(0, min(mask.shape[0], int(round(y1 * mask.shape[0]))))
        if x1_px > x0_px and y1_px > y0_px:
            mask[y0_px:y1_px, x0_px:x1_px] = 0
            applied_excluded_regions.append([x0, y0, x1, y1])
    applied_forced_regions = []
    for region in forced_thumbnail_regions or ():
        x0, y0, x1, y1 = (float(value) for value in region)
        x0_px = max(0, min(mask.shape[1], int(round(x0 * mask.shape[1]))))
        x1_px = max(0, min(mask.shape[1], int(round(x1 * mask.shape[1]))))
        y0_px = max(0, min(mask.shape[0], int(round(y0 * mask.shape[0]))))
        y1_px = max(0, min(mask.shape[0], int(round(y1 * mask.shape[0]))))
        if x1_px > x0_px and y1_px > y0_px:
            mask[y0_px:y1_px, x0_px:x1_px] = 1
            applied_forced_regions.append([x0, y0, x1, y1])

    min_area = max(1, int(round(mask.size * float(min_component_area_fraction))))
    mask, kept_components = _filter_connected_components(mask, min_area=min_area)
    attrs = {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "min_signal": min_signal,
        "blur_sigma": blur_sigma,
        "close_radius": close_radius,
        "open_radius": open_radius,
        "dilate_radius": dilate_radius,
        "min_component_area_fraction": min_component_area_fraction,
        "min_component_area_px": min_area,
        "kept_components": kept_components,
        "foreground_fraction": float(mask.mean()) if mask.size else 0.0,
        "remove_border_frame_artifacts": bool(remove_border_frame_artifacts),
        "border_frame_max_density": border_frame_max_density,
        "border_frame_min_span_fraction": border_frame_min_span_fraction,
        "border_components_removed": border_components_removed,
        "remove_border_strip_artifacts": bool(remove_border_strip_artifacts),
        "border_strip_components_removed": border_strip_components_removed,
        "remove_small_edge_artifacts": bool(remove_small_edge_artifacts),
        "small_edge_components_removed": small_edge_components_removed,
        "small_edge_max_area_fraction": small_edge_max_area_fraction,
        "small_edge_margin_fraction": small_edge_margin_fraction,
        "remove_sparse_peripheral_artifacts": bool(remove_sparse_peripheral_artifacts),
        "sparse_peripheral_max_area_fraction": sparse_peripheral_max_area_fraction,
        "sparse_peripheral_max_density": sparse_peripheral_max_density,
        "sparse_peripheral_edge_margin_fraction": sparse_peripheral_edge_margin_fraction,
        "sparse_peripheral_components_removed": sparse_peripheral_components_removed,
        "max_hole_area_fraction": max_hole_area_fraction,
        "excluded_thumbnail_regions": applied_excluded_regions,
        "forced_thumbnail_regions": applied_forced_regions,
    }
    return (mask.astype(np.uint8) * 255), attrs


def add_roi_max_fusion_foreground(
    mask: np.ndarray,
    max_fusion_thumbnail_rgb: np.ndarray,
    *,
    regions: Sequence[Tuple[float, float, float, float]],
    threshold_percentile: float = 70.0,
    min_signal: float = 8.0,
    blur_sigma: float = 2.0,
    close_radius: int = 7,
    open_radius: int = 1,
    dilate_radius: int = 2,
    min_component_area_fraction: float = 0.002,
    max_hole_area_fraction: Optional[float] = None,
    replace_existing_mask: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Recover weak tissue by segmenting max marker fusion *inside* ROIs.

    The ROI is a computational boundary only.  Its pixels are never added
    directly: each crop is thresholded and morphology-filtered independently,
    then only its detected foreground is ORed back into the global mask.
    """
    base = (np.asarray(mask) > 0).astype(np.uint8)
    thumbnail = np.asarray(max_fusion_thumbnail_rgb)
    if thumbnail.ndim != 3 or thumbnail.shape[:2] != base.shape:
        raise ValueError(
            "Max-fusion thumbnail must be RGB and have the same H/W as the global mask; "
            f"got thumbnail={thumbnail.shape}, mask={base.shape}."
        )

    # Derive the cutoff from the complete thumbnail rather than each ROI.
    # A mostly-black ROI has a deceptively low local percentile; estimating
    # the threshold inside it can make its boundary become a foreground slab.
    full_mask, full_attrs = make_sp_fluorescence_foreground(
        thumbnail,
        threshold_percentile=threshold_percentile,
        min_signal=min_signal,
        blur_sigma=blur_sigma,
        close_radius=close_radius,
        open_radius=open_radius,
        dilate_radius=dilate_radius,
        min_component_area_fraction=min_component_area_fraction,
        max_hole_area_fraction=max_hole_area_fraction,
    )
    full_binary = full_mask > 0
    applied: List[Dict[str, Any]] = []
    height, width = base.shape
    for region in regions:
        x0, y0, x1, y1 = (float(value) for value in region)
        x0_px = max(0, min(width, int(round(x0 * width))))
        x1_px = max(0, min(width, int(round(x1 * width))))
        y0_px = max(0, min(height, int(round(y0 * height))))
        y1_px = max(0, min(height, int(round(y1 * height))))
        if x1_px <= x0_px or y1_px <= y0_px:
            continue
        crop_binary = full_binary[y0_px:y1_px, x0_px:x1_px]
        if replace_existing_mask:
            base[y0_px:y1_px, x0_px:x1_px] = crop_binary
        else:
            base[y0_px:y1_px, x0_px:x1_px] |= crop_binary
        applied.append(
            {
                "region": [x0, y0, x1, y1],
                "foreground_fraction": float(crop_binary.mean()) if crop_binary.size else 0.0,
                "foreground_pixels": int(crop_binary.sum()),
                "replaced_existing_mask": bool(replace_existing_mask),
                "segmentation": full_attrs,
            }
        )
    return (base * 255).astype(np.uint8), applied


def replace_roi_max_fusion_foreground(
    mask: np.ndarray,
    max_fusion_thumbnail_rgb: np.ndarray,
    *,
    regions: Sequence[Tuple[float, float, float, float]],
    threshold_percentile: float = 55.0,
    min_signal: float = 6.0,
    blur_sigma: float = 2.0,
    close_radius: int = 3,
    open_radius: int = 0,
    dilate_radius: int = 2,
    min_component_area_fraction: float = 0.0001,
    max_hole_area_fraction: Optional[float] = 0.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Re-segment selected ROIs and replace only those mask pixels.

    Used for explicit false-positive regions. Unlike an exclusion rectangle,
    each selected region is independently segmented from the fluorescence
    signal, then replaces (rather than clears) just that local part of mask.
    """
    base = (np.asarray(mask) > 0).astype(np.uint8)
    thumbnail = np.asarray(max_fusion_thumbnail_rgb)
    if thumbnail.ndim != 3 or thumbnail.shape[:2] != base.shape:
        raise ValueError(
            "Max-fusion thumbnail must be RGB and have the same H/W as the global mask; "
            f"got thumbnail={thumbnail.shape}, mask={base.shape}."
        )

    # Estimate the cutoff on the complete thumbnail.  Estimating it per ROI
    # makes a mostly black crop look relatively bright and can retain its crop
    # boundary as foreground.
    full_mask, full_attrs = make_sp_fluorescence_foreground(
        thumbnail,
        threshold_percentile=threshold_percentile,
        min_signal=min_signal,
        blur_sigma=blur_sigma,
        close_radius=close_radius,
        open_radius=open_radius,
        dilate_radius=dilate_radius,
        min_component_area_fraction=min_component_area_fraction,
        max_hole_area_fraction=max_hole_area_fraction,
    )
    full_binary = full_mask > 0
    applied: List[Dict[str, Any]] = []
    height, width = base.shape
    for region in regions:
        x0, y0, x1, y1 = (float(value) for value in region)
        x0_px = max(0, min(width, int(round(x0 * width))))
        x1_px = max(0, min(width, int(round(x1 * width))))
        y0_px = max(0, min(height, int(round(y0 * height))))
        y1_px = max(0, min(height, int(round(y1 * height))))
        if x1_px <= x0_px or y1_px <= y0_px:
            continue
        crop_binary = full_binary[y0_px:y1_px, x0_px:x1_px]
        base[y0_px:y1_px, x0_px:x1_px] = crop_binary
        applied.append(
            {
                "region": [x0, y0, x1, y1],
                "foreground_fraction": float(crop_binary.mean()) if crop_binary.size else 0.0,
                "foreground_pixels": int(crop_binary.sum()),
                "segmentation": full_attrs,
            }
        )
    return (base * 255).astype(np.uint8), applied


def fill_roi_component_concavities(
    mask: np.ndarray,
    *,
    regions: Sequence[Tuple[float, float, float, float]],
    min_component_area_fraction: float = 0.01,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Fill only concavities of detected tissue components inside explicit ROIs.

    This is a shape-based recovery for a documented weak-signal U-shaped gap.
    It takes the convex hull of each sufficiently large detected connected
    component; it never writes the ROI rectangle itself.
    """
    import cv2

    base = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = base.shape
    applied: List[Dict[str, Any]] = []
    for region in regions:
        x0, y0, x1, y1 = (float(value) for value in region)
        x0_px = max(0, min(width, int(round(x0 * width))))
        x1_px = max(0, min(width, int(round(x1 * width))))
        y0_px = max(0, min(height, int(round(y0 * height))))
        y1_px = max(0, min(height, int(round(y1 * height))))
        if x1_px <= x0_px or y1_px <= y0_px:
            continue
        crop = base[y0_px:y1_px, x0_px:x1_px]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
        min_area = max(1, int(round(crop.size * float(min_component_area_fraction))))
        recovered = np.zeros_like(crop)
        components = 0
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
                continue
            ys, xs = np.where(labels == label)
            if xs.size < 3:
                continue
            hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
            cv2.fillConvexPoly(recovered, hull, 1)
            components += 1
        added = recovered & ~crop
        base[y0_px:y1_px, x0_px:x1_px] |= recovered
        applied.append(
            {
                "region": [x0, y0, x1, y1],
                "components": components,
                "pixels_added": int(added.sum()),
            }
        )
    return (base * 255).astype(np.uint8), applied


def force_thumbnail_polygons(
    mask: np.ndarray,
    *,
    polygons: Sequence[Sequence[Tuple[float, float]]],
) -> Tuple[np.ndarray, List[List[List[float]]]]:
    """Apply explicitly supplied polygonal foreground corrections."""
    import cv2

    base = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = base.shape
    applied: List[List[List[float]]] = []
    for polygon in polygons:
        points = np.asarray(
            [
                [
                    max(0, min(width - 1, round(float(x) * width))),
                    max(0, min(height - 1, round(float(y) * height))),
                ]
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        if len(points) < 3:
            continue
        cv2.fillPoly(base, [points], 1)
        applied.append([[float(x), float(y)] for x, y in polygon])
    return (base * 255).astype(np.uint8), applied


def _safe_file_stem(path: str | Path) -> str:
    name = Path(path).name
    name = re.sub(r"\.(ome\.)?(tif|tiff|qptiff|ims)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_") or "sample"


def _channel_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def _normalize_thumbnail_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image.astype(np.uint8)
    positive = image[image > 0]
    if positive.size:
        lo = float(np.percentile(positive, 1.0))
        hi = float(np.percentile(positive, 99.5))
    else:
        lo = float(image.min())
        hi = float(image.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _read_tiled_tiff_page_thumbnail(page: Any, *, thumbnail_max_size: int) -> np.ndarray:
    """Downsample a tiled TIFF page without materializing the whole page.

    Some CODEX OME-TIFFs have no reduced-resolution SubIFDs. Decoding their
    level-0 hyperstack would require tens of gigabytes, so decode one tile at
    a time and place its downsampled representation into a thumbnail canvas.
    """
    import cv2

    source_height, source_width = _tiff_page_spatial_shape(page)
    if source_width >= source_height:
        target_width = int(thumbnail_max_size)
        target_height = max(1, int(round(target_width * source_height / source_width)))
    else:
        target_height = int(thumbnail_max_size)
        target_width = max(1, int(round(target_height * source_width / source_height)))

    if not bool(getattr(page, "is_tiled", False)):
        # This fallback is intended for the large tiled CODEX pages. A small
        # non-tiled page is still safe to decode once and resize.  Use the
        # region decoder rather than page.asarray(): old tifffile releases
        # call ndarray.newbyteorder(), removed in NumPy 2, for this ImageJ
        # TIFF layout.
        source = read_tiff_page_region(page, 0, 0, source_height, source_width)
        if source.ndim == 3:
            source = np.max(source, axis=-1)
        return cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)

    thumbnail = np.zeros((target_height, target_width), dtype=np.float32)
    tile_height = int(getattr(page, "tilelength"))
    tile_width = int(getattr(page, "tilewidth"))
    for y0 in range(0, source_height, tile_height):
        y1 = min(source_height, y0 + tile_height)
        ty0 = int(round(y0 * target_height / source_height))
        ty1 = max(ty0 + 1, int(round(y1 * target_height / source_height)))
        for x0 in range(0, source_width, tile_width):
            x1 = min(source_width, x0 + tile_width)
            tx0 = int(round(x0 * target_width / source_width))
            tx1 = max(tx0 + 1, int(round(x1 * target_width / source_width)))
            tile = read_tiff_page_region(page, y0, x0, y1, x1)
            if tile.ndim == 3:
                tile = np.max(tile, axis=-1)
            thumbnail[ty0:ty1, tx0:tx1] = cv2.resize(
                np.asarray(tile, dtype=np.float32),
                (tx1 - tx0, ty1 - ty0),
                interpolation=cv2.INTER_AREA,
            )
    return thumbnail


def _tiff_page_spatial_shape(page: Any) -> Tuple[int, int]:
    """Return a TIFF page's Y/X shape without decoding its pixels."""
    height = getattr(page, "imagelength", None)
    width = getattr(page, "imagewidth", None)
    if height is not None and width is not None:
        return int(height), int(width)
    shape = tuple(getattr(page, "shape", ()))
    if len(shape) < 2:
        raise ValueError(f"TIFF page has no spatial shape: {shape}")
    return int(shape[-2]), int(shape[-1])


def _page_to_grayscale_plane(page: Any) -> np.ndarray:
    """Decode one low-resolution TIFF page as a 2D fluorescence intensity plane."""
    array = np.asarray(page.asarray())
    if array.ndim == 2:
        return array
    axes = str(getattr(page, "axes", ""))
    if "Y" in axes and "X" in axes and len(axes) == array.ndim:
        y_axis, x_axis = axes.index("Y"), axes.index("X")
        moved = np.moveaxis(array, [y_axis, x_axis], [-2, -1])
        # Most QPTIFF marker pages are YX. Retain a safe path for a sample or
        # other non-spatial axis: a marker is foreground when any value is
        # bright at the pixel.
        return np.max(moved, axis=tuple(range(moved.ndim - 2)))
    samples = int(getattr(page, "samplesperpixel", 1) or 1)
    if array.ndim == 3 and array.shape[-1] == samples:
        return np.max(array, axis=-1)
    raise ValueError(f"Cannot identify Y/X axes for TIFF page shaped {array.shape} with axes={axes!r}")


def _qptiff_page_channel_name(page: Any, fallback: str) -> str:
    description = str(getattr(page, "description", "") or "")
    match = re.search(r"<Name>([^<]+)</Name>", description)
    return match.group(1).strip() if match else fallback


def read_qptiff_multichannel_thumbnail(
    image_path: str | Path,
    *,
    expected_channel_count: int,
    thumbnail_max_size: int,
    fusion: str = "median",
    fusion_percentile: float = 100.0,
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    """Build a QPTIFF foreground thumbnail from all raw marker pages.

    OpenSlide's generic-TIFF backend exposes PerkinElmer QPTIFF channel pages
    as duplicate-resolution *levels*. Its ``get_thumbnail`` call therefore
    reads one arbitrary page at a selected resolution rather than combining
    marker channels. Here we instead select one pyramid resolution containing
    every marker page, robust-normalize each page independently, and exclude
    the technical autofluorescence page. ``median`` is used for the global
    mask because it rejects one-channel illumination artifacts. ``percentile``
    preserves tissue visible in only a subset of markers, while ``max`` is
    available for a small, manually located ROI when weak tissue is visible in
    only one marker.
    """
    if int(expected_channel_count) <= 0:
        raise ValueError(f"expected_channel_count must be positive, got {expected_channel_count}")
    if int(thumbnail_max_size) <= 0:
        raise ValueError(f"thumbnail_max_size must be positive, got {thumbnail_max_size}")
    fusion = str(fusion).lower()
    if fusion not in {"median", "percentile", "max"}:
        raise ValueError(
            f"Unsupported QPTIFF thumbnail fusion {fusion!r}; expected 'median', 'percentile', or 'max'."
        )
    if not 0.0 <= float(fusion_percentile) <= 100.0:
        raise ValueError(f"QPTIFF fusion percentile must be in [0, 100], got {fusion_percentile}")

    import tifffile

    image_path = Path(image_path)
    with tifffile.TiffFile(str(image_path)) as handle:
        pages_by_shape: Dict[Tuple[int, int], List[Tuple[int, Any]]] = {}
        for index, page in enumerate(handle.pages):
            try:
                shape_yx = _tiff_page_spatial_shape(page)
            except (TypeError, ValueError):
                continue
            if min(shape_yx) <= 0:
                continue
            pages_by_shape.setdefault(shape_yx, []).append((index, page))

        eligible = [
            (shape_yx, pages)
            for shape_yx, pages in pages_by_shape.items()
            if len(pages) == int(expected_channel_count)
        ]
        if not eligible:
            available = ", ".join(
                f"{width}x{height}:{len(pages)}" for (height, width), pages in sorted(pages_by_shape.items())
            )
            raise ValueError(
                "Could not find a QPTIFF pyramid resolution containing exactly "
                f"{expected_channel_count} marker pages in {image_path}. "
                f"Observed page groups (widthxheight:pages): {available or 'none'}."
            )

        def long_edge(item: Tuple[Tuple[int, int], List[Tuple[int, Any]]]) -> int:
            return max(item[0])

        # Prefer the highest-resolution group that is already no larger than
        # the requested thumbnail. If every stored pyramid level is larger,
        # use the coarsest level and downsample it once more.
        within_target = [item for item in eligible if long_edge(item) <= int(thumbnail_max_size)]
        selected_shape, selected_pages = (
            max(within_target, key=long_edge) if within_target else min(eligible, key=long_edge)
        )
        selected_height, selected_width = selected_shape
        normalized_planes = []
        included_channel_names = []
        excluded_channel_names = []
        for index, page in selected_pages:
            channel_name = _qptiff_page_channel_name(page, fallback=f"page_{index}")
            if channel_name.strip().lower() in {"sample af", "autofluorescence"}:
                excluded_channel_names.append(channel_name)
                continue
            plane = _page_to_grayscale_plane(page)
            if plane.shape != selected_shape:
                raise ValueError(
                    f"Inconsistent QPTIFF page shape in {image_path}: "
                    f"expected {selected_shape}, got {plane.shape}"
                )
            normalized_plane = _normalize_thumbnail_u8(plane)
            normalized_planes.append(normalized_plane)
            included_channel_names.append(channel_name)
        if not normalized_planes:
            raise ValueError(f"No usable biological marker pages found in {image_path}")
        if len(normalized_planes) == 1:
            composite = normalized_planes[0]
            fusion = "single_channel"
        elif fusion == "max":
            composite = np.max(np.stack(normalized_planes, axis=0), axis=0).astype(np.uint8)
        elif fusion == "percentile":
            composite = np.percentile(
                np.stack(normalized_planes, axis=0), float(fusion_percentile), axis=0
            ).astype(np.uint8)
        else:
            composite = np.median(np.stack(normalized_planes, axis=0), axis=0).astype(np.uint8)
            fusion = "median"

        source_height, source_width = max(pages_by_shape, key=lambda shape: shape[0] * shape[1])

    if max(composite.shape) > int(thumbnail_max_size):
        from PIL import Image

        height, width = composite.shape
        if width >= height:
            new_size = (int(thumbnail_max_size), max(1, int(round(thumbnail_max_size * height / width))))
        else:
            new_size = (max(1, int(round(thumbnail_max_size * width / height))), int(thumbnail_max_size))
        resampling = getattr(Image, "Resampling", Image)
        composite = np.asarray(Image.fromarray(composite).resize(new_size, resample=resampling.BILINEAR))

    thumbnail_rgb = np.repeat(composite[..., np.newaxis], 3, axis=2)
    attrs = {
        "thumbnail_source": "qptiff_raw_marker_pages_robust_fusion",
        "thumbnail_fusion": fusion,
        "thumbnail_fusion_percentile": float(fusion_percentile) if fusion == "percentile" else None,
        "thumbnail_marker_page_count": int(len(selected_pages)),
        "thumbnail_marker_page_indices": [int(index) for index, _ in selected_pages],
        "thumbnail_pyramid_page_shape_yx": [int(selected_height), int(selected_width)],
        "thumbnail_included_channel_names": included_channel_names,
        "thumbnail_excluded_channel_names": excluded_channel_names,
    }
    return thumbnail_rgb, (int(source_height), int(source_width)), attrs


def _ims_resolution_shapes(handle: h5py.File, *, timepoint: int = 0) -> Dict[int, Tuple[int, int]]:
    shapes: Dict[int, Tuple[int, int]] = {}
    dataset = handle.get("DataSet")
    if dataset is None:
        return shapes
    for key in dataset.keys():
        match = re.search(r"ResolutionLevel\s+(\d+)$", key)
        if not match:
            continue
        level = int(match.group(1))
        tp = dataset[key].get(f"TimePoint {timepoint}")
        if tp is None:
            continue
        channel_keys = sorted([k for k in tp.keys() if k.startswith("Channel ")], key=_channel_sort_key)
        if not channel_keys:
            continue
        data = tp[channel_keys[0]].get("Data")
        if data is None:
            continue
        shapes[level] = (int(data.shape[-2]), int(data.shape[-1]))
    return shapes


def _choose_thumbnail_level(shapes: Dict[int, Tuple[int, int]], thumbnail_max_size: int) -> int:
    if not shapes:
        raise ValueError("IMS file has no readable DataSet/ResolutionLevel image data.")
    levels = sorted(shapes)
    fitting = [level for level in levels if max(shapes[level]) <= int(thumbnail_max_size)]
    if fitting:
        return fitting[0]
    return levels[-1]


def _read_ims_projection_thumbnail(
    image_path: str | Path,
    *,
    thumbnail_max_size: int,
    timepoint: int = 0,
    fusion_percentile: float = 75.0,
) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Read a robust multi-marker thumbnail directly from an IMS pyramid.

    Each marker is first projected along Z and independently normalized.  We
    then use an upper quantile across markers rather than a raw channel-wise
    maximum: a raw maximum promotes the background of whichever channel has
    the largest numeric range, while a median can miss weak tissue visible in
    only a subset of markers.  The 75th percentile is deliberately lenient at
    the tissue boundary without admitting one-channel background everywhere.
    """
    if not 0.0 <= float(fusion_percentile) <= 100.0:
        raise ValueError(f"IMS fusion percentile must be in [0, 100], got {fusion_percentile}")
    with h5py.File(str(image_path), "r") as handle:
        shapes = _ims_resolution_shapes(handle, timepoint=timepoint)
        original_shape = shapes.get(0)
        if original_shape is None:
            raise ValueError(f"IMS file has no ResolutionLevel 0 image data: {image_path}")
        level = _choose_thumbnail_level(shapes, thumbnail_max_size)
        tp = handle["DataSet"][f"ResolutionLevel {level}"][f"TimePoint {timepoint}"]
        channel_keys = sorted([k for k in tp.keys() if k.startswith("Channel ")], key=_channel_sort_key)

        normalized_planes: List[np.ndarray] = []
        for key in channel_keys:
            data = tp[key]["Data"]
            arr = data[()]
            if arr.ndim >= 3:
                plane = arr.max(axis=0)
            else:
                plane = arr
            normalized_planes.append(_normalize_thumbnail_u8(np.asarray(plane, dtype=np.float32)))
        if not normalized_planes:
            raise ValueError(f"IMS file has no readable channels: {image_path}")
        projection = np.percentile(np.stack(normalized_planes, axis=0), float(fusion_percentile), axis=0).astype(np.uint8)

    thumb_gray = projection
    if max(thumb_gray.shape) > int(thumbnail_max_size):
        from PIL import Image

        height, width = thumb_gray.shape
        if width >= height:
            new_size = (int(thumbnail_max_size), max(1, int(round(thumbnail_max_size * height / width))))
        else:
            new_size = (max(1, int(round(thumbnail_max_size * width / height))), int(thumbnail_max_size))
        resampling = getattr(Image, "Resampling", Image)
        thumb_gray = np.asarray(Image.fromarray(thumb_gray).resize(new_size, resample=resampling.BILINEAR))
    thumbnail_rgb = np.repeat(thumb_gray[..., np.newaxis], 3, axis=2)
    return thumbnail_rgb, original_shape, level


def _read_ome_tiff_projection_thumbnail(
    image_path: str | Path,
    *,
    thumbnail_max_size: int,
    fusion_percentile: float = 75.0,
    channel_names: Optional[Sequence[str]] = None,
    technical_channel_fusion: str = "percentile",
    force_median_marker_fusion: bool = False,
    include_nucleus_thumbnail: bool = False,
    treat_z_as_channels: bool = False,
    flatten_nonspatial_axes_as_channels: bool = False,
    max_thumbnail_marker_planes: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int], int, Dict[str, Any], Optional[np.ndarray]]:
    """Read a multi-marker OME-TIFF thumbnail without relying on OpenSlide.

    This avoids relying on OpenSlide when TIFF compression or layout is not
    supported. At a stored low-resolution pyramid level, planes can be read
    directly with tifffile, max-projected over Z and robust-normalized. AF,
    blank and control planes are excluded before fusion.
    """
    if not 0.0 <= float(fusion_percentile) <= 100.0:
        raise ValueError(f"OME-TIFF fusion percentile must be in [0, 100], got {fusion_percentile}")
    technical_channel_fusion = str(technical_channel_fusion).lower()
    if technical_channel_fusion not in {"median", "percentile"}:
        raise ValueError(
            "OME-TIFF technical-channel fusion must be 'median' or 'percentile', "
            f"got {technical_channel_fusion!r}"
        )
    import tifffile

    with tifffile.TiffFile(str(image_path)) as handle:
        series = handle.series[0]
        levels = list(series.levels)
        if not levels:
            raise ValueError(f"OME-TIFF has no image series levels: {image_path}")

        def level_shape_yx(level: Any) -> Tuple[int, int]:
            axes = str(level.axes)
            shape = tuple(int(value) for value in level.shape)
            if "Y" not in axes or "X" not in axes:
                raise ValueError(f"OME-TIFF level has no Y/X axes: axes={axes!r}, shape={shape}")
            return shape[axes.index("Y")], shape[axes.index("X")]

        level_shapes = [level_shape_yx(level) for level in levels]
        fitting = [index for index, shape in enumerate(level_shapes) if max(shape) <= int(thumbnail_max_size)]
        level_index = min(fitting) if fitting else len(levels) - 1
        level = levels[level_index]
        axes = str(level.axes)
        y_axis, x_axis = axes.index("Y"), axes.index("X")
        channel_axes = (
            [index for index, axis in enumerate(axes) if axis not in {"Y", "X", "Z"}]
            if flatten_nonspatial_axes_as_channels
            else [
                axes.index("C")
                if "C" in axes
                else (axes.index("I") if "I" in axes else (axes.index("Z") if treat_z_as_channels and "Z" in axes else -1))
            ]
        )
        channel_axes = [index for index in channel_axes if index >= 0]
        channel_count = int(np.prod([level.shape[index] for index in channel_axes])) if channel_axes else 1
        use_tiled_page_thumbnail = max(level_shapes[level_index]) > int(thumbnail_max_size)
        if use_tiled_page_thumbnail:
            # No low-resolution pyramid is available. By default sample DAPI
            # plus evenly spaced marker pages; a non-positive cap explicitly
            # requests every available biological plane.
            available_indices = list(range(channel_count))
            if channel_names is not None and len(channel_names) == channel_count:
                biological_indices = [
                    index
                    for index, channel_name in enumerate(channel_names)
                    if not (
                        bool(re.search(r"(^|[_\s-])af\d*($|[_\s-])", str(channel_name).lower()))
                        or "autofluorescence" in str(channel_name).lower()
                        or "blank" in str(channel_name).lower()
                    )
                ]
                if biological_indices:
                    available_indices = biological_indices
            if int(max_thumbnail_marker_planes) <= 0:
                selected_source_indices = available_indices
            else:
                sample_count = min(max(1, int(max_thumbnail_marker_planes)), len(available_indices))
                selected_source_indices = sorted(
                    {
                        available_indices[int(round(position * (len(available_indices) - 1) / max(sample_count - 1, 1)))]
                        for position in range(sample_count)
                    }
                )
            if len(level.pages) >= channel_count:
                normalized_planes = [
                    _normalize_thumbnail_u8(
                        _read_tiled_tiff_page_thumbnail(
                            level.pages[source_index], thumbnail_max_size=thumbnail_max_size
                        )
                    )
                    for source_index in selected_source_indices
                ]
            else:
                # ImageJ may store a TCYX hyperstack as one contiguous TIFF
                # page even though its logical series has T*C planes.  Such
                # stacks are memmappable, allowing us to sample only the
                # selected planes for a thumbnail instead of materializing a
                # multi-gigabyte array or incorrectly falling back to
                # OpenSlide (which cannot open these files).
                try:
                    mapped = tifffile.memmap(str(image_path), series=0, mode="r")
                except Exception as error:
                    raise ValueError(
                        f"TIFF has {channel_count} logical channels but only {len(level.pages)} pages, "
                        f"and is not a readable contiguous hyperstack: {image_path}"
                    ) from error
                if tuple(mapped.shape) != tuple(level.shape):
                    raise ValueError(
                        f"Contiguous TIFF shape {tuple(mapped.shape)} does not match series shape "
                        f"{tuple(level.shape)}: {image_path}"
                    )
                channel_shape = tuple(int(level.shape[index]) for index in channel_axes)
                step = max(1, int(math.ceil(max(level_shapes[level_index]) / float(thumbnail_max_size))))

                def read_mapped_plane(source_index: int) -> np.ndarray:
                    channel_coordinate = np.unravel_index(int(source_index), channel_shape)
                    selection: List[Any] = [0] * mapped.ndim
                    for axis, coordinate in zip(channel_axes, channel_coordinate):
                        selection[axis] = int(coordinate)
                    # Preserve a non-channel Z axis long enough to max-project
                    # it, which also handles contiguous TCZYX stacks.
                    for axis, name in enumerate(axes):
                        if name == "Z" and axis not in channel_axes:
                            selection[axis] = slice(None)
                    selection[y_axis] = slice(None, None, step)
                    selection[x_axis] = slice(None, None, step)
                    plane = np.asarray(mapped[tuple(selection)])
                    while plane.ndim > 2:
                        plane = np.max(plane, axis=0)
                    return plane

                normalized_planes = [
                    _normalize_thumbnail_u8(read_mapped_plane(source_index))
                    for source_index in selected_source_indices
                ]
        else:
            array = np.asarray(level.asarray())
            if not channel_axes:
                moved = np.moveaxis(array, [y_axis, x_axis], [-2, -1])
                plane = np.max(moved, axis=tuple(range(moved.ndim - 2))) if moved.ndim > 2 else moved
                normalized_planes = [_normalize_thumbnail_u8(plane)]
            else:
                destination_axes = list(range(len(channel_axes))) + [-2, -1]
                moved = np.moveaxis(array, channel_axes + [y_axis, x_axis], destination_axes)
                reduction_axes = tuple(range(len(channel_axes), moved.ndim - 2))
                projected = np.max(moved, axis=reduction_axes) if reduction_axes else moved
                projected = projected.reshape(channel_count, projected.shape[-2], projected.shape[-1])
                normalized_planes = [_normalize_thumbnail_u8(projected[index]) for index in range(projected.shape[0])]
            selected_source_indices = list(range(len(normalized_planes)))
        if not normalized_planes:
            raise ValueError(f"OME-TIFF has no readable marker planes: {image_path}")
        usable_indices = list(range(len(normalized_planes)))
        technical_indices: List[int] = []
        nucleus_indices: List[int] = []
        if channel_names is not None and len(channel_names) == channel_count:
            for index, source_index in enumerate(selected_source_indices):
                channel_name = channel_names[source_index]
                text = str(channel_name).lower()
                is_af = bool(re.search(r"(^|[_\s-])af\d*($|[_\s-])", text)) or "autofluorescence" in text
                # CyCIF panels label the non-biological reference exposures
                # as Control-<wavelength>.  Including them in a marker fusion
                # turns their scattered field/background signal into apparent
                # tissue, just like AF and Blank channels do in Orion panels.
                if is_af or "blank" in text or "control" in text:
                    technical_indices.append(index)
                # Treat common nuclear labels as a nucleus channel for optional
                # support gating.
                if any(token in text for token in ("nucleus", "hoechst", "dapi", "dna")):
                    nucleus_indices.append(index)
            usable_indices = [index for index in usable_indices if index not in technical_indices]
        if not usable_indices:
            usable_indices = list(range(len(normalized_planes)))
            technical_indices = []
        stack = np.stack([normalized_planes[index] for index in usable_indices], axis=0)
        if force_median_marker_fusion or (technical_indices and technical_channel_fusion == "median"):
            composite = np.median(stack, axis=0).astype(np.uint8)
            fusion_method = "median_all_markers" if force_median_marker_fusion else "median_biological_markers"
        else:
            composite = np.percentile(stack, float(fusion_percentile), axis=0).astype(np.uint8)
            fusion_method = (
                "percentile_biological_markers" if technical_indices else "percentile_all_markers"
            )
        source_shape_yx = level_shapes[0]

    thumbnail_rgb = np.repeat(composite[..., np.newaxis], 3, axis=2)
    # Several cyclic CODEX panels repeat DNA after every marker round. Their
    # individual DNA planes can vary in local quality, so use their union as
    # the tissue-support image instead of choosing the first one.
    nucleus_thumbnail_rgb = (
        np.repeat(
            np.max(np.stack([normalized_planes[index] for index in nucleus_indices], axis=0), axis=0)[..., np.newaxis],
            3,
            axis=2,
        )
        if nucleus_indices and (technical_indices or include_nucleus_thumbnail)
        else None
    )
    return thumbnail_rgb, source_shape_yx, level_index, {
        "thumbnail_fusion_method": fusion_method,
        "thumbnail_fusion_percentile": float(fusion_percentile) if fusion_method.startswith("percentile") else None,
        "thumbnail_marker_count": int(len(normalized_planes)),
        "thumbnail_total_marker_count": int(channel_count),
        "thumbnail_biological_marker_count": int(len(usable_indices)),
        "thumbnail_source_channel_indices": [int(index) for index in selected_source_indices],
        "thumbnail_read_strategy": "tiled_page_downsample" if use_tiled_page_thumbnail else "pyramid_level",
        "thumbnail_technical_channel_indices": technical_indices,
        "thumbnail_nucleus_channel_indices": [int(index) for index in nucleus_indices],
    }, nucleus_thumbnail_rgb


def _patch_coords_from_foreground_mask(
    mask: np.ndarray,
    *,
    spatial_shape_yx: Tuple[int, int],
    patch_size: int,
    overlap: int,
    min_tissue_proportion: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    height, width = int(spatial_shape_yx[0]), int(spatial_shape_yx[1])
    mask_bool = (np.asarray(mask) > 0).astype(np.uint8)
    mask_h, mask_w = mask_bool.shape
    if mask_h == 0 or mask_w == 0:
        return np.zeros((0, 2), dtype=np.int64), {"mask_width": mask_w, "mask_height": mask_h}

    step = int(patch_size) - int(overlap)
    if step <= 0:
        raise ValueError(f"overlap must be smaller than patch_size, got patch_size={patch_size}, overlap={overlap}")
    scale_y = height / float(mask_h)
    scale_x = width / float(mask_w)
    integral = np.pad(mask_bool.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")

    def region_fraction(x: int, y: int) -> float:
        tx0 = max(0, min(mask_w, int(math.floor(x / scale_x))))
        ty0 = max(0, min(mask_h, int(math.floor(y / scale_y))))
        tx1 = max(tx0 + 1, min(mask_w, int(math.ceil((x + patch_size) / scale_x))))
        ty1 = max(ty0 + 1, min(mask_h, int(math.ceil((y + patch_size) / scale_y))))
        area = max(1, (tx1 - tx0) * (ty1 - ty0))
        total = integral[ty1, tx1] - integral[ty0, tx1] - integral[ty1, tx0] + integral[ty0, tx0]
        return float(total) / float(area)

    x_stop = max(0, width - int(patch_size))
    y_stop = max(0, height - int(patch_size))
    coords: List[Tuple[int, int]] = []
    for y in range(0, y_stop + 1, step):
        for x in range(0, x_stop + 1, step):
            if region_fraction(x, y) >= float(min_tissue_proportion):
                coords.append((x, y))
    attrs = {
        "mask_width": mask_w,
        "mask_height": mask_h,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "step": step,
        "min_tissue_proportion": min_tissue_proportion,
    }
    return np.asarray(coords, dtype=np.int64).reshape(-1, 2), attrs


def _save_native_foreground_qc(
    *,
    trident_root: str | Path,
    job_dir: Path,
    slide_name: str,
    thumbnail_rgb: np.ndarray,
    mask: np.ndarray,
    contour_scale: float,
    mpp: float,
    min_contour_area: float,
) -> None:
    _ensure_trident_import(trident_root)
    from PIL import Image
    from trident.IO import mask_to_gdf, overlay_gdf_on_thumbnail

    thumbnails_dir = job_dir / "thumbnails"
    contours_dir = job_dir / "contours"
    geojson_dir = job_dir / "contours_geojson"
    mask_dir = job_dir / "foreground_masks"
    for directory in [thumbnails_dir, contours_dir, geojson_dir, mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    thumbnail = Image.fromarray(thumbnail_rgb.astype(np.uint8)).convert("RGB")
    thumbnail.save(thumbnails_dir / f"{slide_name}.jpg")
    Image.fromarray(mask.astype(np.uint8)).save(mask_dir / f"{slide_name}.png")

    gdf_contours = mask_to_gdf(
        mask=mask,
        max_nb_holes=0,
        min_contour_area=float(min_contour_area),
        pixel_size=float(mpp),
        contour_scale=float(contour_scale),
    )
    geojson_path = geojson_dir / f"{slide_name}.geojson"
    try:
        gdf_contours.set_crs("EPSG:3857", inplace=True)
        gdf_contours.to_file(geojson_path, driver="GeoJSON")
    except Exception:
        geojson_path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")

    if len(gdf_contours) == 0:
        thumbnail.save(contours_dir / f"{slide_name}.jpg")
    else:
        annotated = thumbnail_rgb.copy()
        overlay_gdf_on_thumbnail(
            gdf_contours,
            annotated,
            str(contours_dir / f"{slide_name}.jpg"),
            thumbnail_rgb.shape[1] / (thumbnail_rgb.shape[1] * contour_scale),
        )


def _extract_native_ims_sp_fluorescence_coords(
    image_path: str | Path,
    *,
    trident_root: str | Path,
    job_dir: str | Path,
    data_root: str | Path | None,
    reader_type: str,
    dataset: str,
    patch_size: int,
    mag: int,
    raw_mpp: Optional[float],
    normalized_mpp: float,
    overlap: int,
    channel_names: Sequence[str],
    marker_names: Sequence[str],
    min_tissue_proportion: float,
    thumbnail_max_size: int,
    threshold_percentile: float,
    min_signal: float,
    blur_sigma: float,
    close_radius: int,
    open_radius: int,
    dilate_radius: int,
    min_component_area_fraction: float,
    min_contour_area: float,
    fusion_percentile: float,
    max_hole_area_fraction: float,
    forced_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]],
) -> Path:
    thumbnail_rgb, spatial_shape_yx, thumbnail_level = _read_ims_projection_thumbnail(
        image_path,
        thumbnail_max_size=thumbnail_max_size,
        fusion_percentile=fusion_percentile,
    )
    mask, mask_attrs = make_sp_fluorescence_foreground(
        thumbnail_rgb,
        threshold_percentile=threshold_percentile,
        min_signal=min_signal,
        blur_sigma=blur_sigma,
        close_radius=close_radius,
        open_radius=open_radius,
        dilate_radius=dilate_radius,
        min_component_area_fraction=min_component_area_fraction,
        max_hole_area_fraction=max_hole_area_fraction,
        forced_thumbnail_regions=forced_thumbnail_regions,
    )
    coords_xy, coord_attrs = _patch_coords_from_foreground_mask(
        mask,
        spatial_shape_yx=spatial_shape_yx,
        patch_size=patch_size,
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
    )

    job_dir = Path(job_dir)
    slide_name = _safe_file_stem(image_path)
    contour_scale = spatial_shape_yx[1] / float(thumbnail_rgb.shape[1])
    _save_native_foreground_qc(
        trident_root=trident_root,
        job_dir=job_dir,
        slide_name=slide_name,
        thumbnail_rgb=thumbnail_rgb,
        mask=mask,
        contour_scale=contour_scale,
        mpp=normalized_mpp,
        min_contour_area=min_contour_area,
    )

    mag_str = f"{float(mag):g}"
    coords_h5 = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap" / "patches" / f"{slide_name}_patches.h5"
    coords_h5 = write_sp_coords_h5(
        coords_h5,
        coords_xy,
        source_path=str(image_path),
        reader_type=reader_type,
        dataset=dataset,
        spatial_shape_yx=spatial_shape_yx,
        patch_size=patch_size,
        overlap=overlap,
        channel_names=channel_names,
        marker_names=marker_names,
        foreground_method="sp_fluorescence_native_ims",
        foreground_attrs={
            **mask_attrs,
            **coord_attrs,
            "mag": mag,
            "raw_mpp": raw_mpp,
            "mpp": normalized_mpp,
            "thumbnail_size": list(thumbnail_rgb.shape[:2]),
            "thumbnail_resolution_level": thumbnail_level,
            "thumbnail_fusion": "per_channel_robust_normalized_percentile",
            "thumbnail_fusion_percentile": fusion_percentile,
            "native_reader": "ims",
        },
    )
    with h5py.File(coords_h5, "a") as handle:
        handle["coords"].attrs["level0_magnification"] = int(mag)
        handle["coords"].attrs["target_magnification"] = int(mag)
    return Path(coords_h5)


def _extract_native_ome_tiff_sp_fluorescence_coords(
    image_path: str | Path,
    *,
    trident_root: str | Path,
    job_dir: str | Path,
    data_root: str | Path | None,
    reader_type: str,
    dataset: str,
    patch_size: int,
    mag: int,
    raw_mpp: Optional[float],
    normalized_mpp: float,
    overlap: int,
    channel_names: Sequence[str],
    marker_names: Sequence[str],
    min_tissue_proportion: float,
    thumbnail_max_size: int,
    threshold_percentile: float,
    min_signal: float,
    blur_sigma: float,
    close_radius: int,
    open_radius: int,
    dilate_radius: int,
    min_component_area_fraction: float,
    min_contour_area: float,
    fusion_percentile: float,
    max_hole_area_fraction: float,
    max_fusion_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    max_fusion_threshold_percentile: float = 35.0,
    max_fusion_min_signal: float = 2.0,
    max_fusion_min_component_area_fraction: float = 0.0001,
    max_fusion_replace_existing_mask: bool = False,
    support_recovery_fusion_percentile: Optional[float] = None,
    support_recovery_threshold_percentile: float = 20.0,
    support_recovery_min_signal: float = 3.0,
    support_recovery_dilation_radius: int = 40,
    excluded_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    excluded_thumbnail_polygons: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    roi_recovery_uses_base_fusion: bool = False,
    strict_threshold_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    strict_threshold_percentile: float = 70.0,
    strict_min_signal: float = 8.0,
    strict_min_component_area_fraction: Optional[float] = None,
    post_strict_recovery_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    post_strict_recovery_fusion_percentile: float = 75.0,
    post_strict_recovery_threshold_percentile: float = 60.0,
    post_strict_recovery_min_signal: float = 6.0,
    post_strict_recovery_close_radius: int = 4,
    post_strict_recovery_dilate_radius: int = 3,
    component_concavity_fill_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    local_background_resegmentation_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    local_background_resegmentation_threshold_percentile: float = 55.0,
    local_background_resegmentation_min_signal: float = 6.0,
    forced_foreground_polygons: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    final_forced_foreground_polygons: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    technical_channel_fusion: str = "percentile",
    include_nucleus_thumbnail: bool = False,
    treat_z_as_channels: bool = False,
    flatten_nonspatial_axes_as_channels: bool = False,
    max_thumbnail_marker_planes: int = 0,
    force_median_marker_fusion: bool = False,
    remove_border_frame_artifacts: bool = False,
    border_frame_max_density: float = 0.30,
    border_frame_min_span_fraction: float = 0.65,
    remove_border_strip_artifacts: bool = False,
    nucleus_gate_threshold_percentile: float = 70.0,
    nucleus_gate_dilation_radius: int = 25,
    apply_nucleus_seed_gate: bool = True,
    remove_small_edge_artifacts: bool = False,
    small_edge_max_area_fraction: float = 0.005,
    small_edge_margin_fraction: float = 0.15,
    remove_sparse_peripheral_artifacts: bool = False,
    sparse_peripheral_max_area_fraction: float = 0.03,
    sparse_peripheral_max_density: float = 0.35,
    sparse_peripheral_edge_margin_fraction: float = 0.15,
    remove_thin_grid_artifacts: bool = False,
    thin_grid_min_aspect_ratio: float = 5.0,
    thin_grid_max_thickness_fraction: float = 0.025,
    thin_grid_max_area_fraction: float = 0.008,
    qc_fusion_percentile: Optional[float] = None,
    edge_cleanup_fusion_percentile: Optional[float] = None,
    edge_cleanup_threshold_percentile: float = 45.0,
    edge_cleanup_min_signal: float = 8.0,
    edge_cleanup_margin_fraction: float = 0.08,
    dense_foreground_rescue: bool = False,
    dense_foreground_rescue_radius: int = 20,
    dense_foreground_rescue_min_density: float = 0.35,
) -> Path:
    """Native foreground extraction for compressed multichannel OME-TIFF WSI."""
    thumbnail_rgb, spatial_shape_yx, thumbnail_level, thumbnail_attrs, nucleus_thumbnail_rgb = _read_ome_tiff_projection_thumbnail(
        image_path,
        thumbnail_max_size=thumbnail_max_size,
        fusion_percentile=fusion_percentile,
        channel_names=channel_names,
        technical_channel_fusion=technical_channel_fusion,
        force_median_marker_fusion=force_median_marker_fusion,
        include_nucleus_thumbnail=include_nucleus_thumbnail,
        treat_z_as_channels=treat_z_as_channels,
        flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
        max_thumbnail_marker_planes=max_thumbnail_marker_planes,
    )
    mask, mask_attrs = make_sp_fluorescence_foreground(
        thumbnail_rgb,
        threshold_percentile=threshold_percentile,
        min_signal=min_signal,
        blur_sigma=blur_sigma,
        close_radius=close_radius,
        open_radius=open_radius,
        dilate_radius=dilate_radius,
        min_component_area_fraction=min_component_area_fraction,
        max_hole_area_fraction=max_hole_area_fraction,
        remove_border_frame_artifacts=remove_border_frame_artifacts,
        border_frame_max_density=border_frame_max_density,
        border_frame_min_span_fraction=border_frame_min_span_fraction,
        remove_border_strip_artifacts=remove_border_strip_artifacts,
    )
    base_mask = mask > 0
    nucleus_gate_applied = False
    dense_foreground_pixels_added = 0
    if apply_nucleus_seed_gate and nucleus_thumbnail_rgb is not None:
        import cv2

        nucleus_mask, _ = make_sp_fluorescence_foreground(
            nucleus_thumbnail_rgb,
            threshold_percentile=max(float(threshold_percentile), float(nucleus_gate_threshold_percentile)),
            min_signal=min_signal,
            blur_sigma=blur_sigma,
            close_radius=close_radius,
            open_radius=0,
            dilate_radius=dilate_radius,
            min_component_area_fraction=min_component_area_fraction,
            max_hole_area_fraction=max_hole_area_fraction,
        )
        nucleus_gate_dilation_radius = max(0, int(nucleus_gate_dilation_radius))
        kernel_size = 2 * nucleus_gate_dilation_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        nearby_nuclei = cv2.dilate((nucleus_mask > 0).astype(np.uint8), kernel) > 0
        gated = base_mask & nearby_nuclei
        if dense_foreground_rescue:
            rescue_radius = max(0, int(dense_foreground_rescue_radius))
            kernel_size = 2 * rescue_radius + 1
            local_density = cv2.blur(base_mask.astype(np.float32), (kernel_size, kernel_size))
            dense_foreground = base_mask & (local_density >= float(dense_foreground_rescue_min_density))
            dense_foreground_pixels_added = int((dense_foreground & ~gated).sum())
            gated |= dense_foreground
        min_area = max(1, int(round(gated.size * float(min_component_area_fraction))))
        gated, _ = _filter_connected_components(gated, min_area=min_area)
        mask = (gated * 255).astype(np.uint8)
        nucleus_gate_applied = True
    max_fusion_recovery = []
    if max_fusion_thumbnail_regions:
        if roi_recovery_uses_base_fusion:
            max_thumbnail_rgb = thumbnail_rgb
        else:
            max_thumbnail_rgb, _, _, _, _ = _read_ome_tiff_projection_thumbnail(
                image_path,
                thumbnail_max_size=thumbnail_max_size,
                fusion_percentile=100.0,
                channel_names=channel_names,
                technical_channel_fusion="percentile",
                force_median_marker_fusion=False,
                include_nucleus_thumbnail=False,
                treat_z_as_channels=treat_z_as_channels,
                flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
                max_thumbnail_marker_planes=max_thumbnail_marker_planes,
            )
        mask, max_fusion_recovery = add_roi_max_fusion_foreground(
            mask,
            max_thumbnail_rgb,
            regions=max_fusion_thumbnail_regions,
            threshold_percentile=max_fusion_threshold_percentile,
            min_signal=max_fusion_min_signal,
            blur_sigma=2.0,
            # A large closing can bridge tile seams into an ROI-spanning ring
            # whose hole-fill looks like a rectangular foreground block.
            close_radius=3,
            open_radius=0,
            dilate_radius=2,
            min_component_area_fraction=max_fusion_min_component_area_fraction,
            max_hole_area_fraction=0.0,
            replace_existing_mask=max_fusion_replace_existing_mask,
        )
    support_recovery_pixels_added = 0
    if support_recovery_fusion_percentile is not None:
        # Keep the conservative primary mask as the spatial anchor, then use
        # a higher marker quantile only to recover nearby weak tissue. This
        # is useful for Orion: its median biological-marker fusion is clean
        # but can omit tissue expressed in a small subset of the panel.
        import cv2

        recovery_thumbnail_rgb, _, _, _, _ = _read_ome_tiff_projection_thumbnail(
            image_path,
            thumbnail_max_size=thumbnail_max_size,
            fusion_percentile=float(support_recovery_fusion_percentile),
            channel_names=channel_names,
            technical_channel_fusion="percentile",
            force_median_marker_fusion=False,
            include_nucleus_thumbnail=False,
            treat_z_as_channels=treat_z_as_channels,
            flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
        )
        recovery_mask, _ = make_sp_fluorescence_foreground(
            recovery_thumbnail_rgb,
            threshold_percentile=support_recovery_threshold_percentile,
            min_signal=support_recovery_min_signal,
            blur_sigma=blur_sigma,
            close_radius=close_radius,
            open_radius=0,
            dilate_radius=dilate_radius,
            min_component_area_fraction=min_component_area_fraction,
            max_hole_area_fraction=max_hole_area_fraction,
        )
        radius = max(0, int(support_recovery_dilation_radius))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        anchor = cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0
        recovered = (recovery_mask > 0) & anchor & ~(mask > 0)
        support_recovery_pixels_added = int(recovered.sum())
        mask = (((mask > 0) | recovered).astype(np.uint8) * 255)
    applied_strict_regions = []
    if strict_threshold_regions:
        # Apply this after any weak-tissue recovery so the final explicit
        # region cannot be repopulated by a low-signal compensation pass.
        strict_mask, _ = make_sp_fluorescence_foreground(
            thumbnail_rgb,
            threshold_percentile=strict_threshold_percentile,
            min_signal=strict_min_signal,
            blur_sigma=blur_sigma,
            close_radius=close_radius,
            open_radius=open_radius,
            dilate_radius=dilate_radius,
            min_component_area_fraction=(
                min_component_area_fraction
                if strict_min_component_area_fraction is None
                else strict_min_component_area_fraction
            ),
            max_hole_area_fraction=max_hole_area_fraction,
        )
        height, width = mask.shape
        for region in strict_threshold_regions:
            x0, y0, x1, y1 = (float(value) for value in region)
            x0_px = max(0, min(width, int(round(x0 * width))))
            x1_px = max(0, min(width, int(round(x1 * width))))
            y0_px = max(0, min(height, int(round(y0 * height))))
            y1_px = max(0, min(height, int(round(y1 * height))))
            if x1_px > x0_px and y1_px > y0_px:
                mask[y0_px:y1_px, x0_px:x1_px] = strict_mask[y0_px:y1_px, x0_px:x1_px]
                applied_strict_regions.append([x0, y0, x1, y1])
    post_strict_recovery = []
    component_concavity_fill = []
    local_background_resegmentation = []
    applied_forced_foreground_polygons = []
    if post_strict_recovery_regions or local_background_resegmentation_regions or forced_foreground_polygons:
        # User-supplied weak tissue is restored only after strict cleanup, so
        # the recovery cannot be erased by an image-wide artifact threshold.
        recovery_thumbnail_rgb, _, _, _, _ = _read_ome_tiff_projection_thumbnail(
            image_path,
            thumbnail_max_size=thumbnail_max_size,
            fusion_percentile=post_strict_recovery_fusion_percentile,
            channel_names=channel_names,
            technical_channel_fusion="percentile",
            force_median_marker_fusion=False,
            include_nucleus_thumbnail=False,
            treat_z_as_channels=treat_z_as_channels,
            flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
        )
        mask, post_strict_recovery = add_roi_max_fusion_foreground(
            mask,
            recovery_thumbnail_rgb,
            regions=post_strict_recovery_regions,
            threshold_percentile=post_strict_recovery_threshold_percentile,
            min_signal=post_strict_recovery_min_signal,
            blur_sigma=2.0,
            # Do not close/fill across the boundary of a weak-tissue ROI.
            # The prior 17-pixel closing plus unrestricted hole filling could
            # turn a broadly positive ROI into a rectangular foreground slab.
            # A small closing only joins adjacent real signal; hole filling is
            # limited to single pixels so the crop boundary remains background.
            close_radius=post_strict_recovery_close_radius,
            open_radius=0,
            dilate_radius=post_strict_recovery_dilate_radius,
            min_component_area_fraction=0.0001,
            max_hole_area_fraction=0.0,
        )
        if component_concavity_fill_regions:
            mask, component_concavity_fill = fill_roi_component_concavities(
                mask,
                regions=component_concavity_fill_regions,
                min_component_area_fraction=0.01,
            )
        if local_background_resegmentation_regions:
            mask, local_background_resegmentation = replace_roi_max_fusion_foreground(
                mask,
                recovery_thumbnail_rgb,
                regions=local_background_resegmentation_regions,
                threshold_percentile=local_background_resegmentation_threshold_percentile,
                min_signal=local_background_resegmentation_min_signal,
                blur_sigma=2.0,
                close_radius=3,
                open_radius=0,
                dilate_radius=2,
                min_component_area_fraction=0.0001,
                max_hole_area_fraction=0.0,
            )
        if forced_foreground_polygons:
            mask, applied_forced_foreground_polygons = force_thumbnail_polygons(
                mask,
                polygons=forced_foreground_polygons,
            )
    applied_excluded_regions = []
    if excluded_thumbnail_regions:
        # Apply explicit exclusions after every recovery pass so a local
        # rescue cannot reintroduce a known scanner-residue region.
        height, width = mask.shape
        for region in excluded_thumbnail_regions:
            x0, y0, x1, y1 = (float(value) for value in region)
            x0_px = max(0, min(width, int(round(x0 * width))))
            x1_px = max(0, min(width, int(round(x1 * width))))
            y0_px = max(0, min(height, int(round(y0 * height))))
            y1_px = max(0, min(height, int(round(y1 * height))))
            if x1_px > x0_px and y1_px > y0_px:
                mask[y0_px:y1_px, x0_px:x1_px] = 0
                applied_excluded_regions.append([x0, y0, x1, y1])
    applied_excluded_polygons = []
    if excluded_thumbnail_polygons:
        # Polygonal exclusions preserve a slanted tissue/artifact boundary
        # that would be incorrectly removed by an axis-aligned rectangle.
        import cv2

        binary = (mask > 0).astype(np.uint8)
        height, width = binary.shape
        for polygon in excluded_thumbnail_polygons:
            points = np.asarray(
                [
                    [
                        max(0, min(width - 1, round(float(x) * width))),
                        max(0, min(height - 1, round(float(y) * height))),
                    ]
                    for x, y in polygon
                ],
                dtype=np.int32,
            )
            if len(points) >= 3:
                cv2.fillPoly(binary, [points], 0)
                applied_excluded_polygons.append([[float(x), float(y)] for x, y in polygon])
        mask = (binary * 255).astype(np.uint8)
    edge_cleanup_pixels_removed = 0
    if edge_cleanup_fusion_percentile is not None:
        # A max-marker fusion can reveal weak stroma, but it may also light up
        # a stitched scan frame. Re-segment only the
        # outer band with the conservative fusion and subtract unsupported
        # max-fusion pixels; this never force-adds a border rectangle.
        conservative_thumbnail_rgb, _, _, _, _ = _read_ome_tiff_projection_thumbnail(
            image_path,
            thumbnail_max_size=thumbnail_max_size,
            fusion_percentile=float(edge_cleanup_fusion_percentile),
            channel_names=channel_names,
            technical_channel_fusion="percentile",
            force_median_marker_fusion=False,
            include_nucleus_thumbnail=False,
            treat_z_as_channels=treat_z_as_channels,
            flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
        )
        conservative_mask, _ = make_sp_fluorescence_foreground(
            conservative_thumbnail_rgb,
            threshold_percentile=edge_cleanup_threshold_percentile,
            min_signal=edge_cleanup_min_signal,
            blur_sigma=blur_sigma,
            close_radius=close_radius,
            open_radius=open_radius,
            dilate_radius=dilate_radius,
            min_component_area_fraction=min_component_area_fraction,
            max_hole_area_fraction=max_hole_area_fraction,
            remove_border_frame_artifacts=True,
            remove_border_strip_artifacts=True,
        )
        base = mask > 0
        conservative = conservative_mask > 0
        height, width = base.shape
        margin_y = max(1, int(round(height * float(edge_cleanup_margin_fraction))))
        margin_x = max(1, int(round(width * float(edge_cleanup_margin_fraction))))
        edge = np.zeros_like(base, dtype=bool)
        edge[:margin_y, :] = True
        edge[-margin_y:, :] = True
        edge[:, :margin_x] = True
        edge[:, -margin_x:] = True
        removed = base & edge & ~conservative
        edge_cleanup_pixels_removed = int(removed.sum())
        mask = ((base & ~removed).astype(np.uint8) * 255)
    mask_attrs["max_fusion_recovery"] = max_fusion_recovery
    mask_attrs["roi_recovery_uses_base_fusion"] = bool(roi_recovery_uses_base_fusion)
    mask_attrs["strict_threshold_regions"] = applied_strict_regions
    mask_attrs["strict_threshold_percentile"] = float(strict_threshold_percentile)
    mask_attrs["strict_min_component_area_fraction"] = strict_min_component_area_fraction
    mask_attrs["post_strict_recovery"] = post_strict_recovery
    mask_attrs["component_concavity_fill"] = component_concavity_fill
    mask_attrs["local_background_resegmentation"] = local_background_resegmentation
    mask_attrs["forced_foreground_polygons"] = applied_forced_foreground_polygons
    mask_attrs["support_recovery_fusion_percentile"] = support_recovery_fusion_percentile
    mask_attrs["support_recovery_dilation_radius"] = int(support_recovery_dilation_radius)
    mask_attrs["support_recovery_pixels_added"] = support_recovery_pixels_added
    mask_attrs["excluded_thumbnail_regions"] = applied_excluded_regions
    mask_attrs["excluded_thumbnail_polygons"] = applied_excluded_polygons
    mask_attrs["edge_cleanup_fusion_percentile"] = edge_cleanup_fusion_percentile
    mask_attrs["edge_cleanup_margin_fraction"] = float(edge_cleanup_margin_fraction)
    mask_attrs["edge_cleanup_pixels_removed"] = edge_cleanup_pixels_removed
    mask_attrs["nucleus_seed_gate_applied"] = nucleus_gate_applied
    mask_attrs["remove_border_frame_artifacts"] = bool(remove_border_frame_artifacts)
    mask_attrs["border_frame_max_density"] = float(border_frame_max_density)
    mask_attrs["border_frame_min_span_fraction"] = float(border_frame_min_span_fraction)
    mask_attrs["remove_border_strip_artifacts"] = bool(remove_border_strip_artifacts)
    mask_attrs["nucleus_seed_gate_threshold_percentile"] = (
        float(max(float(threshold_percentile), float(nucleus_gate_threshold_percentile)))
        if nucleus_gate_applied
        else None
    )
    mask_attrs["nucleus_seed_gate_dilation_radius"] = nucleus_gate_dilation_radius if nucleus_gate_applied else None
    mask_attrs["dense_foreground_rescue"] = bool(dense_foreground_rescue)
    mask_attrs["dense_foreground_rescue_radius"] = int(dense_foreground_rescue_radius)
    mask_attrs["dense_foreground_rescue_min_density"] = float(dense_foreground_rescue_min_density)
    mask_attrs["dense_foreground_pixels_added"] = int(dense_foreground_pixels_added)
    small_edge_components_removed = 0
    if remove_small_edge_artifacts:
        filtered_mask, small_edge_components_removed = _remove_small_edge_components(
            (mask > 0).astype(np.uint8),
            max_area_fraction=small_edge_max_area_fraction,
            edge_margin_fraction=small_edge_margin_fraction,
        )
        mask = (filtered_mask * 255).astype(np.uint8)
    mask_attrs["remove_small_edge_artifacts"] = bool(remove_small_edge_artifacts)
    mask_attrs["small_edge_components_removed"] = int(small_edge_components_removed)
    mask_attrs["small_edge_max_area_fraction"] = float(small_edge_max_area_fraction)
    mask_attrs["small_edge_margin_fraction"] = float(small_edge_margin_fraction)
    sparse_peripheral_components_removed = 0
    if remove_sparse_peripheral_artifacts:
        filtered_mask, sparse_peripheral_components_removed = _remove_sparse_peripheral_components(
            (mask > 0).astype(np.uint8),
            max_area_fraction=sparse_peripheral_max_area_fraction,
            max_density=sparse_peripheral_max_density,
            edge_margin_fraction=sparse_peripheral_edge_margin_fraction,
        )
        mask = (filtered_mask * 255).astype(np.uint8)
    mask_attrs["remove_sparse_peripheral_artifacts"] = bool(remove_sparse_peripheral_artifacts)
    mask_attrs["sparse_peripheral_components_removed"] = int(sparse_peripheral_components_removed)
    mask_attrs["sparse_peripheral_max_area_fraction"] = float(sparse_peripheral_max_area_fraction)
    mask_attrs["sparse_peripheral_max_density"] = float(sparse_peripheral_max_density)
    mask_attrs["sparse_peripheral_edge_margin_fraction"] = float(sparse_peripheral_edge_margin_fraction)
    thin_grid_components_removed = 0
    if remove_thin_grid_artifacts:
        filtered_mask, thin_grid_components_removed = _remove_thin_elongated_components(
            (mask > 0).astype(np.uint8),
            min_aspect_ratio=thin_grid_min_aspect_ratio,
            max_thickness_fraction=thin_grid_max_thickness_fraction,
            max_area_fraction=thin_grid_max_area_fraction,
        )
        mask = (filtered_mask * 255).astype(np.uint8)
    mask_attrs["remove_thin_grid_artifacts"] = bool(remove_thin_grid_artifacts)
    mask_attrs["thin_grid_components_removed"] = int(thin_grid_components_removed)
    mask_attrs["thin_grid_min_aspect_ratio"] = float(thin_grid_min_aspect_ratio)
    mask_attrs["thin_grid_max_thickness_fraction"] = float(thin_grid_max_thickness_fraction)
    mask_attrs["thin_grid_max_area_fraction"] = float(thin_grid_max_area_fraction)
    final_forced_foreground = []
    if final_forced_foreground_polygons:
        # Apply after every artifact-removal stage. This is reserved for
        # explicitly supplied all-tissue regions whose foreground status must
        # not be undone by generic cleanup.
        mask, final_forced_foreground = force_thumbnail_polygons(
            mask,
            polygons=final_forced_foreground_polygons,
        )
    mask_attrs["final_forced_foreground_polygons"] = final_forced_foreground
    coords_xy, coord_attrs = _patch_coords_from_foreground_mask(
        mask,
        spatial_shape_yx=spatial_shape_yx,
        patch_size=patch_size,
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
    )

    job_dir = Path(job_dir)
    slide_name = _safe_file_stem(image_path)
    qc_thumbnail_rgb = thumbnail_rgb
    if qc_fusion_percentile is not None and float(qc_fusion_percentile) != float(fusion_percentile):
        # Keep the segmentation fusion conservative enough to reject the
        # one-marker grid extrema, while displaying the maximum fusion for
        # QC on sparse-marker tissue.  This changes neither the mask nor the
        # patch coordinates.
        qc_thumbnail_rgb, _, _, qc_thumbnail_attrs, _ = _read_ome_tiff_projection_thumbnail(
            image_path,
            thumbnail_max_size=thumbnail_max_size,
            fusion_percentile=float(qc_fusion_percentile),
            channel_names=channel_names,
            technical_channel_fusion=technical_channel_fusion,
            force_median_marker_fusion=force_median_marker_fusion,
            include_nucleus_thumbnail=False,
            treat_z_as_channels=treat_z_as_channels,
            flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
        )
        thumbnail_attrs["qc_fusion_percentile"] = float(qc_fusion_percentile)
        thumbnail_attrs["qc_thumbnail_reader"] = qc_thumbnail_attrs.get("native_reader")
    contour_scale = spatial_shape_yx[1] / float(qc_thumbnail_rgb.shape[1])
    _save_native_foreground_qc(
        trident_root=trident_root,
        job_dir=job_dir,
        slide_name=slide_name,
        thumbnail_rgb=qc_thumbnail_rgb,
        mask=mask,
        contour_scale=contour_scale,
        mpp=normalized_mpp,
        min_contour_area=min_contour_area,
    )

    mag_str = f"{float(mag):g}"
    coords_h5 = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap" / "patches" / f"{slide_name}_patches.h5"
    coords_h5 = write_sp_coords_h5(
        coords_h5,
        coords_xy,
        source_path=str(image_path),
        reader_type=reader_type,
        dataset=dataset,
        spatial_shape_yx=spatial_shape_yx,
        patch_size=patch_size,
        overlap=overlap,
        channel_names=channel_names,
        marker_names=marker_names,
        foreground_method="sp_fluorescence_native_ome_tiff",
        foreground_attrs={
            **mask_attrs,
            **coord_attrs,
            "mag": mag,
            "raw_mpp": raw_mpp,
            "mpp": normalized_mpp,
            "thumbnail_size": list(qc_thumbnail_rgb.shape[:2]),
            "thumbnail_resolution_level": thumbnail_level,
            **thumbnail_attrs,
            "native_reader": "ome_tiff",
        },
    )
    with h5py.File(coords_h5, "a") as handle:
        handle["coords"].attrs["level0_magnification"] = int(mag)
        handle["coords"].attrs["target_magnification"] = int(mag)
    return Path(coords_h5)


def extract_trident_coords(
    image_path: str | Path,
    *,
    trident_root: str | Path,
    job_dir: str | Path,
    marker_registry_path: str | Path,
    data_root: str | Path | None,
    reader_type: str,
    dataset: str,
    patch_size: int = 224,
    mag: Optional[int] = None,
    segmenter: str = "hest",
    seg_conf_thresh: float = 0.5,
    gpu: int = 0,
    remove_holes: bool = False,
    overlap: int = 0,
    trident_reader_type: Optional[str] = None,
    channel_names_override: Optional[Sequence[str]] = None,
    mpp: Optional[float] = None,
    min_tissue_proportion: float = 0.0,
) -> Path:
    """Run TRIDENT segmentation + coordinate extraction, then append SP channel metadata."""
    _ensure_trident_import(trident_root)
    from trident import load_wsi
    from trident.segmentation_models import segmentation_model_factory

    channel_names, marker_names = _read_channel_metadata(
        image_path,
        data_root=data_root,
        reader_type=reader_type,
        marker_registry_path=marker_registry_path,
        channel_names_override=channel_names_override,
    )

    job_dir = Path(job_dir)
    seg_device = "cpu" if segmenter == "otsu" or gpu < 0 else f"cuda:{gpu}"
    if trident_reader_type is None and str(image_path).lower().endswith(".qptiff"):
        trident_reader_type = "openslide"
    raw_mpp, mag, normalized_mpp = infer_mag_and_mpp(image_path, mag=mag, mpp=mpp)
    load_kwargs = {"mpp": float(normalized_mpp)}
    with load_wsi(slide_path=str(image_path), reader_type=trident_reader_type, lazy_init=False, **load_kwargs) as slide:
        segmentation_model = segmentation_model_factory(model_name=segmenter, confidence_thresh=seg_conf_thresh)
        slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=str(job_dir),
            device=seg_device,
            holes_are_tissue=not remove_holes,
        )
        mag_str = f"{float(mag):g}"
        save_coords = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap"
        coords_h5 = slide.extract_tissue_coords(
            target_mag=mag,
            patch_size=patch_size,
            save_coords=str(save_coords),
            overlap=overlap,
            min_tissue_proportion=min_tissue_proportion,
        )
    append_channel_metadata_to_h5(
        coords_h5,
        source_path=str(image_path),
        reader_type=reader_type,
        dataset=dataset,
        channel_names=channel_names,
        marker_names=marker_names,
    )
    with h5py.File(coords_h5, "a") as handle:
        handle["coords"].attrs["foreground_method"] = f"trident_{segmenter}"
        handle["coords"].attrs["foreground_attrs_json"] = json.dumps(
            {
                "seg_conf_thresh": seg_conf_thresh,
                "mag": mag,
                "raw_mpp": raw_mpp,
                "mpp": normalized_mpp,
                "trident_reader_type": trident_reader_type,
            },
            ensure_ascii=True,
        )
    return Path(coords_h5)


def extract_sp_fluorescence_coords(
    image_path: str | Path,
    *,
    trident_root: str | Path,
    job_dir: str | Path,
    marker_registry_path: str | Path,
    data_root: str | Path | None,
    reader_type: str,
    dataset: str,
    patch_size: int = 224,
    mag: Optional[int] = None,
    overlap: int = 0,
    trident_reader_type: Optional[str] = None,
    channel_names_override: Optional[Sequence[str]] = None,
    mpp: Optional[float] = None,
    min_tissue_proportion: float = 0.10,
    thumbnail_max_size: int = 1600,
    threshold_percentile: float = 50.0,
    min_signal: float = 8.0,
    blur_sigma: float = 4.0,
    close_radius: int = 20,
    open_radius: int = 2,
    dilate_radius: int = 3,
    min_component_area_fraction: float = 0.001,
    min_contour_area: float = 1000.0,
    excluded_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    forced_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    max_fusion_thumbnail_regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    max_fusion_threshold_percentile: float = 70.0,
    max_fusion_min_signal: float = 8.0,
    qptiff_max_hole_area_fraction: float = 0.05,
    qptiff_small_edge_max_area_fraction: float = 0.005,
    qptiff_small_edge_margin_fraction: float = 0.15,
    qptiff_remove_sparse_peripheral_artifacts: bool = False,
    ims_fusion_percentile: float = 75.0,
    ims_max_hole_area_fraction: float = 0.02,
    ims_threshold_percentile: float = 20.0,
    ims_min_signal: float = 6.0,
    ims_blur_sigma: float = 3.0,
    ims_close_radius: int = 6,
    ims_open_radius: int = 0,
    ims_dilate_radius: int = 2,
    ims_min_component_area_fraction: float = 0.0002,
) -> Path:
    """Extract patch coordinates using a fluorescence-specific foreground mask."""
    _ensure_trident_import(trident_root)
    from trident import load_wsi
    from trident.IO import mask_to_gdf, overlay_gdf_on_thumbnail

    channel_names, marker_names = _read_channel_metadata(
        image_path,
        data_root=data_root,
        reader_type=reader_type,
        marker_registry_path=marker_registry_path,
        channel_names_override=channel_names_override,
    )

    job_dir = Path(job_dir)
    # Format-level defaults are intentionally independent of dataset and file
    # names. Use the CLI parameters or a named generic preset to change the
    # recall/artifact trade-off explicitly.
    treat_z_as_channels = str(reader_type).strip().lower() == "tiff_z_as_channels"
    flatten_nonspatial_axes_as_channels = str(reader_type).strip().lower() == "tiff_hyperstack"

    # OME/IMS images use every available biological plane in a pixelwise max
    # fusion. Technical AF/blank/control planes are removed by the reader.
    ome_threshold_percentile, ome_min_signal = threshold_percentile, min_signal
    ome_blur_sigma, ome_close_radius, ome_open_radius, ome_dilate_radius = (
        blur_sigma, close_radius, open_radius, dilate_radius
    )
    ome_min_component_area_fraction = min_component_area_fraction
    ome_fusion_percentile = 100.0
    ome_qc_fusion_percentile = None
    ome_max_hole_area_fraction = 0.02

    # QPTIFF receives the caller-specified morphology. The common edge/frame
    # cleanup is applied consistently to every QPTIFF, never by collection.
    qptiff_threshold_percentile, qptiff_min_signal = threshold_percentile, min_signal
    qptiff_blur_sigma, qptiff_close_radius, qptiff_open_radius, qptiff_dilate_radius = (
        blur_sigma, close_radius, open_radius, dilate_radius
    )
    qptiff_min_component_area_fraction = min_component_area_fraction
    if trident_reader_type is None and str(image_path).lower().endswith(".qptiff"):
        trident_reader_type = "openslide"
    raw_mpp, mag, normalized_mpp = infer_mag_and_mpp(image_path, mag=mag, mpp=mpp)

    if reader_type == "ims" or str(image_path).lower().endswith(".ims"):
        return _extract_native_ims_sp_fluorescence_coords(
            image_path,
            trident_root=trident_root,
            job_dir=job_dir,
            data_root=data_root,
            reader_type=reader_type,
            dataset=dataset,
            patch_size=patch_size,
            mag=mag,
            raw_mpp=raw_mpp,
            normalized_mpp=normalized_mpp,
            overlap=overlap,
            channel_names=channel_names,
            marker_names=marker_names,
            min_tissue_proportion=min_tissue_proportion,
            thumbnail_max_size=thumbnail_max_size,
            threshold_percentile=ome_threshold_percentile,
            min_signal=ome_min_signal,
            blur_sigma=ome_blur_sigma,
            close_radius=ome_close_radius,
            open_radius=ome_open_radius,
            dilate_radius=ome_dilate_radius,
            min_component_area_fraction=ome_min_component_area_fraction,
            min_contour_area=min_contour_area,
            fusion_percentile=ome_fusion_percentile,
            qc_fusion_percentile=ome_qc_fusion_percentile,
            max_hole_area_fraction=ome_max_hole_area_fraction,
            forced_thumbnail_regions=forced_thumbnail_regions,
        )

    image_name_lower = str(image_path).lower()
    if image_name_lower.endswith((".ome.tif", ".ome.tiff")) or flatten_nonspatial_axes_as_channels:
        return _extract_native_ome_tiff_sp_fluorescence_coords(
            image_path,
            trident_root=trident_root,
            job_dir=job_dir,
            data_root=data_root,
            reader_type=reader_type,
            dataset=dataset,
            patch_size=patch_size,
            mag=mag,
            raw_mpp=raw_mpp,
            normalized_mpp=normalized_mpp,
            overlap=overlap,
            channel_names=channel_names,
            marker_names=marker_names,
            min_tissue_proportion=min_tissue_proportion,
            thumbnail_max_size=thumbnail_max_size,
            threshold_percentile=ome_threshold_percentile,
            min_signal=ome_min_signal,
            blur_sigma=ome_blur_sigma,
            close_radius=ome_close_radius,
            open_radius=ome_open_radius,
            dilate_radius=ome_dilate_radius,
            min_component_area_fraction=ome_min_component_area_fraction,
            min_contour_area=min_contour_area,
            fusion_percentile=ome_fusion_percentile,
            max_hole_area_fraction=ome_max_hole_area_fraction,
            max_fusion_thumbnail_regions=max_fusion_thumbnail_regions or [],
            max_fusion_threshold_percentile=max_fusion_threshold_percentile,
            max_fusion_min_signal=max_fusion_min_signal,
            max_fusion_min_component_area_fraction=0.0001,
            max_fusion_replace_existing_mask=False,
            support_recovery_fusion_percentile=None,
            support_recovery_threshold_percentile=20.0,
            support_recovery_min_signal=3.0,
            support_recovery_dilation_radius=40,
            excluded_thumbnail_regions=excluded_thumbnail_regions or [],
            excluded_thumbnail_polygons=[],
            roi_recovery_uses_base_fusion=False,
            strict_threshold_regions=[],
            strict_threshold_percentile=70.0,
            strict_min_signal=8.0,
            strict_min_component_area_fraction=None,
            post_strict_recovery_regions=[],
            post_strict_recovery_fusion_percentile=75.0,
            post_strict_recovery_threshold_percentile=60.0,
            post_strict_recovery_min_signal=6.0,
            component_concavity_fill_regions=[],
            local_background_resegmentation_regions=[],
            local_background_resegmentation_threshold_percentile=55.0,
            local_background_resegmentation_min_signal=6.0,
            forced_foreground_polygons=[],
            final_forced_foreground_polygons=[],
            technical_channel_fusion="percentile",
            force_median_marker_fusion=False,
            include_nucleus_thumbnail=False,
            treat_z_as_channels=treat_z_as_channels,
            flatten_nonspatial_axes_as_channels=flatten_nonspatial_axes_as_channels,
            max_thumbnail_marker_planes=0,
            remove_border_frame_artifacts=False,
            remove_border_strip_artifacts=False,
            remove_thin_grid_artifacts=False,
            thin_grid_min_aspect_ratio=5.0,
            thin_grid_max_thickness_fraction=0.025,
            edge_cleanup_fusion_percentile=None,
            edge_cleanup_threshold_percentile=45.0,
            edge_cleanup_min_signal=8.0,
            edge_cleanup_margin_fraction=0.08,
            nucleus_gate_threshold_percentile=70.0,
            nucleus_gate_dilation_radius=25,
            apply_nucleus_seed_gate=False,
            remove_small_edge_artifacts=False,
            small_edge_max_area_fraction=qptiff_small_edge_max_area_fraction,
            small_edge_margin_fraction=qptiff_small_edge_margin_fraction,
            remove_sparse_peripheral_artifacts=False,
            dense_foreground_rescue=False,
        )

    load_kwargs = {"mpp": float(normalized_mpp)}

    with load_wsi(slide_path=str(image_path), reader_type=trident_reader_type, lazy_init=False, **load_kwargs) as slide:
        width, height = slide.get_dimensions()
        thumbnail_attrs: Dict[str, Any] = {}
        if str(image_path).lower().endswith(".qptiff"):
            thumbnail_rgb, raw_shape_yx, thumbnail_attrs = read_qptiff_multichannel_thumbnail(
                image_path,
                expected_channel_count=len(channel_names),
                thumbnail_max_size=thumbnail_max_size,
                fusion="percentile",
                fusion_percentile=ims_fusion_percentile,
            )
            max_fusion_thumbnail_rgb = None
            if max_fusion_thumbnail_regions:
                max_fusion_thumbnail_rgb, _, max_fusion_attrs = read_qptiff_multichannel_thumbnail(
                    image_path,
                    expected_channel_count=len(channel_names),
                    thumbnail_max_size=thumbnail_max_size,
                    fusion="max",
                )
                thumbnail_attrs["roi_max_fusion_thumbnail"] = max_fusion_attrs
            raw_height, raw_width = raw_shape_yx
            if (raw_width, raw_height) != (int(width), int(height)):
                raise ValueError(
                    "Raw QPTIFF page dimensions do not match OpenSlide geometry: "
                    f"raw={(raw_width, raw_height)}, openslide={(width, height)} for {image_path}"
                )
            from PIL import Image

            thumbnail = Image.fromarray(thumbnail_rgb).convert("RGB")
        else:
            if width >= height:
                thumb_size = (int(thumbnail_max_size), max(1, int(round(thumbnail_max_size * height / width))))
            else:
                thumb_size = (max(1, int(round(thumbnail_max_size * width / height))), int(thumbnail_max_size))
            thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
            thumbnail_rgb = np.asarray(thumbnail)
            thumbnail_attrs = {"thumbnail_source": "openslide_rendered_rgb"}
            max_fusion_thumbnail_rgb = None
        is_qptiff = str(image_path).lower().endswith(".qptiff")
        mask, mask_attrs = make_sp_fluorescence_foreground(
            thumbnail_rgb,
            threshold_percentile=qptiff_threshold_percentile,
            min_signal=qptiff_min_signal,
            blur_sigma=qptiff_blur_sigma,
            close_radius=qptiff_close_radius,
            open_radius=qptiff_open_radius,
            dilate_radius=qptiff_dilate_radius,
            min_component_area_fraction=qptiff_min_component_area_fraction,
            remove_border_frame_artifacts=is_qptiff,
            remove_border_strip_artifacts=is_qptiff,
            remove_small_edge_artifacts=is_qptiff,
            max_hole_area_fraction=qptiff_max_hole_area_fraction if is_qptiff else None,
            excluded_thumbnail_regions=excluded_thumbnail_regions,
            forced_thumbnail_regions=forced_thumbnail_regions,
            small_edge_max_area_fraction=qptiff_small_edge_max_area_fraction,
            small_edge_margin_fraction=qptiff_small_edge_margin_fraction,
            remove_sparse_peripheral_artifacts=is_qptiff and qptiff_remove_sparse_peripheral_artifacts,
        )
        roi_max_fusion_attrs: List[Dict[str, Any]] = []
        if max_fusion_thumbnail_regions:
            if max_fusion_thumbnail_rgb is None:
                raise ValueError("ROI max-fusion segmentation is supported only for QPTIFF inputs.")
            mask, roi_max_fusion_attrs = add_roi_max_fusion_foreground(
                mask,
                max_fusion_thumbnail_rgb,
                regions=max_fusion_thumbnail_regions,
                threshold_percentile=max_fusion_threshold_percentile,
                min_signal=max_fusion_min_signal,
            )
        mask_attrs["roi_max_fusion_regions"] = roi_max_fusion_attrs
        contour_scale = width / float(thumbnail_rgb.shape[1])
        gdf_contours = mask_to_gdf(
            mask=mask,
            max_nb_holes=0,
            min_contour_area=float(min_contour_area),
            pixel_size=slide.mpp if slide.mpp is not None else 1,
            contour_scale=contour_scale,
        )

        thumbnails_dir = job_dir / "thumbnails"
        contours_dir = job_dir / "contours"
        geojson_dir = job_dir / "contours_geojson"
        mask_dir = job_dir / "foreground_masks"
        for directory in [thumbnails_dir, contours_dir, geojson_dir, mask_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        thumbnail.save(thumbnails_dir / f"{slide.name}.jpg")

        geojson_path = geojson_dir / f"{slide.name}.geojson"
        gdf_contours.set_crs("EPSG:3857", inplace=True)
        gdf_contours.to_file(geojson_path, driver="GeoJSON")
        from PIL import Image

        Image.fromarray(mask).save(mask_dir / f"{slide.name}.png")
        annotated = thumbnail_rgb.copy()
        overlay_gdf_on_thumbnail(gdf_contours, annotated, str(contours_dir / f"{slide.name}.jpg"), thumbnail_rgb.shape[1] / width)
        slide.gdf_contours = gdf_contours
        slide.tissue_seg_path = str(geojson_path)

        mag_str = f"{float(mag):g}"
        save_coords = job_dir / f"{mag_str}x_{patch_size}px_{overlap}px_overlap"
        coords_h5 = slide.extract_tissue_coords(
            target_mag=mag,
            patch_size=patch_size,
            save_coords=str(save_coords),
            overlap=overlap,
            min_tissue_proportion=min_tissue_proportion,
        )

    append_channel_metadata_to_h5(
        coords_h5,
        source_path=str(image_path),
        reader_type=reader_type,
        dataset=dataset,
        channel_names=channel_names,
        marker_names=marker_names,
    )
    with h5py.File(coords_h5, "a") as handle:
        handle["coords"].attrs["foreground_method"] = "sp_fluorescence"
        handle["coords"].attrs["foreground_attrs_json"] = json.dumps(
            {
                **mask_attrs,
                "mag": mag,
                "raw_mpp": raw_mpp,
                "mpp": normalized_mpp,
                "trident_reader_type": trident_reader_type,
                "thumbnail_size": list(thumbnail_rgb.shape[:2]),
                "min_tissue_proportion": min_tissue_proportion,
                **thumbnail_attrs,
            },
            ensure_ascii=True,
        )
    return Path(coords_h5)

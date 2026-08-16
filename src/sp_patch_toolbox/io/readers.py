import json
import math
import re
from pathlib import Path
from itertools import product
from typing import List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import tifffile

try:
    import openslide
except Exception:  # pragma: no cover - optional runtime dependency
    openslide = None


IMAGE_SUFFIXES = (
    ".ome.tiff",
    ".ome.tif",
    ".qptiff",
    ".tiff",
    ".tif",
    ".ims",
)


def expand_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    p = Path(str(path).replace("\\", "/")).expanduser()
    if not p.is_absolute() and data_root is not None:
        p = Path(data_root).expanduser() / p
    return p


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "O", "U"}:
            parts = []
            for item in value.ravel():
                if isinstance(item, bytes):
                    parts.append(item.decode("utf-8", errors="replace"))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        return " ".join(str(x) for x in value.ravel())
    return str(value)


def _channels_from_sidecar(path: Path) -> List[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    channels = data.get("hpac_conf", {}).get("channel_imgs", [])
    return [str(ch.get("name", "")).strip() for ch in channels if ch.get("name")]


def _channels_from_ome_xml(ome_xml: str | bytes | None) -> List[str]:
    if not ome_xml:
        return []
    try:
        root = ET.fromstring(ome_xml.encode("utf-8") if isinstance(ome_xml, str) else ome_xml)
    except Exception:
        return []
    names = []
    for channel in root.findall(".//{*}Channel"):
        name = channel.attrib.get("Name") or channel.attrib.get("Fluor")
        if name:
            names.append(name)
    return names


def _channels_from_perkinelmer_pages(pages: Sequence[tifffile.TiffPage], count: int) -> List[str]:
    names = []
    for page in list(pages)[:count]:
        # Pyramid levels can be exposed by tifffile as TiffFrame objects,
        # which intentionally do not duplicate the base-page description.
        # Treat those as unnamed rather than failing metadata discovery.
        desc = getattr(page, "description", "") or ""
        found = re.findall(r"<Name>([^<]+)</Name>", desc)
        names.append(found[0].strip() if found else "")
    return names


def _infer_sidecar(path: Path) -> Optional[Path]:
    candidates = [
        Path(str(path) + ".metadata.json"),
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(".metadata.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _fit_channel_names(names: Sequence[str], count: int) -> List[str]:
    fitted = [str(name) for name in list(names)[: int(count)]]
    while len(fitted) < int(count):
        fitted.append(f"C{len(fitted) + 1}")
    return fitted


def _decoded_segment_to_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def _page_attr(page, name: str, default=None):
    if hasattr(page, name):
        return getattr(page, name)
    keyframe = getattr(page, "keyframe", None)
    if keyframe is not None and hasattr(keyframe, name):
        return getattr(keyframe, name)
    return default


def _page_hw(page) -> Tuple[int, int]:
    height = _page_attr(page, "imagelength")
    width = _page_attr(page, "imagewidth")
    if height is not None and width is not None:
        return int(height), int(width)
    shape = tuple(page.shape)
    return int(shape[0]), int(shape[1])


def _empty_region(page, height: int, width: int) -> np.ndarray:
    samples = int(_page_attr(page, "samplesperpixel", 1) or 1)
    if samples > 1:
        return np.zeros((height, width, samples), dtype=page.dtype)
    return np.zeros((height, width), dtype=page.dtype)


def read_tiff_page_region(page, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    """Read a rectangular region from a single TIFF page without full-page decode."""
    image_height, image_width = _page_hw(page)
    y0 = max(0, int(y0))
    x0 = max(0, int(x0))
    y1 = min(image_height, int(y1))
    x1 = min(image_width, int(x1))
    out_h = max(0, y1 - y0)
    out_w = max(0, x1 - x0)
    out = _empty_region(page, out_h, out_w)
    if out_h == 0 or out_w == 0:
        return out

    filehandle = page.parent.filehandle
    offsets = page.dataoffsets
    bytecounts = page.databytecounts

    if bool(_page_attr(page, "is_tiled", False)):
        tile_h = int(_page_attr(page, "tilelength"))
        tile_w = int(_page_attr(page, "tilewidth"))
        tiles_across = math.ceil(image_width / tile_w)
        row_start = y0 // tile_h
        row_stop = (y1 - 1) // tile_h
        col_start = x0 // tile_w
        col_stop = (x1 - 1) // tile_w
        for tile_row in range(row_start, row_stop + 1):
            for tile_col in range(col_start, col_stop + 1):
                index = tile_row * tiles_across + tile_col
                filehandle.seek(offsets[index])
                data = filehandle.read(bytecounts[index])
                decoded, _, _ = page.decode(data, index)
                if decoded is None:
                    continue
                tile = _decoded_segment_to_array(decoded)
                gy0 = tile_row * tile_h
                gx0 = tile_col * tile_w
                gy1 = min(gy0 + tile.shape[0], image_height)
                gx1 = min(gx0 + tile.shape[1], image_width)
                sy0 = max(y0, gy0)
                sx0 = max(x0, gx0)
                sy1 = min(y1, gy1)
                sx1 = min(x1, gx1)
                if sy1 <= sy0 or sx1 <= sx0:
                    continue
                ty0 = sy0 - gy0
                tx0 = sx0 - gx0
                out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = tile[ty0 : ty0 + sy1 - sy0, tx0 : tx0 + sx1 - sx0]
        return out

    rows_per_strip = int(_page_attr(page, "rowsperstrip", 0) or image_height)
    strip_start = y0 // rows_per_strip
    strip_stop = (y1 - 1) // rows_per_strip
    for strip_index in range(strip_start, strip_stop + 1):
        filehandle.seek(offsets[strip_index])
        data = filehandle.read(bytecounts[strip_index])
        decoded, _, _ = page.decode(data, strip_index)
        if decoded is None:
            continue
        strip = _decoded_segment_to_array(decoded)
        gy0 = strip_index * rows_per_strip
        gy1 = min(gy0 + strip.shape[0], image_height)
        sy0 = max(y0, gy0)
        sy1 = min(y1, gy1)
        if sy1 <= sy0:
            continue
        out[sy0 - y0 : sy1 - y0, :] = strip[sy0 - gy0 : sy1 - gy0, x0:x1]
    return out


class BaseImageReader:
    def __init__(self, path: str | Path, **kwargs):
        self.path = Path(path)

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    @property
    def channel_count(self) -> int:
        raise NotImplementedError

    @property
    def z_count(self) -> int:
        return 1

    def channel_names(self) -> List[str]:
        return [f"C{i + 1}" for i in range(self.channel_count)]

    def read_patch(self, y: int, x: int, size: int) -> np.ndarray:
        raise NotImplementedError

    def read_patch_channels(self, y: int, x: int, size: int, channels: Sequence[int]) -> np.ndarray:
        patch = self.read_patch(y, x, size)
        return patch[np.asarray(list(channels), dtype=np.int64)]

    def close(self) -> None:
        return None


class ImsImageReader(BaseImageReader):
    def __init__(
        self,
        path: str | Path,
        resolution_level: int = 0,
        timepoint: int = 0,
        z_strategy: str = "middle",
        z_index: int | None = None,
        **kwargs,
    ):
        super().__init__(path)
        self.resolution_level = int(resolution_level)
        self.timepoint = int(timepoint)
        self.z_strategy = z_strategy
        self.z_index = None if z_index is None else int(z_index)
        self._file: Optional[h5py.File] = None

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    @property
    def _tp_group(self):
        return self.file["DataSet"][f"ResolutionLevel {self.resolution_level}"][f"TimePoint {self.timepoint}"]

    def _channel_keys(self) -> List[str]:
        def idx(name: str) -> int:
            match = re.search(r"(\d+)$", name)
            return int(match.group(1)) if match else 10**9

        return sorted([k for k in self._tp_group.keys() if k.startswith("Channel ")], key=idx)

    @property
    def channel_count(self) -> int:
        return len(self._channel_keys())

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        first = self._tp_group[self._channel_keys()[0]]["Data"]
        return int(first.shape[-2]), int(first.shape[-1])

    @property
    def z_count(self) -> int:
        first = self._tp_group[self._channel_keys()[0]]["Data"]
        return int(first.shape[0])

    def channel_names(self) -> List[str]:
        names = []
        info = self.file.get("DataSetInfo")
        for key in self._channel_keys():
            name = ""
            if info is not None and key in info:
                name = _decode_attr(info[key].attrs.get("Name", ""))
            names.append(name or key)
        return names

    def _z_indices(self, z_count: int) -> Sequence[int]:
        if z_count <= 1:
            return [0]
        if self.z_index is not None:
            if self.z_index < 0 or self.z_index >= int(z_count):
                raise IndexError(f"z_index={self.z_index} out of range for z_count={z_count} in {self.path}")
            return [self.z_index]
        if self.z_strategy == "middle":
            return [z_count // 2]
        if self.z_strategy in {"max", "mean"}:
            return list(range(z_count))
        raise ValueError(f"Unsupported z_strategy={self.z_strategy!r}")

    def read_patch(self, y: int, x: int, size: int) -> np.ndarray:
        height, width = self.spatial_shape
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(height, y + size), min(width, x + size)
        planes = []
        for key in self._channel_keys():
            data = self._tp_group[key]["Data"]
            z_ids = self._z_indices(int(data.shape[0]))
            stack = np.stack([data[z, y0:y1, x0:x1] for z in z_ids], axis=0)
            if self.z_strategy == "max":
                planes.append(stack.max(axis=0))
            elif self.z_strategy == "mean":
                planes.append(stack.mean(axis=0))
            else:
                planes.append(stack[0])
        return np.stack(planes, axis=0)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class TiffImageReader(BaseImageReader):
    FULL_READ_FALLBACK_MAX_ELEMENTS = 250_000_000

    def __init__(
        self,
        path: str | Path,
        sidecar_path: str | Path | None = None,
        z_strategy: str = "middle",
        z_index: int | None = None,
        treat_z_as_channels: bool = False,
        flatten_nonspatial_axes_as_channels: bool = False,
        allow_full_read: bool = False,
        **kwargs,
    ):
        super().__init__(path)
        self.sidecar_path = Path(sidecar_path) if sidecar_path else _infer_sidecar(self.path)
        self.z_strategy = z_strategy
        self.z_index = None if z_index is None else int(z_index)
        self.treat_z_as_channels = bool(treat_z_as_channels)
        self.flatten_nonspatial_axes_as_channels = bool(flatten_nonspatial_axes_as_channels)
        self.allow_full_read = bool(allow_full_read)
        self._tf: Optional[tifffile.TiffFile] = None
        self._memmap: Optional[np.ndarray] = None

    @property
    def tf(self) -> tifffile.TiffFile:
        if self._tf is None:
            self._tf = tifffile.TiffFile(str(self.path))
        return self._tf

    @property
    def series(self):
        return self.tf.series[0]

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        axes = self.series.axes
        shape = self.series.shape
        return int(shape[axes.index("Y")]), int(shape[axes.index("X")])

    @property
    def channel_count(self) -> int:
        axes = self.series.axes
        shape = self.series.shape
        if self.treat_z_as_channels and axes == "ZYX":
            return int(shape[axes.index("Z")])
        if "Y" in axes and "X" in axes:
            channel_axes = [axis for axis in axes if axis not in {"Y", "X", "Z"}]
            if not channel_axes:
                return 1
            count = 1
            for axis, dim in zip(axes, shape):
                if axis in channel_axes:
                    count *= int(dim)
            return int(count)
        if "C" in axes:
            return int(shape[axes.index("C")])
        if "I" in axes:
            return int(shape[axes.index("I")])
        if "S" in axes and "Y" in axes and "X" in axes:
            return int(shape[axes.index("S")])
        if len(self.series.pages) > 1:
            return len(self.series.pages)
        return 1

    @property
    def z_count(self) -> int:
        axes = self.series.axes
        if "Z" not in axes:
            return 1
        return int(self.series.shape[axes.index("Z")])

    def channel_names(self) -> List[str]:
        # Some ImageJ hyperstacks have TCYX axes with repeated detector labels.
        # Preserve acquisition coordinates instead of inventing marker names.
        if self.flatten_nonspatial_axes_as_channels and self.series.axes == "TCYX":
            timepoints, channels = (int(value) for value in self.series.shape[:2])
            return [
                f"cycle_{timepoint + 1:02d}_channel_{channel + 1}"
                for timepoint in range(timepoints)
                for channel in range(channels)
            ]
        names_file = self.path.parent / "channelNames.txt"
        if names_file.exists():
            names = [line.strip() for line in names_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            if names:
                return _fit_channel_names(names, self.channel_count)
        if self.sidecar_path:
            names = _channels_from_sidecar(self.sidecar_path)
            if names:
                return _fit_channel_names(names, self.channel_count)
        names = _channels_from_ome_xml(getattr(self.tf, "ome_metadata", None))
        if names:
            return _fit_channel_names(names, self.channel_count)
        names = _channels_from_perkinelmer_pages(self.series.pages, self.channel_count)
        if any(names):
            return _fit_channel_names([name or f"C{i + 1}" for i, name in enumerate(names)], self.channel_count)
        if "S" in self.series.axes:
            return _fit_channel_names(["RGB_R", "RGB_G", "RGB_B"], self.channel_count)
        return super().channel_names()

    @property
    def memmap(self) -> np.ndarray:
        if self._memmap is None:
            self._memmap = tifffile.memmap(str(self.path), series=0, mode="r")
        return self._memmap

    def _page_index(self, c: int) -> Optional[int]:
        axes = self.series.axes
        shape = self.series.shape
        pages = self.series.pages
        if self.treat_z_as_channels and axes == "ZYX":
            return c if c < len(pages) else None
        if "Y" in axes and "X" in axes and "Z" not in axes and len(pages) >= self.channel_count:
            return c
        if axes == "CZYX":
            z_count = int(shape[axes.index("Z")])
            z_values = self._z_values(z_count)
            if len(z_values) != 1:
                return None
            z = int(z_values[0])
            index = c * z_count + z
            return index if index < len(pages) else None
        if axes == "ZCYX":
            z_count = int(shape[axes.index("Z")])
            z_values = self._z_values(z_count)
            if len(z_values) != 1:
                return None
            z = int(z_values[0])
            index = z * self.channel_count + c
            return index if index < len(pages) else None
        return None

    def _z_values(self, z_count: int) -> Sequence[int]:
        if z_count <= 1:
            return [0]
        if self.z_index is not None:
            if self.z_index < 0 or self.z_index >= int(z_count):
                raise IndexError(f"z_index={self.z_index} out of range for z_count={z_count} in {self.path}")
            return [self.z_index]
        if self.z_strategy == "middle":
            return [z_count // 2]
        if self.z_strategy in {"max", "mean"}:
            return list(range(z_count))
        raise ValueError(f"Unsupported z_strategy={self.z_strategy!r}")

    def _full_to_cyx(self, arr: np.ndarray, axes: str) -> np.ndarray:
        axes_list = list(axes)
        if "Z" in axes_list:
            idx = axes_list.index("Z")
            z_values = self._z_values(arr.shape[idx])
            if len(z_values) == 1:
                arr = np.take(arr, int(z_values[0]), axis=idx)
            elif self.z_strategy == "max":
                arr = arr.max(axis=idx)
            elif self.z_strategy == "mean":
                arr = arr.mean(axis=idx)
            axes_list.pop(idx)

        channel_axes = [i for i, axis in enumerate(axes_list) if axis not in {"Y", "X"}]
        if not channel_axes:
            arr = arr[np.newaxis, ...]
            return arr

        y_axis = axes_list.index("Y")
        x_axis = axes_list.index("X")
        arr = np.transpose(arr, channel_axes + [y_axis, x_axis])
        channel_count = int(np.prod(arr.shape[: len(channel_axes)]))
        return arr.reshape(channel_count, arr.shape[-2], arr.shape[-1])

    def _full_read_patch(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        element_count = int(np.prod(self.series.shape))
        if not self.allow_full_read and element_count > self.FULL_READ_FALLBACK_MAX_ELEMENTS:
            raise RuntimeError(
                f"{self.path} has axes={self.series.axes} shape={self.series.shape} and is not memory-mappable. "
                "Refusing full read because the image is too large; add a region-readable reader for this layout."
            )
        arr = self._full_to_cyx(self.series.asarray(), self.series.axes)
        return arr[:, y0:y1, x0:x1]

    def _hyperstack_patch(self, y0: int, x0: int, y1: int, x1: int) -> Optional[np.ndarray]:
        axes = self.series.axes
        if "Y" not in axes or "X" not in axes:
            return None
        channel_axes = [axis for axis in axes if axis not in {"Y", "X", "Z"}]
        if not channel_axes:
            return None
        axis_sizes = dict(zip(axes, self.series.shape))
        if "Z" in axes:
            z_values = self._z_values(int(axis_sizes["Z"]))
        else:
            z_values = [None]

        try:
            arr = self.memmap
        except ValueError as exc:
            if "memory-mappable" not in str(exc):
                raise
            return self._full_read_patch(y0, x0, y1, x1)
        planes = []
        channel_ranges = [range(int(axis_sizes[axis])) for axis in channel_axes]
        for combo in product(*channel_ranges):
            z_planes = []
            for z in z_values:
                slicer = []
                combo_pos = 0
                for axis in axes:
                    if axis == "Y":
                        slicer.append(slice(y0, y1))
                    elif axis == "X":
                        slicer.append(slice(x0, x1))
                    elif axis == "Z":
                        slicer.append(int(z))
                    else:
                        slicer.append(int(combo[combo_pos]))
                        combo_pos += 1
                z_planes.append(np.asarray(arr[tuple(slicer)]))
            stack = np.stack(z_planes, axis=0)
            if self.z_strategy == "max":
                planes.append(stack.max(axis=0))
            elif self.z_strategy == "mean":
                planes.append(stack.mean(axis=0))
            else:
                planes.append(stack[0])
        return np.stack(planes, axis=0)

    def read_patch(self, y: int, x: int, size: int) -> np.ndarray:
        height, width = self.spatial_shape
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(height, y + size), min(width, x + size)

        if self.series.axes == "YXS" and len(self.series.pages) == 1:
            region = read_tiff_page_region(self.series.pages[0], y0, x0, y1, x1)
            if region.ndim == 2:
                region = region[..., np.newaxis]
            return np.moveaxis(region, -1, 0)

        planes = []
        for c in range(self.channel_count):
            page_index = self._page_index(c)
            if page_index is None:
                break
            plane = read_tiff_page_region(self.series.pages[page_index], y0, x0, y1, x1)
            if plane.ndim == 3:
                plane = plane[..., 0]
            planes.append(plane)
        if len(planes) == self.channel_count:
            return np.stack(planes, axis=0)

        hyperstack = self._hyperstack_patch(y0, x0, y1, x1)
        if hyperstack is not None:
            return hyperstack

        if not self.allow_full_read:
            raise RuntimeError(
                f"{self.path} has axes={self.series.axes} shape={self.series.shape}. "
                "This layout is not mapped to region-readable TIFF pages. Set allow_full_read=True "
                "only for small images, or add a format-specific reader."
            )
        arr = self._full_to_cyx(self.series.asarray(), self.series.axes)
        return arr[:, y0:y1, x0:x1]

    def read_patch_channels(self, y: int, x: int, size: int, channels: Sequence[int]) -> np.ndarray:
        height, width = self.spatial_shape
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(height, y + size), min(width, x + size)
        channel_ids = [int(c) for c in channels]

        if self.series.axes == "YXS" and len(self.series.pages) == 1:
            region = read_tiff_page_region(self.series.pages[0], y0, x0, y1, x1)
            if region.ndim == 2:
                region = region[..., np.newaxis]
            cyx = np.moveaxis(region, -1, 0)
            return cyx[np.asarray(channel_ids, dtype=np.int64)]

        planes = []
        for c in channel_ids:
            page_index = self._page_index(c)
            if page_index is None:
                break
            plane = read_tiff_page_region(self.series.pages[page_index], y0, x0, y1, x1)
            if plane.ndim == 3:
                plane = plane[..., 0]
            planes.append(plane)
        if len(planes) == len(channel_ids):
            return np.stack(planes, axis=0)

        hyperstack = self._hyperstack_patch(y0, x0, y1, x1)
        if hyperstack is not None:
            return hyperstack[np.asarray(channel_ids, dtype=np.int64)]

        return super().read_patch_channels(y, x, size, channel_ids)

    def close(self) -> None:
        if self._tf is not None:
            self._tf.close()
            self._tf = None
        self._memmap = None


class OpenSlideRgbReader(BaseImageReader):
    """OpenSlide fallback for WSI-like files.

    This returns rendered RGB channels, not protein marker planes. Prefer
    TiffImageReader for QPTIFF files when page-per-channel data are available.
    """

    def __init__(self, path: str | Path, level: int = 0, **kwargs):
        if openslide is None:
            raise ImportError("openslide is not installed in this environment")
        super().__init__(path)
        self.level = int(level)
        self._slide = openslide.OpenSlide(str(self.path))

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        width, height = self._slide.level_dimensions[self.level]
        return int(height), int(width)

    @property
    def channel_count(self) -> int:
        return 3

    def channel_names(self) -> List[str]:
        return ["RGB_R", "RGB_G", "RGB_B"]

    def read_patch(self, y: int, x: int, size: int) -> np.ndarray:
        downsample = float(self._slide.level_downsamples[self.level])
        loc = (int(round(x * downsample)), int(round(y * downsample)))
        rgba = np.asarray(self._slide.read_region(loc, self.level, (size, size)))
        rgb = rgba[..., :3]
        return np.moveaxis(rgb, -1, 0)

    def close(self) -> None:
        self._slide.close()


def open_image_reader(
    path: str | Path,
    data_root: str | Path | None = None,
    reader_type: str | None = None,
    **kwargs,
) -> BaseImageReader:
    resolved = expand_path(path, data_root)
    if kwargs.get("sidecar_path") is not None:
        kwargs["sidecar_path"] = expand_path(kwargs["sidecar_path"], data_root)
    name = resolved.name.lower()
    kind = reader_type or ("ims" if name.endswith(".ims") else "tiff")
    if kind == "ims":
        return ImsImageReader(resolved, **kwargs)
    if kind in {"tiff", "ome-tiff", "qptiff"}:
        return TiffImageReader(resolved, **kwargs)
    if kind == "tiff_z_as_channels":
        return TiffImageReader(resolved, treat_z_as_channels=True, **kwargs)
    if kind == "tiff_hyperstack":
        return TiffImageReader(resolved, flatten_nonspatial_axes_as_channels=True, **kwargs)
    if kind in {"openslide", "openslide_rgb"}:
        return OpenSlideRgbReader(resolved, **kwargs)
    raise ValueError(f"Unsupported reader_type={kind!r} for {resolved}")

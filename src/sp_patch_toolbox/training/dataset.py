import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..metadata.markers import MarkerRegistry
from ..coordinates import load_sp_coords_h5
from ..io.readers import open_image_reader


BAD_IMAGE_REL_PATHS = {
    "immunoatlas/NOLN/210920-1/NOLN21102/A01/download/NOLN21102_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21109/A01/download/NOLN21109_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21111/A01/download/NOLN21111_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21113/A01/download/NOLN21113_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21121/A01/download/NOLN21121_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21126/A01/download/NOLN21126_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21157/A01/download/NOLN21157_A01.tif",
    "immunoatlas/NOLN/210920-1/NOLN21163/A01/download/NOLN21163_A01.tif",
}


def _is_bad_image_row(row: Dict[str, Any]) -> bool:
    path = str(row.get("path", "")).replace("\\", "/")
    return any(path == bad or path.endswith("/" + bad) for bad in BAD_IMAGE_REL_PATHS)


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            return [row for row in payload.get("images", []) if not _is_bad_image_row(row)]
        return [row for row in payload if not _is_bad_image_row(row)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            row = json.loads(line)
            if "error" not in row and not _is_bad_image_row(row):
                rows.append(row)
    return rows


def _pad_to_patch(arr: np.ndarray, patch_size: int) -> np.ndarray:
    c, h, w = arr.shape
    if h == patch_size and w == patch_size:
        return arr
    out = np.zeros((c, patch_size, patch_size), dtype=arr.dtype)
    out[:, : min(h, patch_size), : min(w, patch_size)] = arr[:, :patch_size, :patch_size]
    return out


def _normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return arr.astype(np.float32, copy=False)
    source_dtype = arr.dtype
    arr = arr.astype(np.float32, copy=False)
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        plane = arr[i]
        if mode == "uint":
            scale = np.iinfo(source_dtype).max if np.issubdtype(source_dtype, np.integer) else plane.max()
            out[i] = plane / float(scale or 1.0)
            continue
        if mode == "log_percentile":
            plane = np.log1p(np.maximum(plane, 0))
        if mode in {"percentile", "log_percentile"}:
            lo, hi = np.percentile(plane, [1.0, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[i] = 0.0
            else:
                out[i] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
            continue
        raise ValueError(f"Unsupported normalize={mode!r}")
    return out


class SpatialProteomicsPatchDataset(Dataset):
    """PyTorch Dataset yielding variable-channel spatial proteomics patches.

    Each item returns:
    - pixels/image: FloatTensor [C, patch_size, patch_size]
    - marker_ids: LongTensor [C]
    - valid_channels: BoolTensor [C]
    - patch_grid: LongTensor [2], grid for downstream ViT patch tokenization
    - marker_names/raw_channel_names: lists aligned with channels
    """

    def __init__(
        self,
        manifest_path: str | Path,
        marker_registry_path: str | Path,
        data_root: str | Path | None = None,
        patch_size: int = 224,
        samples_per_epoch: Optional[int] = None,
        normalize: str = "percentile",
        drop_blank: bool = True,
        drop_unknown: bool = False,
        sort_channels_by_marker_id: bool = True,
        max_channels: Optional[int] = None,
        max_reader_cache_size: int = 2,
        vit_patch_size: int = 16,
        index_coords: bool = True,
        sample_entry_block_size: int = 1,
        reader_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.entries = load_manifest(self.manifest_path)
        if not self.entries:
            raise ValueError(f"No valid entries found in manifest: {self.manifest_path}")
        self.registry = MarkerRegistry.from_json(marker_registry_path)
        self.data_root = data_root
        self.patch_size = int(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.normalize = normalize
        self.drop_blank = drop_blank
        self.drop_unknown = drop_unknown
        self.sort_channels_by_marker_id = sort_channels_by_marker_id
        self.max_channels = max_channels
        self.max_reader_cache_size = max(0, int(max_reader_cache_size))
        self.vit_patch_size = int(vit_patch_size)
        self.index_coords = bool(index_coords)
        # Keep the legacy fully independent sampling behavior by default. A
        # small value is useful for read-heavy evaluation: nearby requested
        # items reuse the same raw-image reader while retaining random coords.
        self.sample_entry_block_size = max(1, int(sample_entry_block_size))
        self.reader_kwargs = reader_kwargs or {}
        self.seed = seed
        self._reader_cache: OrderedDict[str, Any] = OrderedDict()
        self._coords_cache: Dict[str, Any] = {}
        self._coord_counts: List[int] = []
        self._coord_cumsum: List[int] = []
        if self.index_coords and any(entry.get("coords_path") for entry in self.entries):
            total = 0
            for entry in self.entries:
                count = 1
                if entry.get("coords_path"):
                    count = self._coords_count(entry)
                self._coord_counts.append(count)
                total += count
                self._coord_cumsum.append(total)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader_cache"] = {}
        state["_coords_cache"] = {}
        return state

    def __len__(self) -> int:
        if self.samples_per_epoch:
            return int(self.samples_per_epoch)
        if self._coord_cumsum:
            return int(self._coord_cumsum[-1])
        return int(self.samples_per_epoch or len(self.entries))

    def close(self) -> None:
        for reader in self._reader_cache.values():
            reader.close()
        self._reader_cache.clear()

    def _rng(self, index: int) -> random.Random:
        if self.seed is None:
            return random
        return random.Random(self.seed + int(index))

    def _coords_path_for(self, entry: Dict[str, Any]) -> Path:
        path = entry.get("coords_path") or entry.get("coords_h5")
        if not path:
            raise KeyError("entry does not contain coords_path")
        p = Path(str(path)).expanduser()
        if p.is_absolute():
            return p
        if self.data_root is not None:
            candidate = Path(self.data_root).expanduser() / p
            if candidate.exists():
                return candidate
        return (self.manifest_path.parent / p).resolve()

    def _coords_payload_for(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        path = str(self._coords_path_for(entry))
        if path not in self._coords_cache:
            self._coords_cache[path] = load_sp_coords_h5(path)
        return self._coords_cache[path]

    def _coords_count(self, entry: Dict[str, Any]) -> int:
        payload = self._coords_payload_for(entry)
        return int(len(payload["coords"]))

    def _entry_for_index(self, index: int, rng: random.Random) -> tuple[Dict[str, Any], Optional[int]]:
        if self.samples_per_epoch:
            entry_rng = rng
            if self.sample_entry_block_size > 1:
                entry_rng = self._rng(index // self.sample_entry_block_size)
            entry = entry_rng.choice(self.entries)
            coord_index = None
            if entry.get("coords_path"):
                count = self._coords_count(entry)
                coord_index = None if count == 0 else rng.randrange(count)
            return entry, coord_index
        if self._coord_cumsum:
            import bisect

            entry_index = bisect.bisect_right(self._coord_cumsum, index)
            prev = 0 if entry_index == 0 else self._coord_cumsum[entry_index - 1]
            return self.entries[entry_index], int(index - prev)
        return self.entries[index % len(self.entries)], None

    def _reader_for(self, entry: Dict[str, Any]):
        key = json.dumps(
            {
                "path": entry["path"],
                "reader_type": entry.get("reader_type"),
                "kwargs": entry.get("reader_kwargs", {}),
                "sidecar_path": entry.get("sidecar_path"),
            },
            sort_keys=True,
        )
        if key in self._reader_cache:
            self._reader_cache.move_to_end(key)
            return self._reader_cache[key]
        if key not in self._reader_cache:
            kwargs = dict(self.reader_kwargs)
            kwargs.update(entry.get("reader_kwargs", {}))
            if entry.get("sidecar_path"):
                kwargs["sidecar_path"] = entry["sidecar_path"]
            self._reader_cache[key] = open_image_reader(
                entry["path"],
                data_root=self.data_root,
                reader_type=entry.get("reader_type"),
                **kwargs,
            )
            if self.max_reader_cache_size and len(self._reader_cache) > self.max_reader_cache_size:
                _, old_reader = self._reader_cache.popitem(last=False)
                old_reader.close()
        return self._reader_cache[key]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        current_index = int(index)
        for attempt in range(9):
            try:
                return self._getitem_once(current_index)
            except OSError as exc:
                last_error = exc
                if attempt == 8:
                    break
                rng = self._rng(int(index) + 1_000_003 * (attempt + 1))
                current_index = rng.randrange(max(1, len(self)))
        raise RuntimeError(f"Failed to read a valid patch after retries; last error: {last_error}") from last_error

    def _getitem_once(self, index: int) -> Dict[str, Any]:
        rng = self._rng(index)
        entry, coord_index = self._entry_for_index(index, rng)
        reader = self._reader_for(entry)
        height, width = reader.spatial_shape
        if entry.get("coords_path"):
            payload = self._coords_payload_for(entry)
            coords = payload["coords"]
            if len(coords) == 0:
                raise IndexError(f"No coordinates in {entry.get('coords_path')}")
            if coord_index is None:
                coord_index = rng.randrange(len(coords))
            x, y = coords[int(coord_index)]
            x, y = int(x), int(y)
            if "channel_names" not in entry and "channel_names" in payload:
                entry = dict(entry)
                entry["channel_names"] = list(payload["channel_names"])
            if "marker_names" not in entry and "marker_names" in payload:
                entry = dict(entry)
                entry["marker_names"] = list(payload["marker_names"])
        else:
            y = 0 if height <= self.patch_size else rng.randint(0, height - self.patch_size)
            x = 0 if width <= self.patch_size else rng.randint(0, width - self.patch_size)
        try:
            image = _pad_to_patch(reader.read_patch(y, x, self.patch_size), self.patch_size)
        except (OSError, ValueError) as exc:
            raise OSError(
                "Failed to read patch "
                f"path={entry.get('path')} image_id={entry.get('image_id', entry.get('path'))} "
                f"coords_path={entry.get('coords_path', '')} coord_index={coord_index} "
                f"xy=({x}, {y}) z_index={entry.get('z_index', entry.get('reader_kwargs', {}).get('z_index', -1))}"
            ) from exc

        raw_names = entry.get("channel_names") or reader.channel_names()
        if len(raw_names) < image.shape[0]:
            raw_names = list(raw_names) + [f"C{i + 1}" for i in range(len(raw_names), image.shape[0])]
        raw_names = list(raw_names[: image.shape[0]])
        display_marker_names = list(entry.get("marker_names") or raw_names)
        if len(display_marker_names) < image.shape[0]:
            display_marker_names = display_marker_names + raw_names[len(display_marker_names) : image.shape[0]]
        display_marker_names = list(display_marker_names[: image.shape[0]])
        # Prefer biological marker names when metadata supplied them. Raw
        # fluorophore/filter labels are useful for provenance, not identity.
        marker_ids = np.asarray(self.registry.ids_for(display_marker_names), dtype=np.int64)

        keep = np.ones_like(marker_ids, dtype=bool)
        if self.drop_blank:
            keep &= marker_ids != self.registry.blank_marker_id
        if self.drop_unknown:
            keep &= marker_ids != self.registry.unknown_marker_id
        if keep.any():
            image = image[keep]
            marker_ids = marker_ids[keep]
            raw_names = [name for name, ok in zip(raw_names, keep) if bool(ok)]
            display_marker_names = [name for name, ok in zip(display_marker_names, keep) if bool(ok)]

        if self.sort_channels_by_marker_id and marker_ids.size:
            sort_ids = marker_ids.copy()
            sort_ids[sort_ids == self.registry.unknown_marker_id] = np.iinfo(np.int64).max
            order = np.lexsort((np.arange(marker_ids.size), sort_ids))
            image = image[order]
            marker_ids = marker_ids[order]
            raw_names = [raw_names[int(i)] for i in order]
            display_marker_names = [display_marker_names[int(i)] for i in order]

        if self.max_channels is not None and image.shape[0] > int(self.max_channels):
            image = image[: int(self.max_channels)]
            marker_ids = marker_ids[: int(self.max_channels)]
            raw_names = raw_names[: int(self.max_channels)]
            display_marker_names = display_marker_names[: int(self.max_channels)]

        image = _normalize(image, self.normalize)
        pixels = torch.from_numpy(np.ascontiguousarray(image))
        patch_grid = torch.tensor(
            [self.patch_size // self.vit_patch_size, self.patch_size // self.vit_patch_size],
            dtype=torch.long,
        )
        valid_channels = torch.ones((pixels.shape[0],), dtype=torch.bool)
        return {
            "pixels": pixels,
            "image": pixels,
            "marker_ids": torch.from_numpy(marker_ids),
            "valid_channels": valid_channels,
            "patch_grid": patch_grid,
            "marker_names": display_marker_names,
            "raw_channel_names": raw_names,
            "dataset": entry.get("dataset", ""),
            "path": entry["path"],
            "image_id": entry.get("image_id", entry["path"]),
            "reader_type": entry.get("reader_type", ""),
            "coords_path": entry.get("coords_path", ""),
            "coord_index": torch.tensor(-1 if coord_index is None else int(coord_index), dtype=torch.long),
            "z_index": torch.tensor(int(entry.get("z_index", -1)), dtype=torch.long),
            "z_count": torch.tensor(int(entry.get("z_count", 1)), dtype=torch.long),
            "origin_yx": torch.tensor([y, x], dtype=torch.long),
            "spatial_shape": torch.tensor([height, width], dtype=torch.long),
        }


def _generate_training_masks(
    valid_channels: torch.Tensor,
    num_patches: int,
    patch_mask_ratio: float,
    channel_mask_ratio: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, max_c = valid_channels.shape
    patch_mask = torch.zeros((batch, max_c, num_patches), dtype=torch.bool)
    channel_mask = torch.zeros((batch, max_c), dtype=torch.bool)
    if channel_mask_ratio > 0:
        channel_rand = torch.rand((batch, max_c), generator=generator)
        channel_mask = (channel_rand < float(channel_mask_ratio)) & valid_channels
    if patch_mask_ratio > 0:
        patch_rand = torch.rand((batch, max_c, num_patches), generator=generator)
        patch_mask = (patch_rand < float(patch_mask_ratio)) & valid_channels.unsqueeze(-1)
    loss_mask = patch_mask | channel_mask.unsqueeze(-1)
    return patch_mask, channel_mask, loss_mask


def collate_variable_channels(
    samples: Iterable[Dict[str, Any]],
    vit_patch_size: int = 16,
    patch_mask_ratio: float = 0.0,
    channel_mask_ratio: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    samples = list(samples)
    batch = len(samples)
    max_c = max(int(sample["pixels"].shape[0]) for sample in samples)
    _, h, w = samples[0]["pixels"].shape
    pixels = samples[0]["pixels"].new_zeros((batch, max_c, h, w))
    marker_ids = torch.full((batch, max_c), -1, dtype=torch.long)
    valid_channels = torch.zeros((batch, max_c), dtype=torch.bool)
    for i, sample in enumerate(samples):
        c = int(sample["pixels"].shape[0])
        pixels[i, :c] = sample["pixels"]
        marker_ids[i, :c] = sample["marker_ids"]
        valid_channels[i, :c] = True

    grid_y = h // int(vit_patch_size)
    grid_x = w // int(vit_patch_size)
    patch_grid = torch.tensor([[grid_y, grid_x] for _ in range(batch)], dtype=torch.long)
    num_patches = int(grid_y * grid_x)
    patch_mask, channel_mask, loss_mask = _generate_training_masks(
        valid_channels,
        num_patches=num_patches,
        patch_mask_ratio=patch_mask_ratio,
        channel_mask_ratio=channel_mask_ratio,
        generator=generator,
    )
    return {
        "pixels": pixels,
        "image": pixels,
        "marker_ids": marker_ids,
        "valid_channels": valid_channels,
        "patch_grid": patch_grid,
        "patch_mask": patch_mask,
        "channel_mask": channel_mask,
        "loss_mask": loss_mask,
        "marker_names": [sample["marker_names"] for sample in samples],
        "raw_channel_names": [sample["raw_channel_names"] for sample in samples],
        "dataset": [sample["dataset"] for sample in samples],
        "path": [sample["path"] for sample in samples],
        "image_id": [sample.get("image_id", sample["path"]) for sample in samples],
        "reader_type": [sample["reader_type"] for sample in samples],
        "coords_path": [sample["coords_path"] for sample in samples],
        "coord_index": torch.stack([sample["coord_index"] for sample in samples]),
        "z_index": torch.stack([sample.get("z_index", torch.tensor(-1, dtype=torch.long)) for sample in samples]),
        "z_count": torch.stack([sample.get("z_count", torch.tensor(1, dtype=torch.long)) for sample in samples]),
        "origin_yx": torch.stack([sample["origin_yx"] for sample in samples]),
        "spatial_shape": torch.stack([sample["spatial_shape"] for sample in samples]),
    }


def make_sp_dino_collate(
    vit_patch_size: int = 16,
    patch_mask_ratio: float = 0.0,
    channel_mask_ratio: float = 0.0,
    generator: Optional[torch.Generator] = None,
):
    """Create a collate_fn that emits DINO/iBOT-style masks.

    `patch_mask` masks individual image tokens. `channel_mask` masks every image
    token in selected channels. Marker tokens are not represented here and should
    remain unmasked inside the model.
    """

    def _collate(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_variable_channels(
            samples,
            vit_patch_size=vit_patch_size,
            patch_mask_ratio=patch_mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            generator=generator,
        )

    return _collate

"""Small, serializable contracts shared by the toolbox and agent harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ReaderType = Literal["tiff", "tiff_hyperstack", "tiff_z_as_channels", "qptiff", "ims", "openslide_rgb"]
FusionMode = Literal["median", "percentile", "max"]


@dataclass(frozen=True)
class ImageSpec:
    """One input image and the metadata required to process it reproducibly."""

    path: str
    dataset: str = ""
    reader_type: ReaderType = "tiff"
    output_stem: str | None = None
    channel_names: tuple[str, ...] = ()
    mpp: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return Path(self.path).name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channel_names"] = list(self.channel_names)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageSpec":
        known = {"path", "dataset", "reader_type", "output_stem", "channel_names", "mpp", "extras"}
        extras = {**dict(data.get("extras") or {}), **{key: value for key, value in data.items() if key not in known}}
        return cls(
            path=str(data["path"]),
            dataset=str(data.get("dataset", "")),
            reader_type=str(data.get("reader_type", "tiff")),  # type: ignore[arg-type]
            output_stem=data.get("output_stem"),
            channel_names=tuple(str(value) for value in data.get("channel_names", [])),
            mpp=float(data["mpp"]) if data.get("mpp") is not None else None,
            extras=extras,
        )


@dataclass(frozen=True)
class SegmentationProfile:
    """Generic fluorescence foreground parameters, deliberately data-agnostic."""

    name: str = "fluorescence-default"
    fusion: FusionMode = "max"
    threshold_percentile: float = 50.0
    min_signal: float = 8.0
    blur_sigma: float = 4.0
    close_radius: int = 20
    open_radius: int = 2
    dilate_radius: int = 3
    min_component_area_fraction: float = 0.001
    min_foreground_fraction: float = 0.10

    def to_cli_flags(self) -> list[str]:
        """Translate generic settings to the compatibility CLI flags."""
        return [
            "--sp-threshold-percentile", str(self.threshold_percentile),
            "--sp-min-signal", str(self.min_signal),
            "--sp-blur-sigma", str(self.blur_sigma),
            "--sp-close-radius", str(self.close_radius),
            "--sp-open-radius", str(self.open_radius),
            "--sp-dilate-radius", str(self.dilate_radius),
            "--sp-min-component-area-fraction", str(self.min_component_area_fraction),
            "--min-foreground-fraction", str(self.min_foreground_fraction),
        ]


@dataclass(frozen=True)
class IntegrityIssue:
    """A strict decoding failure discovered before segmentation."""

    path: str
    stage: str
    error_type: str
    message: str
    channel: str | None = None
    page_index: int | None = None


@dataclass
class IntegrityReport:
    """Serializable result of strict image decode validation."""

    total: int = 0
    passed: int = 0
    issues: list[IntegrityIssue] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "issues": [asdict(issue) for issue in self.issues],
        }

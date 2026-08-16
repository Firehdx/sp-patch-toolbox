"""Named generic profiles with no source-specific behavior."""

from __future__ import annotations

from ..models import SegmentationProfile


DEFAULT_PROFILES = {
    "fluorescence-default": SegmentationProfile(),
    "fluorescence-recall": SegmentationProfile(
        name="fluorescence-recall",
        threshold_percentile=35.0,
        min_signal=5.0,
        close_radius=12,
        open_radius=0,
        dilate_radius=3,
        min_component_area_fraction=0.0002,
    ),
    "fluorescence-artifact-strict": SegmentationProfile(
        name="fluorescence-artifact-strict",
        threshold_percentile=70.0,
        min_signal=12.0,
        close_radius=8,
        open_radius=3,
        dilate_radius=1,
        min_component_area_fraction=0.002,
    ),
    "fluorescence-grid-artifact": SegmentationProfile(
        name="fluorescence-grid-artifact",
        threshold_percentile=65.0,
        min_signal=12.0,
        close_radius=6,
        open_radius=2,
        dilate_radius=2,
        min_component_area_fraction=0.0015,
    ),
    "fluorescence-sparse-tissue": SegmentationProfile(
        name="fluorescence-sparse-tissue",
        threshold_percentile=20.0,
        min_signal=4.0,
        close_radius=10,
        open_radius=0,
        dilate_radius=5,
        min_component_area_fraction=0.00005,
    ),
}

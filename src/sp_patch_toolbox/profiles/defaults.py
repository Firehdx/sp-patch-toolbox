"""Named generic profiles.  These contain no dataset-specific filenames."""

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
}

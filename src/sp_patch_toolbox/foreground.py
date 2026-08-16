"""Public fluorescence-foreground operations.

The functions are thin, named exports around the validated implementation.
They form the stable API while implementation details continue to be split out
of the legacy compatibility module.
"""

from .compat.legacy_preprocessing import (
    add_roi_max_fusion_foreground,
    fill_roi_component_concavities,
    force_thumbnail_polygons,
    make_sp_fluorescence_foreground,
    read_qptiff_multichannel_thumbnail,
    replace_roi_max_fusion_foreground,
)

__all__ = [
    "make_sp_fluorescence_foreground",
    "read_qptiff_multichannel_thumbnail",
    "add_roi_max_fusion_foreground",
    "replace_roi_max_fusion_foreground",
    "fill_roi_component_concavities",
    "force_thumbnail_polygons",
]

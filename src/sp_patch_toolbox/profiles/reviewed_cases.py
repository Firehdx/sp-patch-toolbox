"""Catalog of reviewed extreme-case corrections.

These corrections are intentionally explicit and opt-in through a dataset
preset.  They are not learned rules and must never be applied merely because a
new image has a similar filename.  The exact validated values remain in the
compatibility implementation during migration; this module exposes a stable,
auditable catalogue for tools and coding agents.
"""

from __future__ import annotations

from typing import Any


REVIEWED_CASE_GROUPS: dict[str, tuple[str, ...]] = {
    "DFCI": (
        "_DFCI_RECALL_RESCUE_STEMS",
        "_DFCI_MXIF_FRAME_ARTIFACT_STEMS",
        "_DFCI_CODEX_PERIPHERAL_ARTIFACT_STEMS",
        "_DFCI_EXTRA_RECALL_STEMS",
        "_DFCI_MAX_FUSION_RESCUE_REGIONS",
    ),
    "WUSTL": (
        "_WUSTL_MAX_FUSION_RECALL_STEMS",
        "_WUSTL_THIN_GRID_CLEANUP_STEMS",
        "_WUSTL_LOCAL_MAX_RESCUE_REGIONS",
        "_WUSTL_LOW_PERCENTILE_RECALL_STEMS",
        "_WUSTL_LOCAL_EXCLUDED_REGIONS",
        "_WUSTL_FORCED_FOREGROUND_POLYGONS",
    ),
    "Vanderbilt": ("_VANDERBILT_CODEX_THIN_STRIP_CLEANUP_STEMS",),
    "OHSU": (
        "_OHSU_CYCIF_ARTIFACT_CLEANUP_STEMS",
        "_OHSU_CYCIF_EXTRA_STRICT_STEMS",
        "_OHSU_CYCIF_EMPTY_MASK_RECALL_STEMS",
        "_OHSU_CYCIF_LOCAL_RECALL_STEMS",
        "_OHSU_CYCIF_MAX_FUSION_RESCUE_REGIONS",
        "_OHSU_CYCIF_MAX_FUSION_REPLACE_STEMS",
        "_OHSU_CYCIF_MAX_FUSION_PROFILES",
        "_OHSU_CYCIF_WEAK_TISSUE_REGIONS",
        "_OHSU_CYCIF_LOCAL_ARTIFACT_EXCLUSIONS",
        "_OHSU_CYCIF_LOCAL_ARTIFACT_POLYGONS",
    ),
    "Stanford": (
        "_STANFORD_RECALL_STEMS",
        "_STANFORD_WEAK_SIGNAL_STEMS",
        "_STANFORD_STRICT_ARTIFACT_STEMS",
        "_STANFORD_LOCAL_MAX_RESCUE_REGIONS",
        "_STANFORD_LOCAL_MAX_RESCUE_PROFILES",
        "_STANFORD_WEAK_SIGNAL_ARTIFACT_STEMS",
        "_STANFORD_EXTRA_STRICT_ARTIFACT_STEMS",
        "_STANFORD_EXTRA_RECALL_STEMS",
        "_STANFORD_MID_RECALL_STEMS",
        "_STANFORD_FINAL_FORCED_FOREGROUND_POLYGONS",
    ),
    "TNP-Sardana": (
        "_TNP_SARDANA_CYCIF_STRICT_STEMS",
        "_TNP_SARDANA_CYCIF_RECALL_STEMS",
        "_TNP_SARDANA_CYCIF_EXTRA_STRICT_STEMS",
    ),
    "TNP-TMA": (
        "_TNP_TMA_CYCIF_STRICT_STEMS",
        "_TNP_TMA_CYCIF_RECALL_STEMS",
        "_TNP_TMA_CYCIF_DENOISE_STEMS",
        "_TNP_TMA_MIHC_STRICT_DENOISE_STEMS",
        "_TNP_TMA_MIHC_EXTRA_STRICT_DENOISE_STEMS",
        "_TNP_TMA_MIHC_BUBBLE_POLYGONS",
        "_TNP_TMA_MIHC_BUBBLE_STEMS",
        "_TNP_TMA_CYCIF_FINAL_FORCED_FOREGROUND_POLYGONS",
    ),
    "HMS": (
        "_HMS_CYCIF_RECALL_STEMS",
        "_HMS_CYCIF_LOCAL_MAX_RESCUE_REGIONS",
        "_HMS_CYCIF_LOCAL_EXCLUDED_REGIONS",
        "_HMS_CYCIF_STRICT_REGIONS",
        "_HMS_ORION_STRICT_PROFILES",
        "_HMS_ORION_WEAK_TISSUE_REGIONS",
        "_HMS_ORION_WEAK_TISSUE_RECOVERY_PROFILES",
        "_HMS_ORION_LOCAL_BACKGROUND_RESEGMENTATION",
        "_HMS_ORION_COMPONENT_CONCAVITY_FILL",
        "_HMS_ORION_FORCED_FOREGROUND_POLYGONS",
        "_HMS_ORION_LOCAL_BACKGROUND_RESEGMENTATION_PROFILES",
    ),
}


def reviewed_case_values() -> dict[str, dict[str, Any]]:
    """Return exact reviewed values from the compatibility implementation."""
    from ..compat import legacy_preprocessing

    values: dict[str, dict[str, Any]] = {}
    for collection, names in REVIEWED_CASE_GROUPS.items():
        missing = [name for name in names if not hasattr(legacy_preprocessing, name)]
        if missing:
            raise RuntimeError(f"Reviewed-case compatibility values missing for {collection}: {missing}")
        values[collection] = {name: getattr(legacy_preprocessing, name) for name in names}
    return values


def reviewed_case_summary() -> dict[str, int]:
    """Return number of rule objects per reviewed collection."""
    return {collection: len(names) for collection, names in REVIEWED_CASE_GROUPS.items()}

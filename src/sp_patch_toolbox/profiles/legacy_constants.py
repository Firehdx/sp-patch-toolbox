"""Exact, reviewed slide-scoped corrections migrated from the legacy pipeline.

Names remain private because callers should select a dataset profile rather
than depend on an individual slide identifier.  Values are maintained here so
they can be reviewed independently from generic image-processing code.
"""

_DFCI_RECALL_RESCUE_STEMS = {
    "1787-1205_1789-1305_Scan1",
    "HTA5_1340_1002_Scan1",
    "HTA5_1380_1002_Scan1",
    "HTA5_1654_1002_Scan1",
    "HTA5_1847_1003_Scan1",
    "HTA5_1876_1006_Scan1",
    "HTA5_1876_1007_Scan1",
    "HTA5_1942_1003_Scan1",
    "melanoma_tma2_15_2200",
    "melanoma_tma2_23_2200",
    "melanoma_tma2_24_2200",
    "melanoma_tma2_30_2200",
    "melanoma_tma2_35_2200",
    "melanoma_tma2_36_2200",
}
_DFCI_MXIF_FRAME_ARTIFACT_STEMS = {"HTA5_1868_1004_Scan1", "HTA5_1872_1003_Scan1"}
_DFCI_CODEX_PERIPHERAL_ARTIFACT_STEMS = {"melanoma_tma2_20_2200"}
_WUSTL_MAX_FUSION_RECALL_STEMS = {
    "HT206B1-H1__20220625",
    "HT243B1-S1H4A4_left__20230512",
    "HT308B1-S1H5A4__20230420",
    "HT308B1-S1H5A4_left__20230512",
    "HT323B1-A1__20220210",
    "HT323B1-H3__20220211",
    "HT339B1-H1A1__20220302",
    "HT339B1-H2A1__20220302",
    "HT397B1-H2A2__20220505",
    "HT480B1-S1H2__20220915",
    "HT565B1-S1H2_left__20230701",
    "HT565B1-S1H2_right__20230701",
}
_WUSTL_THIN_GRID_CLEANUP_STEMS = {"HT271B1-S1H6A5_middle__20230526"}
# S074TLF has ten disconnected, narrow vertical scan bars along the left
# canvas.  Its tissue remains one broad component, so a slide-scoped shape
# cleanup removes only the bars without cropping the left tissue edge.
_VANDERBILT_CODEX_THIN_STRIP_CLEANUP_STEMS = {"S074TLF"}
_OHSU_CYCIF_ARTIFACT_CLEANUP_STEMS = {
    "0000270371",
    "0000384160",
    "0000384176",
    "0000384184",
    "LSP13171",
    "LSP13172",
    "LSP13447",
    "LSP13448",
    "LSP13450",
    "LSP13452",
    "LSP13454",
    "LSP13456",
    "LSP13462",
    "LSP13464",
    "LSP16025",
    "LSP16026",
    "LSP16710",
    "LSP16712",
    "LSP16713",
    "LSP16714",
    "LSP16715",
    "LSP16716",
    "LSP16717",
}
_OHSU_CYCIF_EXTRA_STRICT_STEMS = {"LSP13171", "LSP16026", "LSP16716", "LSP16714"}
_OHSU_CYCIF_EMPTY_MASK_RECALL_STEMS = {"LSP13450", "LSP13454"}
_OHSU_CYCIF_LOCAL_RECALL_STEMS = {"LSP13456", "LSP16710", "LSP16712"}
_OHSU_CYCIF_MAX_FUSION_RESCUE_REGIONS = {
    # Each region is re-segmented from high-percentile marker fusion; for the
    # reviewed failed masks this is the full image, still based on measured
    # signal rather than a force-filled rectangle.
    "LSP13456": [(0.00, 0.00, 1.00, 1.00)],
    "LSP13450": [(0.00, 0.00, 1.00, 1.00)],
    "LSP13454": [(0.00, 0.00, 1.00, 1.00)],
    "LSP16710": [(0.00, 0.00, 1.00, 1.00)],
    "LSP16712": [(0.00, 0.00, 1.00, 1.00)],
}
_OHSU_CYCIF_MAX_FUSION_REPLACE_STEMS = {"LSP13450", "LSP13454", "LSP13456", "LSP16710", "LSP16712"}
_OHSU_CYCIF_MAX_FUSION_PROFILES = {
    # percentile, minimum intensity, minimum connected component fraction
    "LSP13450": (70.0, 12.0, 0.010),
    # Restore LSP13454's pre-recall strict mask: its weaker profile admitted
    # broad scanner residue rather than a genuine tissue edge.
    "LSP13454": (75.0, 15.0, 0.025),
    # Keep the conservative signal threshold for the remainder of the slide;
    # the lower-right scanner residue is removed separately below.
    "LSP13456": (75.0, 15.0, 0.005),
    "LSP16710": (75.0, 15.0, 0.012),
    "LSP16712": (75.0, 15.0, 0.012),
}
_OHSU_CYCIF_WEAK_TISSUE_REGIONS = {
    # Blue annotations: each ROI contains dim tissue and some true background.
    # It is re-segmented from the 75th-percentile marker fusion, never filled.
    "LSP16710": [(0.00, 0.00, 0.64, 0.37), (0.38, 0.58, 0.79, 0.93)],
    "LSP16712": [
        (0.31, 0.00, 0.72, 0.31),
        (0.00, 0.42, 0.27, 0.98),
        (0.60, 0.00, 1.00, 0.50),
        (0.17, 0.75, 0.57, 1.00),
        (0.54, 0.48, 0.68, 0.63),
    ],
}
_OHSU_CYCIF_LOCAL_ARTIFACT_EXCLUSIONS = {
    # Review of the rendered contour shows that these are dark, tiled scanner
    # residue rather than tissue.  They are subtracted after all recovery
    # passes, so no weak-signal rescue can put them back.
    "LSP13171": [(0.53, 0.42, 0.73, 0.62), (0.56, 0.62, 0.73, 0.99)],
    # User-reviewed scanner residue: the complete upper-right quadrant.
    "LSP13450": [(0.50, 0.00, 1.00, 0.60)],
    "LSP13456": [(0.50, 0.50, 1.00, 1.00)],
    # Red annotations: all marked regions are scanner residue without tissue.
    "LSP16710": [(0.00, 0.36, 0.15, 0.86), (0.66, 0.00, 0.81, 0.32), (0.76, 0.12, 1.00, 0.62)],
    "LSP16712": [(0.50, 0.72, 0.60, 0.90), (0.60, 0.60, 1.00, 1.00)],
}
_OHSU_CYCIF_LOCAL_ARTIFACT_POLYGONS = {
    # Red annotation for LSP16712 is a slanted wedge, not a rectangular area.
    # This keeps the tissue immediately below-left of the diagonal boundary.
    "LSP16712": [[(0.66, 0.50), (0.90, 0.18), (1.00, 0.18), (1.00, 0.60), (0.66, 0.60)]],
}
_WUSTL_LOCAL_MAX_RESCUE_REGIONS = {
    # The tissue is visible only in the maximum fusion, while the surrounding
    # stitched canvas has the same bright seam response.  Re-segment only the
    # reviewed tissue-bearing islands instead of using max fusion globally.
    "HT271B1-S1H6A5_middle__20230526": [
        (0.22, 0.12, 0.67, 0.58),
        (0.25, 0.42, 0.64, 0.78),
        (0.35, 0.68, 0.80, 0.98),
    ],
}
_WUSTL_LOW_PERCENTILE_RECALL_STEMS = {"HT339B1-H2A1__20220302"}
_WUSTL_LOCAL_EXCLUDED_REGIONS = {
    # Verified scan-grid segments outside the tissue in this sparse-marker
    # section.  The exclusions are normalized thumbnail ROIs and apply after
    # every recovery pass.
    "HT243B1-S1H4A4_left__20230512": [
        (0.00, 0.66, 0.25, 0.90),
        (0.10, 0.05, 0.95, 0.20),
        (0.86, 0.14, 1.00, 0.68),
    ],
}
_WUSTL_FORCED_FOREGROUND_POLYGONS = {
    # HT397: retain the ordinary signal-derived mask in the top fifth, then
    # union it with the user-reviewed all-tissue lower four fifths.
    "HT397B1-H2A2__20220505": [[(0.0, 0.20), (1.0, 0.20), (1.0, 1.0), (0.0, 1.0)]],
}
_STANFORD_RECALL_STEMS = {
    "A001-C-002.DUPLICATE",
    "A014-C-002",
    "A002-C-016",
    "A002-C-204",
    "A014-C-114",
    "A015-C-010",
    "A015-C-204",
    "A015-C-208",
    "A018-E-021",
    "A055-C-212",
    "A055-C-213",
    "G055",
}
_STANFORD_WEAK_SIGNAL_STEMS = {
    "A002-C-025",
    "A002-C-106",
    "A002-C-205",
    "G025",
    "G044",
    "G051",
}
_STANFORD_STRICT_ARTIFACT_STEMS = {"A002-C-203", "A015-C-005", "A015-C-206"}
_STANFORD_LOCAL_MAX_RESCUE_REGIONS = {
    # Local segmentation only: neither rectangle is filled directly.
    "A001-C-043": [(0.00, 0.50, 0.55, 1.00)],
    "A002-C-021": [(0.00, 0.00, 0.50, 1.00)],
}
_STANFORD_LOCAL_MAX_RESCUE_PROFILES = {
    # A001-C-043's lower-left ROI contains weak tissue but also a broad dim
    # canvas; use a stricter local threshold than the left-half rescue below.
    "A001-C-043": (65.0, 12.0, 0.001),
    "A002-C-021": (35.0, 4.0, 0.00005),
}
_STANFORD_WEAK_SIGNAL_ARTIFACT_STEMS = {"A002-C-025", "A002-C-106", "A002-C-205", "G044"}
_STANFORD_EXTRA_STRICT_ARTIFACT_STEMS = {"A002-C-203", "A015-C-005"}
_STANFORD_EXTRA_RECALL_STEMS = {"A015-C-010"}
_STANFORD_MID_RECALL_STEMS = {"A015-C-206"}

# TNP-Sardana CyCIF: these reviewed scenes retain diffuse scanner/canvas
# signal under the standard OME threshold.  Keep this deliberately narrow
# list so the regular profile remains the default for the rest of the cohort.
_TNP_SARDANA_CYCIF_STRICT_STEMS = {
    "SDA684-5_scene005",
    "SDA684-89_scene006",
    "SDA845-5_scene001",
    "SDA845-5_scene002",
    "SDA845-5_scene005",
    "SDA845-5_scene007",
    "SDA845-5_scene008",
    "SDA845-5_scene017",
    "SDA845-5_scene022",
    "SDA845-89_scene001",
    "SDA845-89_scene002",
    "SDA845-89_scene004",
    "SDA845-89_scene007",
    "SDA845-89_scene008",
    "SDA847-5_scene001",
}
_TNP_SARDANA_CYCIF_RECALL_STEMS = {
    "SDA684-89_scene006",
    "SDA845-89_scene022",
}
_TNP_SARDANA_CYCIF_EXTRA_STRICT_STEMS = {
    "SDA845-5_scene005",
    "SDA845-89_scene007",
    "SDA847-5_scene001",
}

# TNP-TMA CyCIF review.  These sets are deliberately slide-scoped because the
# all-marker maximum projection produces distinct failure modes among TMA cores.
_TNP_TMA_CYCIF_STRICT_STEMS = {
    "OHSU_TMA1_004-A2",
    "OHSU_TMA1_004-B2",
    "OHSU_TMA1_004-B8",
    "OHSU_TMA1_004-D5",
}
_TNP_TMA_CYCIF_RECALL_STEMS = {
    "OHSU_TMA1_004-A3",
    "OHSU_TMA1_004-C6",
    "OHSU_TMA1_004-D2",
    "OHSU_TMA1_004-E1",
}
_TNP_TMA_CYCIF_DENOISE_STEMS = {
    "OHSU_TMA1_004-A2",
    "OHSU_TMA1_004-A3",
    "OHSU_TMA1_004-B8",
    "OHSU_TMA1_004-C6",
    "OHSU_TMA1_004-C11",
    "OHSU_TMA1_004-D1",
    "OHSU_TMA1_004-D2",
    "OHSU_TMA1_004-D5",
    "OHSU_TMA1_004-D11",
    "OHSU_TMA1_004-E1",
}
_TNP_TMA_MIHC_STRICT_DENOISE_STEMS = {
    "TMA06_ROI02_A2",
    "TMA06_ROI07_A7",
    "TMA06_ROI10_A10",
    "TMA06_ROI11_A11",
    "TMA06_ROI13_B2",
    "TMA06_ROI19_B8",
    "TMA06_ROI22_B11",
    "TMA06_ROI33_C11",
    "TMA06_ROI38_D5",
    "TMA06_ROI42_D9",
    "TMA06_ROI44_D11",
}
_TNP_TMA_MIHC_EXTRA_STRICT_DENOISE_STEMS = {"TMA06_ROI19_B8"}
# E2's bubble rims are connected to the core by the closing/dilation stage,
# so a component-size filter alone cannot remove them. These two polygons
# trace only the manually reviewed lower bubble cluster and central bubble.
_TNP_TMA_MIHC_BUBBLE_POLYGONS = {
    "TMA06_ROI46_E2": [
        # Bottom edge bubble cluster, including the upper arcs that remained
        # after the first conservative exclusion.
        [(0.00, 0.82), (0.14, 0.80), (0.25, 0.81), (0.34, 0.79), (0.46, 0.82), (0.58, 0.88), (0.70, 0.88), (0.74, 1.00), (0.00, 1.00)],
        [(0.12, 0.78), (0.24, 0.78), (0.30, 0.83), (0.28, 0.90), (0.16, 0.90), (0.12, 0.84)],
        # Central bubble, expanded to its full visible rim while retaining
        # the adjacent tissue bridge above and to the right.
        [(0.45, 0.54), (0.56, 0.49), (0.69, 0.49), (0.80, 0.55), (0.83, 0.65), (0.80, 0.75), (0.72, 0.86), (0.62, 0.91), (0.51, 0.90), (0.44, 0.83), (0.42, 0.71)],
    ],
}
_TNP_TMA_MIHC_BUBBLE_STEMS = {"TMA06_ROI45_E1", "TMA06_ROI46_E2"}

# Manually reviewed TNP-TMA CyCIF tissue.  Coordinates are normalized blue
# annotation boundaries from trident_job/ref.  They are applied only after
# ordinary segmentation and therefore *union* with, rather than replace, the
# existing foreground mask.
_TNP_TMA_CYCIF_FINAL_FORCED_FOREGROUND_POLYGONS = {
    "OHSU_TMA1_004-A3": [
        [(0.882, 0.273), (0.851, 0.307), (0.781, 0.601), (0.847, 0.650), (0.916, 0.642), (0.939, 0.608), (0.940, 0.391)],
    ],
    "OHSU_TMA1_004-C6": [
        [(0.097, 0.366), (0.102, 0.610), (0.240, 0.858), (0.595, 0.776), (0.763, 0.617), (0.706, 0.476), (0.460, 0.338), (0.189, 0.300)],
    ],
    "OHSU_TMA1_004-D2": [
        [(0.461, 0.046), (0.409, 0.335), (0.667, 0.447), (0.762, 0.706), (0.843, 0.727), (0.918, 0.514), (0.879, 0.221), (0.697, 0.082)],
        [(0.181, 0.599), (0.163, 0.648), (0.242, 0.821), (0.309, 0.861), (0.369, 0.855), (0.410, 0.765), (0.369, 0.641), (0.290, 0.599)],
    ],
    "OHSU_TMA1_004-E1": [
        [(0.420, 0.162), (0.110, 0.489), (0.157, 0.743), (0.362, 0.870), (0.486, 0.747), (0.537, 0.538), (0.533, 0.228)],
    ],
}
_STANFORD_FINAL_FORCED_FOREGROUND_POLYGONS = {
    # Explicitly user-reviewed all-tissue images/regions. These are applied
    # after cleanup so no generic artifact filter can remove them.
    "A001-C-007": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
    "B001-A-301": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
    "B001-A-401": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
    "B001-A-101": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0 / 3.0), (0.0, 1.0 / 3.0)]],
}
_DFCI_EXTRA_RECALL_STEMS = {
    "1787-1205_1789-1305_Scan1",
    "HTA5_1340_1002_Scan1",
    "HTA5_1654_1002_Scan1",
    "HTA5_1942_1003_Scan1",
}
_DFCI_MAX_FUSION_RESCUE_REGIONS = {
    # Reviewed low-signal tissue regions. Each region is normalized x0,y0,x1,y1
    # on the thumbnail; only max-fusion pixels that independently pass the
    # local segmentation are added back.
    "HTA5_1654_1002_Scan1": [(0.35, 0.00, 1.00, 0.60)],
    "HTA5_1942_1003_Scan1": [(0.00, 0.50, 1.00, 1.00)],
}

# These HMS CyCIF sections contain broad, genuine low-DNA tissue.  They are
# explicitly reviewed exceptions to the general CyCIF DNA-gated artifact
# profile; foreground is still fluorescence-derived, never rectangle-filled.
_HMS_CYCIF_RECALL_STEMS = {"CRC03", "CRC04", "CRC05", "CRC06"}
_HMS_CYCIF_LOCAL_MAX_RESCUE_REGIONS = {
    # LSP11060: two detached, marker-positive tissue fragments at the far
    # left.  These are computational ROIs only; local segmentation decides
    # which pixels are restored.
    "LSP11060": [(0.00, 0.08, 0.17, 0.48), (0.00, 0.48, 0.15, 0.80)],
}
_HMS_CYCIF_LOCAL_EXCLUDED_REGIONS = {
    # Intentionally empty: direct rectangular exclusions created hard mask
    # edges through real tissue. LSP11060 uses a right-half stricter signal
    # segmentation instead (see _HMS_CYCIF_STRICT_REGIONS).
    "LSP11060": [],
}
_HMS_CYCIF_STRICT_REGIONS = {"LSP11060": [(0.50, 0.00, 1.00, 1.00)]}
_HMS_ORION_STRICT_PROFILES = {
    # foreground percentile, minimum retained connected-component fraction
    "LSP14398_P54_A31_C100_HMS_Orion7_20221201_201503_606038-zlib": (70.0, 0.001),
    "LSP15284_P54_A31_C100_HMS_Orion7_20230126_171736_478335-zlib": (70.0, 0.002),
    "P37_S31_A24_C59kX_E15_20220106_014409_014236-zlib": (70.0, 0.002),
    "P37_S62_Full_A24_C59nX_E15_20220224_011204_784145-zlib": (85.0, 0.003),
    "P37_S82_Full_A24_C59qX_E15_20220304_200614_832683-zlib": (80.0, 0.002),
}
_HMS_ORION_WEAK_TISSUE_REGIONS = {
    # Reviewed low-signal tissue zones. Coordinates are normalized thumbnail
    # x0,y0,x1,y1. Each zone is locally re-segmented; it is never filled.
    "LSP14398_P54_A31_C100_HMS_Orion7_20221201_201503_606038-zlib": [(0.00, 0.00, 1.00, 0.38)],
    "LSP15304_P54_A31_C100_HMS_Orion7_20230124_023055_240792-zlib": [(0.00, 0.05, 0.40, 0.95), (0.60, 0.58, 1.00, 1.00)],
    "P37_S30_A24_C59kX_E15_20220106_014319_409148-zlib": [(0.42, 0.18, 1.00, 0.82)],
    "P37_S31_A24_C59kX_E15_20220106_014409_014236-zlib": [(0.00, 0.00, 1.00, 0.38)],
    "P37_S32_A24_C59kX_E15_20220106_014630_553652-zlib": [(0.00, 0.55, 1.00, 1.00)],
    "P37_S34_A24_C59kX_E15_20220107_202112_212579-zlib": [(0.00, 0.00, 1.00, 0.35), (0.00, 0.52, 0.48, 1.00)],
    "P37_S38_A24_C59kX_E15_20220108_012130_664519-zlib": [(0.00, 0.00, 1.00, 0.38)],
    "P37_S45_Full_A24_C59mX_E15_20220128_171409_633341-zlib": [(0.60, 0.05, 1.00, 1.00)],
    "P37_S48_Full_A24_C59mX_E15_20220129_015105_865195-zlib": [(0.25, 0.12, 0.78, 0.72)],
    "P37_S52_Full_A24_C59mX_E15_20220129_015324_574779-zlib": [(0.25, 0.00, 0.82, 0.48), (0.62, 0.00, 1.00, 0.35)],
    "P37_S60_Full_A24_C59nX_E15_20220224_011127_971497-zlib": [(0.28, 0.00, 0.82, 0.48), (0.62, 0.00, 1.00, 0.42)],
    "P37_S65_Full_A24_C59nX_E15_20220224_011333_386280-zlib": [(0.00, 0.50, 1.00, 1.00)],
    "P37_S74_Full_A24_C59qX_E15_20220302_234837_137590-zlib": [(0.00, 0.00, 1.00, 0.38), (0.60, 0.00, 1.00, 0.45)],
    "P37_S75_Full_A24_C59qX_E15_20220302_235001_586560-zlib": [(0.58, 0.00, 1.00, 1.00)],
    "P37_S80_Full_A24_C59qX_E15_20220307_235159_333000-zlib": [(0.42, 0.04, 0.78, 0.58)],
    "P37_S81_Full_A24_C59qX_E15_20220302_235331_704703-zlib": [
        (0.70, 0.25, 1.00, 0.72),  # reviewed weak tissue at the right
        (0.58, 0.55, 1.00, 1.00),  # previously reviewed lower-right tissue
    ],
    "P37_S83_Full_A24_C59qX_E15_20220304_200429_490805-zlib": [(0.25, 0.00, 0.82, 0.48), (0.62, 0.00, 1.00, 0.42)],
}

# Per-slide local recall thresholds for the weak, but genuine, Orion tissue
# zones reviewed after the first local pass.  They only affect the listed ROI.
_HMS_ORION_WEAK_TISSUE_RECOVERY_PROFILES = {
    "P37_S30_A24_C59kX_E15_20220106_014319_409148-zlib": (20.0, 3.0),
    "P37_S48_Full_A24_C59mX_E15_20220129_015105_865195-zlib": (60.0, 6.0),
    "P37_S80_Full_A24_C59qX_E15_20220307_235159_333000-zlib": (45.0, 5.0),
    "P37_S81_Full_A24_C59qX_E15_20220302_235331_704703-zlib": (60.0, 6.0),
}

# P37_S74's upper recovery is correct.  These lower areas instead contain
# background swallowed by the original global hole fill, so re-segment only
# those regions and replace their local mask values.
_HMS_ORION_LOCAL_BACKGROUND_RESEGMENTATION = {
    "P37_S74_Full_A24_C59qX_E15_20220302_234837_137590-zlib": [
        (0.00, 0.62, 0.38, 1.00),
        (0.30, 0.50, 0.76, 1.00),
    ],
    "P37_S80_Full_A24_C59qX_E15_20220307_235159_333000-zlib": [
        (0.00, 0.78, 0.36, 1.00),
        (0.00, 0.88, 1.00, 1.00),
    ],
}

# Documented U-shaped weak-signal gaps.  Filling is based on the convex hull
# of already detected tissue components, not on the rectangular ROI extent.
_HMS_ORION_COMPONENT_CONCAVITY_FILL = {
    "P37_S30_A24_C59kX_E15_20220106_014319_409148-zlib": [(0.54, 0.38, 1.00, 1.00)],
    "P37_S48_Full_A24_C59mX_E15_20220129_015105_865195-zlib": [(0.34, 0.10, 0.68, 0.90)],
}

# P37_S48's remaining central U is confirmed tissue but carries no usable
# marker signal.  The requested correction follows the U-shaped tissue gap,
# rather than filling the surrounding rectangular ROI.
_HMS_ORION_FORCED_FOREGROUND_POLYGONS = {
    "P37_S48_Full_A24_C59mX_E15_20220129_015105_865195-zlib": [
        [(0.37, 0.00), (0.61, 0.00), (0.34, 0.70)],
    ],
}

# Re-segmentation of manually reviewed lower-background ROIs uses a stricter
# whole-thumbnail cutoff.  It does not touch each slide's rescued upper ROI.
_HMS_ORION_LOCAL_BACKGROUND_RESEGMENTATION_PROFILES = {
    "P37_S74_Full_A24_C59qX_E15_20220302_234837_137590-zlib": (72.0, 7.0),
    "P37_S80_Full_A24_C59qX_E15_20220307_235159_333000-zlib": (70.0, 7.0),
}

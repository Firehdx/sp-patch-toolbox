# Extreme-case correction guide

## Principle

Generic segmentation should solve normal images. A reviewed exception is a
small, documented operation layered on top of an already valid global mask.
Every reviewed rule in `profiles.reviewed_cases` corresponds to a named image
or explicit dataset profile from prior QC; it is not a heuristic template.

| Failure mode | Evidence | Safe response |
|---|---|---|
| LZW/decode failure | integrity report | exclude/re-export source; never zero-fill or drop a channel |
| QPTIFF outer scanner ring | contour shows peripheral frame | remove thin/sparse edge component, preserving connected tissue |
| tile-grid extrema | periodic thin lines across thumbnail | local/grid artifact cleanup before hole fill |
| weak genuine tissue | annotated low-signal region | maximum-channel ROI re-segmentation, then paste only detected components |
| local artifact | annotated artifact-only region | local stricter threshold, replacement or polygon exclusion |
| U-shaped tissue gap | annotation confirms concavity | component-based concavity fill, not rectangle fill |
| bubbles | circular annotated non-tissue region | exact bubble polygon exclusion |

## Required review record

For each new exception, keep:

1. input slide path and checksum/mtime;
2. original QC overlay and an annotated reference;
3. profile and normalized polygon/ROI parameters;
4. new QC overlay, coordinate count before/after and reasoning;
5. a statement that no unrelated slide is affected.

## Existing reviewed groups

The catalogue contains corrections for DFCI, WUSTL, Vanderbilt, OHSU, Stanford,
TNP-Sardana, TNP-TMA and HMS. Use `sppatch profiles` to enumerate groups. The
exact values are preserved in the compatibility layer during the migration and
are checked by tests so that a refactor cannot silently delete them.

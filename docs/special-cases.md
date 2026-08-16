# Extreme-case correction guide

## Principle

Generic segmentation should solve normal images. An exceptional image is a
small, documented operation layered on top of an already valid global mask.
Every correction must be explicitly supplied with the run configuration; it
must not be inferred from a filename, directory, collection or database.

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
3. selected generic profile and normalized polygon/ROI parameters;
4. new QC overlay, coordinate count before/after and reasoning;
5. a statement that no unrelated slide is affected.

## Generic presets

Use `sppatch profiles` to list the available generic presets. They describe
observable image conditions—such as weak sparse tissue, diffuse background or
grid-like artifacts—rather than a source collection. Apply them only after
reviewing representative QC thumbnails.

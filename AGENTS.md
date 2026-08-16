# Agent operating contract

This repository processes scientific source data. Correctness and traceability
are more important than throughput.

## Non-negotiable rules

1. Do not write to raw images. Do not overwrite output outside the exact slide
   list requested by the user.
2. Run `sppatch integrity` before generating coordinates for a new collection.
   A slide with any strict decoding error is excluded until its source data is
   replaced or re-exported.
3. Never infer marker identities from a dye or filter name alone. Preserve the
   raw channel label unless an image-aligned metadata mapping is explicitly
   supplied.
4. Keep the generic foreground profile generic. A manual correction requires
   a named slide, documented QA evidence, normalized ROI coordinates and a
   reason. A rectangle is never automatic foreground.
5. Save/inspect QC artifacts before declaring a segmentation successful:
   thumbnail, binary mask, contour overlay, GeoJSON and coordinate count.
6. Prefer a small representative run before an entire dataset. Preserve
   successful outputs when retrying failed or reviewed slides.

## Required task sequence

Follow these cards in order:

1. [`harness/tasks/01-discovery.md`](harness/tasks/01-discovery.md)
2. [`harness/tasks/02-integrity-and-baseline.md`](harness/tasks/02-integrity-and-baseline.md)
3. [`harness/tasks/03-reviewed-corrections.md`](harness/tasks/03-reviewed-corrections.md)
4. [`harness/tasks/04-release-and-handoff.md`](harness/tasks/04-release-and-handoff.md)

## Definition of done

A run is complete only when the manifest, HDF5 output, channel/marker metadata,
QC artifacts, integrity report and exception log agree on the same slide set.
Do not describe an empty mask, decoder failure, unresolved channel names or an
unreviewed special-case correction as successful.

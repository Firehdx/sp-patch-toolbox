# Task 01 — Dataset discovery and modality audit

## Goal

Produce a manifest only for images that are in scope and have sufficient
channel/marker provenance.

## Required checks

- Enumerate directories and extensions; distinguish raw WSIs from ROI/tile,
  H&E/RGB and derived images.
- Inspect image axes, shape, native channel count, page/pyramid layout and mpp.
- Extract embedded channel labels from OME XML, ImageJ metadata, QPTIFF page
  descriptions or IMS metadata.
- If channel labels are dye/filter names, locate image-aligned panel/protocol
  metadata. Do not map a dye or filter label to a marker without that evidence.
- Emit unresolved images as an explicit exclusion report, not guessed channels.

## Deliverables

- `images.jsonl`
- `channel_audit.md` with per-format counts and unresolved items
- a small representative selection for baseline segmentation

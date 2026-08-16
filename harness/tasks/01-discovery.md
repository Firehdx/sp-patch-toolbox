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
- If channel labels are dye/filter names, locate the case and panel protocol
  metadata. Do not map an Opal wavelength globally across cases.
- Emit unresolved images as an explicit exclusion report, not guessed channels.

## Deliverables

- `images.jsonl`
- `channel_audit.md` with per-format counts and unresolved items
- a small representative selection for baseline segmentation

# Task 02 — Strict integrity gate and generic baseline

## Goal

Prove that source pixels can be read and assess the generic profile before a
collection-wide run.

## Procedure

1. Run `sppatch integrity` on the complete candidate manifest.
2. Stop and report failed slides. Do not zero-fill, skip a damaged channel or
   create coordinate HDF5 for them.
3. Run one representative image per modality with the generic profile.
4. Inspect thumbnail, foreground mask, contour overlay and coordinate count.
5. Record parameters and exact command in the run log.

## Acceptance

- all processed slides passed strict integrity;
- channel count/order in HDF5 equals the native source;
- output paths are collision-free and reproducible from the manifest;
- no profile contains filename-specific settings.

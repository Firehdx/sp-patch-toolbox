# Architecture

## Design boundary

The old project grew around a successful operational workflow. Its readers,
metadata logic, segmentation functions, presets and manually reviewed patches
live in one module, which made it hard to reuse safely. This repository splits
responsibilities without invalidating prior results.

```text
manifest + verified metadata
            |
            v
  io readers / channel inspection ----> strict integrity gate
            |                                  |
            v                                  v
 marker normalization                    failure report (stop)
            |
            v
 generic all-channel thumbnail fusion
            |
            v
 generic foreground mask ---> reviewed local rule layer (optional, named only)
            |
            v
 contours / GeoJSON / coordinate HDF5 / QC artifacts
```

## Modules

- `io.readers`: native region readers. It owns TIFF axis interpretation,
  QPTIFF page access, IMS projections and OpenSlide RGB inspection.
- `metadata.markers`: normalization and marker IDs. Dataset metadata adapters
  must prove a mapping before overriding image labels.
- `foreground`: public mask, QPTIFF thumbnail fusion and local re-segmentation
  operations.
- `coordinates`: HDF5 contract and TRIDENT-coordinate extraction.
- `pipeline.integrity`: strict, full-page decoding at the actual thumbnail
  level. It is the gate between discovery and processing.
- `profiles`: generic named parameter presets and explicit reviewed rules.
- `compat`: exact, validated implementation retained while source functions are
  migrated in small commits. It is a compatibility boundary, not a place for
  new special cases.

## HDF5 coordinate contract

Every coordinate HDF5 contains at least:

- `coords`: int64 `[N, 2]`, level-0 `(x, y)` patch top-left coordinates;
- `channel_names`: UTF-8 native source channel names;
- `marker_names`: UTF-8 resolved marker names in exactly the same order;
- attributes for source path, reader type, patch size, image dimensions, mpp,
  foreground method and foreground parameters.

Patch pixels are read lazily from the original source. Therefore a successful
coordinate file is valid only if strict source integrity passed first.

## Refactor roadmap

1. Compatibility baseline (current): preserve all working formats, presets and
   reviewed rules under `compat/`, with new public APIs and harness.
2. Move format thumbnail builders and morphology primitives into `io/` and
   `foreground/` with synthetic tests.
3. Convert reviewed constants into versioned JSON/YAML records with an
   evidence field and a rule executor.
4. Retire `compat/` only after parity tests compare mask/coordinate outputs on
   a frozen representative corpus.

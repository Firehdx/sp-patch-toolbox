# SP Patch Toolbox

`sp-patch-toolbox` is a reproducible toolbox for multi-format spatial-proteomics images:

- format-aware marker image readers: OME-TIFF/TIFF, QPTIFF and IMS;
- marker and channel metadata normalization;
- fluorescence foreground segmentation from all available biological channels;
- TRIDENT-compatible contour, coordinate and patch-coordinate HDF5 generation;
- strict source-integrity gates before segmentation;
- generic, opt-in artifact and weak-signal handling options;
- an agent harness that makes discovery, QA and exception handling auditable.

It is intentionally **fail closed**: a decoding failure in one QPTIFF marker
page invalidates the whole slide for patch generation. Pixels are never
silently replaced, channels are never silently omitted, and an ROI rectangle is
never itself treated as foreground.

## Repository layout

```text
src/sp_patch_toolbox/
  io/           format-aware readers and safe region reads
  metadata/     marker normalization and registry
  foreground.py public foreground API
  coordinates.py coordinate-HDF5 and extraction API
  training/     lazy pixel reader and variable-channel batch collation
  pipeline/     manifests and strict integrity preflight
  profiles/     generic segmentation presets
  compat/       validated legacy implementation during incremental migration
harness/        instructions, task cards and acceptance criteria for agents
docs/           architecture, modality and special-case guidance
tests/          dependency-light regression tests
```

## Installation

```bash
git clone <your-github-url>/sp-patch-toolbox.git
cd sp-patch-toolbox
python -m pip install -e '.[fluorescence,dev]'
```

The optional TRIDENT extraction stage also requires a compatible local TRIDENT
checkout and its dependencies. The toolbox deliberately does not vendor
TRIDENT.

## Recommended workflow

1. Build or review a JSONL manifest. Each row declares the image path,
   `reader_type`, dataset and any verified marker/mpp overrides.
2. Run a strict integrity gate before segmentation.
3. Segment one representative slide with the generic profile and inspect the
   thumbnail, mask, contour and coordinates.
4. Process the collection with the selected generic profile.
5. For an exceptional image, save QA evidence and pass explicit ROI/polygon
   parameters in its manifest or invocation; never add filename branches.

```bash
sppatch integrity --manifest images.jsonl --data-root /data/sp --out integrity.json

# Existing, validated coordinate pipeline. `--` separates toolbox and legacy flags.
sppatch segment -- --manifest images.jsonl --data-root /data/sp --out-dir /data/patches \
  --method sp-fluorescence --patch-size 224 --overlap 0
```

## Input manifest

```json
{"dataset":"example","path":"collection/slide.qptiff","reader_type":"qptiff","mpp":0.5}
```

Use an explicit `channel_names` override only when it comes from a verified
panel/metadata source and the number and order exactly match native image
planes. See [`docs/metadata-and-modality.md`](docs/metadata-and-modality.md).

## Strict QPTIFF integrity gate

For a QPTIFF, the preflight locates the same pyramid group used to build the
foreground thumbnail and fully decodes **every native channel page**, including
technical channels. One invalid LZW tile/strip is enough to report the slide as
bad. This protects later lazy patch reads from failing halfway through training.

The command returns non-zero if any image fails and emits machine-readable JSON
containing source path, channel, page index and decoder exception.

## Agent harness

Start with [`AGENTS.md`](AGENTS.md), then use the task cards under
[`harness/tasks`](harness/tasks). The harness separates discovery, baseline
processing, QA and special-case work; it prevents agents from guessing marker
identities, applying a correction to an entire dataset, or overwriting known
good output.

## Migration status

The validated implementation is retained under `compat/` while public modules
are migrated. New work must go into the public modules or configuration; do
not add source-specific branches to `compat/`.

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/special-cases.md`](docs/special-cases.md).

## Licensing

No license is selected yet. Choose one only after confirming that every copied
or contributed component may be distributed under that license.

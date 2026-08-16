# Coding-agent harness

This directory is a reusable execution contract rather than a prompt dump.
An agent receives a manifest plus a task card and returns the listed artifacts.

| Stage | Input | Required output | Stop condition |
|---|---|---|---|
| Discovery | raw directory + metadata | manifest, modality/channel audit | marker identity unresolved |
| Integrity | manifest | strict decode report | any failed source image |
| Baseline | one representative slide | thumbnail, mask, contour, coordinate HDF5 | QC not accepted |
| Correction | annotated QC evidence | explicit manifest-scoped rule + rerun evidence | rule expands beyond declared scope |
| Release | completed output | counts, checksums/manifest, exception log | output and manifest disagree |

Use the root [`AGENTS.md`](../AGENTS.md) as the authoritative safety contract.

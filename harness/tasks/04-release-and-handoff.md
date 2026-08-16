# Task 04 — Release and handoff

## Required artifacts

- input manifest and strict integrity JSON;
- output coordinate manifest, with failures retained explicitly;
- marker registry version and any external panel metadata used;
- per-manifest patch/slide/channel totals;
- explicit-correction change log and QC references;
- exact command(s), package commit and environment details.

## Final checks

- no successful row points to a missing HDF5;
- no failed source has a newly generated coordinate HDF5;
- all output stems are unique;
- every channel-to-marker assignment has an evidence source;
- special-case rules are slide-scoped and documented.

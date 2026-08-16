# Metadata and modality guide

## Channel is not marker

An image channel may be a biological marker (`CD8`), a dye (`Opal 570`), a
filter (`Cy5 MSI`), a cycle/channel coordinate, DAPI, autofluorescence or RGB.
Only biological markers should be presented as resolved marker identities.

For case-specific Opal panels, the only safe resolution chain is:

```text
raw slide -> case directory -> panel/protocol -> metadata dye row -> marker
```

Never reuse an `Opal 480 -> marker` mapping across cases without evidence.

## Format-specific rules

| Format | Typical axes/layout | Safe source of channel names | Notes |
|---|---|---|---|
| QPTIFF | page-per-channel, repeated pyramid groups | PerkinElmer page XML + case/panel metadata | Strictly decode all channel pages before use. |
| OME-TIFF | CYX/CZYX/TCYX or page stack | OME XML `Channel@Name` | Inspect axes; Z is not automatically marker C. |
| ImageJ TIFF | labels may enumerate T×C planes | ImageJ labels | Validate label count against non-spatial plane count. |
| IMS | HDF5 channel groups and resolution levels | Imaris channel metadata | Prefer native lower resolution for thumbnails. |
| RGB/OpenSlide | rendered 3-channel image | not SP marker metadata | Inspection/brightfield only unless separately curated. |

## Z versus C decision

Do not call Z a marker channel based on dimensionality alone. Use OME XML,
ImageJ labels, acquisition metadata, Minerva/panel records and the expected
marker count. If the evidence is incomplete, stop before SP patch generation.

## mpp

Prefer physical pixel size embedded in the raw image. Viewer display settings
or a portal card may be a rendering scale rather than source mpp. When a
fallback is necessary, record the source and override explicitly in the
manifest.

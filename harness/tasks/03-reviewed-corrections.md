# Task 03 — Reviewed local corrections

## When this task is allowed

Only after baseline QC identifies a named failure and the user supplies an
annotation or unambiguous written ROI description.

## Allowed correction patterns

- stricter local artifact threshold or local exclusion polygon;
- local maximum-channel re-segmentation for weak tissue;
- replacement of a local mask region after re-segmentation;
- component concavity fill when a tissue U-shape is explicitly documented;
- verified bubble or frame-artifact polygon exclusion.

## Prohibited shortcuts

- filling an ROI rectangle as foreground;
- applying a named-slide rule to similar-looking slides;
- reducing source-integrity requirements;
- changing global defaults to solve one slide.

## Deliverables

For every correction: source slide, QC reference, normalized geometry, before /
after overlay, parameter values, rationale and rerun command. Add it to the
reviewed-case registry only after acceptance.

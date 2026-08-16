import argparse
import json
import re
import sys
from pathlib import Path


from .legacy_preprocessing import extract_sp_fluorescence_coords, extract_trident_coords, load_sp_coords_h5
from ..profiles.defaults import DEFAULT_PROFILES


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            row = json.loads(line)
            if "error" not in row:
                rows.append(row)
    return rows


def safe_stem(path: str) -> str:
    stem = Path(path).name
    stem = re.sub(r"\.(ome\.)?(tif|tiff|qptiff|ims)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem.strip("_") or "sample"


def artifact_stem(entry: dict) -> str:
    """Return one filesystem-safe stem for every generated per-slide artifact."""
    return safe_stem(str(entry.get("output_stem") or entry["path"]))


def contour_exists(contours_dir: Path, entry: dict) -> bool:
    stem = artifact_stem(entry)
    return any((contours_dir / f"{stem}{suffix}").exists() for suffix in [".jpg", ".jpeg", ".png"])


def infer_reader_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".ims"):
        return "ims"
    if lower.endswith(".qptiff"):
        return "qptiff"
    return "tiff"


def parse_thumbnail_regions(values: list[str]) -> list[tuple[float, float, float, float]]:
    regions = []
    for value in values:
        try:
            region = tuple(float(item.strip()) for item in value.split(","))
        except ValueError as exc:
            raise ValueError(f"Invalid thumbnail region {value!r}; expected x0,y0,x1,y1 in [0,1].") from exc
        if len(region) != 4 or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
            raise ValueError(f"Invalid thumbnail region {value!r}; expected x0<x1 and y0<y1 in [0,1].")
        regions.append(region)
    return regions


def make_single_entry(args) -> dict:
    path = Path(args.image).expanduser()
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = str(path.resolve())
    else:
        rel = str(path.resolve())
    return {
        "dataset": args.dataset_name or "single",
        "path": rel,
        "reader_type": args.reader_type or infer_reader_type(rel),
    }


def source_path_for(entry: dict, data_root: Path | None) -> Path:
    path = Path(str(entry["path"])).expanduser()
    if path.is_absolute():
        return path
    if data_root is None:
        return path.resolve()
    return (data_root / path).resolve()


def write_coords_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def merge_manifest_rows(base_rows: list[dict], update_rows: list[dict]) -> list[dict]:
    updates = {row.get("path", ""): row for row in update_rows if row.get("path")}
    merged = []
    seen = set()
    for row in base_rows:
        key = row.get("path", "")
        if key in updates:
            merged.append(updates[key])
            seen.add(key)
        else:
            merged.append(row)
    for key, row in updates.items():
        if key not in seen:
            merged.append(row)
    return merged


def rename_trident_artifacts(produced_coords: Path, job_dir: Path, output_stem: str) -> Path:
    """Give all per-slide TRIDENT artifacts the same collision-free stem.

    TRIDENT derives its own artifact name from the raw filename, while the
    final patch HDF5 may use a case/directory prefix.  Renaming the generated
    artifacts immediately after a slide finishes keeps thumbnails, contours,
    masks, GeoJSON and the intermediate coordinate HDF5 mutually traceable.
    """
    suffix = "_patches.h5"
    raw_stem = produced_coords.name[: -len(suffix)] if produced_coords.name.endswith(suffix) else produced_coords.stem
    if raw_stem == output_stem:
        return produced_coords

    for directory_name, extension in [
        ("thumbnails", ".jpg"),
        ("contours", ".jpg"),
        ("foreground_masks", ".png"),
        ("contours_geojson", ".geojson"),
    ]:
        source = job_dir / directory_name / f"{raw_stem}{extension}"
        target = job_dir / directory_name / f"{output_stem}{extension}"
        if source.exists():
            source.replace(target)

    target_coords = produced_coords.with_name(f"{output_stem}{suffix}")
    if produced_coords.exists():
        produced_coords.replace(target_coords)
    return target_coords


def process_entry(entry: dict, args, data_root: Path | None, out_dir: Path) -> dict:
    reader_type = entry.get("reader_type") or infer_reader_type(entry["path"])
    source_path = source_path_for(entry, data_root)
    slide_name = artifact_stem(entry)
    coords_dir = out_dir / f"patch_{args.patch_size}_overlap_{args.overlap}" / "patches"
    coords_path = coords_dir / f"{slide_name}_patches.h5"
    channel_names_override = entry.get("channel_names")

    if args.method == "sp-fluorescence":
        produced = extract_sp_fluorescence_coords(
            source_path,
            trident_root=args.trident_root,
            job_dir=out_dir / "trident_job",
            marker_registry_path=args.registry,
            data_root=data_root,
            reader_type=reader_type,
            dataset=entry.get("dataset", ""),
            patch_size=args.patch_size,
            mag=args.mag,
            overlap=args.overlap,
            trident_reader_type=args.trident_reader_type,
            channel_names_override=channel_names_override,
            mpp=entry.get("mpp", args.mpp),
            min_tissue_proportion=args.min_foreground_fraction,
            thumbnail_max_size=args.sp_thumbnail_max_size,
            threshold_percentile=args.sp_threshold_percentile,
            min_signal=args.sp_min_signal,
            blur_sigma=args.sp_blur_sigma,
            close_radius=args.sp_close_radius,
            open_radius=args.sp_open_radius,
            dilate_radius=args.sp_dilate_radius,
            min_component_area_fraction=args.sp_min_component_area_fraction,
            min_contour_area=args.sp_min_contour_area,
            excluded_thumbnail_regions=args.sp_exclude_thumbnail_region,
            forced_thumbnail_regions=args.sp_force_include_thumbnail_region,
            max_fusion_thumbnail_regions=args.sp_max_fusion_thumbnail_region,
            max_fusion_threshold_percentile=args.sp_max_fusion_threshold_percentile,
            max_fusion_min_signal=args.sp_max_fusion_min_signal,
            qptiff_max_hole_area_fraction=args.sp_max_hole_area_fraction,
            qptiff_small_edge_max_area_fraction=args.sp_small_edge_max_area_fraction,
            qptiff_small_edge_margin_fraction=args.sp_small_edge_margin_fraction,
            qptiff_remove_sparse_peripheral_artifacts=args.sp_remove_sparse_peripheral_artifacts,
            ims_fusion_percentile=args.ims_fusion_percentile,
            ims_max_hole_area_fraction=args.ims_max_hole_area_fraction,
            ims_threshold_percentile=args.ims_threshold_percentile,
            ims_min_signal=args.ims_min_signal,
            ims_blur_sigma=args.ims_blur_sigma,
            ims_close_radius=args.ims_close_radius,
            ims_open_radius=args.ims_open_radius,
            ims_dilate_radius=args.ims_dilate_radius,
            ims_min_component_area_fraction=args.ims_min_component_area_fraction,
        )
    else:
        segmenter = args.method.replace("trident-", "")
        produced = extract_trident_coords(
            source_path,
            trident_root=args.trident_root,
            job_dir=out_dir / "trident_job",
            marker_registry_path=args.registry,
            data_root=data_root,
            reader_type=reader_type,
            dataset=entry.get("dataset", ""),
            patch_size=args.patch_size,
            mag=args.mag,
            segmenter=segmenter,
            seg_conf_thresh=args.seg_conf_thresh,
            gpu=args.gpu,
            remove_holes=args.remove_holes,
            overlap=args.overlap,
            trident_reader_type=args.trident_reader_type,
            channel_names_override=channel_names_override,
            mpp=args.mpp,
            min_tissue_proportion=args.min_foreground_fraction,
        )
    produced = rename_trident_artifacts(produced, out_dir / "trident_job", slide_name)
    if produced.resolve() != coords_path.resolve():
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        coords_path.write_bytes(produced.read_bytes())

    payload = load_sp_coords_h5(coords_path)
    return {
        "dataset": entry.get("dataset", ""),
        "path": entry["path"],
        "reader_type": reader_type,
        "coords_path": coords_path.relative_to(out_dir).as_posix(),
        "patch_size": args.patch_size,
        "num_patches": int(len(payload["coords"])),
        "channel_count": int(payload["attrs"].get("channel_count", len(payload.get("channel_names", [])))),
        "channel_names": list(payload.get("channel_names", [])),
        "marker_names": list(payload.get("marker_names", [])),
    }


def keep_by_name(entry: dict, args, data_root: Path | None) -> bool:
    path = source_path_for(entry, data_root)
    name = path.name
    if args.include_name_regex and re.search(args.include_name_regex, name) is None:
        return False
    if args.exclude_name_regex and re.search(args.exclude_name_regex, name) is not None:
        return False
    if args.skip_roi_tiles and re.search(r"(^|[_-])ROI\\d+", name, flags=re.IGNORECASE):
        return False
    return True


def keep_by_qc_contours(entry: dict, args) -> bool:
    if not args.qc_contours_dir:
        return True
    exists = contour_exists(Path(args.qc_contours_dir).expanduser(), entry)
    if args.only_missing_qc_contours:
        return not exists
    if args.only_existing_qc_contours:
        return exists
    return True


def _option_was_supplied(argv: list[str], option: str) -> bool:
    """Recognize both ``--option value`` and ``--option=value`` spellings."""
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def apply_foreground_profile(args, argv: list[str]) -> None:
    """Apply profile defaults without discarding explicit CLI overrides."""
    profile = DEFAULT_PROFILES[args.foreground_preset]
    values = [
        ("sp_threshold_percentile", "--sp-threshold-percentile", profile.threshold_percentile),
        ("sp_min_signal", "--sp-min-signal", profile.min_signal),
        ("sp_blur_sigma", "--sp-blur-sigma", profile.blur_sigma),
        ("sp_close_radius", "--sp-close-radius", profile.close_radius),
        ("sp_open_radius", "--sp-open-radius", profile.open_radius),
        ("sp_dilate_radius", "--sp-dilate-radius", profile.dilate_radius),
        ("sp_min_component_area_fraction", "--sp-min-component-area-fraction", profile.min_component_area_fraction),
        ("min_foreground_fraction", "--min-foreground-fraction", profile.min_foreground_fraction),
    ]
    for attribute, option, profile_value in values:
        if not _option_was_supplied(argv, option):
            setattr(args, attribute, profile_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment foreground tissue and write 224x224 patch coordinate HDF5 files.",
        allow_abbrev=False,
    )
    parser.add_argument("--manifest", default=None, help="Input image manifest JSONL from build_manifest.py.")
    parser.add_argument("--image", default=None, help="Process one image instead of a manifest.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "configs" / "marker_registry.json"))
    parser.add_argument("--trident-root", default=str(Path(__file__).resolve().parents[2] / "TRIDENT"))
    parser.add_argument("--method", choices=["sp-fluorescence", "trident-hest", "trident-otsu", "trident-grandqc"], default="sp-fluorescence")
    parser.add_argument("--reader-type", default=None, choices=["tiff", "tiff_hyperstack", "qptiff", "ims", "openslide_rgb"])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument(
        "--foreground-preset",
        choices=sorted(DEFAULT_PROFILES),
        default="fluorescence-default",
        help="Generic image-condition preset; it never inspects dataset or filename.",
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.10)
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print selected slides without writing contours, HDF5, or manifest files.")
    parser.add_argument("--include-name-regex", default=None, help="Only process image basenames matching this regex.")
    parser.add_argument("--exclude-name-regex", default=None, help="Skip image basenames matching this regex.")
    parser.add_argument("--skip-roi-tiles", action="store_true", help="Skip obvious ROI/tile filenames such as P01_ROI01.tiff.")
    parser.add_argument("--qc-contours-dir", default=None, help="Directory of manually QC'd contour thumbnails from a previous run.")
    qc_group = parser.add_mutually_exclusive_group()
    qc_group.add_argument("--only-missing-qc-contours", action="store_true", help="Only process slides whose previous QC contour thumbnail is missing.")
    qc_group.add_argument("--only-existing-qc-contours", action="store_true", help="Only process slides whose previous QC contour thumbnail still exists.")
    parser.add_argument(
        "--mag",
        type=int,
        default=None,
        choices=[5, 10, 20, 40],
        help="Optional TRIDENT target magnification. If omitted, inferred as round(10 / raw_mpp).",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seg-conf-thresh", type=float, default=0.5)
    parser.add_argument("--remove-holes", action="store_true")
    parser.add_argument("--trident-reader-type", default=None)
    parser.add_argument(
        "--mpp",
        type=float,
        default=None,
        help="Optional raw microns per pixel override. If omitted, tries to infer from image metadata.",
    )
    parser.add_argument("--sp-thumbnail-max-size", type=int, default=1600, help="Long edge of the thumbnail used for SP fluorescence foreground segmentation.")
    parser.add_argument("--sp-threshold-percentile", type=float, default=50.0, help="Percentile of positive thumbnail intensities used as fluorescence foreground threshold.")
    parser.add_argument("--sp-min-signal", type=float, default=8.0, help="Absolute lower bound for fluorescence foreground threshold on 8-bit thumbnails.")
    parser.add_argument("--sp-blur-sigma", type=float, default=4.0, help="Gaussian blur sigma before thresholding the fluorescence thumbnail.")
    parser.add_argument("--sp-close-radius", type=int, default=20, help="Morphological closing radius on the thumbnail mask.")
    parser.add_argument("--sp-open-radius", type=int, default=2, help="Morphological opening radius on the thumbnail mask.")
    parser.add_argument("--sp-dilate-radius", type=int, default=3, help="Dilation radius after hole filling.")
    parser.add_argument("--sp-min-component-area-fraction", type=float, default=0.001, help="Drop foreground components smaller than this fraction of thumbnail pixels.")
    parser.add_argument("--sp-min-contour-area", type=float, default=1000.0, help="Minimum contour area passed to TRIDENT mask_to_gdf.")
    parser.add_argument("--sp-max-hole-area-fraction", type=float, default=0.05, help="For QPTIFF, fill enclosed holes no larger than this thumbnail fraction.")
    parser.add_argument("--sp-small-edge-max-area-fraction", type=float, default=0.005, help="For QPTIFF, drop edge components below this thumbnail fraction; use 0 to disable.")
    parser.add_argument("--sp-small-edge-margin-fraction", type=float, default=0.15, help="For QPTIFF, edge-margin fraction used for small-component filtering.")
    parser.add_argument("--sp-remove-sparse-peripheral-artifacts", action="store_true", help="For QPTIFF, remove sparse medium-sized components near the thumbnail boundary.")
    parser.add_argument("--ims-fusion-percentile", type=float, default=75.0, help="For IMS, percentile across independently normalized marker projections used to build the foreground thumbnail.")
    parser.add_argument("--ims-max-hole-area-fraction", type=float, default=0.02, help="For IMS, fill only enclosed holes no larger than this thumbnail fraction.")
    parser.add_argument("--ims-threshold-percentile", type=float, default=20.0, help="For IMS, positive-pixel percentile used for the lenient tissue threshold.")
    parser.add_argument("--ims-min-signal", type=float, default=6.0, help="For IMS, absolute lower bound for the foreground threshold.")
    parser.add_argument("--ims-blur-sigma", type=float, default=3.0, help="For IMS, Gaussian blur sigma before thresholding.")
    parser.add_argument("--ims-close-radius", type=int, default=6, help="For IMS, light morphological closing radius on the thumbnail mask.")
    parser.add_argument("--ims-open-radius", type=int, default=0, help="For IMS, opening radius; 0 avoids eroding weak tissue edges.")
    parser.add_argument("--ims-dilate-radius", type=int, default=2, help="For IMS, final edge-tolerant dilation radius.")
    parser.add_argument("--ims-min-component-area-fraction", type=float, default=0.0002, help="For IMS, minimum retained foreground component fraction.")
    parser.add_argument(
        "--sp-exclude-thumbnail-region",
        action="append",
        default=[],
        metavar="X0,Y0,X1,Y1",
        help="Normalized thumbnail rectangle to exclude after segmentation; may be supplied more than once.",
    )
    parser.add_argument(
        "--sp-force-include-thumbnail-region",
        action="append",
        default=[],
        metavar="X0,Y0,X1,Y1",
        help="Normalized thumbnail rectangle to force into the foreground mask; may be supplied more than once.",
    )
    parser.add_argument(
        "--sp-max-fusion-thumbnail-region",
        action="append",
        default=[],
        metavar="X0,Y0,X1,Y1",
        help="Normalized ROI to segment from the QPTIFF marker maximum fusion; the rectangle itself is never forced into foreground.",
    )
    parser.add_argument(
        "--sp-max-fusion-threshold-percentile",
        type=float,
        default=70.0,
        help="Positive-pixel percentile used only inside each QPTIFF maximum-fusion ROI.",
    )
    parser.add_argument(
        "--sp-max-fusion-min-signal",
        type=float,
        default=8.0,
        help="Minimum 8-bit threshold used only inside each QPTIFF maximum-fusion ROI.",
    )
    args = parser.parse_args()
    apply_foreground_profile(args, sys.argv[1:])
    args.sp_exclude_thumbnail_region = parse_thumbnail_regions(args.sp_exclude_thumbnail_region)
    args.sp_force_include_thumbnail_region = parse_thumbnail_regions(args.sp_force_include_thumbnail_region)
    args.sp_max_fusion_thumbnail_region = parse_thumbnail_regions(args.sp_max_fusion_thumbnail_region)

    input_count = sum(bool(value) for value in [args.manifest, args.image])
    if input_count != 1:
        raise ValueError("Pass exactly one of --manifest or --image.")
    if (args.only_missing_qc_contours or args.only_existing_qc_contours) and not args.qc_contours_dir:
        raise ValueError("Pass --qc-contours-dir when using QC contour filters.")

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        rows = load_jsonl(Path(args.manifest))
    else:
        rows = [make_single_entry(args)]
    before_filter = len(rows)
    rows = [row for row in rows if keep_by_name(row, args, data_root)]
    if before_filter != len(rows):
        print(f"Filtered images by name: {before_filter} -> {len(rows)}")
    before_qc_filter = len(rows)
    rows = [row for row in rows if keep_by_qc_contours(row, args)]
    if before_qc_filter != len(rows):
        print(f"Filtered images by QC contours: {before_qc_filter} -> {len(rows)}")
    if not rows:
        if args.dry_run:
            print("Dry run selected 0 slide(s).")
            return
        raise ValueError(
            "No images left after filtering. Relax --include-name-regex/"
            "--exclude-name-regex/--qc-contours-dir filters or remove --skip-roi-tiles."
        )
    if args.max_slides is not None:
        rows = rows[: int(args.max_slides)]

    if args.dry_run:
        print(f"Dry run selected {len(rows)} slide(s):")
        for entry in rows:
            print(f"  {entry.get('dataset', '')}: {entry['path']}")
        return

    coords_rows = []
    for i, entry in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {entry.get('dataset', '')}: {entry['path']}")
        try:
            row = process_entry(entry, args, data_root, out_dir)
            print(f"  kept {row['num_patches']} patches -> {row['coords_path']}")
        except Exception as exc:
            row = {
                "dataset": entry.get("dataset", ""),
                "path": entry.get("path", ""),
                "reader_type": entry.get("reader_type") or infer_reader_type(entry.get("path", "")),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR {row['error']}")
        coords_rows.append(row)

    manifest_out = out_dir / f"coords_manifest_patch{args.patch_size}.jsonl"
    # A targeted rerun must preserve the rows for slides which were not
    # selected.  Name filters are routinely used for manual QC fixes.
    should_merge_manifest = bool(
        args.only_missing_qc_contours
        or args.only_existing_qc_contours
        or args.include_name_regex
        or args.exclude_name_regex
    )
    if should_merge_manifest:
        base_rows = load_jsonl(manifest_out) if manifest_out.exists() else rows
        output_rows = merge_manifest_rows(base_rows, coords_rows)
        print(f"Merged {len(coords_rows)} updated row(s) into existing manifest rows: {len(base_rows)} -> {len(output_rows)}")
    else:
        output_rows = coords_rows
    write_coords_manifest(output_rows, manifest_out)
    print(f"Wrote coords manifest to {manifest_out}")


if __name__ == "__main__":
    main()

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


from .legacy_preprocessing import extract_sp_fluorescence_coords, extract_trident_coords, load_sp_coords_h5


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


def contour_stem(path: str) -> str:
    return re.sub(r"\.(ome\.)?(tif|tiff|qptiff|ims)$", "", Path(path).name, flags=re.IGNORECASE)


def contour_exists(contours_dir: Path, entry: dict) -> bool:
    stem = contour_stem(entry["path"])
    return any((contours_dir / f"{stem}{suffix}").exists() for suffix in [".jpg", ".jpeg", ".png"])


def infer_reader_type(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".ims"):
        return "ims"
    if lower.endswith(".qptiff"):
        return "qptiff"
    return "tiff"


def load_dfci_codex_panel(data_root: Path) -> list[str]:
    """Load the one shared Minerva-derived channel order for DFCI CODEX."""
    panel_path = data_root / "HTAN" / "DFCI" / "dfci_codex_panel.json"
    try:
        payload = json.loads(panel_path.read_text(encoding="utf-8"))
        channel_names = [str(name) for name in payload["channel_names"]]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load the DFCI CODEX channel panel {panel_path}: {error}") from error
    if len(channel_names) != 59:
        raise ValueError(f"Expected 59 DFCI CODEX channel names in {panel_path}, found {len(channel_names)}")
    return channel_names


_STANFORD_UNALIGNED_CHANNEL_METADATA = {
    # The portal points these three image records to metadata files whose
    # plane counts disagree with their actual TCYX dimensions.  Do not assign
    # a truncated/shifted marker panel and do not process them as SP patches.
    "A001-C-002.ome.tiff",
    "A001-C-023.ome.tiff",
    "CRC_TB15564.ome.tiff",
}

_STANFORD_MPP_FALLBACKS = {
    # These four OME-TIFFs lack a physical-size field. Their acquisition
    # dimensions match Stanford's ImageJ TCYX scans (all ~0.3774 mpp).
    "B001-A-101.ome.tiff": 0.3774,
    "B001-A-301.ome.tiff": 0.3774,
    "B001-A-401.ome.tiff": 0.3774,
    "F072B.ome.tiff": 0.3774,
}


def load_stanford_channel_names(data_root: Path, image_path: Path) -> list[str] | None:
    """Return verified Synapse channel labels for Stanford's unnamed OME-TIFFs.

    The HTAN portal exposes per-image channel files separately from the OME
    XML.  The downloader writes both the raw files and their image mapping in
    ``HTAN/Stanford/channel_metadata``.  Only use an override if its length
    exactly matches the image's nonspatial plane count.
    """
    metadata_dir = data_root / "HTAN" / "Stanford" / "channel_metadata"
    mapping_path = metadata_dir / "image_to_channel_metadata.json"
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else []
        record = next((item for item in mapping if item["image_file"] == image_path.name), None)
        import tifffile

        with tifffile.TiffFile(image_path) as handle:
            series = handle.series[0]
            expected = 1
            for axis, size in zip(series.axes, series.shape):
                if axis not in {"Y", "X", "Z"}:
                    expected *= int(size)
            if record is not None:
                labels = [
                    line.strip()
                    for line in (metadata_dir / record["channel_metadata_file"]).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                imagej = handle.imagej_metadata or {}
                labels = [str(label).strip() for label in imagej.get("Labels", []) if str(label).strip()]
                if not labels and handle.ome_metadata:
                    pixels = ET.fromstring(handle.ome_metadata).find(".//{*}Pixels")
                    ome_names = (
                        [channel.get("Name") or channel.get("ID") for channel in pixels.findall("{*}Channel")]
                        if pixels is not None
                        else []
                    )
                    if len(ome_names) == expected:
                        labels = ome_names
                    elif series.axes == "TCYX" and len(ome_names) == int(series.shape[1]):
                        # F072B stores only generic C1--C4 labels in OME, so
                        # preserve their cycle coordinate rather than falsely
                        # claiming marker identities.
                        labels = [
                            f"cycle_{timepoint + 1:02d}_{ome_name}"
                            for timepoint in range(int(series.shape[0]))
                            for ome_name in ome_names
                        ]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load Stanford channel metadata for {image_path.name}: {error}") from error
    if not labels:
        return None
    if len(labels) != expected:
        raise ValueError(
            f"Stanford metadata mismatch for {image_path.name}: {len(labels)} labels for {expected} image planes"
        )
    return labels


def require_named_ome_channels(paths: list[Path], *, collection: str) -> None:
    """Fail closed when a curated OME-TIFF collection lacks marker names.

    OHSU contains a small set of RGB/unnamed TIFFs which were moved out of the
    protein-imaging folders.  Do not silently turn any such file into C1/C2
    placeholders: valid SP files must expose one non-empty OME Channel Name
    for every native channel before a preset is allowed to process them.
    """
    import tifffile
    import xml.etree.ElementTree as ET

    invalid: list[str] = []
    for path in paths:
        try:
            with tifffile.TiffFile(path) as handle:
                ome_xml = handle.ome_metadata
                pixels = ET.fromstring(ome_xml).find(".//{*}Pixels") if ome_xml else None
                channel_names = (
                    [channel.get("Name") for channel in pixels.findall("{*}Channel")]
                    if pixels is not None
                    else []
                )
                channel_count = int(pixels.get("SizeC", "0")) if pixels is not None else 0
        except Exception as error:
            invalid.append(f"{path.name} (could not read OME metadata: {error})")
            continue
        if channel_count <= 0 or len(channel_names) != channel_count or not all(
            name and name.strip() for name in channel_names
        ):
            invalid.append(path.name)
    if invalid:
        examples = ", ".join(invalid[:8])
        suffix = " ..." if len(invalid) > 8 else ""
        raise ValueError(
            f"{collection} contains {len(invalid)} OME-TIFF(s) without complete named channel metadata: "
            f"{examples}{suffix}. Move them out of the SP Level_2 directory or supply their marker metadata first."
        )


def preset_entries(name: str, data_root: Path | None) -> list[dict]:
    """Return curated raw-image entries for a named SP preprocessing preset.

    A preset is deliberately narrower than a recursive manifest build: HTAN/BU
    contains both H&E SVS images and MxIF WSIs, while this pipeline is for the
    multichannel protein images only.  Keeping the selection here also makes
    the mixed OME-TIFF/QPTIFF reader choice explicit and reproducible.
    """
    if data_root is None:
        raise ValueError(f"--data-root is required for --dataset-preset {name!r}.")
    if name == "htan-bu-mxif":
        source_dir = data_root / "HTAN" / "BU" / "MxIF" / "Level_2"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"HTAN BU MxIF directory does not exist: {source_dir}")
        paths = sorted([*source_dir.glob("*.ome.tiff"), *source_dir.glob("*.qptiff")])
        if not paths:
            raise FileNotFoundError(f"No HTAN BU MxIF OME-TIFF/QPTIFF files found in {source_dir}")
        return [
            {
                "dataset": "htan_bu_mxif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "qptiff" if path.name.lower().endswith(".qptiff") else "tiff",
            }
            for path in paths
        ]

    if name == "htan-chop-codex":
        source_dir = data_root / "HTAN" / "CHOP" / "CODEX" / "Level_2"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"HTAN CHOP CODEX Level_2 directory does not exist: {source_dir}")
        paths = sorted(source_dir.glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No CHOP CODEX Level_2 OME-TIFF files found in {source_dir}")
        return [
            {
                "dataset": "htan_chop_codex",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in paths
        ]

    if name == "htan-dfci":
        codex_dir = data_root / "HTAN" / "DFCI" / "CODEX" / "Level_2"
        mxif_dir = data_root / "HTAN" / "DFCI" / "MxIF" / "Level_2"
        if not codex_dir.is_dir() or not mxif_dir.is_dir():
            raise FileNotFoundError("HTAN DFCI CODEX/MxIF Level_2 directories do not both exist")
        codex_paths = sorted(codex_dir.glob("*.tif"))
        mxif_paths = sorted(mxif_dir.glob("*.tiff"))
        if not codex_paths or not mxif_paths:
            raise FileNotFoundError("No DFCI Level_2 CODEX or MxIF images found")
        codex_channel_names = load_dfci_codex_panel(data_root)
        return [
            {
                "dataset": "htan_dfci",
                "path": path.relative_to(data_root).as_posix(),
                # These ImageJ ZYX files are a 59-marker page stack, not a
                # focal Z stack. The field name (2200 um / 4400 px) gives 0.5 mpp.
                "reader_type": "tiff_z_as_channels",
                "mpp": 0.5,
                "channel_names": codex_channel_names,
            }
            for path in codex_paths
        ] + [
            {
                "dataset": "htan_dfci",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in mxif_paths
        ]

    if name == "htan-hms":
        # HMS contains H&E Level_2 whole-slide images and CyCIF Level_3 cell
        # masks in addition to spatial-proteomics images.  Select only the
        # two protein-imaging Level_2 collections.
        cycif_dir = data_root / "HTAN" / "HMS" / "CyCIF" / "Level_2"
        orion_dir = data_root / "HTAN" / "HMS" / "RareCyte_Orion" / "Level_2"
        if not cycif_dir.is_dir() or not orion_dir.is_dir():
            raise FileNotFoundError("HTAN HMS CyCIF/RareCyte_Orion Level_2 directories do not both exist")
        cycif_paths = sorted([*cycif_dir.glob("*.ome.tif"), *cycif_dir.glob("*.ome.tiff")])
        orion_paths = sorted([*orion_dir.glob("*.ome.tif"), *orion_dir.glob("*.ome.tiff")])
        if not cycif_paths or not orion_paths:
            raise FileNotFoundError("No HMS CyCIF or RareCyte_Orion Level_2 OME-TIFF images found")
        return [
            {
                "dataset": "htan_hms_cycif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in cycif_paths
        ] + [
            {
                "dataset": "htan_hms_orion",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in orion_paths
        ]

    if name == "htan-htapp":
        # HTAPP Level_2 contains both the complete TZCYX acquisition stacks
        # and best-focus TCYX hyperstacks.  A *_Z<n>.tif filename identifies
        # the latter: Z has already been selected, while all cycle x detector
        # channels remain in the file and must be retained as output channels.
        source_dir = data_root / "HTAN" / "HTAPP" / "CODEX" / "Level_2"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"HTAPP CODEX Level_2 directory does not exist: {source_dir}")
        paths = sorted(path for path in source_dir.glob("*_Z*.tif") if re.search(r"_Z\d+\.tif$", path.name, re.IGNORECASE))
        if not paths:
            raise FileNotFoundError(f"No HTAPP best-focus *_Z<n>.tif files found in {source_dir}")
        return [
            {
                "dataset": "htan_htapp",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff_hyperstack",
                # The HTAPP CODEX acquisition protocol reports 396 nm/px.
                "mpp": 0.396,
            }
            for path in paths
        ]

    if name == "htan-wustl-codex":
        # Keep this narrow: WUSTL also has H&E Level_2 images and CODEX
        # segmentation/feature products at Levels 3 and 4.  Only these
        # multichannel CODEX Level_2 OME-TIFFs are source images for patches.
        source_dir = data_root / "HTAN" / "WUSTL" / "CODEX" / "Level_2"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"WUSTL CODEX Level_2 directory does not exist: {source_dir}")
        paths = sorted([*source_dir.glob("*.ome.tif"), *source_dir.glob("*.ome.tiff")])
        if not paths:
            raise FileNotFoundError(f"No WUSTL CODEX Level_2 OME-TIFF files found in {source_dir}")
        return [
            {
                "dataset": "htan_wustl_codex",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in paths
        ]

    if name == "htan-vanderbilt":
        # Select only spatial-proteomics Level_2 source imagery.  H&E Level_2
        # files and all higher-level derivative products are deliberately not
        # included.  CODEX is OME-CYX; Vanderbilt MxIF is a non-OME IYX page
        # stack, so its image axis is flattened into channels by the native
        # TIFF reader.  No dataset-specific foreground profile is attached.
        codex_dir = data_root / "HTAN" / "Vanderbilt" / "CODEX" / "Level_2"
        mxif_dir = data_root / "HTAN" / "Vanderbilt" / "MxIF" / "Level_2"
        if not codex_dir.is_dir() or not mxif_dir.is_dir():
            raise FileNotFoundError("HTAN Vanderbilt CODEX/MxIF Level_2 directories do not both exist")
        codex_paths = sorted([*codex_dir.glob("*.ome.tif"), *codex_dir.glob("*.ome.tiff")])
        mxif_paths = sorted([*mxif_dir.glob("*.tif"), *mxif_dir.glob("*.tiff")])
        if not codex_paths or not mxif_paths:
            raise FileNotFoundError("No Vanderbilt Level_2 CODEX or MxIF source images found")
        return [
            {
                "dataset": "htan_vanderbilt_codex",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
                # Present in each CODEX OME-XML PhysicalSizeX/Y field.  Keep
                # this per-modality value independent of a user-supplied MxIF
                # --mpp override, because Vanderbilt MxIF TIFFs lack one.
                "mpp": 0.5100762527233116,
            }
            for path in codex_paths
        ] + [
            {
                "dataset": "htan_vanderbilt_mxif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff_hyperstack",
            }
            for path in mxif_paths
        ]

    if name == "htan-ohsu":
        # OHSU Level_2 holds both mIHC and CyCIF source OME-TIFFs.  The
        # non-protein RGB/unnamed images are intentionally stored in HE and
        # are excluded by selecting only these two SP directories.  Keep the
        # validation here so a later misplaced file cannot silently enter the
        # patch set with fabricated channel names.
        mihc_dir = data_root / "HTAN" / "OHSU" / "mIHC" / "Level_2"
        cycif_dir = data_root / "HTAN" / "OHSU" / "CyCIF" / "Level_2"
        if not mihc_dir.is_dir() or not cycif_dir.is_dir():
            raise FileNotFoundError("HTAN OHSU mIHC/CyCIF Level_2 directories do not both exist")
        mihc_paths = sorted([*mihc_dir.glob("*.ome.tif"), *mihc_dir.glob("*.ome.tiff")])
        cycif_paths = sorted([*cycif_dir.glob("*.ome.tif"), *cycif_dir.glob("*.ome.tiff")])
        if not mihc_paths or not cycif_paths:
            raise FileNotFoundError("No OHSU mIHC or CyCIF Level_2 OME-TIFF source images found")
        require_named_ome_channels(mihc_paths, collection="HTAN OHSU mIHC Level_2")
        require_named_ome_channels(cycif_paths, collection="HTAN OHSU CyCIF Level_2")
        return [
            {
                "dataset": "htan_ohsu_mihc",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in mihc_paths
        ] + [
            {
                "dataset": "htan_ohsu_cycif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in cycif_paths
        ]

    if name == "htan-tnp-sardana":
        # TNP-Sardana Level_2 contains protein-imaging CODEX, CyCIF and mIHC
        # ROI files alongside H&E.  Select only the three SP modalities.  The
        # three unnamed three-plane mIHC overviews were moved out of this
        # directory; retain an explicit ROI filter so they cannot be admitted
        # later by mistake.  Do not assign a dataset-specific segmentation
        # profile: this preset intentionally uses the standard fluorescence
        # settings.
        root = data_root / "HTAN" / "TNP-SARDANA"
        codex_dir = root / "CODEX" / "Level_2"
        cycif_dir = root / "CyCIF" / "Level_2"
        mihc_dir = root / "mIHC" / "Level_2"
        if not codex_dir.is_dir() or not cycif_dir.is_dir() or not mihc_dir.is_dir():
            raise FileNotFoundError("HTAN TNP-Sardana CODEX/CyCIF/mIHC Level_2 directories do not all exist")
        codex_paths = sorted([*codex_dir.glob("*.ome.tif"), *codex_dir.glob("*.ome.tiff")])
        cycif_paths = sorted([*cycif_dir.glob("*.ome.tif"), *cycif_dir.glob("*.ome.tiff")])
        mihc_paths = sorted(
            path
            for path in [*mihc_dir.glob("*.ome.tif"), *mihc_dir.glob("*.ome.tiff")]
            if "_ROI" in path.stem.upper()
        )
        if not codex_paths or not cycif_paths or not mihc_paths:
            raise FileNotFoundError("No TNP-Sardana Level_2 CODEX, CyCIF, or named mIHC ROI OME-TIFF images found")
        require_named_ome_channels(codex_paths, collection="HTAN TNP-Sardana CODEX Level_2")
        require_named_ome_channels(cycif_paths, collection="HTAN TNP-Sardana CyCIF Level_2")
        require_named_ome_channels(mihc_paths, collection="HTAN TNP-Sardana mIHC ROI Level_2")
        return [
            {
                "dataset": "htan_tnp_sardana_codex",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in codex_paths
        ] + [
            {
                "dataset": "htan_tnp_sardana_cycif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in cycif_paths
        ] + [
            {
                "dataset": "htan_tnp_sardana_mihc",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in mihc_paths
        ]

    if name == "htan-tnp-tma":
        # TNP-TMA Level_2 consists solely of CyCIF and mIHC OME-TIFF source
        # images.  Level_3 holds derived segmentation products and is
        # deliberately not traversed.  Use the ordinary fluorescence profile;
        # no cohort-specific foreground overrides belong in this preset.
        root = data_root / "HTAN" / "TNP-TMA"
        cycif_dir = root / "CyCIF" / "Level_2"
        mihc_dir = root / "mIHC" / "Level_2"
        if not cycif_dir.is_dir() or not mihc_dir.is_dir():
            raise FileNotFoundError("HTAN TNP-TMA CyCIF/mIHC Level_2 directories do not both exist")
        cycif_paths = sorted([*cycif_dir.glob("*.ome.tif"), *cycif_dir.glob("*.ome.tiff")])
        mihc_paths = sorted([*mihc_dir.glob("*.ome.tif"), *mihc_dir.glob("*.ome.tiff")])
        if not cycif_paths or not mihc_paths:
            raise FileNotFoundError("No TNP-TMA CyCIF or mIHC Level_2 OME-TIFF source images found")
        require_named_ome_channels(cycif_paths, collection="HTAN TNP-TMA CyCIF Level_2")
        require_named_ome_channels(mihc_paths, collection="HTAN TNP-TMA mIHC Level_2")
        return [
            {
                "dataset": "htan_tnp_tma_cycif",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in cycif_paths
        ] + [
            {
                "dataset": "htan_tnp_tma_mihc",
                "path": path.relative_to(data_root).as_posix(),
                "reader_type": "tiff",
            }
            for path in mihc_paths
        ]

    if name == "htan-stanford-codex":
        # Stanford CODEX Level_2 contains 50 source images.  Thirteen images
        # without OME/ImageJ names have now been resolved from HTAN's linked
        # Synapse channel metadata. Three other OME-TIFFs point to metadata
        # with a mismatched plane count and are excluded rather than guessed.
        source_dir = data_root / "HTAN" / "Stanford" / "CODEX" / "Level_2"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"HTAN Stanford CODEX Level_2 directory does not exist: {source_dir}")
        paths = sorted([*source_dir.glob("*.tif"), *source_dir.glob("*.tiff")])
        paths = [path for path in paths if path.name not in _STANFORD_UNALIGNED_CHANNEL_METADATA]
        if len(paths) != 47:
            raise ValueError(
                f"Expected 47 Stanford CODEX images after excluding the three unaligned metadata files, found {len(paths)}"
            )
        entries = []
        for path in paths:
            channel_names = load_stanford_channel_names(data_root, path)
            entry = {
                "dataset": "htan_stanford_codex",
                "path": path.relative_to(data_root).as_posix(),
                # All Stanford source files are TCYX/CYX fluorescence stacks.
                # Flatten every nonspatial plane so ImageJ and Synapse labels
                # remain aligned with the patch reader and thumbnail fusion.
                "reader_type": "tiff_hyperstack",
            }
            if channel_names is not None:
                entry["channel_names"] = channel_names
            if path.name in _STANFORD_MPP_FALLBACKS:
                entry["mpp"] = _STANFORD_MPP_FALLBACKS[path.name]
            entries.append(entry)
        return entries

    raise ValueError(f"Unsupported dataset preset {name!r}.")


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
    slide_name = str(entry.get("output_stem") or safe_stem(entry["path"]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment foreground tissue and write 224x224 patch coordinate HDF5 files.")
    parser.add_argument("--manifest", default=None, help="Input image manifest JSONL from build_manifest.py.")
    parser.add_argument("--image", default=None, help="Process one image instead of a manifest.")
    parser.add_argument(
        "--dataset-preset",
        choices=["htan-bu-mxif", "htan-chop-codex", "htan-dfci", "htan-hms", "htan-htapp", "htan-wustl-codex", "htan-vanderbilt", "htan-ohsu", "htan-stanford-codex", "htan-tnp-sardana", "htan-tnp-tma"],
        default=None,
        help="Curated raw-image selection for reviewed HTAN Level_2 collections.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--registry", default=str(Path(__file__).resolve().parents[1] / "configs" / "marker_registry.json"))
    parser.add_argument("--trident-root", default=str(Path(__file__).resolve().parents[2] / "TRIDENT"))
    parser.add_argument("--method", choices=["sp-fluorescence", "trident-hest", "trident-otsu", "trident-grandqc"], default="sp-fluorescence")
    parser.add_argument("--reader-type", default=None, choices=["tiff", "tiff_hyperstack", "qptiff", "ims", "openslide_rgb"])
    parser.add_argument("--dataset-name", default=None)
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
    args.sp_exclude_thumbnail_region = parse_thumbnail_regions(args.sp_exclude_thumbnail_region)
    args.sp_force_include_thumbnail_region = parse_thumbnail_regions(args.sp_force_include_thumbnail_region)
    args.sp_max_fusion_thumbnail_region = parse_thumbnail_regions(args.sp_max_fusion_thumbnail_region)

    input_count = sum(bool(value) for value in [args.manifest, args.image, args.dataset_preset])
    if input_count != 1:
        raise ValueError("Pass exactly one of --manifest, --image, or --dataset-preset.")
    if (args.only_missing_qc_contours or args.only_existing_qc_contours) and not args.qc_contours_dir:
        raise ValueError("Pass --qc-contours-dir when using QC contour filters.")

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        rows = load_jsonl(Path(args.manifest))
    elif args.dataset_preset:
        rows = preset_entries(args.dataset_preset, data_root)
        print(f"Selected {len(rows)} image(s) from dataset preset {args.dataset_preset!r}.")
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

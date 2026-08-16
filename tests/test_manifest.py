import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sp_patch_toolbox.compat.legacy_cli import apply_foreground_profile, artifact_stem
from sp_patch_toolbox.models import ImageSpec
from sp_patch_toolbox.pipeline.manifest import load_manifest, write_manifest


class ManifestTests(unittest.TestCase):
    def test_artifact_stem_blocks_path_traversal_and_matches_qc_name(self):
        self.assertEqual(artifact_stem({"path": "case/slide.ome.tiff", "output_stem": "../outside"}), "outside")
        self.assertEqual(artifact_stem({"path": "case/slide with spaces.ome.tiff"}), "slide_with_spaces")

    def test_explicit_foreground_flag_overrides_selected_profile(self):
        args = SimpleNamespace(
            foreground_preset="fluorescence-artifact-strict",
            sp_threshold_percentile=55.0,
            sp_min_signal=8.0,
            sp_blur_sigma=4.0,
            sp_close_radius=20,
            sp_open_radius=2,
            sp_dilate_radius=3,
            sp_min_component_area_fraction=0.001,
            min_foreground_fraction=0.25,
        )
        apply_foreground_profile(args, ["--sp-threshold-percentile", "55", "--min-foreground-fraction=0.25"])
        self.assertEqual(args.sp_threshold_percentile, 55.0)
        self.assertEqual(args.min_foreground_fraction, 0.25)
        self.assertEqual(args.sp_min_signal, 12.0)

    def test_manifest_roundtrip_preserves_unknown_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "images.jsonl"
            source.write_text(
                '{"path":"a.qptiff","dataset":"demo","reader_type":"qptiff","panel":"P1"}\n',
                encoding="utf-8",
            )
            spec = load_manifest(source)[0]
            self.assertEqual(spec.extras["panel"], "P1")
            target = write_manifest([spec], root / "roundtrip.jsonl")
            self.assertEqual(load_manifest(target), [ImageSpec.from_dict(spec.to_dict())])

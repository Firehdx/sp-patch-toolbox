import tempfile
import unittest
from pathlib import Path

from sp_patch_toolbox.models import ImageSpec
from sp_patch_toolbox.pipeline.manifest import load_manifest, write_manifest


class ManifestTests(unittest.TestCase):
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

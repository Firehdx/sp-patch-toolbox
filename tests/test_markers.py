import unittest

from sp_patch_toolbox.metadata.markers import normalize_marker_text


class MarkerTests(unittest.TestCase):
    def test_marker_normalization_handles_technical_and_alias_labels(self):
        self.assertEqual(normalize_marker_text("DAPI_04"), "DAPI")
        self.assertEqual(normalize_marker_text("empty 3"), "blank")
        self.assertEqual(normalize_marker_text("CD20/MS4A1"), "CD20")
        self.assertEqual(normalize_marker_text("Dye 570"), "Dye 570")

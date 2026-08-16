import tempfile
import unittest
from pathlib import Path

import numpy as np

from sp_patch_toolbox.coordinates import load_sp_coords_h5, write_sp_coords_h5


class CoordinateTests(unittest.TestCase):
    def test_coordinate_h5_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_sp_coords_h5(
                Path(directory) / "coords.h5",
                np.array([[0, 0], [224, 448]]),
                source_path="raw/slide.ome.tif",
                reader_type="tiff",
                dataset="demo",
                spatial_shape_yx=(1000, 1200),
                patch_size=224,
                overlap=0,
                channel_names=["DAPI", "CD8"],
                marker_names=["DAPI", "CD8"],
                foreground_method="synthetic",
            )
            result = load_sp_coords_h5(path)
            self.assertEqual(result["coords"].tolist(), [[0, 0], [224, 448]])
            self.assertEqual(result["attrs"]["channel_count"], 2)
            self.assertEqual(result["channel_names"], ["DAPI", "CD8"])

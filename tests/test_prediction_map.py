import os
import tempfile
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image

from prediction_map import default_layout, patch_position, save_region_map, main as remerge


class PredictionMapTests(unittest.TestCase):
    def test_shuffled_rectangular_grid_with_missing_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, "np_array"))
            ids = ["5", "0", "2", "3", "1"]
            sentinel_paths = []
            for sample_id in ids:
                patch = np.full((1, 2, 3), int(sample_id) / 10, dtype=np.float32)
                np.save(os.path.join(directory, "np_array", sample_id + ".npy"), patch)
                sentinel = np.ones((12, 2, 3), dtype=np.float32)
                sentinel[:, 0, 0] = 0
                path = os.path.join(directory, sample_id + ".npy")
                np.save(path, sentinel)
                sentinel_paths.append(path)
            # A stale patch from a previous run must not fill a missing cell.
            np.save(os.path.join(directory, "np_array", "4.npy"), np.ones((2, 3)))
            dataset = SimpleNamespace(file_idx=list(range(5)), sample_regions=["test"] * 5,
                                      sample_ids=ids, sentinel_paths=sentinel_paths, max_height=30)
            save_region_map(dataset, "test", {"rows": 2, "columns": 3}, directory)
            result = np.load(os.path.join(directory, "map.npy"))
            self.assertEqual(result.shape, (4, 9))
            for sample_id in ids:
                row, column = divmod(int(sample_id), 3)
                np.testing.assert_allclose(result[row*2:(row+1)*2, column*3:(column+1)*3],
                                           int(sample_id) / 10)
            self.assertTrue(np.isnan(result[2:4, 3:6]).all())
            self.assertGreater(os.path.getsize(os.path.join(directory, "map.png")), 0)
            with Image.open(os.path.join(directory, "rgb.png")) as image:
                rgb = np.array(image)
            self.assertEqual(rgb.shape, (4, 9, 3))
            np.testing.assert_array_equal(rgb[2:4, 3:6], 255)
            np.testing.assert_array_equal(rgb[0, 0], [255, 255, 255])
            np.testing.assert_array_equal(rgb[0, 1], [255, 255, 255])
            # A bounded preview must not reduce the resolution of map.npy.
            save_region_map(dataset, "test", {"rows": 2, "columns": 3}, directory,
                            max_preview_size=6)
            np.testing.assert_equal(np.load(os.path.join(directory, "map.npy")), result)
            with Image.open(os.path.join(directory, "rgb.png")) as image:
                self.assertEqual(image.size, (6, 3))

    def test_region_metadata_and_row_boundaries(self):
        for zone, rows, columns in [("zone1", 52, 76), ("zone2", 52, 54),
                                    ("zone4", 126, 165)]:
            with self.subTest(zone=zone):
                layout = default_layout(zone, (256, 256))
                self.assertEqual(layout, {"rows": rows, "columns": columns})
                self.assertEqual(patch_position(str(columns), layout), (1, 0))
                self.assertEqual(patch_position(str(rows * columns - 1), layout),
                                 (rows - 1, columns - 1))
                with self.assertRaises(ValueError):
                    patch_position(str(rows * columns), layout)

    def test_layout_origin_order_and_bounds(self):
        layout = {"rows": 2, "columns": 3, "first_id": 1, "order": "column-major"}
        self.assertEqual(patch_position("4", layout), (1, 1))
        for sample_id in ("0", "7"):
            with self.assertRaises(ValueError):
                patch_position(sample_id, layout)
        with self.assertRaises(ValueError):
            patch_position("0", {"rows": 2, "columns": 0})

    def test_remerge_changes_layout_without_changing_source_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            patch_dir = os.path.join(directory, "patches")
            sentinel_dir = os.path.join(directory, "sentinel")
            os.makedirs(patch_dir)
            os.makedirs(sentinel_dir)
            for i in range(6):
                np.save(os.path.join(patch_dir, f"{i}.npy"),
                        np.full((2, 2), i / 10, dtype=np.float32))
                np.save(os.path.join(sentinel_dir, f"{i}.npy"),
                        np.ones((12, 2, 2), dtype=np.float32))
            for rows, columns, order in [(2, 3, "row-major"), (3, 2, "row-major"),
                                         (2, 3, "column-major")]:
                output_dir = os.path.join(directory, f"{rows}x{columns}_{order}")
                remerge(["--zone", "test", "--patch-dir", patch_dir,
                         "--sentinel-dir", sentinel_dir, "--output-dir", output_dir,
                         "--rows", str(rows), "--columns", str(columns), "--order", order])
                result = np.load(os.path.join(output_dir, "map.npy"))
                layout = {"rows": rows, "columns": columns, "order": order}
                self.assertEqual(result.shape, (rows * 2, columns * 2))
                for i in range(6):
                    row, column = patch_position(str(i), layout)
                    self.assertAlmostEqual(float(result[row * 2, column * 2]), i / 10)
                for name in ("map.png", "rgb.png", "layout.json"):
                    self.assertTrue(os.path.isfile(os.path.join(output_dir, name)))
            np.testing.assert_allclose(np.load(os.path.join(patch_dir, "3.npy")), 0.3)


if __name__ == "__main__":
    unittest.main()

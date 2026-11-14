import os
import tempfile
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from prediction_map import patch_position, save_region_map


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

    def test_layout_origin_order_and_bounds(self):
        layout = {"rows": 2, "columns": 3, "first_id": 1, "order": "column-major"}
        self.assertEqual(patch_position("4", layout), (1, 1))
        for sample_id in ("0", "7"):
            with self.assertRaises(ValueError):
                patch_position(sample_id, layout)
        with self.assertRaises(ValueError):
            patch_position("0", {"rows": 2, "columns": 0})


if __name__ == "__main__":
    unittest.main()

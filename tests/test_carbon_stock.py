import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
from rasterio.transform import from_origin

from carbon_stock import save_patch_carbon, save_region_carbon
from prediction_map import save_region_map


class CarbonStockTests(unittest.TestCase):
    def test_native_rasters_and_paired_gedi_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("np_array", "image", "GEDI/zone", "gedi_source/zone"):
                (root / folder).mkdir(parents=True)
            paths, sentinel_paths = [], []
            for sid in ("0", "1"):
                paths.append(root / "GEDI" / "zone" / f"{sid}.npy")
                sentinel_paths.append(root / f"sentinel{sid}.npy")
                np.save(paths[-1], [[45, np.nan], [3, np.nan]])
                sentinel = np.ones((12, 2, 2))
                sentinel[:, 1, 0] = 0
                np.save(sentinel_paths[-1], sentinel)
                np.save(root / "np_array" / f"{sid}.npy", np.full((2, 2), 0.5))
                np.savez(root / "gedi_source" / "zone" / f"{sid}.npz",
                         **{key: np.ones((2, 2), dtype=np.int64)
                            for key in ("day", "second", "microsecond", "beam")})
            dataset = SimpleNamespace(file_idx=[0, 1], sample_regions=["zone"] * 2,
                                      sample_ids=["0", "1"], gedi_paths=paths,
                                      sentinel_paths=sentinel_paths, max_height=20)
            transform = from_origin(500000, 1000000, 10, 10)
            layout = dict(rows=1, columns=2, crs="EPSG:32648", transform=list(transform)[:6])
            save_region_map(dataset, "zone", layout, root, mask_mosaic=True)
            rows = save_region_carbon(dataset, "zone", layout, root, max_preview_size=2)
            rows = {(row["source"], row["quantity"]): row for row in rows}
            self.assertEqual(rows["prediction_region", "canopy_height"]["count"], 6)
            self.assertEqual(rows["prediction_at_gedi", "canopy_height"]["count"], 1)
            self.assertEqual(rows["gedi_reference", "canopy_height"]["mean"], 45)
            agb = rows["prediction_region", "agb"]
            self.assertAlmostEqual(agb["mean"], 142.994)
            self.assertAlmostEqual(agb["total_mg"], 0.06 * 142.994)
            with rasterio.open(root / "carbon_stock.tif") as raster:
                self.assertEqual(raster.transform, transform)
                self.assertEqual(raster.crs.to_epsg(), 32648)
                self.assertEqual(raster.shape, (2, 4))
                values = raster.read(1, masked=True)
                self.assertEqual(values.count(), 6)
                np.testing.assert_allclose(values.compressed(), 2.58 * 0.48 * 142.994 + 81.06)
            save_patch_carbon(root, "0", np.full((2, 2), 0.5),
                              np.load(paths[0]), np.load(sentinel_paths[0])[0] > 0, 20)
            gedi_agb = np.load(root / "agb" / "np_array" / "0_gedi.npy")
            self.assertAlmostEqual(gedi_agb[0, 0], 10.8 * 45 + 34.994)
            self.assertTrue(np.isnan(gedi_agb[1, 0]))

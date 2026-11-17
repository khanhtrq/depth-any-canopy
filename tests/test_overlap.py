import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import ast
import os

import matplotlib
matplotlib.use('Agg')
import numpy as np

from overlap_layout import make_layout, blend_weights, patch_window
from prediction_map import save_region_map


class OverlapTests(unittest.TestCase):
    def test_transfer_windows_with_and_without_overlap(self):
        for stride in [128, 256]:
            side = 256+stride
            data = np.arange(side*side).reshape(side, side)
            for row in range(2):
                for col in range(2):
                    tile = data[patch_window(row, col, 256, stride)]
                    self.assertEqual(tile.shape, (256, 256))
                    self.assertEqual(tile[0, 0], data[row*stride, col*stride])
                    self.assertEqual(tile[-1, -1], data[row*stride+255, col*stride+255])
            if stride == 128:
                np.testing.assert_equal(data[patch_window(0, 0, 256, stride)][:, 128:],
                                        data[patch_window(0, 1, 256, stride)][:, :128])

    def test_merge_across_output_stripes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'np_array').mkdir()
            paths = []
            for i in range(2):
                np.save(root/'np_array'/f'{i}.npy', np.full((130, 130), i, np.float32))
                path = root/f's{i}.npy'
                np.save(path, np.ones((12, 130, 130), np.float32))
                paths.append(str(path))
            data = SimpleNamespace(file_idx=[0, 1], sample_regions=['z']*2,
                                   sample_ids=['0', '1'], sentinel_paths=paths, max_height=30)
            save_region_map(data, 'z', make_layout(130, 195, 130, 65), str(root), max_preview_size=32)
            weights = blend_weights(130, 130)
            expected = np.zeros((195, 130), np.float32)
            denom = np.zeros_like(expected)
            denom[:130] += weights
            denom[65:] += weights
            expected[65:] += weights
            expected /= denom
            np.testing.assert_allclose(np.load(root/'map.npy'), expected)

    def test_overlap_spatial_split_has_no_shared_pixels(self):
        source = Path(__file__).resolve().parents[1]/'dataset'/'sentinel_gedi_canopy.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GediSentinelDataset')
        namespace = dict(Dataset=object, np=np, os=os, Path=Path, json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ['Sentinel', 'GEDI']:
                (root/folder/'z').mkdir(parents=True)
                for i in range(20):
                    np.save(root/folder/'z'/f'{i}.npy', np.ones((16, 16)))
            layout = make_layout(16, 168, 16, 8)
            (root/'layout.json').write_text(json.dumps(layout))
            cls = namespace['GediSentinelDataset']
            args = dict(gedi_folder=str(root/'GEDI'), sentinel_folder=str(root/'Sentinel'), regions=['z'], ratio_train=.5)
            train, test = cls(**args), cls(**args, mode='test')
            train_end = max(int(train.sample_ids[i])*8+16 for i in train.file_idx)
            test_start = min(int(test.sample_ids[i])*8 for i in test.file_idx)
            self.assertLess(train_end, test_start)
            with self.assertRaises(ValueError):
                cls(**args, all_train_data=True)

    def test_grid_covers_extent(self):
        layout = make_layout(11, 7, 4, 2)
        self.assertEqual((layout['rows'], layout['columns']), (3, 5))
        self.assertGreaterEqual((layout['columns']-1)*2+4, 11)
        self.assertTrue((blend_weights(4, 4) > 0).all())
        with self.assertRaises(ValueError):
            make_layout(11, 7, 4, 5)

    def test_weighted_overlap_mask_nan_and_clipped_extent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'np_array').mkdir()
            paths = []
            for i in range(2):
                pred = np.full((4, 4), float(i), np.float32)
                if i == 1:
                    pred[0, 0] = np.nan
                np.save(root/'np_array'/f'{i}.npy', pred)
                sentinel = np.ones((12, 4, 4), np.float32)
                sentinel[:, 3, :] = 0
                path = root/f's{i}.npy'
                np.save(path, sentinel)
                paths.append(str(path))
            data = SimpleNamespace(file_idx=[1, 0], sample_regions=['z']*2,
                                   sample_ids=['0', '1'], sentinel_paths=paths, max_height=30)
            layout = make_layout(5, 4, 4, 2)
            save_region_map(data, 'z', layout, str(root))
            result = np.load(root/'map.npy')
            self.assertEqual(result.shape, (4, 5))
            np.testing.assert_allclose(result[1, 2:4], [1/3, 2/3])
            self.assertEqual(result[0, 2], 0)
            self.assertTrue(np.isnan(result[3]).all())
            self.assertTrue((root/'rgb.png').exists())

    def test_intersecting_zones_share_spatial_assignments(self):
        source = Path(__file__).resolve().parents[1]/'dataset'/'sentinel_gedi_canopy.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GediSentinelDataset')
        namespace = dict(Dataset=object, np=np, os=os, Path=Path, json=json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            roots = []
            for zone, offset in [('a', 0), ('b', 2)]:
                root = Path(temporary)/zone
                roots.append(root)
                for folder in ['Sentinel', 'GEDI']:
                    (root/folder/zone).mkdir(parents=True)
                    for i in range(100):
                        np.save(root/folder/zone/f'{i}.npy', np.ones((16, 16)))
                layout = make_layout(16, 100*128+128, 256, 128, crs='EPSG:32648', transform=[10, 0, 0, 0, -10, 2400000-offset*1280])
                (root/'layout.json').write_text(json.dumps(layout))
            selections = {}
            for root in roots:
                for mode in ['train', 'test']:
                    data = namespace['GediSentinelDataset'](str(root/'GEDI'), str(root/'Sentinel'), [root.name], mode=mode)
                    layout = json.loads((root/'layout.json').read_text())
                    selections[root.name, mode] = {layout['transform'][5]-int(data.sample_ids[i])*1280 for i in data.file_idx}
            self.assertFalse(selections['a', 'train'] & selections['b', 'test'])
            self.assertFalse(selections['b', 'train'] & selections['a', 'test'])
            self.assertTrue(selections['a', 'train'] & selections['b', 'train'])
            self.assertTrue(selections['a', 'test'] & selections['b', 'test'])


if __name__ == '__main__':
    unittest.main()

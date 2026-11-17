"""Halo alignment, central-only supervision, random splitting and map placement."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Literal
import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image

from halo import crop_core, prediction_core, make_halo_layout
from prediction_map import save_region_map, resolve_prediction_layouts

try:
    import torch
    import torch.nn.functional as functional
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class HaloGeometryTests(unittest.TestCase):
    def test_layout_describes_target_not_input_extent(self):
        layout = make_halo_layout(513, 257, crs='EPSG:32648', transform=[10, 0, 0, 0, -10, 0])
        self.assertEqual((layout['rows'], layout['columns']), (2, 3))
        self.assertEqual((layout['height'], layout['width']), (257, 513))
        self.assertEqual(layout['stride'], 256)
        self.assertEqual(layout['input_shape'], [320, 320])
        self.assertEqual(layout['merge'], 'core-stitch')
        with self.assertRaises(ValueError):
            make_halo_layout(0, 257)

    def test_crop_and_invalid_geometry(self):
        array = np.arange(8*8).reshape(8, 8)
        np.testing.assert_equal(crop_core(array, (4, 4), 2), array[2:6, 2:6])
        self.assertEqual(crop_core(array, (8, 8), 0).shape, (8, 8))
        for halo in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                crop_core(array, (4, 4), halo)
        with self.assertRaises(ValueError):
            crop_core(array, (4, 4), 1)
        with self.assertRaises(ValueError):
            prediction_core(array, (8, 6), (4, 4), lambda *a, **kw: array)

    def test_halo_map_copies_centres_and_excludes_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'np_array').mkdir()
            paths = []
            for sid, value in [(0, .2), (1, .8)]:
                np.save(root/'np_array'/f'{sid}.npy', np.full((4, 4), value, np.float32))
                x = np.full((12, 8, 8), 10000, np.float32)
                x[:, 2:6, 2:6] = sid+1
                if sid == 0:
                    x[:, 2, 2] = 0
                path = root/f's{sid}.npy'
                np.save(path, x)
                paths.append(str(path))
            dataset = SimpleNamespace(file_idx=[1, 0], sample_regions=['z']*2,
                                      sample_ids=['0', '1'], sentinel_paths=paths, max_height=30)
            layout = dict(rows=2, columns=2, patch_size=4, stride=4, input_halo=2,
                          height=7, width=7, order='row-major')
            save_region_map(dataset, 'z', layout, str(root))
            result = np.load(root/'map.npy')
            self.assertEqual(result.shape, (7, 7))
            np.testing.assert_allclose(result[1:4, :4], .2)
            np.testing.assert_allclose(result[:4, 4:], .8)
            self.assertTrue(np.isnan(result[0, 0]))
            self.assertTrue(np.isnan(result[4:]).all())
            with Image.open(root/'rgb.png') as image:
                rgb = np.array(image)
            np.testing.assert_array_equal(rgb[4:], 255)
            self.assertEqual(json.loads((root/'layout.json').read_text())['input_halo'], 2)
            with self.assertRaises(ValueError):
                save_region_map(dataset, 'z', dict(layout, stride=2), str(root))


@unittest.skipIf(torch is None, 'PyTorch is required for CPU model-path tests')
class HaloModelTests(unittest.TestCase):
    @staticmethod
    def resize(value, size, interpolation='bilinear'):
        squeezed = value.ndim == 3
        if squeezed:
            value = value.unsqueeze(1)
        value = functional.interpolate(value, size=size, mode=interpolation, align_corners=False)
        return value.squeeze(1) if squeezed else value

    def test_prediction_alignment_and_zero_halo_loss_gradient(self):
        for leading in [(1,), (1, 1)]:
            prediction = torch.full((*leading, 518, 518), .5, requires_grad=True)
            with torch.no_grad():
                prediction[..., :40, :] = 9
                prediction[..., -40:, :] = 9
                prediction[..., :, :40] = 9
                prediction[..., :, -40:] = 9
            centre = prediction_core(prediction, (320, 320), (256, 256), self.resize)
            self.assertEqual(tuple(centre.shape), (*leading, 256, 256))
            torch.testing.assert_close(centre, torch.full_like(centre, .5))
            centre.square().mean().backward()
            self.assertEqual(torch.count_nonzero(prediction.grad[..., :40, :]).item(), 0)
            self.assertEqual(torch.count_nonzero(prediction.grad[..., :, -40:]).item(), 0)
            self.assertGreater(torch.count_nonzero(prediction.grad[..., 100:400, 100:400]).item(), 0)

    def test_original_resize_path_is_unchanged(self):
        prediction = torch.linspace(0, 1, 518*518).reshape(1, 518, 518)
        expected = self.resize(prediction, (256, 256))
        actual = prediction_core(prediction, (256, 256), (256, 256), self.resize)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_lightning_steps_use_centre_and_save_core_predictions(self):
        self._exercise_lightning_steps(32)

    def test_original_all_lightning_steps_keep_full_target(self):
        self._exercise_lightning_steps(0)

    def _exercise_lightning_steps(self, halo):
        cls = load_class(ROOT/'lightning_model.py', 'DepthAnythingV2Module', dict(
            LightningModule=object, Literal=Literal, torch=torch, np=np, os=os,
            resize=self.resize, crop_core=crop_core, prediction_core=prediction_core,
            plt=SimpleNamespace(close=lambda _: None)))

        class Metric:
            def __init__(self, prefix):
                self.prefix = prefix
            def __call__(self, predicted, target):
                self.count = predicted.numel()
                self.mse = ((predicted-target)**2).mean()
            def compute(self):
                return {self.prefix+'_MSE': self.mse}
            def reset(self):
                pass

        for huggingface in [False, True]:
            module = cls.__new__(cls)
            module.use_huggingface = huggingface
            raw = torch.full((1, 518, 518), .5, requires_grad=True)
            if halo:
                with torch.no_grad():
                    raw[:, :40] = 9
                    raw[:, -40:] = 9
            seen = []
            def forward(image):
                seen.append(tuple(image.shape))
                return SimpleNamespace(predicted_depth=raw) if huggingface else raw
            module.model = forward
            module.hparams = SimpleNamespace(min_depth=0, max_depth=30)
            module.loss = torch.nn.MSELoss()
            module.train_metric, module.val_metric = Metric('train'), Metric('val')
            module.log = lambda *a, **kw: None
            module.log_dict = lambda *a, **kw: None
            module.current_epoch, module.logger = 1, None
            plotted = []
            def plot(image, *args, **kwargs):
                plotted.append(tuple(image.shape))
                return SimpleNamespace(savefig=lambda *a, **kw: None)
            module.trainer = SimpleNamespace(datamodule=SimpleNamespace(
                predict_dataset=SimpleNamespace(plot=plot)))
            batch = dict(image=torch.ones((1, 12, 256+2*halo, 256+2*halo)),
                         mask=torch.full((1, 256, 256), .25),
                         mangrove_mask=torch.ones((1, 256, 256), dtype=torch.bool),
                         zone=['z'], sample_id=['0'])
            batch['mask'][0, 0, 0] = float('nan')
            batch['mangrove_mask'][0, 0, 0] = False
            loss = module.training_step(batch, 0)
            self.assertAlmostEqual(loss.item(), .0625)
            loss.backward()
            if halo:
                self.assertEqual(torch.count_nonzero(raw.grad[:, :40]).item(), 0)
            else:
                self.assertGreater(torch.count_nonzero(raw.grad[:, :40]).item(), 0)
            self.assertEqual(module.train_metric.count, 256*256-1)
            self.assertAlmostEqual(module.validation_step(batch, 0).item(), .0625)
            module.on_test_epoch_start()
            module.test_step(batch, 0)
            self.assertEqual(module.test_records['z', '0'][0], 256*256-1)
            with tempfile.TemporaryDirectory() as temporary:
                previous = os.getcwd()
                try:
                    os.chdir(temporary)
                    predicted = module.predict_step(batch, 0)
                    saved = np.load(Path('inference_predictions/z/np_array/0.npy'))
                finally:
                    os.chdir(previous)
            self.assertEqual(tuple(predicted.shape), (1, 256, 256))
            self.assertEqual(saved.shape, (256, 256))
            self.assertEqual(saved[0, 0], 0)
            np.testing.assert_allclose(saved[1:], .5)
            self.assertEqual(plotted, [(12, 256, 256)] if halo else [(12, 518, 518)])
            self.assertEqual(seen, [(1, 12, 518, 518)]*4)

    def test_dataset_random_split_core_mask_and_legacy(self):
        class Compose:
            def __init__(self, transforms): self.transforms = transforms
            def __call__(self, value):
                for transform in self.transforms: value = transform(value)
                return value
        class Normalize:
            def __init__(self, mean, std):
                self.mean, self.std = mean, std
            def __call__(self, value):
                return (value-torch.tensor(self.mean)[:, None, None])/torch.tensor(self.std)[:, None, None]
        transforms = SimpleNamespace(Compose=Compose, Normalize=Normalize,
            ToTensor=lambda: lambda value: torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))))
        cls = load_class(ROOT/'dataset/sentinel_gedi_canopy.py', 'GediSentinelDataset',
                         dict(Dataset=object, np=np, torch=torch, os=os, json=json, T=transforms))
        with tempfile.TemporaryDirectory() as temporary:
            selected = []
            for halo in [0, 2]:
                root = Path(temporary)/str(halo)
                for folder in ['GEDI', 'Sentinel']: (root/folder/'z').mkdir(parents=True)
                for sid in range(10):
                    y = np.full((4, 4), 7.5, np.float32)
                    y[0, 0] = np.nan
                    x = np.full((12, 4+2*halo, 4+2*halo), 5, np.float32)
                    x[:, halo:halo+4, halo:halo+4] = 1
                    x[:, halo, halo] = 0
                    np.save(root/'GEDI'/'z'/f'{sid}.npy', y)
                    np.save(root/'Sentinel'/'z'/f'{sid}.npy', x)
                (root/'layout.json').write_text(json.dumps(dict(rows=2, columns=5,
                    patch_size=4, stride=4, input_halo=halo, height=8, width=20)))
                args = dict(gedi_folder=str(root/'GEDI'), sentinel_folder=str(root/'Sentinel'),
                            regions=['z'], input_halo=halo)
                train, test, val = cls(**args), cls(**args, mode='test'), cls(**args, mode='val')
                train_ids = [train.sample_ids[i] for i in train.file_idx]
                test_ids = [test.sample_ids[i] for i in test.file_idx]
                self.assertEqual((len(train), len(test)), (8, 2))
                self.assertFalse(set(train_ids) & set(test_ids))
                self.assertEqual(test.file_idx, val.file_idx)
                selected.append((train_ids, test_ids))
                sample = train[0]
                self.assertEqual(tuple(sample['image'].shape), (12, 4+2*halo, 4+2*halo))
                self.assertEqual(tuple(sample['mask'].shape), (4, 4))
                self.assertEqual(tuple(sample['mangrove_mask'].shape), (4, 4))
                self.assertFalse(sample['mangrove_mask'][0, 0])
                self.assertEqual(sample['mangrove_mask'].sum().item(), 15)
                self.assertAlmostEqual(sample['mask'][1, 1].item(), .25)
                self.assertEqual(len(cls(**args, predict=True)), 10)
                dataset = cls(**args, predict=True)
                resolved = resolve_prediction_layouts(dataset, {})['z']
                self.assertEqual(resolved.get('input_halo', 0), halo)
            self.assertEqual(selected[0], selected[1])


if __name__ == '__main__':
    unittest.main()

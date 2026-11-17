"""Check path resolution and setup wiring without loading training dependencies."""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def load_class(name, namespace):
    source = Path(__file__).resolve().parents[1] / "dataset" / "sentinel_gedi_canopy.py"
    tree = ast.parse(source.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class RegionSubdirTests(unittest.TestCase):
    def test_halo_reaches_all_stages(self):
        factory = Mock()
        module_class = load_class('GediSentinelDataModule', {
            'L': SimpleNamespace(LightningDataModule=object), 'GediSentinelDataset': factory,
        })
        module = module_class(['z'], 'GEDI', 'Sentinel', input_halo=32)
        module.setup('fit')
        module.setup('test')
        module.setup('predict')
        self.assertEqual(factory.call_count, 4)
        self.assertTrue(all(call.kwargs['input_halo'] == 32 for call in factory.call_args_list))

    def test_paths_and_region_identity(self):
        dataset_class = load_class("GediSentinelDataset", {"Dataset": object, "os": os})
        for mapping in (None, {}, {"named_run": "disk_region"}, {"other": "unused"}):
            with self.subTest(mapping=mapping), tempfile.TemporaryDirectory() as root:
                subdir = (mapping or {}).get("named_run", "named_run")
                for kind in ("GEDI", "Sentinel"):
                    folder = Path(root) / kind / subdir
                    folder.mkdir(parents=True)
                    (folder / "7.npy").touch()
                for mapped_roots in (False, True):
                    gedi = os.path.join(root, "GEDI")
                    sentinel = os.path.join(root, "Sentinel")
                    dataset = dataset_class(
                        {"named_run": gedi} if mapped_roots else gedi,
                        {"named_run": sentinel} if mapped_roots else sentinel,
                        ["named_run"], predict=True, region_subdirs=mapping,
                    )
                    self.assertEqual(dataset.sample_regions, ["named_run"])
                    self.assertEqual(dataset.sample_ids, ["7"])
                    self.assertEqual(dataset.gedi_paths, [os.path.join(gedi, subdir, "7.npy")])
                    self.assertEqual(dataset.sentinel_paths, [os.path.join(sentinel, subdir, "7.npy")])

    def test_all_stages_receive_mapping(self):
        factory = Mock()
        module_class = load_class("GediSentinelDataModule", {
            "L": SimpleNamespace(LightningDataModule=object), "GediSentinelDataset": factory,
        })
        for mapping in (None, {"named_run": "disk_region"}):
            factory.reset_mock()
            module = module_class(["named_run"], "GEDI", "Sentinel", region_subdirs=mapping)
            module.setup("fit")
            module.setup("test")
            module.setup("predict")
            self.assertEqual(factory.call_count, 4)
            for call in factory.call_args_list:
                self.assertEqual(call.kwargs["region_subdirs"], mapping or {})


if __name__ == "__main__":
    unittest.main()

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from distributed_runtime import launch_workers, worker_count


class DistributedRuntimeTests(unittest.TestCase):
    def test_device_selection(self):
        for devices, expected in [(1, 1), (2, 2), ("auto", 2), (-1, 2),
                                  ([0, 1], 2), ("0,1", 2)]:
            with self.subTest(devices=devices):
                self.assertEqual(worker_count({"devices": devices}, 2), expected)
        self.assertEqual(worker_count({"devices": "auto"}, 0), 1)
        self.assertEqual(worker_count({"devices": 2, "num_nodes": 2}, 2), 1)
        self.assertEqual(worker_count({"devices": 2, "strategy": "ddp_spawn"}, 2), 1)

    @patch.dict(os.environ, {}, clear=True)
    @patch("distributed_runtime.subprocess.run")
    def test_one_launch_preserves_overrides_and_defers_merge(self, run):
        overrides = ["trainer.devices=2", "+trainer.strategy=ddp_find_unused_parameters_true",
                     "dataset.gedi_folder={zone1:/path with spaces/GEDI}"]
        self.assertTrue(launch_workers({"devices": 2}, 2, "main.py", overrides))
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "torch.distributed.run"])
        self.assertIn("--nproc_per_node=2", command)
        self.assertEqual(command[-3:], overrides)
        self.assertTrue(run.call_args.kwargs["check"])
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CANOPY_DEFER_MAP_MERGE"], "1")
        self.assertEqual(env["HYDRA_FULL_ERROR"], "1")
        self.assertEqual(len(env["CANOPY_RUN_ID"]), 32)

    @patch("distributed_runtime.subprocess.run")
    def test_workers_and_single_device_do_not_launch_recursively(self, run):
        for env in ({"LOCAL_RANK": "0"}, {"LOCAL_RANK": "1"}, {"RANK": "0"}):
            with patch.dict(os.environ, env, clear=True):
                self.assertFalse(launch_workers({"devices": 2}, 2, "main.py", []))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(launch_workers({"devices": 1}, 2, "main.py", []))
        run.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("distributed_runtime.subprocess.run", side_effect=subprocess.CalledProcessError(1, "torchrun"))
    def test_worker_failure_is_not_reported_as_success(self, run):
        with self.assertRaises(subprocess.CalledProcessError):
            launch_workers({"devices": 2}, 2, "main.py", [])


if __name__ == "__main__":
    unittest.main()

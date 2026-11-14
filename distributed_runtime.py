"""Launch one persistent set of DDP workers for the full main.py workflow."""

import os
import subprocess
import sys
import uuid


def worker_count(trainer_config, available_gpus):
    if int(trainer_config.get("num_nodes", 1)) != 1:
        return 1  # Multi-node runs should be launched by the cluster scheduler.
    strategy = str(trainer_config.get("strategy", "auto"))
    if strategy != "auto" and strategy not in {
        "ddp", "ddp_find_unused_parameters_true", "ddp_find_unused_parameters_false"
    }:
        return 1
    accelerator = trainer_config.get("accelerator", "auto")
    if accelerator not in {"auto", "gpu", "cuda", "cpu"}:
        return 1
    devices = trainer_config.get("devices", "auto")
    if devices in ("auto", -1, "-1"):
        return max(1, available_gpus) if accelerator != "cpu" else 1
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, str) and "," in devices:
        return len([device for device in devices.split(",") if device.strip()])
    return int(devices)


def launch_workers(trainer_config, available_gpus, script, overrides):
    """Return True only in the driver, after all distributed workers succeed."""
    if any(key in os.environ for key in ("LOCAL_RANK", "RANK", "SLURM_PROCID")):
        return False
    count = worker_count(trainer_config, available_gpus)
    if count <= 1:
        return False
    env = os.environ.copy()
    env["CANOPY_DEFER_MAP_MERGE"] = "1"
    env["CANOPY_RUN_ID"] = uuid.uuid4().hex
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    command = [sys.executable, "-u", "-m", "torch.distributed.run",
               "--standalone", "--nnodes=1", f"--nproc_per_node={count}",
               os.path.abspath(script), *overrides]
    print(f"Launching {count} persistent DDP workers for training, evaluation and prediction", flush=True)
    # An argv list preserves Hydra mappings and paths without shell interpolation.
    subprocess.run(command, env=env, check=True)
    return True

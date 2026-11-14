"""Reconstruct region maps from numbered, non-overlapping prediction patches."""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Source dimensions are (height, width); only complete patches were exported.
SOURCE_SHAPES = {
    "BaBe": (10980, 10980),
    "HoangLien": (10980, 10980),
    "CucPhuong": (10980, 10980),
    "CaMau": (2609, 6033),
    "zone1": (19482, 13359),
    "zone2": (13915, 13360),
    "zone4": (42302, 32283),
}


def default_layout(zone, patch_shape):
    source_height, source_width = SOURCE_SHAPES[zone]
    height, width = patch_shape
    return {"rows": source_height // height, "columns": source_width // width}


def patch_position(sample_id, layout):
    """IDs include missing grid cells; first_id is the top-left cell's ID."""
    rows, columns = int(layout["rows"]), int(layout["columns"])
    if rows <= 0 or columns <= 0:
        raise ValueError("Prediction layout rows and columns must be positive")
    index = int(sample_id) - int(layout.get("first_id", 0))
    if not 0 <= index < rows * columns:
        raise ValueError(f"Patch {sample_id} is outside the configured prediction grid")
    order = layout.get("order", "row-major")
    if order == "row-major":
        return divmod(index, columns)
    if order == "column-major":
        column, row = divmod(index, rows)
        return row, column
    raise ValueError("Prediction layout order must be row-major or column-major")


def save_region_map(dataset, zone, layout, output_dir):
    """Read this run's patches from disk, keeping uncovered cells as NaN."""
    entries = [i for i in dataset.file_idx if dataset.sample_regions[i] == zone]
    if not entries:
        return
    rows, columns = int(layout["rows"]), int(layout["columns"])
    patch_dir = os.path.join(output_dir, "np_array")
    first = np.load(os.path.join(patch_dir, f"{dataset.sample_ids[entries[0]]}.npy"))
    height, width = first.shape[-2:]
    map_path = os.path.join(output_dir, "map.npy")
    mosaic = np.lib.format.open_memmap(
        map_path, mode="w+", dtype=np.float32,
        shape=(rows * height, columns * width),
    )
    mosaic[:] = np.nan
    visible = np.zeros(mosaic.shape, dtype=bool)
    rgb_image = Image.new("RGB", (columns * width, rows * height), "white")
    for i in entries:
        sample_id = dataset.sample_ids[i]
        row, column = patch_position(sample_id, layout)
        patch = np.load(os.path.join(patch_dir, f"{sample_id}.npy"))
        if patch.shape[-2:] != (height, width) or patch.size != height * width:
            raise ValueError(f"Inconsistent prediction patch shape for {zone}/{sample_id}")
        window = np.s_[row * height:(row + 1) * height,
                       column * width:(column + 1) * width]
        mosaic[window] = patch.reshape(height, width)
        sentinel = np.load(dataset.sentinel_paths[i])
        if sentinel.shape[-2:] != (height, width):
            raise ValueError(f"Sentinel/prediction size mismatch for {zone}/{sample_id}")
        visible[window] = np.isfinite(sentinel).all(axis=0) & np.any(sentinel != 0, axis=0)
        # Same RGB bands and per-patch min/max stretch as dataset.plot(),
        # applied directly to the original, unnormalized reflectance.
        rgb = np.array(sentinel[[2, 1, 0]], dtype=np.float32)
        rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)
        rgb -= rgb.min()
        rgb /= max(float(rgb.max()), 1e-8)
        rgb[:, ~visible[window]] = 1
        rgb_patch = Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
        rgb_image.paste(rgb_patch, (column * width, row * height))
        rgb_patch.close()
    mosaic.flush()
    rgb_image.save(os.path.join(output_dir, "rgb.png"))
    rgb_image.close()
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        image = ax.imshow(np.ma.masked_where(~visible | ~np.isfinite(mosaic), mosaic),
                          cmap=cmap, vmin=0, vmax=1, interpolation="none")
        ax.set_title(f"{zone} - Prediction ({rows} x {columns})")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
            f"Canopy Height (0–{dataset.max_height:g}m)")
        fig.savefig(os.path.join(output_dir, "map.png"), bbox_inches="tight", dpi=200)
    finally:
        plt.close(fig)
        del mosaic

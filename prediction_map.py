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
    "zone1": (13359, 19482),
    "zone2": (13360, 13915),
    "zone4": (32283, 42302),
}


def default_layout(zone, patch_shape):
    source_height, source_width = SOURCE_SHAPES[zone]
    height, width = patch_shape
    return {"rows": source_height // height, "columns": source_width // width}


def resolve_prediction_layouts(dataset, configured):
    layouts = {}
    for zone in dataset.regions:
        layout = configured.get(zone)
        entries = [i for i in dataset.file_idx if dataset.sample_regions[i] == zone]
        if not entries:
            raise ValueError(f"No prediction patches found for {zone}")
        if layout is None and zone in SOURCE_SHAPES:
            shape = np.load(dataset.gedi_paths[entries[0]], mmap_mode="r").shape[-2:]
            layout = default_layout(zone, shape)
        if layout is None:
            raise ValueError(
                f"Set dataset.prediction_layouts.{zone}={{rows: <rows>, columns: <columns>}} "
                "to the original patch grid; numeric IDs alone do not encode its width."
            )
        for i in entries:
            patch_position(dataset.sample_ids[i], layout)
        layouts[zone] = layout
    return layouts


def merge_region_predictions(dataset, configured):
    for zone, layout in resolve_prediction_layouts(dataset, configured).items():
        output_dir = os.path.abspath(os.path.join("inference_predictions", zone))
        print(f"Merging prediction map for {zone}", flush=True)
        save_region_map(dataset, zone, layout, output_dir)
        print(f"Saved {zone}: map.npy, map.png, rgb.png in {output_dir}", flush=True)


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


def save_region_map(dataset, zone, layout, output_dir, max_preview_size=4096, patch_dir=None):
    """Save a full-resolution array and bounded RGB/heatmap visualizations."""
    entries = [i for i in dataset.file_idx if dataset.sample_regions[i] == zone]
    if not entries:
        return
    rows, columns = int(layout["rows"]), int(layout["columns"])
    patch_dir = patch_dir or os.path.join(output_dir, "np_array")
    first = np.load(os.path.join(patch_dir, f"{dataset.sample_ids[entries[0]]}.npy"))
    height, width = first.shape[-2:]
    # One shared RGB stretch for the region, excluding masked/background pixels.
    rgb_min, rgb_max = float("inf"), float("-inf")
    for i in entries:
        sentinel = np.load(dataset.sentinel_paths[i])
        valid = np.isfinite(sentinel).all(axis=0) & np.any(sentinel != 0, axis=0)
        if valid.any():
            values = sentinel[[2, 1, 0]][:, valid]
            rgb_min = min(rgb_min, float(values.min()))
            rgb_max = max(rgb_max, float(values.max()))
    if not np.isfinite(rgb_min):
        rgb_min = rgb_max = 0.0
    rgb_range = max(rgb_max - rgb_min, 1e-8)
    map_path = os.path.join(output_dir, "map.npy")
    mosaic = np.lib.format.open_memmap(
        map_path, mode="w+", dtype=np.float32,
        shape=(rows * height, columns * width),
    )
    mosaic[:] = np.nan
    # Zone4 alone has >1 billion pixels. Never give the full map to Matplotlib
    # or allocate full-map RGB and visibility arrays alongside it.
    scale = min(1.0, max_preview_size / max(mosaic.shape))
    preview_height = max(1, round(mosaic.shape[0] * scale))
    preview_width = max(1, round(mosaic.shape[1] * scale))
    preview = np.full((preview_height, preview_width), np.nan, dtype=np.float32)
    rgb_image = Image.new("RGB", (preview_width, preview_height), "white")
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
        visible = np.isfinite(sentinel).all(axis=0) & np.any(sentinel != 0, axis=0)
        top = row * height * preview_height // mosaic.shape[0]
        bottom = (row + 1) * height * preview_height // mosaic.shape[0]
        left = column * width * preview_width // mosaic.shape[1]
        right = (column + 1) * width * preview_width // mosaic.shape[1]
        if bottom == top or right == left:
            continue
        y = np.linspace(0, height - 1, bottom - top).astype(int)
        x = np.linspace(0, width - 1, right - left).astype(int)
        sample = np.ix_(y, x)
        preview[top:bottom, left:right] = np.where(
            visible[sample], patch.reshape(height, width)[sample], np.nan)
        # Shared region-wide min/max applied to original RGB reflectance.
        rgb = np.array(sentinel[[2, 1, 0]], dtype=np.float32)
        rgb = np.nan_to_num(rgb, nan=0, posinf=0, neginf=0)
        rgb -= rgb_min
        rgb /= rgb_range
        np.clip(rgb, 0, 1, out=rgb)
        rgb[:, ~visible] = 1
        rgb_patch = Image.fromarray((rgb.transpose(1, 2, 0)[sample] * 255).astype(np.uint8))
        rgb_image.paste(rgb_patch, (left, top))
        rgb_patch.close()
    mosaic.flush()
    rgb_image.save(os.path.join(output_dir, "rgb.png"))
    rgb_image.close()
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        image = ax.imshow(np.ma.masked_invalid(preview),
                          cmap=cmap, vmin=0, vmax=1, interpolation="none")
        ax.set_title(f"{zone} - Prediction ({rows} x {columns})")
        ax.axis("off")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label(
            f"Canopy Height (0–{dataset.max_height:g}m)")
        fig.savefig(os.path.join(output_dir, "map.png"), bbox_inches="tight", dpi=200)
    finally:
        plt.close(fig)
        del mosaic


def main(argv=None):
    """Rebuild a chosen layout from existing predictions, without loading a model."""
    import argparse
    import json
    from pathlib import Path
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--sentinel-dir", required=True,
                        help="Directory containing this zone's original Sentinel .npy patches")
    parser.add_argument("--patch-dir", help="Defaults to inference_predictions/<zone>/np_array")
    parser.add_argument("--output-dir", help="Defaults to a separate remerge_<rows>x<columns>_<order> folder")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--order", choices=["row-major", "column-major"], default="row-major")
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--max-height", type=float, default=30)
    parser.add_argument("--max-preview-size", type=int, default=4096)
    args = parser.parse_args(argv)
    root = Path("inference_predictions") / args.zone
    patch_dir = Path(args.patch_dir) if args.patch_dir else root / "np_array"
    files = sorted(patch_dir.glob("*.npy"), key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No saved prediction patches in {patch_dir}")
    layout = {"rows": args.rows, "columns": args.columns,
              "order": args.order, "first_id": args.first_id}
    sentinel_paths = []
    for path in files:
        patch_position(path.stem, layout)
        sentinel = Path(args.sentinel_dir) / path.name
        if not sentinel.is_file():
            raise FileNotFoundError(f"Missing Sentinel patch: {sentinel}")
        sentinel_paths.append(str(sentinel))
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / f"remerge_{args.rows}x{args.columns}_{args.order}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SimpleNamespace(file_idx=list(range(len(files))),
                              sample_regions=[args.zone] * len(files),
                              sample_ids=[path.stem for path in files],
                              sentinel_paths=sentinel_paths, max_height=args.max_height)
    save_region_map(dataset, args.zone, layout, str(output_dir),
                    max_preview_size=args.max_preview_size, patch_dir=str(patch_dir))
    with open(output_dir / "layout.json", "w") as handle:
        json.dump({"zone": args.zone, "layout": layout, "patch_count": len(files),
                   "patch_dir": str(patch_dir.resolve()),
                   "sentinel_dir": str(Path(args.sentinel_dir).resolve())}, handle, indent=2)
    print(f"Rebuilt {args.zone} with {args.rows} rows x {args.columns} columns ({args.order}): "
          f"{output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

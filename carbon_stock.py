"""Mangrove allometry and inference exports."""

import csv
from contextlib import ExitStack
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np


AGB_SLOPE = 10.8
AGB_INTERCEPT = 34.994
CARBON_FRACTION = 0.48
BGC_SLOPE = 1.58
BGC_INTERCEPT = 81.06
CARBON_MULTIPLIER = (1 + BGC_SLOPE) * CARBON_FRACTION
QUANTITIES = (
    ("canopy_height", "m", 1.0, 0.0),
    ("agb", "Mg biomass/ha", AGB_SLOPE, AGB_INTERCEPT),
    ("carbon_stock", "Mg C/ha", CARBON_MULTIPLIER * AGB_SLOPE,
     CARBON_MULTIPLIER * AGB_INTERCEPT + BGC_INTERCEPT),
)


def save_patch_carbon(output, sample_id, prediction, gedi, valid, max_height):
    valid = np.asarray(valid, dtype=bool)
    prediction = prediction.reshape(valid.shape) * max_height
    gedi = gedi.reshape(valid.shape)
    predicted = np.where(valid & np.isfinite(prediction), prediction, np.nan)
    reference = np.where(valid & np.isfinite(gedi) & (gedi >= 0), gedi, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    try:
        for column, (name, unit, slope, intercept) in enumerate(QUANTITIES):
            values = (predicted * slope + intercept, reference * slope + intercept)
            if name != "canopy_height":
                folder = Path(output) / name / "np_array"
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / f"{sample_id}.npy", values[0])
                np.save(folder / f"{sample_id}_gedi.npy", values[1])
            high = max(max_height, np.nanmax(reference, initial=0)) * slope + intercept
            for row, label in enumerate(("Prediction", "GEDI reference")):
                ax = axes[row, column]
                image = ax.imshow(values[row], cmap="Spectral_r", vmin=intercept, vmax=high)
                ax.set_title(f"{label} · {name.replace('_', ' ').title()}")
                ax.set_axis_off()
            fig.colorbar(image, ax=axes[:, column], label=unit, shrink=0.8)
        fig.savefig(Path(output) / "image" / f"{sample_id}_carbon.png")
    finally:
        plt.close(fig)


class HeightStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.squares = 0.0

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.count += values.size
        self.total += values.sum()
        self.squares += np.square(values).sum()

    def rows(self, zone, source, pixel_area=None):
        mean = self.total / self.count if self.count else np.nan
        std = np.sqrt(max(0, self.squares / self.count - mean**2)) if self.count else np.nan
        area = self.count * pixel_area if pixel_area is not None else None
        for name, unit, slope, intercept in QUANTITIES:
            yield dict(region=zone, source=source, quantity=name, unit=unit, count=self.count,
                       mean=mean * slope + intercept, std=std * slope, area_ha=area,
                       total_mg=area * (mean * slope + intercept)
                       if area and name != "canopy_height" else None)


def paired_heights(dataset, zone, layout, mosaic):
    from halo import crop_core
    from prediction_map import patch_position

    predicted, reference = HeightStats(), HeightStats()
    seen_ids, seen_pixels = set(), set()
    entries = sorted((i for i in dataset.file_idx if dataset.sample_regions[i] == zone),
                     key=lambda i: int(dataset.sample_ids[i]))
    fields = ("day", "second", "microsecond", "beam")
    for i in entries:
        path = Path(dataset.gedi_paths[i])
        gedi = np.load(path)
        h, w = gedi.shape[-2:]
        gedi = gedi.reshape(h, w)
        row, col = patch_position(dataset.sample_ids[i], layout)
        top, left = row * layout.get("stride", h), col * layout.get("stride", w)
        prediction = mosaic[top:top+h, left:left+w]
        h, w = prediction.shape
        sentinel = crop_core(np.load(dataset.sentinel_paths[i]), gedi.shape,
                             layout.get("input_halo", 0))[:, :h, :w]
        gedi = gedi[:h, :w]
        valid = (np.isfinite(gedi) & (gedi >= 0) & np.isfinite(prediction)
                 & np.isfinite(sentinel).all(axis=0) & np.any(sentinel != 0, axis=0))
        ys, xs = np.nonzero(valid)
        if not ys.size:
            continue
        ids = None
        if path.parent.parent.name.lower() == "gedi":
            sidecar = path.parents[2] / "gedi_source" / path.parent.name / path.with_suffix(".npz").name
            if sidecar.is_file():
                with np.load(sidecar) as data:
                    if all(field in data for field in fields):
                        ids = np.column_stack([data[field][ys, xs] for field in fields])
        selected = []
        for j, (y, x) in enumerate(zip(ys, xs)):
            pixel = (top + int(y), left + int(x))
            shot = tuple(ids[j]) if ids is not None and (ids[j] >= 0).all() else None
            if pixel in seen_pixels or (shot is not None and shot in seen_ids):
                continue
            selected.append(j)
            seen_pixels.add(pixel)
            if shot is not None:
                seen_ids.add(shot)
        predicted.add(prediction[ys[selected], xs[selected]] * dataset.max_height)
        reference.add(gedi[ys[selected], xs[selected]])
    return predicted, reference


def save_region_carbon(dataset, zone, layout, output, max_preview_size=4096):
    import rasterio
    from rasterio.transform import Affine
    from rasterio.windows import Window

    output = Path(output)
    mosaic = np.load(output / "map.npy", mmap_mode="r")
    height, width = mosaic.shape
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32",
                   nodata=np.nan, tiled=True, blockxsize=256, blockysize=128,
                   compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    pixel_area = None
    if layout.get("crs") and layout.get("transform"):
        crs = rasterio.crs.CRS.from_user_input(layout["crs"])
        transform = Affine(*layout["transform"][:6])
        profile.update(crs=crs, transform=transform)
        if crs.is_projected:
            pixel_area = abs(transform.determinant) * crs.linear_units_factor[1]**2 / 10_000
    else:
        warnings.warn(f"{zone}: layout has no CRS and transform. TIFFs use the native pixel grid; "
                      "mapped area and stock totals require a georeferenced layout.", stacklevel=2)

    stats = HeightStats()
    with ExitStack() as stack:
        rasters = []
        for name, unit, _, _ in QUANTITIES:
            raster = stack.enter_context(rasterio.open(output / f"{name}.tif", "w", **profile))
            raster.set_band_description(1, name)
            raster.set_band_unit(1, unit)
            rasters.append(raster)
        for top in range(0, height, 128):
            heights = mosaic[top:top+128] * dataset.max_height
            valid = np.isfinite(heights)
            stats.add(heights[valid])
            heights[~valid] = np.nan
            window = Window(0, top, width, len(heights))
            for raster, (_, _, slope, intercept) in zip(rasters, QUANTITIES):
                raster.write(heights * slope + intercept, 1, window=window)

    scale = min(1.0, max_preview_size / max(height, width))
    ys = np.linspace(0, height-1, max(1, round(height*scale))).astype(int)
    xs = np.linspace(0, width-1, max(1, round(width*scale))).astype(int)
    preview = mosaic[np.ix_(ys, xs)] * dataset.max_height
    for name, unit, slope, intercept in QUANTITIES[1:]:
        fig, ax = plt.subplots(figsize=(12, 10))
        try:
            image = ax.imshow(preview * slope + intercept, cmap="Spectral_r",
                              vmin=intercept, vmax=dataset.max_height * slope + intercept)
            ax.set_title(f"{zone} · {name.replace('_', ' ').title()}")
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, label=unit)
            fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=150)
        finally:
            plt.close(fig)

    predicted, reference = paired_heights(dataset, zone, layout, mosaic)
    return [*stats.rows(zone, "prediction_region", pixel_area),
            *predicted.rows(zone, "prediction_at_gedi"),
            *reference.rows(zone, "gedi_reference")]


def write_region_statistics(path, rows):
    fields = ("region", "source", "quantity", "unit", "count", "mean", "std",
              "area_ha", "total_mg")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

The active experiment uses non-overlapping 256 x 256 GEDI target tiles, stride
256, and 320 x 320 Sentinel inputs containing a 32-pixel halo on all four sides.
The halo must contain neighboring imagery; the loader does not invent context
by padding an existing 256-pixel array. Data downloading is deferred.

The model still receives 518 x 518 inputs. In every train/validation/test/predict
step, the prediction is resized to the original 320 x 320 footprint, then cropped
to [32:288, 32:288]. Only that 256 x 256 centre enters loss, metrics, saved patch
predictions, target masks and map assembly. The plotted image also shows the
centre. Saved predictions are normalized as before; max_height remains 30 m.

Use `dataset=zones_1_2_4_2019_cloud20_halo32` once the new data are available.
Train/test assignment is a seeded (42) random split of labelled target patches,
independently within each zone, using ratio_train=0.8. Validation shares the test
holdout, preserving the original workflow. Inputs can share context and a GEDI
footprint may cross target boundaries; this is random-patch evaluation, not
geographically independent evaluation. Prediction includes unlabelled patches.
Geographic intersections between zones can also put common locations in both
partitions; the requested per-zone random split does not group those locations.

Future dataset contract:

- `Sentinel/<zone>/<id>.npy`: float32 (12,320,320), same band order and units.
- `GEDI/<zone>/<id>.npy`: float32 (256,256), metres, missing labels NaN.
- As before, invalid/non-mangrove Sentinel pixels in the core are zero. The
  loader derives its core validity mask from those stored Sentinel pixels.
- Target masks and GEDI source IDs cover only the 256 x 256 centre.
- The input halo contains real neighboring pixels; unavailable context at the
  external imagery boundary must have a documented fill rule in the exporter.
- `layout.json` describes the target grid, not the padded input. Its fields are
  rows, columns, patch_size=256, stride=256, input_halo=32, first_id=0,
  order=row-major, exact target-region height/width, CRS and target-grid affine.
  `halo.make_halo_layout(...)` builds this metadata without accessing imagery.
- Patch IDs preserve missing grid cells. The layout transform and patch offsets
  refer to the target centre's upper-left pixel; input origins are 32 pixels
  north/west of those coordinates. Do not reuse stride-128 layout metadata.

The merger crops Sentinel context when constructing the validity mask/RGB,
copies each central prediction to its target-grid position, and clips padded
outer tiles to the exact region extent. No overlapping prediction blending is
performed for halo layouts. Missing/invalid output pixels are NaN.

Original experiments keep input_halo=0 by default. Their 256 x 256 inputs,
seeded random split, 518 x 518 model input and prediction resizing behavior are
preserved. Existing rows/columns layouts and the zone_4_2019_cloud20 config still
work. Historical stride-128 exports and their merge helper remain readable for
comparison; they are not inputs for this new experiment. Existing data have not
been modified or downloaded again.

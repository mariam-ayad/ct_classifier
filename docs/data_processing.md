# Data Processing Guide

Step-by-step pipeline for adding a new reef site to the coral bleaching dataset —
from raw PlanetScope download through tile extraction and training.

**A fully illustrated PDF version of this guide (with QGIS screenshots) is available separately.**
The steps here are a concise reference for users already familiar with the tools.

---

## Pipeline Overview

```
1. Download PlanetScope scenes (Planet Explorer)
2. Spatial alignment (QGIS Georeferencer, if needed)
3. Land masking (utils/masking.py)
4. Define tile locations (QGIS point shapefile)
5. Extract 120×120 px tiles (scripts/extract_tiles.py)
6. Visual QA (scripts/viewtif.py)
7. Generate data split (analysis/01_data_split.ipynb)
8. Train the classifier (ct_classifier/train.py)
9. Evaluate (analysis/02–06 notebooks)
```

---

## 1. Accessing PlanetScope Imagery

PlanetScope SuperDove (PSB.SD) imagery is the foundation of the dataset. Each scene is an
8-band multispectral GeoTIFF at 3 m/pixel. Access requires a Planet account with an
Education & Research license.

You need **two sets of imagery per site**:

| Set | Period | Subfolder |
|---|---|---|
| Bleached | During bleaching event (e.g. 2023 marine heatwave) | `bleached/` |
| Healthy | Earlier years with confirmed healthy reef (e.g. 2021–2022) | `healthy/` |

### In Planet Explorer

1. Draw a tight AOI bounding box around the reef (typically 0.5–2 km²). Keep it as small as
   possible to minimise monthly quota usage (~3,000 km²/month).
2. Set your date range and filter: **Imagery Type → PlanetScope Scene only**.
   Enable **Include only results with ground control**.
3. Review scene thumbnails. Reject scenes with >5% cloud cover, haze, or sun glint.
4. On the order page, select **Analytic Radiance (TOAR) — 8 Band**. This is the only product
   that includes all 8 bands: coastal blue, blue, green I, green II, yellow, red, red edge, NIR.
   Keep output format as **GeoTIFF**.
5. Name orders using the convention: `SITENAME_DATE_BLCH` or `SITENAME_DATE_HTHY`
   (e.g. `DRY_ROCKS_APR_2_2023_HTHY`).

Downloaded scenes arrive as:
```
<order_name>/PSScene/<scene_id>_AnalyticMS_8b_clip.tif
```

Organize them under:
```
planet_superdove/<site_name>/bleached/<YYYYMMDD>/
planet_superdove/<site_name>/healthy/<YYYYMMDD>/
```

---

## 2. Spatial Alignment (QGIS Georeferencer)

Some scenes are spatially offset relative to others at the same site. Before extracting tiles,
check alignment by loading two scenes from the same location in QGIS and toggling between them.
If the reef structure shifts, run the Georeferencer.

### Running the Georeferencer

1. Open **Raster → Georeferencer** in QGIS.
2. Load the misaligned scene as the input raster.
3. Use a well-aligned scene or reference layer as the canvas.
4. Place **8–10 GCP points** on identifiable reef landmarks spread across the full image
   (coral heads, rocky outcrops, sandy patches). Do not cluster points in one corner.
5. For each GCP: click **From Map Canvas** and click the corresponding feature in the reference layer.
   Verify CRS is the appropriate UTM zone (e.g. EPSG:32617 for Florida Keys).
6. Set transformation → **Linear**, resampling → **Nearest Neighbor**. Run.
7. QGIS saves the corrected output with `_modified` appended to the filename.
8. Toggle between the original and `_modified` scenes to verify the reef stays stationary.

> GCP point sets are scene-specific — you cannot reuse points across different dates.

---

## 3. Land Masking

Land masking sets land pixels to NaN so the classifier only sees ocean/reef pixels.

### When it is needed

All sites with visible land (islands, shoreline, mangroves) inside the tile footprints should be
masked. Open-ocean sites with no land in the AOI can skip this step.

### Creating a land mask shapefile (QGIS)

1. Load the scene into QGIS.
2. Create a new polygon layer (**Layer → Create Layer → New Shapefile Layer**), CRS matching the scene.
3. Digitize polygons around all land areas visible in the scene. Cover all dates — use the
   union of land across scenes to create a single conservative mask.
4. Save the shapefile as `<site>_landmask/<site>.shp` (plus `.dbf`, `.prj`, `.shx`).

### Applying the mask

Organize shapefiles under a `land_mask/` directory:
```
land_mask/
  cheeca_flkeys_landmask/cheeca_flkeys.shp
  chachacual_mexico_landmask/chachacual_mexico.shp
  ...
```

Then run from Python:
```python
from utils.masking import apply_landmasks

summary = apply_landmasks(
    planet_root    = "/path/to/planet_superdove",
    landmask_root  = "/path/to/land_mask",
    out_root       = "/path/to/planet_superdove_landmasked",
    invert         = True,          # True = erase land, keep reef
    copy_if_no_mask = True,         # copy tile unchanged if no shapefile found
)
```

This mirrors the full directory tree from `planet_superdove/` into `planet_superdove_landmasked/`
with land pixels replaced by NaN. Sites without a shapefile are copied unchanged.

The function handles site name mismatches between `planet_superdove/` and `land_mask/` automatically
via `SITE_ALIASES` in `utils/masking.py`. If your new site's folder names differ, add an entry there.

See `analysis/07_land_masking_demo.ipynb` to visually verify the mask before running the full pipeline.

---

## 4. Creating Tile Locations (QGIS)

Tile locations are defined by a point shapefile where each point becomes the center of one
120×120 px tile.

1. Load the masked scene into QGIS.
2. Create a new point layer (**Layer → Create Layer → New Shapefile Layer**), CRS matching the scene.
3. Place points on coral reef areas that are clearly healthy or bleached, avoiding:
   - Land and land-adjacent pixels
   - Cloud shadows or haze
   - Scene edges or black-border regions
4. Save as `point_shapes/<site_name>/<site_name>_shape.shp`.

Each point in the shapefile produces one `loc*.tif` tile per acquisition date.
The number of points equals the number of tiles per date for that site.

---

## 5. Extracting Tiles

`scripts/extract_tiles.py` reads each point from your shapefile, computes a 120×120 px window
centered on that point (using the scene's native pixel size), and extracts it from the scene GeoTIFF.
Band names (coastal_blue, blue, green_i, …, nir) are embedded in each tile's metadata.

### Expected output structure

```
planet_superdove_landmasked/
  <site_name>/
    bleached/
      tiled_360m/
        <YYYYMMDD>/
          loc001.tif
          loc002.tif
          ...
    healthy/
      tiled_360m/
        <YYYYMMDD>/
          loc001.tif
          ...
```

> Site names must use lowercase underscores and match exactly what is listed in `01_data_split.ipynb`.
> The `tiled_360m` subfolder name is hardcoded in the script and in the dataset loader — do not rename it.

### Running the script

```bash
# Always dry-run first — no files are written, but it shows exactly what would happen:
python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --dry-run

# Process everything in the downloads folder:
python scripts/extract_tiles.py --downloads-root /path/to/planet_orders

# Filter to one site or one label:
python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --site pulau_kapas
python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --label bleached
```

### Verifying tile integrity

```python
import rasterio
with rasterio.open(
    "planet_superdove_landmasked/chachacual_mexico/bleached/tiled_360m/20230711/loc001.tif"
) as src:
    print(src.count, "bands:", src.descriptions)
    print("Size:", src.width, "x", src.height)
# Expected:
# 8 bands: ('coastal_blue', 'blue', 'green_i', 'green_ii', 'yellow', 'red', 'rededge', 'nir')
# Size: 120 x 120
```

---

## 6. Quality Control

`scripts/viewtif.py` generates a scrollable HTML gallery of all tiles for a site so you can
visually inspect them before training.

> Requires `plotly`: `pip install plotly` (not in `environment.yml` — install separately in the coral env).

```bash
# Show all tiles for a site:
python scripts/viewtif.py --site chachacual_mexico

# Show only bleached tiles:
python scripts/viewtif.py --site chachacual_mexico --label bleached

# Specify a different data root:
python scripts/viewtif.py --site chachacual_mexico --data-root /path/to/planet_superdove_landmasked
```

Output is `<site>_gallery.html` in the current directory — open in any browser.

### What to look for

Remove any tile (delete the `.tif` file) that has:
- **Cloud cover or haze** — blurry or washed-out patches
- **Land contamination** — visible soil, vegetation, or building edges
- **Black borders** — tile window extended outside the scene edge (NaN fill strips)
- **Sun glint** — saturated bright patches masking reef structure
- **Misclassified label** — a "bleached" tile that visually shows healthy reef or vice versa

---

## 7. Data Split

Open `analysis/01_data_split.ipynb` and set the CONFIG cell:

```python
DATA_ROOT = "/path/to/planet_superdove_landmasked"
VAL_SITE  = "chachacual_mexico"   # site to hold out as validation
```

The notebook walks the directory tree, parses site/label/date from paths, and writes:
- `data/splits/train.csv` — all tiles from non-val sites
- `data/splits/val.csv` — all tiles from `VAL_SITE`

Each CSV has columns: `site, label, date, filename, filepath, image_id`.
These are read directly by `BleachDataset` during training and evaluation.

Run the split notebook once per leave-one-site-out experiment. The full 12-site evaluation
requires 12 separate runs (one per site). Each run overwrites `train.csv` and `val.csv`.

---

## 8. Training the Classifier

Edit `configs/exp_resnet18.yaml` to point `data_root` at your splits directory, then run:

```bash
python ct_classifier/train.py --config configs/exp_resnet18.yaml
```

Key config fields:

| Field | Description |
|---|---|
| `data_root` | Directory containing `train.csv` and `val.csv` |
| `layers` | ResNet variant: `18` or `50` |
| `num_classes` | Always `2` (healthy / bleached) |
| `normalization_factor` | Scalar divisor for raw TOAR reflectance (typically `10000`) |
| `image_size` | Tile crop size — default `[120, 120]` |
| `batch_size` | 32 works well on a single A100 |
| `num_workers` | Match available CPU cores |

Checkpoints are saved to `model_states/` (gitignored). The best checkpoint by validation accuracy
is saved as `best.pt`. A copy of the config is saved alongside for reproducibility.

---

## 9. Analysis and Evaluation

Run the notebooks in order from `analysis/`. Each requires only the CONFIG cell at the top
to be updated with your checkpoint and data paths.

```
02_evaluation.ipynb     → loss, accuracy, ROC curve, per-site bar chart
03_gradcam.ipynb        → Grad-CAM attention maps on TP/FP/FN/TN examples
04_spatial_prob_map.ipynb → sub-tile sliding-window bleaching probability heatmap
05_band_importance.ipynb  → per-channel importance (grad×input + occlusion)
06_failure_analysis.ipynb → FP/FN pair viewer, score distributions, leakage checks
```

---

## Appendix: Directory Layout Reference

```
ct_classifier/               ← git repository root
  ct_classifier/             ← Python source module
    train.py  dataset.py  model.py  util.py
  configs/
    exp_resnet18.yaml
  scripts/
    extract_tiles.py         ← tile extraction
    viewtif.py               ← visual QA gallery
  utils/
    masking.py  viz.py  evaluate.py  gradcam.py
  analysis/
    01_data_split.ipynb  …  07_land_masking_demo.ipynb
  docs/
    data_processing.md       ← this file
  environment.yml

planet_superdove/            ← NOT in git (raw scenes, pre-masking)
  <site_name>/
    bleached/<YYYYMMDD>/PSScene/*_AnalyticMS_8b_clip*.tif
    healthy/<YYYYMMDD>/PSScene/*_AnalyticMS_8b_clip*.tif

planet_superdove_landmasked/ ← NOT in git (masked tiles, used for training)
  <site_name>/
    bleached/tiled_360m/<YYYYMMDD>/loc001.tif  …
    healthy/tiled_360m/<YYYYMMDD>/loc001.tif   …

data/splits/                 ← NOT in git (generated by 01_data_split.ipynb)
  train.csv  val.csv

model_states/                ← NOT in git (generated by train.py)
  best.pt  config.yaml

point_shapes/                ← NOT in git (tile center shapefiles)
  <site_name>/<site_name>_shape.shp  .shx  .dbf  .prj

land_mask/                   ← NOT in git (land mask shapefiles)
  <site_name>_landmask/<site_name>.shp  .shx  .dbf  .prj
```

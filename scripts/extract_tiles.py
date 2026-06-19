"""
extract_tiles.py
================
Extracts 120x120 px tiles from downloaded PlanetScope scenes and organizes
them into the expected dataset folder structure.

Usage
-----
    python extract_tiles.py --site chachacual_mexico --label bleached
    python extract_tiles.py --site chachacual_mexico --label healthy
    python extract_tiles.py --site chachacual_mexico --label bleached --dry-run

What it does
------------
For each date folder found under:
    planet_superdove_landmasked/<site>/<label>/

It reads your point shapefile from:
    point_shapes/<site>/<site>_shape.shp

Extracts one 120x120 px tile per point, and writes the final tiles to:
    planet_superdove_landmasked/<site>/<label>/tiled_360m/<YYYYMMDD>/loc001.tif
    planet_superdove_landmasked/<site>/<label>/tiled_360m/<YYYYMMDD>/loc002.tif
    ...

The scene GeoTIFF is found automatically by searching for:
    *AnalyticMS_8b*_clip_modified.tif  (GCP-corrected)
    *AnalyticMS_8b*_clip.tif           (original)
    *_composite.tif                    (composited 2-strip scenes)

Requirements
------------
    pip install geopandas rasterio shapely
"""

import os
import re
import glob
import argparse
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Paths are relative to the repo root. Edit only if your layout differs.

DATA_ROOT   = "planet_superdove_landmasked"   # root of masked tiles
SHAPES_ROOT = "point_shapes"                  # root of point shapefiles
TILE_SUBDIR = "tiled_360m"                    # hardcoded subfolder name (matches dataset loader)
TILE_SIZE   = 120                             # pixels — do not change

BAND_NAMES = [
    "coastal_blue", "blue", "green_i", "green_ii",
    "yellow", "red", "rededge", "nir",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def find_scene_tif(date_folder: str) -> str | None:
    """Find the scene GeoTIFF inside a date folder, trying common naming patterns."""
    patterns = [
        os.path.join(date_folder, "PSScene", "*AnalyticMS_8b*_clip_modified.tif"),
        os.path.join(date_folder, "PSScene", "*AnalyticMS_8b*_clip.tif"),
        os.path.join(date_folder, "1", "*_composite.tif"),
        os.path.join(date_folder, "**", "*AnalyticMS_8b*_clip_modified.tif"),
        os.path.join(date_folder, "**", "*AnalyticMS_8b*_clip.tif"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


def extract_date(folder_name: str) -> str | None:
    """Extract YYYYMMDD from a folder name. Handles YYYY_MM_DD and YYYYMMDD."""
    # YYYY_MM_DD or YYYY-MM-DD
    m = re.search(r"(19|20)\d{2}[_-](0[1-9]|1[0-2])[_-](0[1-9]|[12]\d|3[01])", folder_name)
    if m:
        return m.group(0).replace("_", "").replace("-", "")
    # YYYYMMDD already
    m = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", folder_name)
    if m:
        return m.group(0)
    return None


def make_square(x: float, y: float, size_x: float, size_y: float):
    """Create a square bounding box centered on (x, y)."""
    return box(x - size_x / 2, y - size_y / 2, x + size_x / 2, y + size_y / 2)


def extract_tiles_from_scene(scene_tif: str, points_gdf, out_dir: str, dry_run: bool) -> tuple[int, int]:
    """
    Extract one tile per point from a scene GeoTIFF.

    Returns (tiles_written, tiles_skipped).
    """
    written, skipped = 0, 0

    with rasterio.open(scene_tif) as src:
        if src.crs is None:
            print(f"  [SKIP] Raster has no CRS: {scene_tif}")
            return 0, len(points_gdf)

        # Reproject points to match raster CRS if needed
        gdf = points_gdf.to_crs(src.crs) if points_gdf.crs != src.crs else points_gdf

        # Tile footprint in CRS units (pixel_size * 120)
        px, py = src.res
        size_x = TILE_SIZE * px
        size_y = TILE_SIZE * abs(py)

        for idx, row in enumerate(gdf.itertuples(), start=1):
            try:
                geom = row.geometry
                center = geom if geom.geom_type == "Point" else geom.centroid
                square = make_square(center.x, center.y, size_x, size_y)

                window = from_bounds(*square.bounds, transform=src.transform)
                window = window.round_offsets().round_lengths()

                # Skip tiles that fall outside the raster extent
                if (
                    window.col_off < 0 or window.row_off < 0
                    or window.col_off + window.width > src.width
                    or window.row_off + window.height > src.height
                ):
                    print(f"  [SKIP] loc{idx:03} — window out of raster bounds")
                    skipped += 1
                    continue

                out_path = os.path.join(out_dir, f"loc{idx:03}.tif")

                if dry_run:
                    print(f"  [DRY] would write → {out_path}")
                    written += 1
                    continue

                os.makedirs(out_dir, exist_ok=True)
                if os.path.exists(out_path):
                    os.remove(out_path)

                out_image = src.read(window=window)
                out_transform = rasterio.windows.transform(window, src.transform)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width":  out_image.shape[2],
                    "transform": out_transform,
                })

                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)
                    for band_idx in range(1, src.count + 1):
                        name = BAND_NAMES[band_idx - 1] if band_idx <= len(BAND_NAMES) else f"band_{band_idx}"
                        dst.set_band_description(band_idx, name)

                written += 1

            except Exception as e:
                print(f"  [ERR] loc{idx:03}: {e}")
                skipped += 1

    return written, skipped


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 120x120 px tiles from PlanetScope scenes."
    )
    parser.add_argument(
        "--site", required=True,
        help="Site folder name, e.g. chachacual_mexico"
    )
    parser.add_argument(
        "--label", required=True, choices=["bleached", "healthy"],
        help="Label to process: bleached or healthy"
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help=f"Root of masked tile data (default: {DATA_ROOT})"
    )
    parser.add_argument(
        "--shapes-root", default=SHAPES_ROOT,
        help=f"Root of point shapefiles (default: {SHAPES_ROOT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without creating any files"
    )
    args = parser.parse_args()

    site   = args.site
    label  = args.label

    # ── Locate shapefile ──────────────────────────────────────────────────────
    shp_dir = os.path.join(args.shapes_root, site)
    shp_matches = glob.glob(os.path.join(shp_dir, "*.shp"))

    if not shp_matches:
        print(f"[ERROR] No shapefile found in: {shp_dir}")
        print(f"        Expected: {shp_dir}/<site>_shape.shp")
        raise SystemExit(1)

    shp_path = sorted(shp_matches)[0]
    print(f"Shapefile : {shp_path}")
    points_gdf = gpd.read_file(shp_path)
    print(f"Points    : {len(points_gdf)} tile centers")

    # ── Find date folders ─────────────────────────────────────────────────────
    label_dir = os.path.join(args.data_root, site, label)
    if not os.path.isdir(label_dir):
        print(f"[ERROR] Label folder not found: {label_dir}")
        raise SystemExit(1)

    date_folders = sorted([
        d for d in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, d))
        and d != TILE_SUBDIR  # don't recurse into output we've already written
    ])

    if not date_folders:
        print(f"[ERROR] No date folders found under: {label_dir}")
        raise SystemExit(1)

    print(f"Site      : {site}")
    print(f"Label     : {label}")
    print(f"Dates     : {len(date_folders)} folders")
    if args.dry_run:
        print("Mode      : DRY RUN (no files written)")
    print()

    total_written = 0
    total_skipped = 0

    for folder_name in date_folders:
        folder_path = os.path.join(label_dir, folder_name)
        date = extract_date(folder_name)

        if date is None:
            print(f"[SKIP] Cannot parse date from folder name: {folder_name}")
            continue

        scene_tif = find_scene_tif(folder_path)
        if scene_tif is None:
            print(f"[SKIP] No scene GeoTIFF found in: {folder_path}")
            continue

        out_dir = os.path.join(label_dir, TILE_SUBDIR, date)

        print(f"  {date}  ←  {os.path.relpath(scene_tif, label_dir)}")
        print(f"         →  {os.path.relpath(out_dir, label_dir)}/")

        w, s = extract_tiles_from_scene(scene_tif, points_gdf, out_dir, args.dry_run)
        total_written += w
        total_skipped += s
        print(f"         wrote {w}  skipped {s}")

    print()
    print("=" * 50)
    print(f"Total tiles written : {total_written}")
    print(f"Total tiles skipped : {total_skipped}")
    print(f"Output root         : {os.path.join(label_dir, TILE_SUBDIR)}/")
    if args.dry_run:
        print("(dry run — no files were actually written)")


if __name__ == "__main__":
    main()

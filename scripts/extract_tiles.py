"""
extract_tiles.py
================
Extract 120x120 px tiles from raw PlanetScope order downloads and write them
into the dataset folder structure expected by BleachDataset.

Planet order folders must be named with label, site, and date in the name:

    bleached_Pulau_Kapas_06242024_psscene_analytic_8b_udm2/
    healthy_Pulau_Kapas_03152022_psscene_analytic_8b_udm2/
    bleached_chachacual_mexico_07112023_psscene_analytic_8b_udm2/
    ...

The script parses each folder, finds the AnalyticMS_8b GeoTIFF inside PSScene/,
clips 120x120 px tiles using your point shapefile, and writes them to:

    planet_superdove_landmasked/<site>/<label>/tiled_360m/<YYYYMMDD>/loc001.tif
    planet_superdove_landmasked/<site>/<label>/tiled_360m/<YYYYMMDD>/loc002.tif
    ...

Usage
-----
    # Dry run (no files written) — always do this first:
    python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --dry-run

    # Process all orders:
    python scripts/extract_tiles.py --downloads-root /path/to/planet_orders

    # Filter to one site or label:
    python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --site pulau_kapas
    python scripts/extract_tiles.py --downloads-root /path/to/planet_orders --label bleached

Naming convention for Planet orders
-------------------------------------
When placing an order in Planet Explorer, name it:

    <label>_<SiteName>_<MMDDYYYY>_...

Examples:
    bleached_Pulau_Kapas_06242024_psscene_analytic_8b_udm2
    healthy_chachacual_mexico_03152022_psscene_analytic_8b_udm2

The label must be the very first word (bleached or healthy).
The date must be 8 consecutive digits in MMDDYYYY format.
Everything between label and date becomes the site name (lowercased, underscores).

Requirements
------------
    conda activate coral   (rasterio, geopandas, shapely are in environment.yml)
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

OUT_ROOT    = "planet_superdove_landmasked"   # where tiled output is written
SHAPES_ROOT = "point_shapes"                  # root of per-site point shapefiles
TILE_SUBDIR = "tiled_360m"                    # must match dataset loader — do not rename
TILE_SIZE   = 120                             # pixels — do not change

BAND_NAMES = [
    "coastal_blue", "blue", "green_i", "green_ii",
    "yellow", "red", "rededge", "nir",
]

# ── PARSING ───────────────────────────────────────────────────────────────────

def parse_order_folder(folder_name: str) -> tuple[str, str, str] | None:
    """
    Extract (label, site_name, date_yyyymmdd) from a Planet order folder name.

    Expected format:
        <label>_<SiteName>_<MMDDYYYY>_psscene_...

    Returns None if the folder name does not match the expected pattern.

    Examples:
        bleached_Pulau_Kapas_06242024_psscene_analytic_8b_udm2
            → ('bleached', 'pulau_kapas', '20240624')
        healthy_chachacual_mexico_07112023_psscene_analytic_8b_udm2
            → ('healthy', 'chachacual_mexico', '20230711')
    """
    lower = folder_name.lower()

    # Must start with bleached_ or healthy_
    if lower.startswith("bleached_"):
        label = "bleached"
        rest = folder_name[len("bleached_"):]
    elif lower.startswith("healthy_"):
        label = "healthy"
        rest = folder_name[len("healthy_"):]
    else:
        return None

    # Must contain _psscene_ — strip that suffix
    psscene_idx = rest.lower().find("_psscene_")
    if psscene_idx == -1:
        return None
    site_and_date = rest[:psscene_idx]   # e.g. "Pulau_Kapas_06242024"

    # The date is the last token that is exactly 8 digits (MMDDYYYY)
    parts = site_and_date.split("_")
    date_token_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if re.match(r"^\d{8}$", parts[i]):
            date_token_idx = i
            break

    if date_token_idx is None:
        return None

    date_mmddyyyy = parts[date_token_idx]
    site_name = "_".join(parts[:date_token_idx]).lower()

    if not site_name:
        return None

    # Convert MMDDYYYY → YYYYMMDD
    mm, dd, yyyy = date_mmddyyyy[:2], date_mmddyyyy[2:4], date_mmddyyyy[4:]
    date_yyyymmdd = f"{yyyy}{mm}{dd}"

    return label, site_name, date_yyyymmdd


def find_scene_tif(order_folder: str) -> str | None:
    """
    Find the 8-band AnalyticMS GeoTIFF inside a Planet order folder.

    Tries common naming patterns in order of preference:
      1. PSScene/*AnalyticMS_8b*_clip_modified.tif  (GCP-corrected)
      2. PSScene/*AnalyticMS_8b*_clip.tif           (original clip)
      3. Recursive search for either pattern
    """
    patterns = [
        os.path.join(order_folder, "PSScene", "*AnalyticMS_8b*_clip_modified.tif"),
        os.path.join(order_folder, "PSScene", "*AnalyticMS_8b*_clip.tif"),
        os.path.join(order_folder, "**", "*AnalyticMS_8b*_clip_modified.tif"),
        os.path.join(order_folder, "**", "*AnalyticMS_8b*_clip.tif"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            return sorted(matches)[0]
    return None


def load_shapefile(site_name: str, shapes_root: str):
    """Load the point shapefile for a site. Returns GeoDataFrame or None."""
    shp_dir = os.path.join(shapes_root, site_name)
    matches = glob.glob(os.path.join(shp_dir, "*.shp"))
    if not matches:
        return None, None
    shp_path = sorted(matches)[0]
    return gpd.read_file(shp_path), shp_path

# ── TILE EXTRACTION ───────────────────────────────────────────────────────────

def make_square(x: float, y: float, size_x: float, size_y: float):
    """Bounding box centered on (x, y) with given width/height in CRS units."""
    return box(x - size_x / 2, y - size_y / 2, x + size_x / 2, y + size_y / 2)


def extract_tiles_from_scene(
    scene_tif: str,
    points_gdf,
    out_dir: str,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Extract one 120x120 tile per point from scene_tif and write to out_dir.

    Returns (tiles_written, tiles_skipped).
    """
    written, skipped = 0, 0

    with rasterio.open(scene_tif) as src:
        if src.crs is None:
            print(f"    [SKIP] Raster has no CRS: {scene_tif}")
            return 0, len(points_gdf)

        gdf = points_gdf.to_crs(src.crs) if points_gdf.crs != src.crs else points_gdf

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

                if (
                    window.col_off < 0 or window.row_off < 0
                    or window.col_off + window.width > src.width
                    or window.row_off + window.height > src.height
                ):
                    print(f"    [SKIP] loc{idx:03} — window out of raster bounds")
                    skipped += 1
                    continue

                out_path = os.path.join(out_dir, f"loc{idx:03}.tif")

                if dry_run:
                    print(f"    [DRY]  → {out_path}")
                    written += 1
                    continue

                os.makedirs(out_dir, exist_ok=True)
                if os.path.exists(out_path):
                    os.remove(out_path)

                out_image = src.read(window=window)
                out_transform = rasterio.windows.transform(window, src.transform)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height":    out_image.shape[1],
                    "width":     out_image.shape[2],
                    "transform": out_transform,
                })

                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)
                    for band_idx in range(1, src.count + 1):
                        name = (
                            BAND_NAMES[band_idx - 1]
                            if band_idx <= len(BAND_NAMES)
                            else f"band_{band_idx}"
                        )
                        dst.set_band_description(band_idx, name)

                written += 1

            except Exception as e:
                print(f"    [ERR]  loc{idx:03}: {e}")
                skipped += 1

    return written, skipped

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract 120x120 px tiles from raw PlanetScope order downloads.\n"
            "Order folders must be named: <label>_<SiteName>_<MMDDYYYY>_psscene_..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--downloads-root", required=True,
        help="Directory containing Planet order folders",
    )
    parser.add_argument(
        "--out-root", default=OUT_ROOT,
        help=f"Root for tiled output (default: {OUT_ROOT})",
    )
    parser.add_argument(
        "--shapes-root", default=SHAPES_ROOT,
        help=f"Root of point shapefiles (default: {SHAPES_ROOT})",
    )
    parser.add_argument(
        "--site", default=None,
        help="Process only this site (e.g. pulau_kapas). Default: all sites.",
    )
    parser.add_argument(
        "--label", choices=["bleached", "healthy"], default=None,
        help="Process only this label. Default: both.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without creating any files",
    )
    args = parser.parse_args()

    # ── Discover order folders ────────────────────────────────────────────────
    all_entries = sorted([
        e for e in os.listdir(args.downloads_root)
        if os.path.isdir(os.path.join(args.downloads_root, e))
    ])

    orders = []
    skipped_parse = []
    for entry in all_entries:
        parsed = parse_order_folder(entry)
        if parsed is None:
            skipped_parse.append(entry)
            continue
        label, site_name, date = parsed
        if args.site  and site_name != args.site.lower():
            continue
        if args.label and label    != args.label:
            continue
        orders.append({
            "folder":    os.path.join(args.downloads_root, entry),
            "folder_name": entry,
            "label":     label,
            "site":      site_name,
            "date":      date,
        })

    if skipped_parse:
        print(f"Skipped {len(skipped_parse)} folder(s) with unrecognised names:")
        for name in skipped_parse:
            print(f"  {name}")
        print()

    if not orders:
        print("[ERROR] No matching order folders found.")
        print("        Check that folder names follow: <label>_<SiteName>_<MMDDYYYY>_psscene_...")
        raise SystemExit(1)

    print(f"Found {len(orders)} order folder(s) to process.")
    if args.dry_run:
        print("Mode: DRY RUN — no files will be written.")
    print()

    # Cache shapefiles per site so we load each one only once
    shp_cache: dict[str, object] = {}

    total_written = 0
    total_skipped = 0

    for order in orders:
        label     = order["label"]
        site      = order["site"]
        date      = order["date"]
        folder    = order["folder"]

        print(f"[{label}] {site} / {date}")
        print(f"  order : {order['folder_name']}")

        # ── Find scene GeoTIFF ────────────────────────────────────────────────
        scene_tif = find_scene_tif(folder)
        if scene_tif is None:
            print(f"  [SKIP] No AnalyticMS_8b GeoTIFF found inside order folder.")
            print(f"         Looked in: {os.path.join(folder, 'PSScene', '*AnalyticMS_8b*_clip*.tif')}")
            print()
            continue
        print(f"  scene : {os.path.relpath(scene_tif, folder)}")

        # ── Load point shapefile ──────────────────────────────────────────────
        if site not in shp_cache:
            gdf, shp_path = load_shapefile(site, args.shapes_root)
            if gdf is None:
                print(f"  [SKIP] No shapefile found in: {os.path.join(args.shapes_root, site)}/")
                print(f"         Expected: {args.shapes_root}/{site}/<name>.shp")
                print()
                continue
            shp_cache[site] = gdf
            print(f"  shp   : {shp_path}  ({len(gdf)} points)")
        else:
            gdf = shp_cache[site]

        # ── Extract tiles ─────────────────────────────────────────────────────
        out_dir = os.path.join(args.out_root, site, label, TILE_SUBDIR, date)
        print(f"  out   : {out_dir}/")

        w, s = extract_tiles_from_scene(scene_tif, gdf, out_dir, args.dry_run)
        total_written += w
        total_skipped += s
        print(f"  tiles : wrote {w}  skipped {s}")
        print()

    print("=" * 55)
    print(f"Total tiles written : {total_written}")
    print(f"Total tiles skipped : {total_skipped}")
    print(f"Output root         : {args.out_root}/")
    if args.dry_run:
        print("(dry run — no files were actually written)")


if __name__ == "__main__":
    main()

import os
import glob
import shutil
import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd


# Map planet site folder name → substring to search for in the landmask folder.
# All comparisons are case-insensitive. Needed where the site naming conventions
# between planet_superdove/ and land_mask/ differ.
SITE_ALIASES = {
    "santacruz_mexico":   "santacruz",
    "northpoint_lizard":  "lizard_island_australia",
    "pulau_kapas":        "malaysia",
}


def find_landmask_shp(site_name, landmask_root):
    """
    Locate the shapefile for a site's land mask.

    Search order:
      1. Exact folder match:  <landmask_root>/<site_name>_landmask/*.shp
      2. Alias key match:     SITE_ALIASES[site_name] as substring in folder name
      3. Full site name as substring in any landmask folder
      4. Base token (before first underscore) as substring fallback

    Args:
        site_name: str, e.g. "northpoint_lizard"
        landmask_root: str, path to the directory containing *_landmask/ folders

    Returns path to the first matching .shp file, or None if not found.
    """
    def _first_shp(folder):
        hits = glob.glob(os.path.join(folder, "*.shp"))
        return hits[0] if hits else None

    # 1) exact
    exact = os.path.join(landmask_root, f"{site_name}_landmask")
    if os.path.isdir(exact):
        shp = _first_shp(exact)
        if shp:
            return shp

    # gather all *_landmask dirs
    landmask_dirs = [
        d for d in glob.glob(os.path.join(landmask_root, "*"))
        if os.path.isdir(d) and "landmask" in os.path.basename(d).lower()
    ]
    if not landmask_dirs:
        return None

    site_lower = site_name.lower()

    # 2) alias
    alias_key = SITE_ALIASES.get(site_name)
    if alias_key:
        for d in landmask_dirs:
            if alias_key.lower() in os.path.basename(d).lower():
                shp = _first_shp(d)
                if shp:
                    return shp

    # 3) substring on full site name
    for d in landmask_dirs:
        if site_lower in os.path.basename(d).lower():
            shp = _first_shp(d)
            if shp:
                return shp

    # 4) base token
    base = site_lower.split("_")[0]
    for d in landmask_dirs:
        if base in os.path.basename(d).lower():
            shp = _first_shp(d)
            if shp:
                return shp

    return None


def mask_one_tif(tif_path, shp, out_path, invert=True):
    """
    Apply a GeoDataFrame shapefile mask to a single GeoTIFF tile.

    Args:
        tif_path: str, input GeoTIFF path
        shp: geopandas.GeoDataFrame, the land mask shapefile
        out_path: str, output path (parent dirs created if needed)
        invert: bool (default True) — True removes land inside the polygons
                (sets land pixels to NaN), False keeps only interior pixels.

    Raises ValueError if the raster has no CRS or the shapefile is empty.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        shp_proj = shp.to_crs(src.crs)
        geoms = [g for g in shp_proj.geometry if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError(f"Shapefile has no valid geometries for: {tif_path}")

        out_img, out_transform = rasterio.mask.mask(
            src, geoms, crop=False, invert=invert, filled=False
        )
        out_img = out_img.astype(np.float32).filled(np.nan)

        out_meta = src.meta.copy()
        out_meta.update({
            "dtype":     "float32",
            "height":    out_img.shape[1],
            "width":     out_img.shape[2],
            "transform": out_transform,
        })
        out_meta.pop("nodata", None)

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_img)


def copy_one_tif(src_path, out_path):
    """
    Copy a tile unchanged when no land mask is available for its site.

    Args:
        src_path: str, source GeoTIFF path
        out_path: str, destination path (parent dirs created if needed)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(src_path, out_path)


def apply_landmasks(planet_root, landmask_root, out_root,
                    invert=True, copy_if_no_mask=True):
    """
    Walk all site directories under planet_root and apply land masks.

    Mirrors the directory structure from planet_root into out_root.
    Sites with no matching shapefile are either copied unchanged
    (copy_if_no_mask=True) or skipped.

    Args:
        planet_root:    str, e.g. ".../planet_superdove"
        landmask_root:  str, e.g. ".../land_mask"
        out_root:       str, e.g. ".../planet_superdove_landmasked"
        invert:         bool, passed to mask_one_tif (default True = remove land)
        copy_if_no_mask: bool, if True tiles without a shapefile are copied as-is

    Returns dict mapping site_name → {"masked": int, "copied": int, "errors": int}
    """
    site_dirs = sorted([
        d for d in glob.glob(os.path.join(planet_root, "*")) if os.path.isdir(d)
    ])
    print(f"Found {len(site_dirs)} site folders under {planet_root}")

    summary = {}

    for site_dir in site_dirs:
        site_name = os.path.basename(site_dir)
        tif_paths = glob.glob(
            os.path.join(site_dir, "*", "tiled_360m", "*", "loc*.tif")
        )
        print(f"\n[{site_name}] tiles={len(tif_paths)}")

        shp_path = find_landmask_shp(site_name, landmask_root)
        counts = {"masked": 0, "copied": 0, "errors": 0}

        if shp_path is None:
            print(f"  [NO MASK] {'copying' if copy_if_no_mask else 'skipping'}")
            if not copy_if_no_mask:
                summary[site_name] = counts
                continue
            for tif in tif_paths:
                rel = os.path.relpath(tif, planet_root)
                try:
                    copy_one_tif(tif, os.path.join(out_root, rel))
                    counts["copied"] += 1
                except Exception as e:
                    print(f"  [ERR COPY] {os.path.basename(tif)}: {e}")
                    counts["errors"] += 1
            summary[site_name] = counts
            continue

        try:
            shp = gpd.read_file(shp_path)
        except Exception as e:
            print(f"  [ERR] Cannot read shapefile {shp_path}: {e} — copying instead")
            for tif in tif_paths:
                rel = os.path.relpath(tif, planet_root)
                try:
                    copy_one_tif(tif, os.path.join(out_root, rel))
                    counts["copied"] += 1
                except Exception as ee:
                    print(f"  [ERR COPY] {os.path.basename(tif)}: {ee}")
                    counts["errors"] += 1
            summary[site_name] = counts
            continue

        print(f"  [MASK] {os.path.basename(os.path.dirname(shp_path))} invert={invert}")
        for tif in tif_paths:
            rel = os.path.relpath(tif, planet_root)
            try:
                mask_one_tif(tif, shp, os.path.join(out_root, rel), invert=invert)
                counts["masked"] += 1
            except Exception as e:
                print(f"  [ERR MASK] {os.path.basename(tif)}: {e}")
                counts["errors"] += 1

        summary[site_name] = counts

    return summary

"""
viewtif.py
==========
Generate a scrollable HTML gallery of all tiles for a site,
displayed as natural-color RGB (bands 6, 4, 2 = red, green II, blue).

Use this for a quick visual QA pass before running the data split or training.
Remove any tile that has cloud cover, land contamination, or black borders.

Usage
-----
    # From the repo root:
    python viewtif.py --site chachacual_mexico

    # Specify a different data root (if not using the default):
    python viewtif.py --site chachacual_mexico --data-root /path/to/planet_superdove_landmasked

    # Show only one label:
    python viewtif.py --site chachacual_mexico --label bleached

Output
------
    <site>_gallery.html   (in the current directory — open in any browser)
"""

import os
import glob
import argparse
import numpy as np
import rasterio
import plotly.graph_objects as go
from plotly.offline import plot

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "planet_superdove_landmasked"
RGB_BANDS  = (6, 4, 2)   # 1-based: red, green_ii, blue → natural color
TILE_SUBDIR = "tiled_360m"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_rgb(tif_path: str, bands: tuple = RGB_BANDS) -> np.ndarray | None:
    """Load three bands from a GeoTIFF and return a uint8 [H, W, 3] array."""
    try:
        with rasterio.open(tif_path) as src:
            if src.count < max(bands):
                # Fall back to first three bands if 8-band not available
                band_idx = [min(b, src.count) for b in bands]
            else:
                band_idx = list(bands)
            img = src.read(band_idx).astype(np.float32)  # [3, H, W]

        img = np.transpose(img, (1, 2, 0))  # [H, W, 3]

        # Per-channel percentile stretch (2–98) for display
        for c in range(3):
            band = img[:, :, c]
            lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
            img[:, :, c] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

        return (img * 255).astype(np.uint8)

    except Exception as e:
        print(f"  [ERR] {tif_path}: {e}")
        return None


def collect_tiles(data_root: str, site: str, label: str | None) -> list[dict]:
    """Walk the tiled_360m structure and collect all tile paths + metadata."""
    tiles = []
    labels = [label] if label else ["bleached", "healthy"]

    for lbl in labels:
        pattern = os.path.join(data_root, site, lbl, TILE_SUBDIR, "*", "loc*.tif")
        for tif_path in sorted(glob.glob(pattern)):
            parts = tif_path.replace("\\", "/").split("/")
            # Extract date and filename from path structure
            # ...<site>/<label>/tiled_360m/<date>/<filename>
            try:
                date     = parts[-2]
                filename = parts[-1]
            except IndexError:
                date = "unknown"
                filename = os.path.basename(tif_path)

            tiles.append({
                "path":     tif_path,
                "label":    lbl,
                "date":     date,
                "filename": filename,
                "title":    f"{lbl} / {date} / {filename}",
            })

    return tiles


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML gallery of tiles for visual QA."
    )
    parser.add_argument(
        "--site", required=True,
        help="Site folder name, e.g. chachacual_mexico"
    )
    parser.add_argument(
        "--label", choices=["bleached", "healthy"], default=None,
        help="Limit to one label (default: show both)"
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help=f"Root of masked tile data (default: {DATA_ROOT})"
    )
    parser.add_argument(
        "--out", default=None,
        help="Output HTML filename (default: <site>_gallery.html)"
    )
    args = parser.parse_args()

    site      = args.site
    out_html  = args.out or f"{site}_gallery.html"

    site_dir = os.path.join(args.data_root, site)
    if not os.path.isdir(site_dir):
        print(f"[ERROR] Site folder not found: {site_dir}")
        raise SystemExit(1)

    tiles = collect_tiles(args.data_root, site, args.label)

    if not tiles:
        print(f"[ERROR] No tiles found for site '{site}' under {site_dir}")
        print(f"        Expected: {site_dir}/<label>/tiled_360m/<date>/loc*.tif")
        raise SystemExit(1)

    print(f"Site  : {site}")
    print(f"Tiles : {len(tiles)}")
    print(f"Output: {out_html}")
    print()

    figs = []
    for tile in tiles:
        img_arr = load_rgb(tile["path"])
        if img_arr is None:
            continue

        fig = go.Figure()
        fig.add_trace(go.Image(z=img_arr))
        fig.update_layout(
            title=dict(text=tile["title"], font=dict(size=13)),
            margin=dict(l=0, r=0, t=40, b=0),
            width=260,
            height=300,
        )
        figs.append(fig)

    if not figs:
        print("[ERROR] No tiles could be loaded.")
        raise SystemExit(1)

    # ── Build HTML ────────────────────────────────────────────────────────────
    html_blocks = []
    for i, fig in enumerate(figs):
        div = plot(
            fig,
            include_plotlyjs=(i == 0),   # embed JS only once
            output_type="div",
            show_link=False,
            auto_open=False,
        )
        html_blocks.append(div)

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>{site} — tile gallery</title>
  <style>
    body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 16px; }}
    h1   {{ font-size: 18px; margin-bottom: 4px; color: #9ecfcc; }}
    p    {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 12px;
    }}
    .tile {{ background: #0d1b2a; border-radius: 6px; padding: 4px; }}
  </style>
</head>
<body>
  <h1>{site} — tile gallery</h1>
  <p>{len(figs)} tiles &nbsp;|&nbsp; bands {RGB_BANDS} (R, G_ii, B) &nbsp;|&nbsp;
     2–98 percentile stretch &nbsp;|&nbsp;
     Delete tiles with clouds, land contamination, or black borders before training.</p>
  <div class="grid">
""")
        for block in html_blocks:
            f.write(f'    <div class="tile">{block}</div>\n')
        f.write("  </div>\n</body>\n</html>")

    print(f"Gallery written → {out_html}")
    print(f"Open in a browser to inspect tiles.")


if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
import pandas as pd


# The 12 reef sites used in the paper. lbcaye_bbr is excluded (too few tiles).
SITES = [
    "cheeca_flkeys",
    "looe_flkey",
    "easterndry_flkeys",
    "sand_flkey",
    "rock_flkey",
    "sanagustin_mexico",
    "chachacual_mexico",
    "jicaral_mexico",
    "riscalillo_mexico",
    "santacruz_mexico",
    "northpoint_lizard",
    "pulau_kapas",
]

COLUMNS = ["site", "label", "date", "filename", "filepath", "image_id"]
SORT_COLS = ["site", "label", "date", "filename"]


def build_inventory(data_root, sites=SITES):
    """
    Walk <data_root>/<site>/<label>/tiled_360m/<YYYYMMDD>/locXXX.tif and return
    a DataFrame with one row per tile.

    image_id is assigned after sorting, so it is stable across runs for the
    same set of files.
    """
    rows = []
    for site in sites:
        pattern = os.path.join(data_root, site, "**", "*.tif")
        for fp in glob.glob(pattern, recursive=True):
            if ".ipynb_checkpoints" in fp:
                continue
            parts = os.path.normpath(fp).split(os.sep)
            try:
                site_idx = parts.index(site)
                label = parts[site_idx + 1]   # healthy | bleached
                date = parts[site_idx + 3]    # YYYYMMDD
            except (ValueError, IndexError):
                continue
            if label not in ("healthy", "bleached"):
                continue
            rows.append({
                "site":     site,
                "label":    label,
                "date":     date,
                "filename": os.path.basename(fp),
                "filepath": fp,
            })

    assert rows, f"No .tif files found under {data_root}. Check data_root."

    df = pd.DataFrame(rows).sort_values(SORT_COLS).reset_index(drop=True)
    df["image_id"] = range(len(df))
    return df[COLUMNS]


def make_loso_split(df, test_site, n_val_healthy=2, n_val_bleached=1, seed=0):
    """
    Leave-one-site-out split with an inner validation set.

      test  : every tile from `test_site` (never used for training or model selection)
      val   : from each remaining site, `n_val_healthy` random healthy dates and
              `n_val_bleached` random bleached dates (all locations on those dates)
      train : everything else

    The best epoch is selected on `val`; `test` is only scored once at the end.

    With the defaults (2 healthy + 1 bleached date), every val location yields
    2 healthy-query and 2 bleached-query pairs, so val pairs are class-balanced.
    At least 2 healthy dates are needed for val to contain any healthy pairs,
    since BleachDataset pairs each healthy reference with other images from the
    same location within the same split.
    """
    sites = sorted(df["site"].unique())
    assert test_site in sites, f"test_site '{test_site}' not found. Available: {sites}"

    rng = np.random.default_rng(seed)

    df_test = df[df["site"] == test_site]
    val_mask = pd.Series(False, index=df.index)

    for site in sites:
        if site == test_site:
            continue
        for label, n_val in (("healthy", n_val_healthy), ("bleached", n_val_bleached)):
            dates = np.sort(df.loc[(df.site == site) & (df.label == label), "date"].unique())
            assert len(dates) > n_val, (
                f"{site}/{label} has {len(dates)} dates; need more than {n_val} "
                f"to keep some for training."
            )
            val_dates = rng.choice(dates, size=n_val, replace=False)
            val_mask |= (df.site == site) & (df.label == label) & df.date.isin(val_dates)

    df_val = df[val_mask & (df.site != test_site)]
    df_train = df[~val_mask & (df.site != test_site)]

    ids = [set(d["image_id"]) for d in (df_train, df_val, df_test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2]), \
        "Split leakage detected!"
    assert len(df_train) + len(df_val) + len(df_test) == len(df)

    return df_train, df_val, df_test


def write_split(out_dir, df_train, df_val, df_test):
    """Write train.csv / val.csv / test.csv (the files BleachDataset reads)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, d in (("train", df_train), ("val", df_val), ("test", df_test)):
        d.sort_values(SORT_COLS).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def describe_split(df_train, df_val, df_test):
    """One-line-per-split summary: tiles, sites, dates, and label counts."""
    lines = []
    for name, d in (("train", df_train), ("val", df_val), ("test", df_test)):
        counts = d["label"].value_counts()
        lines.append(
            f"  {name:<5}: {len(d):>4} tiles, {d.site.nunique():>2} sites, "
            f"{d.groupby('site').date.nunique().sum():>3} site-dates  "
            f"(healthy={counts.get('healthy', 0)}, bleached={counts.get('bleached', 0)})"
        )
    return "\n".join(lines)

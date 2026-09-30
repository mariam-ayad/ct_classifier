"""
make_splits.py
==============
Build leave-one-site-out splits for every reef site (or a chosen subset).

For each held-out site, writes:

    <out_dir>/<site>/train.csv   training tiles (11 sites, minus inner-val dates)
    <out_dir>/<site>/val.csv     inner validation: random dates from the training
                                 sites, used only to pick the best epoch
    <out_dir>/<site>/test.csv    all tiles from the held-out site

plus <out_dir>/all.csv with the full inventory.

Usage
-----
    python scripts/make_splits.py --data-root planet_superdove_landmasked
    python scripts/make_splits.py --data-root planet_superdove_landmasked --sites sanagustin_mexico
"""
import os
import sys
import argparse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.splits import SITES, build_inventory, make_loso_split, write_split, describe_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_DIR, "planet_superdove_landmasked"),
                        help="Root of the land-masked tiles")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "data", "splits"),
                        help="Where per-site split folders are written")
    parser.add_argument("--sites", nargs="+", default=SITES,
                        help="Held-out sites to generate folds for (default: all 12)")
    parser.add_argument("--n-val-healthy", type=int, default=2,
                        help="Healthy dates per training site moved to inner val")
    parser.add_argument("--n-val-bleached", type=int, default=1,
                        help="Bleached dates per training site moved to inner val")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = build_inventory(os.path.abspath(args.data_root), SITES)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "all.csv"), index=False)
    print(f"Inventory: {len(df)} tiles from {df.site.nunique()} sites -> {args.out_dir}/all.csv")

    for site in args.sites:
        df_train, df_val, df_test = make_loso_split(
            df, site,
            n_val_healthy=args.n_val_healthy,
            n_val_bleached=args.n_val_bleached,
            seed=args.seed,
        )
        out = os.path.join(args.out_dir, site)
        write_split(out, df_train, df_val, df_test)
        print(f"\n[{site}] -> {out}/")
        print(describe_split(df_train, df_val, df_test))


if __name__ == "__main__":
    main()

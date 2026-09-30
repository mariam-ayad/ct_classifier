"""
summarize_loso.py
=================
Collect runs/<site>/results.json from every leave-one-site-out fold and print
the per-site table (Table 1) plus the summary statistics reported in the paper.

Usage
-----
    python scripts/summarize_loso.py                 # reads runs/
    python scripts/summarize_loso.py --runs-dir runs --out runs/loso_summary.csv
"""
import os
import glob
import json
import argparse
import pandas as pd

REGIONS = {
    "cheeca_flkeys":     ("Florida",   "Cheeca Rocks"),
    "easterndry_flkeys": ("Florida",   "Eastern Dry Rocks"),
    "looe_flkey":        ("Florida",   "Looe Key"),
    "rock_flkey":        ("Florida",   "Rock Key"),
    "sand_flkey":        ("Florida",   "Sand Key"),
    "chachacual_mexico": ("Mexico",    "Chachacual"),
    "jicaral_mexico":    ("Mexico",    "Jicaral"),
    "riscalillo_mexico": ("Mexico",    "Riscalillo"),
    "sanagustin_mexico": ("Mexico",    "San Agustín"),
    "santacruz_mexico":  ("Mexico",    "Santa Cruz"),
    "northpoint_lizard": ("Australia", "North Point Lizard"),
    "pulau_kapas":       ("Malaysia",  "Pulau Kapas"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default=None, help="CSV path (default: <runs-dir>/loso_summary.csv)")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        region, name = REGIONS.get(r["test_site"], ("?", r["test_site"]))
        rows.append({"region": region, "site": name, **r})

    assert rows, f"No results.json found under {args.runs_dir}/*/"

    order = list(REGIONS)
    df = pd.DataFrame(rows)
    df["_order"] = df.test_site.map(lambda s: order.index(s) if s in order else len(order))
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    out = args.out or os.path.join(args.runs_dir, "loso_summary.csv")
    df.to_csv(out, index=False)

    print(f"{'Region':<10} {'Site':<20} {'Tiles':>5} {'Pairs':>5} {'Best ep':>7} "
          f"{'Val OA':>7} {'Test OA':>8} {'Test OA (last)':>14}")
    for _, r in df.iterrows():
        print(f"{r.region:<10} {r.site:<20} {r.n_test_tiles:>5} {r.n_test_pairs:>5} {r.best_epoch:>7} "
              f"{100 * r.oa_val:>6.2f}% {100 * r.oa_test:>7.2f}% {100 * r.oa_test_last_epoch:>13.2f}%")

    acc = 100 * df.oa_test
    print(f"\nFolds: {len(df)}/{len(REGIONS)}")
    print(f"Mean accuracy:                {acc.mean():.1f}%")
    print(f"Sample-weighted mean (tiles): {(acc * df.n_test_tiles).sum() / df.n_test_tiles.sum():.1f}%")
    print(f"Sample-weighted mean (pairs): {(acc * df.n_test_pairs).sum() / df.n_test_pairs.sum():.1f}%")
    print(f"Median:                       {acc.median():.1f}%")
    print(f"SD:                           {acc.std(ddof=1):.1f}")
    print(f"Range:                        {acc.min():.1f}% ({df.site[acc.idxmin()]}) – "
          f"{acc.max():.1f}% ({df.site[acc.idxmax()]})")
    print(f"Sites > 85%: {(acc > 85).sum()}   Sites > 90%: {(acc > 90).sum()}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

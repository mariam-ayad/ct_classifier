#!/usr/bin/env bash
# run_loso.sh — train every leave-one-site-out fold, then summarize.
#
# Run from the repo root, after `python scripts/make_splits.py`:
#   bash scripts/run_loso.sh                          # all 12 sites
#   bash scripts/run_loso.sh sanagustin_mexico        # one or more chosen sites
#   CONFIG=configs/other.yaml bash scripts/run_loso.sh
#
# Each fold writes to runs/<site>/ and resumes from runs/<site>/last.pt if it
# was interrupted. Finished folds (runs/<site>/results.json) are skipped.

set -euo pipefail

CONFIG="${CONFIG:-configs/loso_resnet50.yaml}"
SPLITS_DIR="${SPLITS_DIR:-data/splits}"
RUNS_DIR="${RUNS_DIR:-runs}"

if [[ $# -gt 0 ]]; then
  SITES=("$@")
else
  SITES=(
    cheeca_flkeys looe_flkey easterndry_flkeys sand_flkey rock_flkey
    sanagustin_mexico chachacual_mexico jicaral_mexico riscalillo_mexico santacruz_mexico
    northpoint_lizard pulau_kapas
  )
fi

for site in "${SITES[@]}"; do
  if [[ -f "$RUNS_DIR/$site/results.json" ]]; then
    echo "== $site: already done, skipping"
    continue
  fi
  echo "== $site"
  python ct_classifier/train.py \
    --config "$CONFIG" \
    --data_root "$SPLITS_DIR/$site" \
    --output_dir "$RUNS_DIR/$site"
done

python scripts/summarize_loso.py --runs-dir "$RUNS_DIR"

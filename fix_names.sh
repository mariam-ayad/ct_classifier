#!/usr/bin/env bash
# fix_names.sh — rename "healthy(new)" -> "healthy" and "bleached(new)" -> "bleached"
# Usage:
#   # dry run (recommended first):
#   DRY_RUN=1 bash fix_names.sh "/pscratch/sd/k/kevinval/coraltest/ct_classifier/planet_superdove/cheeca_flkeys"
#   # actual rename:
#   bash fix_names.sh "/pscratch/sd/k/kevinval/coraltest/ct_classifier/planet_superdove/cheeca_flkeys"

set -euo pipefail
ROOT="${1:-.}"
DRY="${DRY_RUN:-0}"

map_name() {
  case "$1" in
    "healthy(new)")  echo "healthy"  ;;
    "bleached(new)") echo "bleached" ;;
    *) return 1 ;;
  esac
}

export -f map_name
export DRY

# Rename deepest directories first so parent renames don't break child paths.
while IFS= read -r -d '' d; do
  base="$(basename "$d")"
  newbase="$(map_name "$base" || true)"
  [[ -z "${newbase:-}" ]] && continue

  parent="$(dirname "$d")"
  target="$parent/$newbase"

  if [[ "$DRY" == "1" ]]; then
    echo "Would rename: $d -> $target"
    continue
  fi

  if [[ -e "$target" ]]; then
    # If target exists, merge contents then remove the old dir.
    shopt -s dotglob nullglob
    for f in "$d"/*; do
      mv -n -- "$f" "$target"/
    done
    shopt -u dotglob nullglob
    rmdir -- "$d" 2>/dev/null || true
    echo "Merged contents of $d into $target"
  else
    mv -- "$d" "$target"
    echo "Renamed: $d -> $target"
  fi
done < <(find "$ROOT" -depth -type d \( -name 'healthy(new)' -o -name 'bleached(new)' \) -print0)

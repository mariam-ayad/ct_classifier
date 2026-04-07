#!/usr/bin/env python3
"""
Move Planet SuperDove tiles into a 'tiled_360m' subfolder that sits directly
under the class label (healthy/bleached) for rock_flkey and sand_flkey.

Dry run by default: prints what would move, counts, and skips already-correct files.
Set DRY_RUN = False to actually move.
"""

from pathlib import Path
import shutil

# ---- config ----
ROOT = Path("/pscratch/sd/k/kevinval/coraltest/ct_classifier/planet_superdove")
SITES = {"rock_flkey", "sand_flkey"}
LABELS = {"healthy", "bleached"}
EXTENSIONS = {".tif", ".tiff"}     # adjust if needed
DRY_RUN = False                     # <<< set to False to perform moves
# ----------------

def needs_move(p: Path) -> bool:
    parts = p.parts
    # quick guards: site + label in path, and not already in .../label/tiled_360m/...
    has_site = any(s in parts for s in SITES)
    has_label = any(l in parts for l in LABELS)
    already_ok = "tiled_360m" in parts
    return has_site and has_label and (not already_ok)

def insert_after_label(p: Path) -> Path:
    parts = list(p.parts)
    # find the first occurrence of a label and insert 'tiled_360m' right after it
    for i, part in enumerate(parts):
        if part in LABELS:
            dest_parts = parts[:i+1] + ["tiled_360m"] + parts[i+1:]
            return Path(*dest_parts)
    # if no label found, return unchanged (shouldn't happen if filtered correctly)
    return p

def main():
    candidates = []
    for site in SITES:
        for label in LABELS:
            # search under ROOT/site/label for image files
            base = ROOT / site / label
            if not base.exists():
                continue
            for src in base.rglob("*"):
                if src.is_file() and src.suffix.lower() in EXTENSIONS and needs_move(src):
                    # also ensure we’re not picking up files already in tiled_360m
                    if "tiled_360m" not in src.parts:
                        candidates.append(src)

    planned = []
    for src in candidates:
        dst = insert_after_label(src)
        planned.append((src, dst))

    # dedupe any accidental duplicates
    seen = set()
    uniq = []
    for s, d in planned:
        key = (s.resolve(), d)
        if key not in seen:
            uniq.append((s, d))
            seen.add(key)

    # report
    print(f"Found {len(uniq)} file(s) needing the 'tiled_360m' insertion.")
    preview = 10
    for i, (s, d) in enumerate(uniq[:preview], 1):
        print(f"[{i:02}] {s}\n   -> {d}")

    # check collisions / missing parents
    collisions = []
    for _, d in uniq:
        if d.exists():
            collisions.append(d)

    if collisions:
        print(f"\nWARNING: {len(collisions)} destination file(s) already exist. They will be SKIPPED.")
        for d in collisions[:10]:
            print(f"   exists: {d}")

    if DRY_RUN:
        print("\nDry run only. No files moved. Set DRY_RUN = False to apply changes.")
        return

    # perform moves
    moved, skipped = 0, 0
    for s, d in uniq:
        if d.exists():
            skipped += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        moved += 1

    print(f"\nDone. Moved: {moved}, Skipped (already existed): {skipped}")

if __name__ == "__main__":
    main()

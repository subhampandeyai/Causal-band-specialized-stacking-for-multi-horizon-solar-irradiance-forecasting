"""
backfill_reference_columns.py
=============================
Add the skill-reference columns to prediction files written before the
reference was carried through.

The supplementary analyses score skill against a reference held in the
prediction files. Files produced by earlier runs carry only the persistence
column, so the analyses fall back to persistence, whose error exceeds the
target standard deviation beyond roughly two hours. This copies ref_clim and
ref_persist across from the matching grid files, which already hold them.

Nothing is retrained: the columns are copied from results/experiments/, and the
existing model predictions are left untouched. Rows are matched only when the
two files agree on y_true, so a mismatched pair is skipped rather than filled
with the wrong values.

Run once after an older grid, then re-run scripts/reproduce_supplementary.py.

    python src/backfill_reference_columns.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results" / "experiments" / "predictions"
LEGACY = REPO / "results" / "predictions"
REF_SEED = 42
REF_W = 16


def main() -> int:
    if not LEGACY.exists():
        print(f"no prediction files at {LEGACY}")
        return 1
    if not GRID.exists():
        print(f"no grid output at {GRID}; run src/run_experiments_full.py first")
        return 1

    files = sorted(LEGACY.glob("*_test_predictions.csv"))
    done = skipped = already = 0

    for p in files:
        m = re.match(r"station_(\d+)_H(\d+)", p.stem)
        if not m:
            continue
        station, horizon = int(m.group(1)), m.group(2)
        src = GRID / f"s{station:02d}_H{horizon}_seed{REF_SEED}_W{REF_W}.csv"
        if not src.exists():
            print(f"  {p.name}: no matching grid file, skipped")
            skipped += 1
            continue

        legacy = pd.read_csv(p)
        if "ref_clim" in legacy.columns:
            already += 1
            continue

        grid = pd.read_csv(src)
        if len(legacy) != len(grid) or not np.allclose(
                legacy.y_true.values, grid.y_true.values, atol=1e-12):
            print(f"  {p.name}: targets do not match {src.name}, skipped")
            skipped += 1
            continue

        for col in ("ref_clim", "ref_persist"):
            if col in grid.columns:
                legacy[col] = grid[col].values
        legacy.to_csv(p, index=False)
        done += 1

    print(f"  files            : {len(files)}")
    print(f"  columns added    : {done}")
    print(f"  already had them : {already}")
    print(f"  skipped          : {skipped}")
    if done or already:
        print("\nnow run: python scripts/reproduce_supplementary.py")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

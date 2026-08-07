#!/usr/bin/env python
"""
Reproduce the manuscript's experimental results.

This runs the full training pipeline: Stage 0 prepares the station series from
the raw dataset, then the causal band-specialized stacking model and its
baselines are fitted for every station and horizon.

Unlike the supplementary analyses, this script trains models and therefore needs
the dataset in place and a substantial amount of compute. See DATA.md for how to
obtain and position the dataset.

Usage
-----
    python scripts/reproduce_manuscript.py --check     # verify inputs only
    python scripts/reproduce_manuscript.py --stage0    # prepare the data
    python scripts/reproduce_manuscript.py             # full pipeline

Runtime: the LSTM band expert dominates the cost. On a CPU-only machine one
(station, horizon) pair takes roughly 16 minutes, so the full 7 x 6 grid is on
the order of 11 hours. A CUDA GPU reduces this by about an order of magnitude;
TensorFlow uses one automatically when available.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
DATA_RAW = REPO / "data" / "raw"
DATA_PROCESSED = REPO / "data" / "processed"


def check_inputs() -> bool:
    print(f"repository root : {REPO}")
    ok = True

    raw = sorted(DATA_RAW.glob("*.xlsx")) if DATA_RAW.exists() else []
    prepared = (sorted(DATA_PROCESSED.glob("station_*_prepared.csv"))
                if DATA_PROCESSED.exists() else [])
    print(f"raw .xlsx files : {len(raw)}  (expected 8)")
    print(f"prepared CSVs   : {len(prepared)}  (expected 8)")

    if not raw and not prepared:
        print("\nNo dataset found. The Chinese State Grid dataset is not "
              "redistributed with this repository.")
        print(f"Place the eight station .xlsx files in: {DATA_RAW}")
        print("See data/DATA.md for the source and the exact file names.")
        ok = False
    elif raw and not prepared:
        print("\nRaw files present but not yet prepared.")
        print("Run: python scripts/reproduce_manuscript.py --stage0")

    for name in ["run_fame_causal.py", "conformal_causal.py"]:
        p = SRC / name
        print(f"{'found  ' if p.exists() else 'MISSING'} src/{name}")
        ok &= p.exists()

    cfg = REPO / "configs" / "config.yaml"
    print(f"{'found  ' if cfg.exists() else 'MISSING'} configs/config.yaml")
    return ok


def run(script: Path, cwd: Path) -> bool:
    print(f"\nrunning {script.relative_to(REPO)}")
    return subprocess.run([sys.executable, str(script)], cwd=str(cwd)).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify inputs and exit without training")
    parser.add_argument("--stage0", action="store_true",
                        help="run data preparation only")
    args = parser.parse_args()

    ok = check_inputs()
    if args.check:
        return 0 if ok else 1
    if not ok:
        print("\ninput check failed; nothing was run")
        return 1

    start = time.time()

    if args.stage0:
        script = SRC / "pipeline" / "stage0_preprocessing.py"
        return 0 if run(script, REPO) else 1

    if not sorted(DATA_PROCESSED.glob("station_*_prepared.csv")):
        print("\nprepared data missing; running Stage 0 first")
        if not run(SRC / "pipeline" / "stage0_preprocessing.py", REPO):
            return 1

    if not run(SRC / "run_fame_causal.py", REPO):
        return 1
    run(SRC / "conformal_causal.py", REPO)

    print(f"\nfinished in {(time.time() - start) / 60:.1f} min")
    print(f"results -> {REPO / 'results'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

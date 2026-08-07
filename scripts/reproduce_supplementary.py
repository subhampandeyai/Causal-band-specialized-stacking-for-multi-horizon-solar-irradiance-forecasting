#!/usr/bin/env python
"""
Reproduce every supplementary table and figure.

Runs the eight supplementary analysis stages in dependency order and writes all
outputs under supplementary/. Every input is a file already inside this
repository, so no model training and no external download is required.

Usage
-----
    python scripts/reproduce_supplementary.py
    python scripts/reproduce_supplementary.py --only s03_uncertainty
    python scripts/reproduce_supplementary.py --list

Runtime: about one minute in total.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "supplementary" / "analysis_code"

# (module, description). Order matters: s02 writes the per-pair metric table that
# s04 consumes, and s06 collects whatever the earlier stages produced.
STAGES = [
    ("s01_consistency_verification",
     "Recompute the reported metrics from the saved predictions"),
    ("s02_statistical_validation",
     "Wilcoxon, Cohen's d_z, confidence intervals, Holm correction, bootstrap"),
    ("s03_uncertainty",
     "Split-conformal intervals for every station and horizon"),
    ("s04_robustness",
     "Station, horizon, failure-case and error-structure analysis"),
    ("s05_implementation_and_baseline_audit",
     "Executed correctness checks and the baseline fairness audit"),
    ("s07_metafusion_and_scope",
     "Post-hoc fusion study and the scope declaration"),
    ("s08_band_spectral_analysis",
     "Frequency content of each wavelet band"),
    ("s06_report",
     "Consolidate the supplementary tables and build the PDF"),
]


def run_stage(module: str) -> tuple[bool, float]:
    start = time.time()
    result = subprocess.run([sys.executable, str(CODE / f"{module}.py")],
                            cwd=str(REPO))
    return result.returncode == 0, time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run a single stage by module name")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    args = parser.parse_args()

    if args.list:
        for module, description in STAGES:
            print(f"  {module:<40} {description}")
        return 0

    stages = [s for s in STAGES if s[0] == args.only] if args.only else STAGES
    if not stages:
        print(f"unknown stage: {args.only}", file=sys.stderr)
        return 1

    print(f"repository root: {REPO}")
    print(f"running {len(stages)} stage(s)\n")

    failures = []
    total = time.time()
    for module, description in stages:
        print("=" * 72)
        print(f"{module}  -  {description}")
        print("=" * 72)
        ok, elapsed = run_stage(module)
        print(f"[{'OK' if ok else 'FAILED'}] {module} ({elapsed:.1f}s)\n")
        if not ok:
            failures.append(module)

    print("=" * 72)
    print(f"finished in {time.time() - total:.1f}s")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all stages completed")
    print(f"tables  -> {REPO / 'supplementary' / 'tables'}")
    print(f"figures -> {REPO / 'supplementary' / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

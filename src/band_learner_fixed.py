"""
band_learner_fixed.py
=====================
Corrected band x learner comparison (Experiment 4).

WHY THE PREVIOUS VERSION WAS INVALID
------------------------------------
It trained on the pipeline's full feature matrix X, which contains the four
reconstructed bands as columns 16-19 (verified: X[:, 16+i] == bands[:, i]
exactly). Predicting band k from a matrix containing band k is an identity map,
so Ridge "won" every band with RMSE ~0. That measured column copying, not
learner suitability.

WHY DROPPING THE BAND COLUMNS IS STILL NOT ENOUGH
-------------------------------------------------
The four bands reconstruct the signal exactly
(max |sum(bands) - sig| = 8.9e-16), so band k = sig(t) - sum_{j!=k} band j.
The pipeline's rolling features rm4/rs4/rm12/rs12/rm32/rs32 use
`rolling(w, min_periods=1)`, whose window ENDS at t and therefore contains
sig(t) itself. Any of them leaks the contemporaneous signal. Measured with the
band columns removed but the rolling features kept, Ridge still reached
RMSE/std = 0.079 on the trend band -- still near-trivial.

WHAT THIS MODULE DOES
---------------------
Each band is predicted from a strictly causal feature set built from
information available BEFORE t:

    lagged irradiance          sig(t-1), sig(t-2), sig(t-4), sig(t-8),
                               sig(t-16), sig(t-32)
    trailing rolling stats     mean/std over windows ending at t-1
    exogenous                  TEMPERATURE, REL_HUMIDITY, ATMOSPHERE, DNI
                               (shifted by one step)
    other bands, lagged        band j != k at t-1 only

No feature contains sig(t) or band k at time t, so no learner can reconstruct
the target by identity. All learners then see exactly the same inputs and
compete fairly.

TARGET
------
The pipeline's band experts forecast the h-step-ahead irradiance from
band-augmented features; they do not forecast the band itself. This module
therefore reports both:

    band_k(t+h)   how well each learner forecasts the band it is assigned to
    y(t+h)        how well each learner forecasts the actual target when given
                  band k's information

The second is the quantity Eq. 11's assignment is about.

Read-only with respect to the repository: run_fame_causal.py is imported, never
modified. Output goes to results/band_learner/band_learner_fixed.csv.

Run (from the repository root). Note that --stations defaults to station 1
alone, so the full grid must be requested explicitly:
    python src/band_learner_fixed.py --stations 1,2,4,5,6,7,8    # full grid
    python src/band_learner_fixed.py --stations 1 --seed 42      # single station
"""
import os
import sys
import time
import argparse
import warnings
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import run_fame_causal as RFC          # noqa: E402  (import only, never modified)

PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "band_learner"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = RFC.SCALE
BAND_NAMES = ["trend", "multi_hour", "hourly", "noise"]   # cA3, cD3, cD2, cD1
HORIZONS = [1, 4, 8, 16, 32, 96]


def causal_features_excluding(df, sig, bands, exclude_band):
    """Strictly causal features that cannot reconstruct `exclude_band` at t.

    Everything is shifted so the value at row t uses information up to t-1 only.
    The excluded band contributes nothing at any lag; the other bands enter at
    t-1, which is genuinely available at forecast time.
    """
    n = len(sig)
    f = {}
    s = pd.Series(sig)

    for L in [1, 2, 4, 8, 16, 32]:
        c = np.full(n, np.nan)
        c[L:] = sig[:-L]
        f[f"lag{L}"] = c

    # rolling windows END at t-1: shift(1) before rolling
    sp = s.shift(1)
    for w in [4, 12, 32]:
        f[f"rm{w}"] = sp.rolling(w, min_periods=1).mean().values
        f[f"rs{w}"] = sp.rolling(w, min_periods=1).std().fillna(0).values

    for col in ["TEMPERATURE", "REL_HUMIDITY", "ATMOSPHERE", "DNI"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
            f[col] = v.shift(1).bfill().values

    # other bands at t-1 only; the excluded band never appears
    for i in range(bands.shape[1]):
        if i == exclude_band:
            continue
        c = np.full(n, np.nan)
        c[1:] = bands[:-1, i]
        f[f"band{i}_lag1"] = c

    return pd.DataFrame(f).values


def split_idx(X, y, h):
    n = len(y)
    tgt = np.full(n, np.nan)
    tgt[:-h] = y[h:]
    v = ~np.isnan(tgt) & ~np.isnan(X).any(1)
    idx = np.where(v)[0]
    a, b = int(len(idx) * .70), int(len(idx) * .85)
    return idx, a, b, tgt


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def skill_vs(y, p, ref):
    sse = float(np.sum((y - ref) ** 2))
    return float(1 - np.sum((y - p) ** 2) / sse) if sse > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="1")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    stations = [int(x) for x in a.stations.split(",")]

    rows = []
    t_start = time.time()

    for st in stations:
        df = pd.read_csv(PROC / f"station_{st:02d}_prepared.csv")
        sig = (pd.to_numeric(df["IRRADIATION"], errors="coerce")
               .ffill().bfill().values.astype(float))
        bands = RFC.bands_causal(sig)
        print(f"\n=== station {st} ({len(sig):,} samples) ===", flush=True)

        for h in HORIZONS:
            for bi, bname in enumerate(BAND_NAMES):
                X = causal_features_excluding(df, sig, bands, exclude_band=bi)

                for target_kind in ("band", "y"):
                    tgt_series = bands[:, bi] if target_kind == "band" else sig
                    idx, ia, ib, tgt = split_idx(X, tgt_series, h)
                    Xv, yv = X[idx], tgt[idx]
                    Xtr, ytr = Xv[:ia], yv[:ia]
                    Xva, yva = Xv[ia:ib], yv[ia:ib]
                    Xte, yte = Xv[ib:], yv[ib:]
                    if len(yte) < 100:
                        continue
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(Xtr)
                    Xva_s, Xte_s = sc.transform(Xva), sc.transform(Xte)

                    # reference: persistence of the quantity being predicted
                    ref = tgt_series[idx[ib:]]

                    cands = {}
                    cands["Ridge"] = RFC.fit_ridge(Xtr_s, ytr, Xva_s, Xte_s)[1]
                    cands["XGBoost"] = RFC.fit_xgb(Xtr_s, ytr, Xva_s, Xte_s)[1]
                    cands["persistence"] = ref
                    if bname == "multi_hour":
                        try:
                            RFC.np.random.seed(a.seed)
                            cands["LSTM"] = RFC.fit_lstm(
                                Xtr_s, ytr, Xva_s, yva, Xte_s)[1]
                        except Exception as e:
                            print(f"      LSTM failed: {e}")

                    sd = float(np.std(yte))
                    for lname, pred in cands.items():
                        pred = np.asarray(pred, float)[:len(yte)]
                        rows.append(dict(
                            station=st, horizon=f"H{h}", band=bname,
                            target=target_kind, learner=lname,
                            rmse_W=round(rmse(yte, pred) * SCALE, 4),
                            target_std_W=round(sd * SCALE, 4),
                            rmse_over_std=round(rmse(yte, pred) / sd, 4) if sd > 0 else np.nan,
                            skill_vs_persist=round(skill_vs(yte, pred, ref), 6),
                            n_test=len(yte)))
                print(f"  H{h} {bname}: done", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(OUT / "band_learner_fixed.csv", index=False)

    print("\n" + "=" * 88)
    print("LEAKAGE SANITY CHECK  (rmse / target_std; ~0 would mean leakage remains)")
    print("=" * 88)
    chk = D.groupby(["target", "learner"]).rmse_over_std.min()
    print(chk.round(4).to_string())
    worst = D[D.learner != "persistence"].rmse_over_std.min()
    print(f"\n  smallest non-persistence rmse/std across all cells: {worst:.4f}")
    print("  -> " + ("SUSPECT: still near zero, leakage not fully removed"
                     if worst < 0.05 else "OK: no trivial fits"))

    for tk, label in [("band", "TARGET = the band itself"),
                      ("y", "TARGET = h-step-ahead irradiance (Eq. 11's question)")]:
        sub = D[D.target == tk]
        if sub.empty:
            continue
        print("\n" + "=" * 88)
        print(label)
        print("=" * 88)
        piv = sub.pivot_table(index=["band", "horizon"], columns="learner",
                              values="rmse_W", aggfunc="mean")
        print(piv.round(2).to_string())
        print("\n  winner per band (lowest mean RMSE over horizons):")
        for b in BAND_NAMES:
            q = sub[sub.band == b].groupby("learner").rmse_W.mean().sort_values()
            if len(q):
                print(f"    {b:<12} {q.index[0]:<12} {q.iloc[0]:8.2f} W   "
                      f"(next: {q.index[1]} {q.iloc[1]:.2f})" if len(q) > 1 else "")

    print(f"\nelapsed {(time.time()-t_start)/60:.1f} min")
    print(f"written -> {OUT/'band_learner_fixed.csv'}")


if __name__ == "__main__":
    main()

"""
run_experiments_full.py
=======================
Full experiment driver for the causal band-specialized stacking study.

Reuses the model path, features, splits and dual skill reference from
run_fame_causal.py without altering any of them. This module only ADDS
computations and output columns.

One unit of work = (station, horizon, seed, W). Every unit is appended to disk
the moment it finishes, so a crash after unit N leaves units 1..N intact and a
restart skips them.

EXPERIMENTS
-----------
  1. 8-seed support            SEEDS, one unit per (station, horizon, seed)
  2. W-sensitivity sweep       W in {8, 16, 32} x coarsest scale
  3. Simple-averaging ablation equal-weight mean of the four band forecasts
  4. Per-band x per-learner    every learner on every band, held-out skill
  5. Persistence-augmented     single models on a residual-over-persistence target
  6. Non-causal (leaky) skill  whole-series operator, skill as well as R2
  7. Classical baselines       MLP and SVR

OUTPUTS -> results/experiments/
    results_units.csv          one row per completed unit (appended live)
    band_learner_table.csv     per-band x per-learner skill (appended live)
    predictions/               per-unit prediction vectors
    manifest.json              config, seeds, versions

Also written:
    results/predictions/       the four-column view the supplementary
                               analyses consume (reference seed and W only)
    supplementary/statistics/Computational_Cost/runtime_probe.json
                               per-model runtime, measured during this run

Run:
    python run_experiments_full.py --stations 1 --seeds 42 --w 16
    python run_experiments_full.py                      # full grid
"""
import os
import sys
import json
import time
import argparse
import platform
import warnings
from pathlib import Path

# determinism must be set before TensorFlow is imported
os.environ["PYTHONHASHSEED"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

# the band experts, features, split and skill references come from the pipeline
import run_fame_causal as RFC          # noqa: E402

PROC_DIR = REPO / "data" / "processed"
OUT_DIR = REPO / "results" / "experiments"
PRED_DIR = OUT_DIR / "predictions"
LEGACY = REPO / "results" / "predictions"     # consumed by supplementary/
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

UNITS_CSV = OUT_DIR / "results_units.csv"
BAND_CSV = OUT_DIR / "band_learner_table.csv"

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = [1, 4, 8, 16, 32, 96]
SEEDS = [42, 7, 123, 2024, 31, 89, 500, 1]
W_MULTS = [8, 16, 32]                  # x coarsest scale 2**LEVEL
SCALE = RFC.SCALE
SEQ_LEN = RFC.SEQ_LEN
BANDS = ["trend", "multi_hour", "hourly", "noise"]


# ---------------------------------------------------------------- determinism
def set_determinism(seed: int):
    import tensorflow as tf
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- decomposition
def bands_causal_W(sig, window):
    """RFC.bands_causal with an explicit window, for the W sweep."""
    n = len(sig)
    nb = RFC.LEVEL + 1
    out = np.full((n, nb), np.nan)
    ml = 8 * 2 ** RFC.LEVEL
    W = max(window, ml)
    for t in range(n):
        seg = sig[max(0, t - W + 1):t + 1]
        if len(seg) < ml:
            out[t, 0] = sig[t]
            out[t, 1:] = 0.0
            continue
        c = pywt.wavedec(seg, RFC.FAMILY, level=RFC.LEVEL)
        for i in range(nb):
            z = [np.zeros_like(x) for x in c]
            z[i] = c[i]
            out[t, i] = pywt.waverec(z, RFC.FAMILY)[:len(seg)][-1]
    return out


def bands_leaky(sig):
    """Whole-series (non-causal) operator, for the leakage comparison."""
    coeffs = pywt.wavedec(sig, RFC.FAMILY, level=RFC.LEVEL)
    cols = []
    for i in range(len(coeffs)):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        cols.append(pywt.waverec(z, RFC.FAMILY)[:len(sig)])
    return np.column_stack(cols)


# ---------------------------------------------------------------- learners
def fit_mlp(Xtr, ytr, Xte, seed):
    m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200,
                     early_stopping=True, n_iter_no_change=8,
                     random_state=seed).fit(Xtr, ytr)
    return m.predict(Xte)


def fit_svr(Xtr, ytr, Xte):
    return SVR(kernel="rbf", C=1.0, epsilon=0.1).fit(Xtr, ytr).predict(Xte)


def fit_lgb(Xtr, ytr, Xte, seed):
    return lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             verbosity=-1).fit(Xtr, ytr).predict(Xte)


def r2(y, p):
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


# ---------------------------------------------------------------- persistence
# Fixed schema. Every row is emitted with exactly these fields in this order,
# so a timing that did not occur for a given unit (e.g. bands_s when the
# decomposition was served from cache) is written as NaN rather than omitted.
# Omitting it shifted every later column left and produced NaN totals.
TIMING_FIELDS = ["bands_s", "ridge_s", "lstm_s", "xgb_s", "lgb_s", "mlp_s",
                 "svr_s", "persaug_s", "leaky_s", "bandtable_s", "unit_total_s"]

ROW_FIELDS = [
    "station", "horizon", "seed", "W_mult", "window", "n_test",
    "sigma_y_W",
    # reference errors, same units and format as the existing persistence pair
    "persist_rmse_W", "persist_mae_W", "clim_rmse_W", "clim_mae_W",
    "persist_is_valid_reference",
    "fame_r2", "fame_rmse_W", "fame_mae_W",
    "fame_skill_clim", "fame_skill_persist",
    "simple_avg_r2", "simple_avg_skill_clim",
    "uxgb_r2", "uxgb_rmse_W", "uxgb_mae_W", "uxgb_skill_clim",
    "lgb_r2", "lgb_skill_clim",
    "mlp_r2", "mlp_rmse_W", "mlp_mae_W", "mlp_skill_clim",
    "svr_r2", "svr_rmse_W", "svr_mae_W", "svr_skill_clim",
    "persaug_xgb_r2", "persaug_xgb_skill_clim",
    "leaky_r2", "leaky_skill_clim", "causal_minus_leaky_r2",
    "bandtable_computed",
] + TIMING_FIELDS


def load_done():
    if not UNITS_CSV.exists():
        return set()
    d = pd.read_csv(UNITS_CSV)
    return set(zip(d.station, d.horizon, d.seed, d.W_mult))


def load_bandtable_done():
    """(station, horizon, W_mult) already covered by the band x learner table.
    That comparison is structural, so it is computed for the first seed only."""
    if not BAND_CSV.exists():
        return set()
    d = pd.read_csv(BAND_CSV)
    return set(zip(d.station, d.horizon, d.W_mult))


def append_row(path, row: dict):
    """Append one row on the fixed schema and flush to disk immediately."""
    full = {k: row.get(k, np.nan) for k in ROW_FIELDS}
    extra = [k for k in row if k not in ROW_FIELDS]
    if extra:
        raise KeyError(f"row has fields outside ROW_FIELDS: {extra}")
    df = pd.DataFrame([full], columns=ROW_FIELDS)
    hdr = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        df.to_csv(fh, header=hdr, index=False)
        fh.flush()
        os.fsync(fh.fileno())


def append_rows(path, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    hdr = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        df.to_csv(fh, header=hdr, index=False)
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------- one unit
def run_unit(s, h, seed, w_mult, df, sig, tod_all, doy_all, cache,
             do_bandtable=True):
    t_unit = time.time()
    set_determinism(seed)
    timings = {k: np.nan for k in TIMING_FIELDS}

    window = w_mult * (2 ** RFC.LEVEL)
    key = (s, w_mult)
    if key not in cache:
        t0 = time.time()
        cache[key] = bands_causal_W(sig, window)
        timings["bands_s"] = time.time() - t0
    bands = cache[key]

    X = RFC.build_features(df, sig, bands)
    (Xtr, ytr, Xva, yva, Xte, yte,
     oidx_tr, oidx_va, oidx_te) = RFC.make_xy(X, sig, h)
    if len(yte) < 100:
        return None, []

    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # ---- band experts (identical to the frozen pipeline) ----
    t0 = time.time(); r_va, r_te = RFC.fit_ridge(Xtr_s, ytr, Xva_s, Xte_s)
    timings["ridge_s"] = time.time() - t0
    t0 = time.time()
    try:
        l_va, l_te = RFC.fit_lstm(Xtr_s, ytr, Xva_s, yva, Xte_s)
    except Exception as e:
        print(f"      LSTM failed ({e}); ridge fallback")
        l_va, l_te = r_va, r_te
    timings["lstm_s"] = time.time() - t0
    t0 = time.time(); g_va, g_te = RFC.fit_xgb(Xtr_s, ytr, Xva_s, Xte_s)
    timings["xgb_s"] = time.time() - t0
    p_va = np.concatenate([[ytr[-1]], yva[:-1]])
    p_te = np.concatenate([[yva[-1]], yte[:-1]])

    Zva = np.column_stack([r_va, l_va, g_va, p_va])
    Zte = np.column_stack([r_te, l_te, g_te, p_te])
    meta = Ridge(alpha=1.0).fit(Zva, yva)
    fame_te = meta.predict(Zte)

    # ---- EXPERIMENT 3: simple averaging ----
    simple_avg = Zte.mean(axis=1)

    # ---- skill references (dual, from the frozen integration) ----
    ref_persist = sig[oidx_te]
    coef, populated, adj = RFC.fit_climatology(sig, tod_all, doy_all, oidx_tr)
    tgt_idx = np.minimum(oidx_te + h, len(sig) - 1)
    ref_clim = RFC.climatology_at(coef, populated, adj,
                                  tod_all[tgt_idx], doy_all[tgt_idx],
                                  float(np.nanmax(sig)))
    sig_y = float(np.std(yte))
    persist_valid = rmse(yte, ref_persist) < sig_y

    def sk(p, ref):
        return RFC.skill_score(yte, p, ref) if ref is not None else np.nan

    row = dict(station=s, horizon=f"H{h}", seed=seed, W_mult=w_mult,
               window=window, n_test=len(yte), sigma_y_W=round(sig_y * SCALE, 2),
               # reference errors: persistence and climatology, like-for-like
               persist_rmse_W=round(rmse(yte, ref_persist) * SCALE, 2),
               persist_mae_W=round(mae(yte, ref_persist) * SCALE, 2),
               clim_rmse_W=(round(rmse(yte, ref_clim) * SCALE, 2)
                            if ref_clim is not None else np.nan),
               clim_mae_W=(round(mae(yte, ref_clim) * SCALE, 2)
                           if ref_clim is not None else np.nan),
               persist_is_valid_reference=bool(persist_valid),
               fame_r2=round(r2(yte, fame_te), 6),
               fame_rmse_W=round(rmse(yte, fame_te) * SCALE, 4),
               fame_mae_W=round(mae(yte, fame_te) * SCALE, 4),
               fame_skill_clim=round(sk(fame_te, ref_clim), 6),
               fame_skill_persist=(round(sk(fame_te, ref_persist), 6)
                                   if h in (1, 4) else np.nan),
               simple_avg_r2=round(r2(yte, simple_avg), 6),
               simple_avg_skill_clim=round(sk(simple_avg, ref_clim), 6),
               bandtable_computed=bool(do_bandtable))

    preds = {"y_true": yte, "fame": fame_te, "simple_avg": simple_avg,
             "persistence_band": p_te, "ref_persist": ref_persist}
    if ref_clim is not None:
        preds["ref_clim"] = ref_clim

    # ---- baselines ----
    ux_te = g_te
    row["uxgb_r2"] = round(r2(yte, ux_te), 6)
    row["uxgb_rmse_W"] = round(rmse(yte, ux_te) * SCALE, 4)
    row["uxgb_mae_W"] = round(mae(yte, ux_te) * SCALE, 4)
    row["uxgb_skill_clim"] = round(sk(ux_te, ref_clim), 6)
    preds["unified_xgb"] = ux_te

    if HAVE_LGB:
        t0 = time.time(); lg = fit_lgb(Xtr_s, ytr, Xte_s, seed)
        timings["lgb_s"] = time.time() - t0
        row["lgb_r2"] = round(r2(yte, lg), 6)
        row["lgb_skill_clim"] = round(sk(lg, ref_clim), 6)
        preds["lightgbm"] = lg

    # ---- EXPERIMENT 7: classical baselines ----
    t0 = time.time(); mlp = fit_mlp(Xtr_s, ytr, Xte_s, seed)
    timings["mlp_s"] = time.time() - t0
    row["mlp_r2"] = round(r2(yte, mlp), 6)
    row["mlp_rmse_W"] = round(rmse(yte, mlp) * SCALE, 4)
    row["mlp_mae_W"] = round(mae(yte, mlp) * SCALE, 4)
    row["mlp_skill_clim"] = round(sk(mlp, ref_clim), 6)
    preds["mlp"] = mlp

    t0 = time.time(); svr = fit_svr(Xtr_s, ytr, Xte_s)
    timings["svr_s"] = time.time() - t0
    row["svr_r2"] = round(r2(yte, svr), 6)
    row["svr_rmse_W"] = round(rmse(yte, svr) * SCALE, 4)
    row["svr_mae_W"] = round(mae(yte, svr) * SCALE, 4)
    row["svr_skill_clim"] = round(sk(svr, ref_clim), 6)
    preds["svr"] = svr

    # ---- EXPERIMENT 5: persistence-augmented target ----
    t0 = time.time()
    base_tr = sig[oidx_tr]
    base_te = ref_persist
    res_tr = ytr - base_tr
    _, res_te = RFC.fit_xgb(Xtr_s, res_tr, Xva_s, Xte_s)
    paug = base_te + res_te
    timings["persaug_s"] = time.time() - t0
    row["persaug_xgb_r2"] = round(r2(yte, paug), 6)
    row["persaug_xgb_skill_clim"] = round(sk(paug, ref_clim), 6)
    preds["persaug_xgb"] = paug

    # ---- EXPERIMENT 6: non-causal (leaky) operator, R2 AND skill ----
    t0 = time.time()
    lk = cache.get((s, "leaky"))
    if lk is None:
        lk = bands_leaky(sig)
        cache[(s, "leaky")] = lk
    Xl = RFC.build_features(df, sig, lk)
    (Ltr, lytr, Lva, lyva, Lte, lyte, _, _, loidx_te) = RFC.make_xy(Xl, sig, h)
    scl = StandardScaler()
    Ltr_s, Lva_s, Lte_s = scl.fit_transform(Ltr), scl.transform(Lva), scl.transform(Lte)
    lr_va, lr_te = RFC.fit_ridge(Ltr_s, lytr, Lva_s, Lte_s)
    lg_va, lg_te2 = RFC.fit_xgb(Ltr_s, lytr, Lva_s, Lte_s)
    lp_va = np.concatenate([[lytr[-1]], lyva[:-1]])
    lp_te = np.concatenate([[lyva[-1]], lyte[:-1]])
    lmeta = Ridge(alpha=1.0).fit(np.column_stack([lr_va, lg_va, lp_va]), lyva)
    leaky_te = lmeta.predict(np.column_stack([lr_te, lg_te2, lp_te]))
    lclim = RFC.climatology_at(coef, populated, adj,
                               tod_all[np.minimum(loidx_te + h, len(sig) - 1)],
                               doy_all[np.minimum(loidx_te + h, len(sig) - 1)],
                               float(np.nanmax(sig)))
    timings["leaky_s"] = time.time() - t0
    row["leaky_r2"] = round(r2(lyte, leaky_te), 6)
    row["leaky_skill_clim"] = (round(RFC.skill_score(lyte, leaky_te, lclim), 6)
                               if lclim is not None else np.nan)
    row["causal_minus_leaky_r2"] = round(row["fame_r2"] - row["leaky_r2"], 6)

    # ---- EXPERIMENT 4: per-band x per-learner ----
    # Structural question: is the Eq. 11 band-to-learner assignment supported by
    # held-out error? That does not vary with the seed, so it is computed for the
    # first seed of each (station, horizon, W) only. This is ~77% of unit cost.
    band_rows = []
    t0 = time.time()
    for bi, bname in enumerate(BANDS if do_bandtable else []):
        btr = bands[oidx_tr, bi]
        bte = bands[oidx_te, bi]
        cands = {}
        cands["Ridge"] = RFC.fit_ridge(Xtr_s, btr, Xva_s, Xte_s)[1]
        cands["XGBoost"] = RFC.fit_xgb(Xtr_s, btr, Xva_s, Xte_s)[1]
        cands["persistence"] = bands[np.maximum(oidx_te - h, 0), bi]
        if bname == "multi_hour":
            try:
                cands["LSTM"] = RFC.fit_lstm(Xtr_s, btr, Xva_s,
                                             bands[oidx_va, bi], Xte_s)[1]
            except Exception:
                pass
        bvar = float(np.sum((bte - bte.mean()) ** 2))
        for lname, pred in cands.items():
            band_rows.append(dict(
                station=s, horizon=f"H{h}", seed=seed, W_mult=w_mult,
                band=bname, learner=lname,
                rmse_W=round(rmse(bte, pred) * SCALE, 4),
                r2=round(1 - np.sum((bte - pred) ** 2) / bvar, 6) if bvar > 0 else np.nan))
    timings["bandtable_s"] = time.time() - t0

    for k, v in timings.items():
        row[k] = round(v, 2)
    row["unit_total_s"] = round(time.time() - t_unit, 2)

    pd.DataFrame(preds).to_csv(
        PRED_DIR / f"s{s:02d}_H{h}_seed{seed}_W{w_mult}.csv", index=False)

    # The supplementary analyses (s01-s11) read results/predictions/ under the
    # original naming and column contract: station_NN_H*_test_predictions.csv
    # with columns y_true, fame, unified_xgb, persistence. Emit that view for
    # the reference configuration so those stages run unchanged.
    if seed == SEEDS[0] and w_mult == 16:
        LEGACY.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"y_true": yte, "fame": fame_te,
                      "unified_xgb": ux_te, "persistence": p_te}).to_csv(
            LEGACY / f"station_{s:02d}_H{h}_test_predictions.csv", index=False)
    return row, band_rows


def write_runtime_probe():
    """Per-model training cost, measured during the grid run.

    The grid times every component of every unit; this reduces those timings to
    the per-pair figures the complexity table reports. Written after the run so
    the runtime numbers are regenerated from the same execution as every other
    result, rather than carried over from a separate measurement.

    Runtimes are hardware-dependent: the values describe the machine that
    produced them, not a portable constant.
    """
    if not UNITS_CSV.exists():
        return
    u = pd.read_csv(UNITS_CSV)
    if u.empty:
        return
    ref = u[(u.W_mult == 16) & (u.seed == SEEDS[0])]
    if ref.empty:
        ref = u

    def med(col):
        return float(ref[col].median()) if col in ref and ref[col].notna().any() else None

    per_pair = sum(v for v in (med("ridge_s"), med("lstm_s"), med("xgb_s")) if v)
    probe = {
        "measured_on": "grid run",
        "n_units_measured": int(len(ref)),
        "n_test_median": int(ref.n_test.median()) if "n_test" in ref else None,
        "t_bands_s": med("bands_s"),
        "t_ridge_s": med("ridge_s"),
        "t_lstm_s": med("lstm_s"),
        "t_xgb_s": med("xgb_s"),
        "t_lgb_s": med("lgb_s"),
        "t_mlp_s": med("mlp_s"),
        "t_svr_s": med("svr_s"),
        "per_pair_proposed_only_s": per_pair if per_pair else None,
        "per_pair_all_models_s": med("unit_total_s"),
        "grid_42_proposed_only_h": (per_pair * 42 / 3600) if per_pair else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    dest = REPO / "supplementary" / "statistics" / "Computational_Cost"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "runtime_probe.json", "w") as fh:
        json.dump(probe, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"  runtime probe -> {dest / 'runtime_probe.json'}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="")
    ap.add_argument("--horizons", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--w", default="", help="W multipliers, e.g. 16 or 8,16,32")
    a = ap.parse_args()
    stations = [int(x) for x in a.stations.split(",")] if a.stations else STATIONS
    horizons = [int(x) for x in a.horizons.split(",")] if a.horizons else HORIZONS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    wmults = [int(x) for x in a.w.split(",")] if a.w else W_MULTS

    done = load_done()
    bt_done = load_bandtable_done()
    total = len(stations) * len(horizons) * len(seeds) * len(wmults)
    print(f"units: {total}  already complete: {len(done)}")
    print(f"band-table cells already covered: {len(bt_done)}")

    t_start = time.time()
    n_new = 0
    for s in stations:
        fp = list(PROC_DIR.glob(f"station_{s:02d}_prepared.csv"))
        if not fp:
            print(f"[skip] station {s}: no prepared.csv")
            continue
        df = pd.read_csv(fp[0])
        sig = (pd.to_numeric(df["IRRADIATION"], errors="coerce")
               .ffill().bfill().values.astype(float))
        tod_all = ((pd.to_numeric(df["HOUR"], errors="coerce").fillna(0).values
                    * 4).astype(int) % RFC.SPD)
        doy_all = pd.to_numeric(df["DOY"], errors="coerce").fillna(1).values.astype(float)
        cache = {}
        print(f"\n=== station {s} ({len(sig):,} samples) ===", flush=True)

        for w_mult in wmults:
            for seed in seeds:
                for h in horizons:
                    if (s, f"H{h}", seed, w_mult) in done:
                        print(f"  s{s} H{h} seed{seed} W{w_mult}: done, skip", flush=True)
                        continue
                    # Cut 1: band x learner table once per (station, horizon, W)
                    need_bt = (s, f"H{h}", w_mult) not in bt_done
                    row, brows = run_unit(s, h, seed, w_mult, df, sig,
                                          tod_all, doy_all, cache,
                                          do_bandtable=need_bt)
                    if row is None:
                        continue
                    append_row(UNITS_CSV, row)
                    if brows:
                        append_rows(BAND_CSV, brows)
                        bt_done.add((s, f"H{h}", w_mult))
                    n_new += 1
                    print(f"  s{s} H{h} seed{seed} W{w_mult}: "
                          f"{row['unit_total_s']/60:.1f} min  "
                          f"skill_clim={row['fame_skill_clim']:+.4f}  "
                          f"r2={row['fame_r2']:.4f}  "
                          f"bandtable={'yes' if need_bt else 'skip'}", flush=True)

    json.dump(dict(stations=stations, horizons=horizons, seeds=seeds,
                   w_mults=wmults, seq_len=SEQ_LEN, level=RFC.LEVEL,
                   family=RFC.FAMILY, units_new=n_new,
                   elapsed_h=round((time.time() - t_start) / 3600, 3),
                   python=sys.version.split()[0],
                   numpy=np.__version__, pandas=pd.__version__,
                   xgboost=xgb.__version__),
              open(OUT_DIR / "manifest.json", "w"), indent=2)

    write_runtime_probe()

    print(f"\nnew units: {n_new}   elapsed {(time.time()-t_start)/60:.1f} min")
    print(f"outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

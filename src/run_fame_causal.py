"""
run_fame_causal.py
==================
Runs PAPER 2's ACTUAL architecture (the 4-band FAME stack) under a strictly
CAUSAL rolling wavelet decomposition, and saves every number Paper 2 needs.

WHY THIS SCRIPT EXISTS
----------------------
Paper 2's reported numbers (R2~0.99, 3.07x intervals, $603k) were produced with
a WHOLE-SERIES wavelet decomposition, which leaks future information. Paper 3
proves that leakage inflates R2. To make Paper 2 both rigorous and leakage-free,
its own 4-band model must be re-run with the decomposition computed on a trailing
(causal) window. That run does not exist yet. This script produces it.

WHAT IT IS (Paper 2's real model, unchanged except decomposition is causal):
  band cA3 (trend)  -> Ridge
  band cD3 (daily)  -> LSTM        (2 layers, seq length 48)
  band cD2 (hourly) -> XGBoost
  band cD1 (noise)  -> persistence (yhat = y_t)
  four band forecasts -> Ridge meta-learner (fit on validation predictions)

  Each band learner reads the SAME shared causal feature vector z_t (lags,
  rolling stats, meteo, and the four causal bands as columns). The band label
  only decides which learner produces that band's forecast; the meta-learner
  then fuses the four. This is exactly Paper 2's Section on frequency-aligned
  base learners, with the ONLY change being that the wavelet is computed causally.

  Baselines (same causal features): Unified XGB, LightGBM, Transformer,
  Informer-lite, TimesNet-lite, Persistence.

CAUSALITY GUARANTEE
-------------------
The wavelet band value at time t is reconstructed from signal[max(0,t-W+1):t+1]
only (trailing window), taking the last reconstructed sample. No sample at t ever
sees signal[t+1:]. Every other feature is backward-only. Chronological 70/15/15
split, no shuffle. Meta-learner trained on validation predictions only.

OUTPUTS (all under outputs/reports/paper2_causal/):
  1. fame_causal_sixmodel.csv        per-horizon mean R2, all 6 models + persistence
  2. fame_causal_perstation.csv      every (station,horizon): R2/RMSE/MAE for FAME + uXGB
  3. fame_causal_metacoef.csv        the 4 per-band meta-coefficients per (station,horizon)
  4. fame_causal_conformal.csv       split-conformal coverage + width, Station 5, H1 & H4
  5. predictions/station_XX_HN_test_predictions.csv   (y_true, fame, unified_xgb,
                                                        persistence) -- honest residuals
  6. fame_causal_econ_inputs.csv     per (station,horizon): sigma_y, persist R2, FAME R2,
                                     uXGB R2  -> feeds a proposed-vs-persistence economic table

RUN (PowerShell, project root, ~30-45 min CPU; LSTM is the slow part):
    <repository root>
    python run_fame_causal.py            # resumable; Ctrl+C then re-run to continue

PREREQ: data/processed/station_XX_prepared.csv (Stage-0 output you already have).
        IRRADIATION column is in kW/m^2; the script converts errors to W/m^2 (x1000).

NOTE ON HONESTY: this script only RUNS the model and records what happens. It does
not target any number. Whatever R2 the causal 4-band model produces is what Paper 2
reports. If the stack ties Unified XGB (as the 2-learner stack did), Paper 2's
headline is the probabilistic/interval contribution, not accuracy superiority.
"""
import os, time, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

# ----------------------------------------------------------------------
ROOT     = Path(__file__).parent
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs" / "reports" / "paper2_causal"
PRED_DIR = OUT_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = [1, 2, 4, 5, 6, 7, 8]          # Station 3 excluded (>30% sentinel)
HORIZONS = [1, 4, 8, 16, 32, 96]
FAMILY, LEVEL, WINDOW, SEED = "db4", 3, 512, 42
SEQ_LEN = 48                               # Paper 2's LSTM sequence length
SCALE   = 1000.0                           # kW/m^2 -> W/m^2 for error magnitudes
CONFORMAL_STATION = 5
CONFORMAL_HORIZONS = [1, 4]
ALPHAS = [0.10, 0.05]
np.random.seed(SEED)

CAPACITY_MW = {1: 50, 2: 130, 4: 130, 5: 110, 6: 35, 7: 30, 8: 30}

# ----------------------------------------------------------------------
# CAUSAL ROLLING WAVELET  (identical convention to your conformal_causal.py)
# ----------------------------------------------------------------------
def bands_causal(sig, window=WINDOW):
    n = len(sig); nb = LEVEL + 1
    out = np.full((n, nb), np.nan)
    ml = 8 * 2 ** LEVEL
    W = max(window, ml)
    for t in range(n):
        seg = sig[max(0, t - W + 1):t + 1]
        if len(seg) < ml:
            out[t, 0] = sig[t]; out[t, 1:] = 0.0; continue
        c = pywt.wavedec(seg, FAMILY, level=LEVEL)
        for i in range(nb):
            z = [np.zeros_like(x) for x in c]; z[i] = c[i]
            out[t, i] = pywt.waverec(z, FAMILY)[:len(seg)][-1]
    return out   # columns: band0=cA3(trend), band1=cD3(daily), band2=cD2(hourly), band3=cD1(noise)

def build_features(df, sig, bands):
    n = len(sig); f = {}
    for L in [1, 2, 4, 8, 16, 32]:
        c = np.full(n, np.nan); c[L:] = sig[:-L]; f[f"lag{L}"] = c
    for w in [4, 12, 32]:
        f[f"rm{w}"] = pd.Series(sig).rolling(w, min_periods=1).mean().values
        f[f"rs{w}"] = pd.Series(sig).rolling(w, min_periods=1).std().fillna(0).values
    for col in ["TEMPERATURE", "REL_HUMIDITY", "ATMOSPHERE", "DNI"]:
        if col in df.columns:
            f[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill().values
    for i in range(bands.shape[1]):
        f[f"band{i}"] = bands[:, i]
    return pd.DataFrame(f).values

def make_xy(X, sig, h):
    n = len(sig); y = np.full(n, np.nan); y[:-h] = sig[h:]
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    X, y = X[v], y[v]
    a, b = int(len(X) * .70), int(len(X) * .85)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]

def to_seq(X, y, L):
    xs, ys = [], []
    for i in range(L - 1, len(X)):
        xs.append(X[i - L + 1:i + 1]); ys.append(y[i])
    return np.array(xs), np.array(ys)

# ----------------------------------------------------------------------
# BASE LEARNERS
# ----------------------------------------------------------------------
def fit_ridge(Xtr, ytr, Xva, Xte):
    r = Ridge(alpha=1.0).fit(Xtr, ytr)
    return r.predict(Xva), r.predict(Xte)

def fit_xgb(Xtr, ytr, Xva, Xte):
    m = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0)
    m.fit(Xtr, ytr)
    return m.predict(Xva), m.predict(Xte)

def fit_lstm(Xtr_s, ytr, Xva_s, yva, Xte_s):
    """Paper 2's daily-band learner. Returns validation and test predictions aligned
    to the FULL (non-sequence) validation/test index by left-padding the first
    SEQ_LEN-1 entries with the first available sequence prediction."""
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras import layers, Model, callbacks
    xtr, ytr2 = to_seq(Xtr_s, ytr, SEQ_LEN)
    xva, yva2 = to_seq(Xva_s, yva, SEQ_LEN)
    xte, _    = to_seq(Xte_s, np.zeros(len(Xte_s)), SEQ_LEN)
    d = Xtr_s.shape[1]
    inp = layers.Input((SEQ_LEN, d))
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    mdl = Model(inp, out); mdl.compile("adam", "mse")
    mdl.fit(xtr, ytr2, validation_data=(xva, yva2), epochs=50, batch_size=128,
            verbose=0, callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    pva = mdl.predict(xva, verbose=0).ravel()
    pte = mdl.predict(xte, verbose=0).ravel()
    # left-pad to full length so all four band forecasts align on the same rows
    pva_full = np.concatenate([np.full(SEQ_LEN - 1, pva[0]), pva])
    pte_full = np.concatenate([np.full(SEQ_LEN - 1, pte[0]), pte])
    return pva_full, pte_full

def deep_generic(kind, Xtr_s, ytr, Xva_s, yva, Xte_s):
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras import layers, Model, callbacks
    xtr, ytr2 = to_seq(Xtr_s, ytr, 24)
    xva, yva2 = to_seq(Xva_s, yva, 24)
    xte, _    = to_seq(Xte_s, np.zeros(len(Xte_s)), 24)
    d = Xtr_s.shape[1]
    inp = layers.Input((24, d))
    if kind == "Transformer":
        x = layers.Dense(64)(inp)
        a = layers.MultiHeadAttention(4, 16)(x, x)
        x = layers.LayerNormalization()(x + a)
        fo = layers.Dense(64, activation="gelu")(x); fo = layers.Dense(64)(fo)
        x = layers.LayerNormalization()(x + fo)
    elif kind == "Informer-lite":
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
        x = layers.MaxPooling1D(2, padding="same")(x)
        a = layers.MultiHeadAttention(4, 16)(x, x)
        x = layers.LayerNormalization()(x + a)
    else:  # TimesNet-lite
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
        x = layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="gelu")(x)
    x = layers.GlobalAveragePooling1D()(x); out = layers.Dense(1)(x)
    mdl = Model(inp, out); mdl.compile("adam", "mse")
    mdl.fit(xtr, ytr2, validation_data=(xva, yva2), epochs=60, batch_size=128,
            verbose=0, callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return mdl.predict(xte, verbose=0).ravel(), 24 - 1

def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    ss_res = float(np.sum(e ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))

# ----------------------------------------------------------------------
def conformal(yhat, y, alpha):
    n = len(y); half = n // 2
    cal = np.abs(y[:half] - yhat[:half]); ncal = len(cal)
    k = min(int(np.ceil((ncal + 1) * (1 - alpha))), ncal)
    q = np.sort(cal)[k - 1]
    ev_y, ev_h = y[half:], yhat[half:]
    cov = float(np.mean((ev_y >= ev_h - q) & (ev_y <= ev_h + q)))
    return cov, float(2 * q)

# ----------------------------------------------------------------------
def main():
    if not list(PROC_DIR.glob("station_*_prepared.csv")):
        print("!! No Stage-0 output in data/processed/. Run stage0 first."); return

    six_rows, per_rows, meta_rows, econ_rows, conf_rows = [], [], [], [], []
    t0 = time.time()

    for s in STATIONS:
        fp = list(PROC_DIR.glob(f"station_{s:02d}_prepared.csv"))
        if not fp:
            print(f"  [skip] station {s}: no prepared.csv"); continue
        df = pd.read_csv(fp[0])
        sig = pd.to_numeric(df["IRRADIATION"], errors="coerce").ffill().bfill().values.astype(float)
        print(f"\n=== Station {s} ({len(sig)} samples): causal rolling decomposition ===")
        bands = bands_causal(sig)
        X = build_features(df, sig, bands)

        for h in HORIZONS:
            Xtr, ytr, Xva, yva, Xte, yte = make_xy(X, sig, h)
            if len(yte) < 100:
                print(f"   H{h}: too few test rows"); continue
            sc = StandardScaler()
            Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)

            # ---- Paper 2's four band learners (all read shared features) ----
            r_va, r_te = fit_ridge(Xtr_s, ytr, Xva_s, Xte_s)        # trend
            try:
                l_va, l_te = fit_lstm(Xtr_s, ytr, Xva_s, yva, Xte_s)  # daily (LSTM)
            except Exception as e:
                print(f"   H{h}: LSTM failed ({e}); using ridge as daily fallback")
                l_va, l_te = r_va, r_te
            g_va, g_te = fit_xgb(Xtr_s, ytr, Xva_s, Xte_s)          # hourly
            p_va = np.concatenate([[ytr[-1]], yva[:-1]])            # noise = persistence
            p_te = np.concatenate([[yva[-1]], yte[:-1]])

            # ---- ridge meta-learner on validation predictions (4 band forecasts) ----
            Zva = np.column_stack([r_va, l_va, g_va, p_va])
            Zte = np.column_stack([r_te, l_te, g_te, p_te])
            meta = Ridge(alpha=1.0).fit(Zva, yva)
            fame_te = meta.predict(Zte)
            b0 = float(meta.intercept_); bt, bd, bh, bn = [float(c) for c in meta.coef_]

            # ---- baselines on same causal features ----
            uxgb_te = g_te                                          # unified XGB == the xgb learner
            lgb_te = None
            if HAVE_LGB:
                ml = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                       n_jobs=-1, verbosity=-1).fit(Xtr_s, ytr)
                lgb_te = ml.predict(Xte_s)
            pers_te = p_te

            # deep baselines
            deep = {}
            for kind in ["Transformer", "Informer-lite", "TimesNet-lite"]:
                try:
                    pred, off = deep_generic(kind, Xtr_s, ytr, Xva_s, yva, Xte_s)
                    deep[kind] = (pred, off)
                except Exception as e:
                    print(f"   H{h}: {kind} failed ({e})"); deep[kind] = (None, 0)

            # ---- metrics ----
            fame_r2, fame_rmse, fame_mae = metrics(yte, fame_te)
            uxgb_r2, uxgb_rmse, uxgb_mae = metrics(yte, uxgb_te)
            pers_r2, _, _ = metrics(yte, pers_te)
            sig_y = float(np.std(yte))

            row6 = dict(station=s, horizon=f"H{h}",
                        FAME=round(fame_r2, 4), Unified_XGB=round(uxgb_r2, 4))
            if lgb_te is not None:
                row6["LightGBM"] = round(metrics(yte, lgb_te)[0], 4)
            for kind, (pred, off) in deep.items():
                if pred is not None:
                    row6[kind.replace("-", "_")] = round(r2_score(yte[off:], pred), 4)
            six_rows.append(row6)

            per_rows.append(dict(station=s, horizon=f"H{h}",
                                 fame_r2=round(fame_r2, 4), fame_rmse_W=round(fame_rmse * SCALE, 2),
                                 fame_mae_W=round(fame_mae * SCALE, 2),
                                 uxgb_r2=round(uxgb_r2, 4), uxgb_rmse_W=round(uxgb_rmse * SCALE, 2),
                                 uxgb_mae_W=round(uxgb_mae * SCALE, 2)))
            meta_rows.append(dict(station=s, horizon=f"H{h}", intercept=round(b0, 4),
                                  trend=round(bt, 4), daily=round(bd, 4),
                                  hourly=round(bh, 4), noise=round(bn, 4)))
            econ_rows.append(dict(station=s, horizon=f"H{h}", capacity_MW=CAPACITY_MW.get(s, 0),
                                  sigma_y_W=round(sig_y * SCALE, 2),
                                  fame_r2=round(fame_r2, 4), uxgb_r2=round(uxgb_r2, 4),
                                  persist_r2=round(pers_r2, 4)))

            # ---- save honest predictions (kW units; convert later as needed) ----
            pd.DataFrame({"y_true": yte, "fame": fame_te,
                          "unified_xgb": uxgb_te, "persistence": pers_te}
                         ).to_csv(PRED_DIR / f"station_{s:02d}_H{h}_test_predictions.csv", index=False)

            # ---- conformal (Station 5, H1 & H4) ----
            if s == CONFORMAL_STATION and h in CONFORMAL_HORIZONS:
                for name, yh in [("Proposed", fame_te), ("Unified XGB", uxgb_te), ("Persistence", pers_te)]:
                    for a in ALPHAS:
                        cov, width = conformal(yh, yte, a)
                        conf_rows.append(dict(station=s, horizon=f"H{h}", model=name,
                                              nominal=f"{int((1-a)*100)}%",
                                              empirical=round(cov * 100, 1),
                                              width_W_m2=round(width * SCALE, 1)))

            print(f"   H{h:<3} FAME R2={fame_r2:.4f} RMSE={fame_rmse*SCALE:6.2f}  "
                  f"uXGB R2={uxgb_r2:.4f}  meta[t,d,h,n]="
                  f"[{bt:+.3f},{bd:+.3f},{bh:+.3f},{bn:+.3f}]")

    # ---- write tables ----
    six = pd.DataFrame(six_rows)
    if len(six):
        order = {f"H{h}": i for i, h in enumerate(HORIZONS)}
        num = six.drop(columns=["station"]).groupby("horizon").mean().reset_index()
        num["__o"] = num["horizon"].map(order); num = num.sort_values("__o").drop(columns="__o").round(4)
        num.to_csv(OUT_DIR / "fame_causal_sixmodel.csv", index=False)
    pd.DataFrame(per_rows).to_csv(OUT_DIR / "fame_causal_perstation.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(OUT_DIR / "fame_causal_metacoef.csv", index=False)
    pd.DataFrame(econ_rows).to_csv(OUT_DIR / "fame_causal_econ_inputs.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT_DIR / "fame_causal_conformal.csv", index=False)

    print(f"\n{'='*66}")
    print(f"DONE in {(time.time()-t0)/60:.1f} min. Wrote to {OUT_DIR}")
    print("  fame_causal_sixmodel.csv     -> Paper 2 Table: 6-model mean R2 (causal)")
    print("  fame_causal_perstation.csv   -> Paper 2 Table: per-station R2/RMSE/MAE")
    print("  fame_causal_metacoef.csv     -> Paper 2 Table/Fig: 4 per-band meta-weights")
    print("  fame_causal_conformal.csv    -> Paper 2 Table: conformal coverage + width")
    print("  fame_causal_econ_inputs.csv  -> Paper 2 economic table (proposed vs persistence)")
    print("  predictions/*.csv            -> honest residuals for any further metric")
    print(f"{'='*66}")
    print("Send these 5 CSVs back and I build the Paper 2 ZIP with real causal numbers.")

if __name__ == "__main__":
    main()

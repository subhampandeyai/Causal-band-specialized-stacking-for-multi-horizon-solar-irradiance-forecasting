"""
s10_seed_sensitivity.py
=======================
CHEAP SEED-SENSITIVITY CHECK (reviewers R1, R2, R3, R7).

Purpose: establish which components of the frozen pipeline are seed-sensitive
and which are deterministic-given-data, WITHOUT retraining the LSTM.

What is executed
----------------
  1. Unified XGBoost and LightGBM are refit on the IDENTICAL data, features,
     split and hyperparameters with three seeds (42, 0, 2024) on one
     representative pair (Station 5, H1), and the spread of skill / RMSE / R2
     across those refits is measured.
  2. Ridge and persistence are checked the same way; both are expected to be
     exactly invariant, and that expectation is verified rather than asserted.
  3. The stochastic sites in the frozen code are enumerated by reading
     run_fame_causal.py, so the claim about which components consume a seed is
     traceable to specific lines.

The LSTM is NOT retrained. Its cost is quoted from the measured runtime probe.

Everything runs on a working copy of the pipeline logic; the frozen root is
read-only and is never written.

Outputs -> Supplementary_Analysis/Seed_Sensitivity/*.csv/.json
           Seed_Sensitivity/seed_sensitivity_note.md
"""
import sys, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s10_seed_sensitivity")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False

PROC = rc.PAPER / "code" / "data" / "processed"
OUTD = rc.OUT / "Seed_Sensitivity"
OUTD.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 2024]
STATION, HORIZON_STEPS, HORIZON = 5, 1, "H1"
FAMILY, LEVEL, WINDOW = "db4", 3, 512


# ---- frozen pipeline pieces, copied verbatim so the check tests SHIPPED logic --
def bands_causal(sig, family=FAMILY, level=LEVEL, window=WINDOW):
    """run_fame_causal.py lines 99-112, unchanged."""
    import pywt
    n = len(sig); nb = level + 1
    out = np.full((n, nb), np.nan)
    ml = 8 * 2 ** level
    W = max(window, ml)
    for t in range(n):
        seg = sig[max(0, t - W + 1):t + 1]
        if len(seg) < ml:
            out[t, 0] = sig[t]; out[t, 1:] = 0.0; continue
        c = pywt.wavedec(seg, family, level=level)
        for i in range(nb):
            z = [np.zeros_like(x) for x in c]; z[i] = c[i]
            out[t, i] = pywt.waverec(z, family)[:len(seg)][-1]
    return out


def build_features(df, sig, bands):
    """run_fame_causal.py lines 114-126, unchanged."""
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
    """run_fame_causal.py lines 128-133, unchanged."""
    n = len(sig); y = np.full(n, np.nan); y[:-h] = sig[h:]
    v = ~np.isnan(y) & ~np.isnan(X).any(1)
    X, y = X[v], y[v]
    a, b = int(len(X) * .70), int(len(X) * .85)
    return X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:]


def set_all_seeds(s: int):
    """Set every seed source the pipeline could consume."""
    random.seed(s)
    np.random.seed(s)


def main():
    t0 = time.time()
    LOG.info("=" * 74)
    LOG.info("SEED-SENSITIVITY CHECK (no LSTM retraining)")
    LOG.info(f"pair: Station {STATION}, {HORIZON}; seeds {SEEDS}")
    LOG.info("=" * 74)

    fp = PROC / f"station_{STATION:02d}_prepared.csv"
    if not fp.exists():
        LOG.error(f"missing {fp}"); return
    df = pd.read_csv(fp)
    sig = pd.to_numeric(df["IRRADIATION"], errors="coerce").ffill().bfill().values.astype(float)

    LOG.info("computing causal bands (once; deterministic, no seed)...")
    B = bands_causal(sig)
    X = build_features(df, sig, B)
    Xtr, ytr, Xva, yva, Xte, yte = make_xy(X, sig, HORIZON_STEPS)
    sc = StandardScaler()
    Xtr_s, Xva_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    pers_te = np.concatenate([[yva[-1]], yte[:-1]])
    LOG.info(f"  n_train={len(ytr)} n_val={len(yva)} n_test={len(yte)} "
             f"n_features={X.shape[1]}")

    rows, preds = [], {}

    for s in SEEDS:
        set_all_seeds(s)

        # --- Ridge: no seed parameter at all ---
        r = Ridge(alpha=1.0).fit(Xtr_s, ytr)
        p_ridge = r.predict(Xte_s)

        # --- unified XGBoost: EXACT frozen hyperparameters (subsample/colsample<1) ---
        m = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=s, n_jobs=-1, tree_method="hist",
                             verbosity=0)
        m.fit(Xtr_s, ytr)
        p_xgb = m.predict(Xte_s)

        # --- LightGBM: EXACT frozen hyperparameters ---
        p_lgb = None
        if HAVE_LGB:
            ml = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=s, n_jobs=-1, verbosity=-1)
            ml.fit(Xtr_s, ytr)
            p_lgb = ml.predict(Xte_s)

        for name, p in [("Ridge (trend expert)", p_ridge),
                        ("Unified XGBoost", p_xgb),
                        ("LightGBM", p_lgb),
                        ("Persistence", pers_te)]:
            if p is None:
                continue
            preds.setdefault(name, {})[s] = np.asarray(p, float)
            rows.append(dict(model=name, seed=s,
                             skill=rc.skill_score(yte, p, pers_te),
                             rmse_W=rc.rmse(yte, p) * rc.SCALE,
                             mae_W=rc.mae(yte, p) * rc.SCALE,
                             r2=rc.r2(yte, p)))
        LOG.info(f"  seed {s:5d} done")

    R = pd.DataFrame(rows)
    rc.save_table(R.round(10), "Seed_Sensitivity", "per_seed_metrics")

    # ---------------- spread across seeds ----------------
    summ = []
    for name, g in R.groupby("model"):
        pv = preds[name]
        stack = np.vstack([pv[s] for s in SEEDS if s in pv])
        max_pred_delta = float(np.max(np.abs(stack - stack[0]))) if len(stack) > 1 else 0.0
        summ.append(dict(
            model=name, n_seeds=int(g.seed.nunique()),
            mean_skill=float(g.skill.mean()),
            max_abs_delta_skill=float(g.skill.max() - g.skill.min()),
            max_abs_delta_rmse_W=float(g.rmse_W.max() - g.rmse_W.min()),
            max_abs_delta_r2=float(g.r2.max() - g.r2.min()),
            max_abs_delta_prediction=max_pred_delta,
            deterministic_given_data=bool(max_pred_delta == 0.0),
            seed_param_in_frozen_code=("no - Ridge takes no random_state"
                                       if "Ridge" in name else
                                       "no - persistence is a deterministic shift"
                                       if "Persistence" in name else
                                       "yes - random_state, with subsample=0.8 "
                                       "and colsample_bytree=0.8")))
    S = pd.DataFrame(summ).sort_values("max_abs_delta_skill", ascending=False)
    rc.save_table(S.round(12), "Seed_Sensitivity", "seed_spread_summary")

    LOG.info("\nSPREAD ACROSS SEEDS (Station 5, H1):\n" +
             S[["model", "n_seeds", "mean_skill", "max_abs_delta_skill",
                "max_abs_delta_rmse_W", "max_abs_delta_prediction",
                "deterministic_given_data"]].to_string(index=False))

    # ---------------- stochastic-site inventory (read from frozen code) ------
    src = (rc.PAPER / "code" / "run_fame_causal.py").read_text(encoding="utf-8",
                                                               errors="replace")
    lines = src.splitlines()
    sites = []
    for i, ln in enumerate(lines, 1):
        for pat, what in [("np.random.seed", "numpy global seed"),
                          ("tf.random.set_seed", "TensorFlow graph-level seed"),
                          ("random_state=SEED", "sklearn/xgboost/lightgbm seed")]:
            if pat in ln:
                sites.append(dict(line=i, code=ln.strip()[:110], site=what))
    SITES = pd.DataFrame(sites)
    rc.save_table(SITES, "Seed_Sensitivity", "stochastic_sites_in_frozen_code")
    LOG.info(f"\nseed-consuming sites in run_fame_causal.py: {len(SITES)}")
    for _, r in SITES.iterrows():
        LOG.info(f"  line {r.line:4d}  {r.site}")

    # ---------------- LSTM cost, quoted from the measured probe ------------
    probe = (rc.OUT /
             "Computational_Cost" / "runtime_probe.json")
    lstm_cost = None
    if probe.exists():
        try:
            lstm_cost = float(json.loads(probe.read_text())["t_lstm_s"])
        except Exception:
            pass

    meta = dict(pair=f"Station {STATION} {HORIZON}", seeds=SEEDS,
                n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)),
                n_features=int(X.shape[1]),
                lstm_measured_s_per_pair=lstm_cost,
                lstm_8seed_grid_hours=(lstm_cost * 42 * 8 / 3600 if lstm_cost else None),
                lstm_retrained="NO - explicitly excluded from this check",
                elapsed_s=round(time.time() - t0, 1))
    (OUTD / "check_metadata.json").write_text(json.dumps(meta, indent=2))

    # ---------------- the note ----------------
    det = S[S.deterministic_given_data]
    nondet = S[~S.deterministic_given_data]
    md = ["# Seed-sensitivity note", "",
          "**Scope.** Measured on the frozen pipeline logic, at "
          f"**Station {STATION}, {HORIZON}**, with seeds **{SEEDS}**. "
          "Every model was refit on the identical data, features, chronological "
          "split and hyperparameters; only the seed changed. "
          "**The LSTM was not retrained** — see the cost note below.", "",
          f"Elapsed for this check: **{meta['elapsed_s']:.0f} s**.", "",
          "## Measured spread across three seeds", "",
          "| Model | max abs Δ skill | max abs Δ RMSE (W/m²) | max abs Δ prediction | deterministic given data? |",
          "|---|---|---|---|---|"]
    for _, r in S.iterrows():
        md.append(f"| {r.model} | {r.max_abs_delta_skill:.3e} | "
                  f"{r.max_abs_delta_rmse_W:.3e} | {r.max_abs_delta_prediction:.3e} | "
                  f"{'**yes**' if r.deterministic_given_data else '**no**'} |")
    md += ["", "## Which components consume a seed", "",
           "Read directly from `code/run_fame_causal.py`:", ""]
    for _, r in SITES.iterrows():
        md.append(f"- line {r.line}: {r.site} — `{r.code}`")
    md += ["", "### Per model", "",
           "| Component | Stochastic? | Why |",
           "|---|---|---|",
           "| Ridge (trend expert) | **No** | `Ridge(alpha=1.0)` takes no `random_state`; "
           "the solution is the closed-form ridge estimator |",
           "| Persistence (noise expert) | **No** | a deterministic one-step shift of the target |",
           "| Ridge meta-learner | **No** | same closed form, fitted on validation predictions |",
           "| Unified XGBoost | **Yes, nominally** | `random_state=SEED` with "
           "`subsample=0.8`, `colsample_bytree=0.8`, so row/column sampling is seeded |",
           "| LightGBM | **Yes, nominally** | same construction |",
           "| **LSTM (daily expert)** | **Yes** | `tf.random.set_seed(SEED)`; weight "
           "initialisation, dropout masks and batch shuffling are all stochastic |",
           "| Transformer / Informer-lite / TimesNet-lite | **Yes** | same, via "
           "`tf.random.set_seed(SEED)` |", ""]

    if len(nondet):
        md += ["## Result", "",
               "The tree models are **not** exactly seed-invariant — subsampling is "
               "seeded, so different seeds draw different rows and columns. The "
               "measured spread is reported above and is the honest figure to quote.", ""]
    if len(det):
        md += [f"Exactly invariant (max |Δ| = 0 across all three seeds): "
               f"{', '.join(det.model)}.", ""]

    if lstm_cost:
        md += ["## The LSTM, and why the full grid was not run", "",
               f"The LSTM daily-band expert is the only component whose cost makes a "
               f"multi-seed grid expensive. Measured on this machine "
               f"(CPU-only, no CUDA GPU): **{lstm_cost:.1f} s per "
               f"(station, horizon) pair**.", "",
               f"A full 8-seed grid over 7 stations × 6 horizons would therefore cost",
               f"**{lstm_cost:.1f} s × 42 pairs × 8 seeds ≈ "
               f"{lstm_cost*42*8/3600:.1f} hours "
               f"({lstm_cost*42*8/86400:.1f} days)** of continuous computation, "
               f"which is why it was not run here.", "",
               "That figure is a measurement from "
               "`Computational_Cost/"
               "runtime_probe.json`, not an estimate.", "",
               "**Consequence for the manuscript.** The seed limitation is confined to "
               "the LSTM band expert and the three deep baselines. The trend expert, "
               "the persistence expert and the meta-learner are deterministic given "
               "the data, and the tree models vary only by the amount measured above. "
               "Any statement about seed robustness should be scoped to the recurrent "
               "and attention components rather than to the framework as a whole.", ""]

    md += ["## Provenance", "",
           "- Pipeline logic copied verbatim from `code/run_fame_causal.py` "
           "(`bands_causal`, `build_features`, `make_xy`, and the exact learner "
           "hyperparameters).",
           "- The frozen implementation was read only; it was never "
           "modified.",
           "- Raw per-seed metrics: "
           "`Supplementary_Analysis/Seed_Sensitivity/per_seed_metrics.csv`.", ""]

    (rc.PKG / "seed_sensitivity_note.md").write_text("\n".join(md), encoding="utf-8")
    LOG.info(f"\nwrote {rc.PKG/'seed_sensitivity_note.md'}")
    LOG.info(f"elapsed {meta['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()

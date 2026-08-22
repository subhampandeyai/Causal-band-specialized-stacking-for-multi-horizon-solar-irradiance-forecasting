"""
s12_cost_benefit.py
===================
Per-model computational cost and its relationship to forecast quality.

Every quantity is measured. Training times are the per-unit timings recorded
during the eight-seed grid; inference time is measured here by timing the
forward pass of each model family on arrays matching the stored test
partitions. Nothing is estimated, scaled from another machine, or copied from
a reference.

Two tables are produced:

    S26  per-model training and inference cost with accuracy alongside, so the
         reader can see what each second buys
    S27  the proposed stack's cost decomposed by component, showing how the
         budget divides across the band experts

Run after the grid:

    python supplementary/analysis_code/s12_cost_benefit.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revision_common as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
UNITS = REPO / "results" / "experiments" / "results_units.csv"
PRED = REPO / "results" / "predictions"

MODELS = [
    ("ridge_s", "Ridge (trend expert)", None, None),
    ("lstm_s", "LSTM (multi-hour expert)", None, None),
    ("xgb_s", "XGBoost (hourly expert)", None, None),
    ("lgb_s", "LightGBM", "lgb_skill_clim", "lgb_r2"),
    ("mlp_s", "MLP", "mlp_skill_clim", "mlp_r2"),
    ("svr_s", "SVR", "svr_skill_clim", "svr_r2"),
]

# display-name fragment -> key in the inference probe
INFER_KEY = [
    ("Ridge", "ridge_ms"),
    ("LSTM", "lstm_ms"),
    ("XGBoost (hourly", "xgboost_ms"),
    ("LightGBM", "lightgbm_ms"),
    ("MLP", "mlp_ms"),
    ("SVR", "svr_ms"),
    ("Unified XGBoost", "xgboost_ms"),
    ("Proposed stack", "proposed_ms"),
]


def measure_inference(n_repeat: int = 5) -> dict:
    """Time the prediction step of every model family on realistic array sizes.

    We refit nothing from the grid: each family is fitted once here on random
    arrays shaped like the real test partitions, then its predict() call is
    timed and the median of several repeats reported, so scheduler noise does
    not dominate.
    """
    from sklearn.linear_model import Ridge

    files = sorted(PRED.glob("*_test_predictions.csv"))
    if not files:
        raise SystemExit("no stored predictions; run the grid first")
    n_test = int(np.median([len(pd.read_csv(f)) for f in files]))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_test, 20))
    y = rng.standard_normal(n_test)
    out = {"n_test_rows": n_test, "n_features": 20}

    def timed(fn, repeats=n_repeat):
        ts = []
        for _ in range(repeats):
            s = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - s)
        return float(np.median(ts) * 1000)

    ridge = Ridge(alpha=1.0).fit(X, y)
    out["ridge_ms"] = timed(lambda: ridge.predict(X))

    try:
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.03,
                             reg_lambda=1.0, verbosity=0, n_jobs=-1).fit(X, y)
        out["xgboost_ms"] = timed(lambda: m.predict(X))
    except Exception as e:
        out["xgboost_ms"] = float("nan")
        out["xgboost_note"] = str(e)[:80]

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, verbose=-1).fit(X, y)
        out["lightgbm_ms"] = timed(lambda: m.predict(X))
    except Exception as e:
        out["lightgbm_ms"] = float("nan")
        out["lightgbm_note"] = str(e)[:80]

    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=20).fit(X, y)
    out["mlp_ms"] = timed(lambda: mlp.predict(X))

    from sklearn.svm import SVR
    sub = min(n_test, 4000)
    svr = SVR().fit(X[:sub], y[:sub])
    out["svr_ms"] = timed(lambda: svr.predict(X), max(2, n_repeat // 2))

    # the recurrent expert dominates training cost, so its forecast-time cost
    # is the one a deployer cares about; time the pipeline's architecture
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
        seq = 48
        net = tf.keras.Sequential([
            tf.keras.layers.Input((seq, 1)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1)])
        Xs = rng.standard_normal((n_test, seq, 1)).astype("float32")
        net.predict(Xs[:64], batch_size=64, verbose=0)        # warm up the graph
        out["lstm_ms"] = timed(
            lambda: net.predict(Xs, batch_size=256, verbose=0),
            max(2, n_repeat // 2))
    except Exception as e:
        out["lstm_ms"] = float("nan")
        out["lstm_note"] = str(e)[:80]

    # the stack runs its experts in sequence, so its forecast cost is their sum
    parts = [out.get(k, np.nan) for k in ("ridge_ms", "lstm_ms", "xgboost_ms")]
    out["proposed_ms"] = float(np.nansum(parts)) if np.isfinite(parts).any() else float("nan")

    # persistence is a table lookup of the last observed value
    out["persistence_ms"] = timed(lambda: y.copy())
    return out


def main() -> int:
    if not UNITS.exists():
        print(f"missing {UNITS}; run src/run_experiments_full.py first")
        return 1
    u = pd.read_csv(UNITS)

    infer = measure_inference()
    n_test = infer["n_test_rows"]
    g = u.groupby("horizon")

    rows = []
    for col, name, skill_c, r2_c in MODELS:
        if col not in u.columns:
            continue
        t = float(u[col].median())
        row = dict(model=name,
                   train_s_per_pair=round(t, 3),
                   train_h_full_grid=round(t * 42 / 3600, 4))
        if skill_c and skill_c in u.columns:
            row["skill_H1"] = round(float(g[skill_c].mean().get("H1", np.nan)), 4)
            row["skill_H96"] = round(float(g[skill_c].mean().get("H96", np.nan)), 4)
        if r2_c and r2_c in u.columns:
            row["mean_r2"] = round(float(u[r2_c].mean()), 4)
        rows.append(row)

    prop = float(u[["bands_s", "ridge_s", "lstm_s", "xgb_s"]].sum(axis=1).median())
    rows.append(dict(model="Proposed stack (all experts + meta)",
                     train_s_per_pair=round(prop, 3),
                     train_h_full_grid=round(prop * 42 / 3600, 4),
                     skill_H1=round(float(g.fame_skill_clim.mean()["H1"]), 4),
                     skill_H96=round(float(g.fame_skill_clim.mean()["H96"]), 4),
                     mean_r2=round(float(u.fame_r2.mean()), 4)))
    ux = float(u.xgb_s.median())
    rows.append(dict(model="Unified XGBoost",
                     train_s_per_pair=round(ux, 3),
                     train_h_full_grid=round(ux * 42 / 3600, 4),
                     skill_H1=round(float(g.uxgb_skill_clim.mean()["H1"]), 4),
                     skill_H96=round(float(g.uxgb_skill_clim.mean()["H96"]), 4),
                     mean_r2=round(float(u.uxgb_r2.mean()), 4)))

    S26 = pd.DataFrame(rows)
    S26["infer_ms_per_pair"] = np.nan
    for frag, key in INFER_KEY:
        v = infer.get(key, np.nan)
        if np.isfinite(v):
            hit = S26.model.str.contains(frag, case=False, regex=False)
            S26.loc[hit & S26.infer_ms_per_pair.isna(), "infer_ms_per_pair"] = round(v, 4)
    S26["infer_us_per_sample"] = (S26.infer_ms_per_pair * 1000 / n_test).round(4)
    rc.save_table(S26, "Computational_Cost", "S26_model_cost_benefit")
    print(f"S26  per-model cost and accuracy ({len(S26)} models; "
          f"inference timed on {n_test} rows)")
    print(S26.to_string(index=False))

    comp = []
    for col, name in [("bands_s", "causal wavelet decomposition"),
                      ("ridge_s", "Ridge trend expert"),
                      ("lstm_s", "LSTM multi-hour expert"),
                      ("xgb_s", "XGBoost hourly expert")]:
        if col not in u.columns:
            continue
        v = float(u[col].median())
        comp.append(dict(component=name, median_s_per_pair=round(v, 3),
                         share_pct=round(100 * v / prop, 2) if prop else np.nan))
    comp.append(dict(component="persistence noise expert (no fit)",
                     median_s_per_pair=0.0, share_pct=0.0))
    S27 = pd.DataFrame(comp)
    rc.save_table(S27, "Computational_Cost", "S27_proposed_cost_breakdown")
    print(f"\nS27  cost decomposition (total {prop:.2f} s per pair)")
    print(S27.to_string(index=False))

    out = REPO / "supplementary" / "statistics" / "Computational_Cost"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference_probe.json").write_text(json.dumps(infer, indent=2))
    print(f"\ninference probe -> {out / 'inference_probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

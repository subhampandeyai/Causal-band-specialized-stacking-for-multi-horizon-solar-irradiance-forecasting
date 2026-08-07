"""
s11_tree_seed_grid.py
=====================
FULL-GRID SEED SPREAD for the two stochastic TREE learners only.

Extends s10 (which measured one pair) to all 7 stations x 6 horizons, so the
tree seed noise is bounded across the whole grid rather than at a single point.

Scope, deliberately narrow:
  * Unified XGBoost and LightGBM only - the two learners whose frozen
    hyperparameters include subsample=0.8 / colsample_bytree=0.8, making
    random_state consequential.
  * Seeds [42, 0, 2024]. Seed 42 first so it ties back to the published run.
  * The LSTM and the three attention baselines are NOT trained. Nothing in this
    script instantiates TensorFlow.

Everything else is held identical to the published run: same 7 admitted
stations, same chronological 70/15/15 split, same causal trailing-window
decomposition, same feature construction, same hyperparameters, same
skill-score-against-persistence metric.

The frozen implementation is read only; it is never modified.

Outputs -> Supplementary_Analysis/Seed_Sensitivity/tree_seed_spread.csv
           Supplementary_Analysis/Seed_Sensitivity/tree_seed_raw.csv
           Supplementary_Analysis/Seed_Sensitivity/tree_seed_model_summary.csv
"""
import sys, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s11_tree_seed_grid")

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
    LOG.warning("lightgbm unavailable - it will be reported as NOT RUN, never estimated")

PROC = rc.REPO / "data" / "processed"
OUTD = rc.OUT / "Seed_Sensitivity"
OUTD.mkdir(parents=True, exist_ok=True)
CKPT = OUTD / "_tree_seed_checkpoint.csv"

SEEDS = [42, 0, 2024]
HORIZON_STEPS = {"H1": 1, "H4": 4, "H8": 8, "H16": 16, "H32": 32, "H96": 96}
FAMILY, LEVEL, WINDOW = "db4", 3, 512


# ---- frozen pipeline pieces, verbatim from code/run_fame_causal.py ----------
def bands_causal(sig, family=FAMILY, level=LEVEL, window=WINDOW):
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


def main():
    t0 = time.time()
    LOG.info("=" * 74)
    LOG.info("TREE-ONLY SEED SPREAD - full 7 station x 6 horizon grid")
    LOG.info(f"seeds {SEEDS}; models: Unified XGBoost"
             f"{' + LightGBM' if HAVE_LGB else ' (LightGBM UNAVAILABLE)'}")
    LOG.info("NO LSTM, NO attention baselines - TensorFlow is never imported")
    LOG.info("=" * 74)

    rows, failures = [], []
    if CKPT.exists():
        try:
            prev = pd.read_csv(CKPT)
            rows = prev.to_dict("records")
            LOG.info(f"resuming: {len(rows)} rows already computed")
        except Exception:
            rows = []
    done = {(int(r["station"]), str(r["horizon"]), int(r["seed"]), str(r["model"]))
            for r in rows}

    n_total = len(rc.STATIONS) * len(HORIZON_STEPS) * len(SEEDS) * (2 if HAVE_LGB else 1)

    for s in rc.STATIONS:
        fp = PROC / f"station_{s:02d}_prepared.csv"
        if not fp.exists():
            failures.append(dict(station=s, horizon="ALL", seed=None, model="ALL",
                                 error=f"missing {fp.name}"))
            LOG.error(f"station {s}: {fp.name} not found - SKIPPED, not estimated")
            continue
        df = pd.read_csv(fp)
        sig = pd.to_numeric(df["IRRADIATION"], errors="coerce") \
                .ffill().bfill().values.astype(float)
        # bands are deterministic and seed-free: compute once per station
        B = bands_causal(sig)
        X = build_features(df, sig, B)
        LOG.info(f"station {s}: n={len(sig)}, features={X.shape[1]}")

        for hname, hsteps in HORIZON_STEPS.items():
            try:
                Xtr, ytr, Xva, yva, Xte, yte = make_xy(X, sig, hsteps)
            except Exception as e:
                failures.append(dict(station=s, horizon=hname, seed=None, model="ALL",
                                     error=f"make_xy: {type(e).__name__}: {e}"))
                continue
            if len(yte) < 100:
                failures.append(dict(station=s, horizon=hname, seed=None, model="ALL",
                                     error=f"only {len(yte)} test rows"))
                continue

            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            pers_te = np.concatenate([[yva[-1]], yte[:-1]])

            for seed in SEEDS:
                random.seed(seed); np.random.seed(seed)

                if (s, hname, seed, "Unified XGBoost") not in done:
                    try:
                        m = xgb.XGBRegressor(
                            n_estimators=800, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=seed, n_jobs=-1, tree_method="hist",
                            verbosity=0)
                        m.fit(Xtr_s, ytr)
                        p = m.predict(Xte_s)
                        rows.append(dict(model="Unified XGBoost", station=s,
                                         horizon=hname, seed=seed,
                                         skill=rc.skill_score(yte, p, pers_te),
                                         rmse_W=rc.rmse(yte, p) * rc.SCALE,
                                         mae_W=rc.mae(yte, p) * rc.SCALE,
                                         r2=rc.r2(yte, p), n_test=len(yte)))
                    except Exception as e:
                        failures.append(dict(station=s, horizon=hname, seed=seed,
                                             model="Unified XGBoost",
                                             error=f"{type(e).__name__}: {e}"))
                        LOG.error(f"  S{s} {hname} seed{seed} XGB FAILED: {e}")

                if HAVE_LGB and (s, hname, seed, "LightGBM") not in done:
                    try:
                        ml = lgb.LGBMRegressor(
                            n_estimators=500, max_depth=8, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, n_jobs=-1, verbosity=-1)
                        ml.fit(Xtr_s, ytr)
                        p = ml.predict(Xte_s)
                        rows.append(dict(model="LightGBM", station=s,
                                         horizon=hname, seed=seed,
                                         skill=rc.skill_score(yte, p, pers_te),
                                         rmse_W=rc.rmse(yte, p) * rc.SCALE,
                                         mae_W=rc.mae(yte, p) * rc.SCALE,
                                         r2=rc.r2(yte, p), n_test=len(yte)))
                    except Exception as e:
                        failures.append(dict(station=s, horizon=hname, seed=seed,
                                             model="LightGBM",
                                             error=f"{type(e).__name__}: {e}"))
                        LOG.error(f"  S{s} {hname} seed{seed} LGB FAILED: {e}")

            pd.DataFrame(rows).to_csv(CKPT, index=False)
            LOG.info(f"  S{s} {hname}: {len(rows)}/{n_total} rows")

    R = pd.DataFrame(rows)
    if R.empty:
        LOG.error("no results produced"); return
    rc.save_table(R.round(10), "Seed_Sensitivity", "tree_seed_raw")

    # ---------------- per (model, horizon) spread over the 42 pairs ----------
    out = []
    for (mdl, hz), g in R.groupby(["model", "horizon"]):
        per_pair = g.groupby("station").agg(
            std_skill=("skill", lambda x: float(np.std(x, ddof=1))),
            rng_skill=("skill", lambda x: float(x.max() - x.min())),
            std_rmse=("rmse_W", lambda x: float(np.std(x, ddof=1))),
            rng_rmse=("rmse_W", lambda x: float(x.max() - x.min())),
            n_seeds=("seed", "nunique"))
        # only stations where all three seeds completed contribute
        ok = per_pair[per_pair.n_seeds == len(SEEDS)]
        out.append(dict(
            model=mdl, horizon=hz,
            n_pairs=int(len(ok)), n_seeds=len(SEEDS),
            mean_std_skill=float(ok.std_skill.mean()) if len(ok) else np.nan,
            max_std_skill=float(ok.std_skill.max()) if len(ok) else np.nan,
            mean_range_skill=float(ok.rng_skill.mean()) if len(ok) else np.nan,
            max_range_skill=float(ok.rng_skill.max()) if len(ok) else np.nan,
            mean_std_rmse_W=float(ok.std_rmse.mean()) if len(ok) else np.nan,
            max_std_rmse_W=float(ok.std_rmse.max()) if len(ok) else np.nan,
            mean_range_rmse_W=float(ok.rng_rmse.mean()) if len(ok) else np.nan,
            max_range_rmse_W=float(ok.rng_rmse.max()) if len(ok) else np.nan))
    S = pd.DataFrame(out)
    order = {h: i for i, h in enumerate(HORIZON_STEPS)}
    S["__o"] = S.horizon.map(order)
    S = S.sort_values(["model", "__o"]).drop(columns="__o")
    rc.save_table(S.round(10), "Seed_Sensitivity", "tree_seed_spread")

    # ---------------- one summary line per model ----------------
    summ = []
    for mdl, g in S.groupby("model"):
        summ.append(dict(
            model=mdl, n_horizons=int(len(g)),
            n_pairs_per_horizon=int(g.n_pairs.max()), n_seeds=len(SEEDS),
            MAX_per_pair_skill_std_across_all_horizons=float(g.max_std_skill.max()),
            horizon_of_that_max=str(g.loc[g.max_std_skill.idxmax(), "horizon"]),
            MAX_per_pair_rmse_std_W=float(g.max_std_rmse_W.max()),
            mean_skill_std_over_grid=float(g.mean_std_skill.mean())))
    SUM = pd.DataFrame(summ)
    rc.save_table(SUM.round(10), "Seed_Sensitivity", "tree_seed_model_summary")

    if failures:
        rc.save_table(pd.DataFrame(failures), "Seed_Sensitivity",
                      "tree_seed_failures")
        LOG.warning(f"{len(failures)} failed run(s) recorded - cells left empty, "
                    f"never estimated")

    el = time.time() - t0
    (OUTD / "tree_seed_metadata.json").write_text(json.dumps(dict(
        seeds=SEEDS, stations=rc.STATIONS, horizons=list(HORIZON_STEPS),
        models_run=sorted(R.model.unique().tolist()),
        lightgbm_available=HAVE_LGB,
        lstm_or_attention_trained=False,
        n_rows=int(len(R)), n_failures=len(failures),
        elapsed_s=round(el, 1)), indent=2))

    LOG.info("\nPER (MODEL, HORIZON) SPREAD across the 42 pairs:\n" +
             S[["model", "horizon", "n_pairs", "mean_std_skill", "max_std_skill",
                "mean_range_skill", "mean_std_rmse_W", "max_std_rmse_W"]]
             .to_string(index=False))
    LOG.info("\nSUMMARY PER MODEL:\n" + SUM.to_string(index=False))
    LOG.info(f"\nelapsed {el/60:.1f} min; failures {len(failures)}")


if __name__ == "__main__":
    main()

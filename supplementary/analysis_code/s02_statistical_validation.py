"""
s02_statistical_validation.py
=============================
STATISTICAL VALIDATION (revision Step 4) computed entirely from the
manuscript's existing prediction files. No retraining; no reported value changed.

Adds the inferential evidence a reviewer expects but the manuscript reports only
partially:

  1. Paired Wilcoxon signed-rank tests, proposed vs each baseline, per horizon
     and aggregated over all 42 station-horizon pairs.
  2. Cohen's d_z with a standard effect-size interpretation.
  3. 95% confidence intervals on every paired difference.
  4. Holm-Bonferroni correction across the per-horizon family.
  5. Bootstrap confidence intervals (10,000 resamples) on the skill score,
     which make no distributional assumption at all.
  6. Per-pair win/loss counts.

Metrics: R2, RMSE, MAE and skill score vs persistence (manuscript Eq. 25).

Outputs -> Supplementary_Analysis/Statistics/*.csv/.json
           Supplementary_Analysis/Figures/statistics_*.png/.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s02_statistical_validation")
N_BOOT = 10000
RNG = np.random.default_rng(42)


def per_pair_metrics() -> pd.DataFrame:
    """One row per (station, horizon): every metric for every model."""
    rows = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        pers = d.persistence.values
        rec = dict(station=s, horizon=h, n_test=len(d),
                   sigma_y_W=float(np.std(y)) * rc.SCALE)
        for m in rc.MODELS:
            p = d[m].values
            rec[f"{m}_r2"] = rc.r2(y, p)
            rec[f"{m}_rmse_W"] = rc.rmse(y, p) * rc.SCALE
            rec[f"{m}_mae_W"] = rc.mae(y, p) * rc.SCALE
            rec[f"{m}_bias_W"] = rc.bias(y, p) * rc.SCALE
            rec[f"{m}_skill"] = rc.skill_score(y, p, pers)
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_test(a, b, higher_is_better=True):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = dict(n=len(a), mean_proposed=float(a.mean()) if len(a) else np.nan,
               mean_baseline=float(b.mean()) if len(b) else np.nan)
    if len(a) < 3 or np.allclose(a, b):
        out.update(mean_diff=np.nan, ci95_lo=np.nan, ci95_hi=np.nan,
                   wilcoxon_W=np.nan, p_value=np.nan, cohens_dz=np.nan,
                   effect_size="n/a", proposed_better_in=np.nan,
                   note="fewer than 3 paired observations or identical vectors")
        return out
    d = a - b
    try:
        w, p = stats.wilcoxon(a, b)
    except Exception:
        w, p = np.nan, np.nan
    m, lo, hi = rc.ci95_mean(d)
    dz = rc.cohens_dz(a, b)
    better = int((d > 0).sum()) if higher_is_better else int((d < 0).sum())
    out.update(mean_diff=m, ci95_lo=lo, ci95_hi=hi,
               wilcoxon_W=float(w) if w == w else np.nan,
               p_value=float(p) if p == p else np.nan,
               cohens_dz=dz, effect_size=rc.interpret_d(dz),
               proposed_better_in=better, note="")
    return out


def bootstrap_ci(vals, n_boot=N_BOOT):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    LOG.info("=" * 72)
    LOG.info("STATISTICAL VALIDATION from existing predictions (no retraining)")
    LOG.info("=" * 72)

    M = per_pair_metrics()
    if M.empty:
        LOG.error("no prediction files found"); return
    rc.save_table(M.round(6), "Raw_Data", "per_pair_metrics")
    LOG.info(f"per-pair metrics computed for {len(M)} station-horizon pairs")

    # ---------------- paired tests ----------------
    comparisons = [
        ("skill", "fame_skill", "unified_xgb_skill", "Unified XGBoost", True),
        ("r2", "fame_r2", "unified_xgb_r2", "Unified XGBoost", True),
        ("r2", "fame_r2", "persistence_r2", "Persistence", True),
        ("rmse_W", "fame_rmse_W", "unified_xgb_rmse_W", "Unified XGBoost", False),
        ("rmse_W", "fame_rmse_W", "persistence_rmse_W", "Persistence", False),
        ("mae_W", "fame_mae_W", "unified_xgb_mae_W", "Unified XGBoost", False),
        ("mae_W", "fame_mae_W", "persistence_mae_W", "Persistence", False),
    ]
    rows = []
    for metric, pcol, bcol, bname, hib in comparisons:
        if pcol not in M.columns or bcol not in M.columns:
            continue
        for scope in ["ALL"] + rc.HORIZONS:
            sub = M if scope == "ALL" else M[M.horizon == scope]
            if sub.empty:
                continue
            r = paired_test(sub[pcol], sub[bcol], higher_is_better=hib)
            r.update(metric=metric, baseline=bname, scope=scope,
                     higher_is_better=hib)
            rows.append(r)
    S = pd.DataFrame(rows)
    cols = ["metric", "baseline", "scope", "n", "mean_proposed", "mean_baseline",
            "mean_diff", "ci95_lo", "ci95_hi", "wilcoxon_W", "p_value",
            "cohens_dz", "effect_size", "proposed_better_in", "note"]
    S = S[[c for c in cols if c in S.columns]]
    S["significant_a05"] = S.p_value < 0.05
    # Round only the non-p-value floats. Rounding a p-value of 7.7e-11 to six
    # decimals would store it as 0.0 and destroy its magnitude, so p-value
    # columns are left at full precision and written in scientific notation.
    _round_cols = [c for c in S.columns
                   if S[c].dtype.kind == "f" and "p_value" not in str(c).lower()]
    S[_round_cols] = S[_round_cols].round(6)
    rc.save_table(S, "Statistics", "paired_tests")

    # ---------------- Holm-Bonferroni per family ----------------
    holm_rows = []
    for (metric, baseline), grp in S[S.scope != "ALL"].groupby(["metric", "baseline"]):
        g = grp.dropna(subset=["p_value"]).copy()
        if g.empty:
            continue
        g["holm_adjusted_p"] = rc.holm_bonferroni(g.p_value.values)
        g["holm_significant"] = g.holm_adjusted_p < 0.05
        holm_rows.append(g)
    if holm_rows:
        H = pd.concat(holm_rows, ignore_index=True)
        _rc2 = [c for c in H.columns
                if H[c].dtype.kind == "f" and "p_value" not in str(c).lower()
                and "holm_adjusted_p" not in str(c).lower()]
        H[_rc2] = H[_rc2].round(6)
        rc.save_table(H, "Statistics", "holm_bonferroni")
        LOG.info(f"Holm-Bonferroni applied to {len(H)} per-horizon tests "
                 f"across {H.groupby(['metric','baseline']).ngroups} families")

    # ---------------- bootstrap CIs on skill ----------------
    boot = []
    for scope in ["ALL"] + rc.HORIZONS:
        sub = M if scope == "ALL" else M[M.horizon == scope]
        if sub.empty:
            continue
        for m in rc.MODELS:
            col = f"{m}_skill"
            if col not in sub.columns:
                continue
            v = sub[col].values
            lo, hi = bootstrap_ci(v)
            boot.append(dict(scope=scope, model=rc.MODEL_LABEL[m], n=len(v),
                             mean_skill=float(np.nanmean(v)),
                             boot_ci95_lo=lo, boot_ci95_hi=hi, n_boot=N_BOOT))
    B = pd.DataFrame(boot)
    rc.save_table(B.round(6), "Statistics", "bootstrap_ci_skill")

    # ---------------- win/loss ----------------
    wl = []
    for scope in ["ALL"] + rc.HORIZONS:
        sub = M if scope == "ALL" else M[M.horizon == scope]
        if sub.empty:
            continue
        for bcol, bname in [("unified_xgb_r2", "Unified XGBoost"),
                            ("persistence_r2", "Persistence")]:
            if bcol not in sub.columns:
                continue
            d = sub["fame_r2"].values - sub[bcol].values
            wl.append(dict(scope=scope, baseline=bname, n=len(d),
                           wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                           ties=int((d == 0).sum()),
                           win_rate_pct=100 * float((d > 0).mean())))
    W = pd.DataFrame(wl)
    rc.save_table(W.round(4), "Statistics", "win_loss_record")

    # ---------------- figures ----------------
    plt = rc.plot_style()
    sk = S[(S.metric == "skill") & (S.scope != "ALL")].set_index("scope").reindex(rc.HORIZONS)
    if sk.mean_diff.notna().any():
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        x = np.arange(len(rc.HORIZONS))
        ax.errorbar(x, sk.mean_diff, yerr=[sk.mean_diff - sk.ci95_lo,
                                           sk.ci95_hi - sk.mean_diff],
                    fmt="o-", capsize=4, color=rc.CB["blue"])
        ax.axhline(0, ls="--", c="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(rc.HORIZONS)
        ax.set_xlabel("forecast horizon")
        ax.set_ylabel("skill difference (proposed − unified XGB)")
        ax.set_title("Paired skill advantage with 95% CI (from existing predictions)")
        rc.save_figure(fig, "statistics_skill_advantage",
                       data=sk.reset_index().round(6))
        plt.close(fig)

    bb = B[(B.scope != "ALL")]
    if not bb.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        for m, c in [("Proposed (FAME)", rc.CB["blue"]),
                     ("Unified XGBoost", rc.CB["orange"])]:
            g = bb[bb.model == m].set_index("scope").reindex(rc.HORIZONS)
            if g.mean_skill.notna().any():
                x = np.arange(len(rc.HORIZONS))
                ax.plot(x, g.mean_skill, "o-", label=m, color=c)
                ax.fill_between(x, g.boot_ci95_lo, g.boot_ci95_hi, alpha=.18, color=c)
        ax.axhline(0, ls="--", c="k", lw=1, label="persistence floor")
        ax.set_xticks(np.arange(len(rc.HORIZONS))); ax.set_xticklabels(rc.HORIZONS)
        ax.set_xlabel("forecast horizon"); ax.set_ylabel("skill score vs persistence")
        ax.set_title(f"Skill with bootstrap 95% CI ({N_BOOT:,} resamples)")
        ax.legend(fontsize=8)
        rc.save_figure(fig, "statistics_bootstrap_skill", data=bb.round(6))
        plt.close(fig)

    key = S[(S.metric == "skill") & (S.baseline == "Unified XGBoost")]
    LOG.info("\nskill, proposed vs unified XGBoost:\n" +
             key[["scope", "n", "mean_diff", "p_value", "cohens_dz",
                  "effect_size", "proposed_better_in"]].round(5).to_string(index=False))
    LOG.info(f"tables written to {rc.OUT/'Statistics'}")


if __name__ == "__main__":
    main()

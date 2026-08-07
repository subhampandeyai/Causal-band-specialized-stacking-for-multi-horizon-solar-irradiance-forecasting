"""
s04_robustness.py
=================
ROBUSTNESS AND ERROR-STRUCTURE ANALYSIS (revision Step 4), computed from the
manuscript's existing prediction files. No retraining; nothing reported changed.

Adds evidence the manuscript does not currently contain:

  1. Station robustness   - spread of skill across the 7 sites, worst case,
                            coefficient of variation.
  2. Horizon robustness   - degradation profile with lead time and a fitted slope.
  3. Failure analysis     - every (station, horizon) where the proposed model
                            fails to beat persistence or the test mean, named
                            explicitly rather than aggregated away.
  4. Error structure      - bias, residual autocorrelation at lag 1, and the
                            MAE/RMSE ratio (a distribution-shape diagnostic the
                            manuscript uses at H1 only).
  5. Conditional skill    - performance stratified by irradiance level
                            (low / medium / high terciles of y_true), which shows
                            where the gain actually comes from.
  6. Per-station consistency of the win/loss record.

Outputs -> Supplementary_Analysis/Robustness/*.csv/.json
           Supplementary_Analysis/Figures/robustness_*.png/.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s04_robustness")


def main():
    LOG.info("=" * 72)
    LOG.info("ROBUSTNESS & ERROR STRUCTURE from existing predictions (no retraining)")
    LOG.info("=" * 72)

    M = pd.read_csv(rc.OUT / "Raw_Data" / "per_pair_metrics.csv") \
        if (rc.OUT / "Raw_Data" / "per_pair_metrics.csv").exists() else None
    if M is None:
        LOG.error("run s02_statistical_validation.py first (needs per_pair_metrics.csv)")
        return

    # ---------------- 1. station robustness ----------------
    ST = (M.groupby("station")
          .agg(n=("fame_skill", "size"),
               mean_skill=("fame_skill", "mean"), std_skill=("fame_skill", "std"),
               min_skill=("fame_skill", "min"), max_skill=("fame_skill", "max"),
               mean_r2=("fame_r2", "mean"), mean_rmse_W=("fame_rmse_W", "mean"),
               mean_uxgb_skill=("unified_xgb_skill", "mean")).reset_index())
    ST["cv_abs"] = ST.std_skill / ST.mean_skill.abs()
    ST["range"] = ST.max_skill - ST.min_skill
    ST["capacity_MW"] = ST.station.map(rc.CAPACITY_MW)
    ST["beats_uxgb"] = ST.mean_skill > ST.mean_uxgb_skill
    rc.save_table(ST.round(6), "Robustness", "station_robustness")
    LOG.info(f"station robustness: skill range across sites "
             f"{ST.mean_skill.min():+.4f} to {ST.mean_skill.max():+.4f}")

    # ---------------- 2. horizon robustness ----------------
    HZ = (M.groupby("horizon")
          .agg(n=("fame_skill", "size"),
               mean_skill=("fame_skill", "mean"), std_skill=("fame_skill", "std"),
               min_skill=("fame_skill", "min"), max_skill=("fame_skill", "max"),
               mean_r2=("fame_r2", "mean"), mean_rmse_W=("fame_rmse_W", "mean"),
               mean_uxgb_rmse_W=("unified_xgb_rmse_W", "mean"),
               mean_pers_rmse_W=("persistence_rmse_W", "mean")).reset_index())
    HZ["__o"] = HZ.horizon.map({h: i for i, h in enumerate(rc.HORIZONS)})
    HZ = HZ.sort_values("__o").drop(columns="__o")
    HZ["horizon_minutes"] = HZ.horizon.map(rc.HORIZON_MIN)
    HZ["rmse_ratio_vs_uxgb"] = HZ.mean_rmse_W / HZ.mean_uxgb_rmse_W
    rc.save_table(HZ.round(6), "Robustness", "horizon_robustness")

    # degradation slope per station
    deg = []
    for s in sorted(M.station.unique()):
        sub = (M[M.station == s].set_index("horizon")
               .reindex(rc.HORIZONS)["fame_skill"].dropna())
        if len(sub) >= 3:
            x = np.arange(len(sub))
            sl, ic, r, p, se = stats.linregress(x, sub.values)
            deg.append(dict(station=int(s), slope_per_horizon_step=sl, intercept=ic,
                            r_value=r, p_value=p, stderr=se,
                            skill_first=float(sub.iloc[0]),
                            skill_last=float(sub.iloc[-1]),
                            total_degradation=float(sub.iloc[0] - sub.iloc[-1])))
    if deg:
        rc.save_table(pd.DataFrame(deg).round(6), "Robustness", "degradation_trends")

    # ---------------- 3. failure cases ----------------
    fails = []
    for _, r in M.iterrows():
        issues = []
        if r.fame_skill < 0:
            issues.append("skill<0 (worse than persistence)")
        if r.fame_r2 < 0:
            issues.append("R2<0 (worse than test mean)")
        if r.fame_r2 < r.unified_xgb_r2:
            issues.append("R2 below unified XGBoost")
        if issues:
            fails.append(dict(station=int(r.station), horizon=r.horizon,
                              fame_skill=r.fame_skill, fame_r2=r.fame_r2,
                              uxgb_r2=r.unified_xgb_r2,
                              fame_rmse_W=r.fame_rmse_W,
                              issues="; ".join(issues)))
    F = pd.DataFrame(fails)
    rc.save_table(F.round(6) if not F.empty else
                  pd.DataFrame([dict(result="no failure cases",
                                     note="the proposed model beat persistence and "
                                          "the test mean in every station-horizon pair")]),
                  "Robustness", "failure_cases")
    LOG.info(f"failure cases: {len(F)} of {len(M)} pairs flagged")

    # ---------------- 4. error structure ----------------
    err = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        for m in rc.MODELS:
            res = (d[m].values - y) * rc.SCALE
            if len(res) < 10:
                continue
            ac1 = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 2 else np.nan
            rm = float(np.sqrt(np.mean(res ** 2))); ma = float(np.mean(np.abs(res)))
            err.append(dict(station=s, horizon=h, model=rc.MODEL_LABEL[m],
                            bias_W=float(np.mean(res)),
                            rmse_W=rm, mae_W=ma,
                            mae_rmse_ratio=(ma / rm if rm else np.nan),
                            residual_std_W=float(np.std(res)),
                            residual_skew=float(stats.skew(res)),
                            residual_kurtosis=float(stats.kurtosis(res)),
                            residual_autocorr_lag1=ac1))
    E = pd.DataFrame(err)
    rc.save_table(E.round(6), "Robustness", "error_structure")
    prop = E[E.model == "Proposed (FAME)"]
    LOG.info(f"error structure: proposed model MAE/RMSE ratio "
             f"{prop.mae_rmse_ratio.mean():.3f} (manuscript uses 0.572 at S5/H1); "
             f"mean |bias| {prop.bias_W.abs().mean():.2f} W/m2")

    # ---------------- 5. conditional skill by irradiance level ----------------
    cond = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
        bins = [("low", y <= q1), ("medium", (y > q1) & (y <= q2)), ("high", y > q2)]
        for name, mask in bins:
            if mask.sum() < 20:
                continue
            yy = y[mask]; pers = d.persistence.values[mask]
            rec = dict(station=s, horizon=h, regime=name, n=int(mask.sum()),
                       mean_irradiance_W=float(yy.mean()) * rc.SCALE)
            for m in rc.MODELS:
                pp = d[m].values[mask]
                rec[f"{m}_rmse_W"] = rc.rmse(yy, pp) * rc.SCALE
                rec[f"{m}_skill"] = rc.skill_score(yy, pp, pers)
            cond.append(rec)
    C = pd.DataFrame(cond)
    rc.save_table(C.round(6), "Robustness", "conditional_skill_by_irradiance")
    if not C.empty:
        summ = (C.groupby("regime")
                .agg(n_pairs=("station", "size"),
                     fame_rmse_W=("fame_rmse_W", "mean"),
                     uxgb_rmse_W=("unified_xgb_rmse_W", "mean"),
                     fame_skill=("fame_skill", "mean")).reset_index())
        rc.save_table(summ.round(6), "Robustness", "conditional_skill_summary")
        LOG.info("\nconditional performance by irradiance regime:\n" +
                 summ.round(4).to_string(index=False))

    # ---------------- figures ----------------
    plt = rc.plot_style()

    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6))
    x = np.arange(len(rc.HORIZONS))
    g = HZ.set_index("horizon").reindex(rc.HORIZONS)
    axs[0].errorbar(x, g.mean_skill, yerr=g.std_skill, fmt="o-", capsize=4,
                    color=rc.CB["blue"], label="proposed")
    axs[0].axhline(0, ls="--", c="k", lw=1, label="persistence floor")
    axs[0].set_ylabel("skill vs persistence"); axs[0].set_title("(a) horizon robustness")
    for m, c, lab in [("mean_rmse_W", rc.CB["blue"], "proposed"),
                      ("mean_uxgb_rmse_W", rc.CB["orange"], "unified XGB"),
                      ("mean_pers_rmse_W", rc.CB["grey"], "persistence")]:
        axs[1].plot(x, g[m], "o-", color=c, label=lab)
    axs[1].set_ylabel("mean RMSE (W/m²)"); axs[1].set_title("(b) error growth")
    for a_ in axs:
        a_.set_xticks(x); a_.set_xticklabels(rc.HORIZONS); a_.set_xlabel("horizon")
        a_.legend(fontsize=7)
    rc.save_figure(fig, "robustness_horizon", data=g.reset_index().round(4))
    plt.close(fig)

    piv = M.pivot_table(index="station", columns="horizon",
                        values="fame_skill").reindex(columns=rc.HORIZONS)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(rc.HORIZONS))); ax.set_xticklabels(rc.HORIZONS)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"S{i}" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="skill vs persistence")
    ax.set_title("Station × horizon skill (from the manuscript's own predictions)")
    rc.save_figure(fig, "robustness_station_horizon_heatmap",
                   data=piv.reset_index().round(4))
    plt.close(fig)

    if not C.empty:
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        order = ["low", "medium", "high"]
        g2 = C.groupby("regime")[["fame_rmse_W", "unified_xgb_rmse_W",
                                  "persistence_rmse_W"]].mean().reindex(order)
        w = 0.26; xx = np.arange(len(order))
        ax.bar(xx - w, g2.fame_rmse_W, w, label="proposed", color=rc.CB["blue"])
        ax.bar(xx, g2.unified_xgb_rmse_W, w, label="unified XGB", color=rc.CB["orange"])
        ax.bar(xx + w, g2.persistence_rmse_W, w, label="persistence", color=rc.CB["grey"])
        ax.set_xticks(xx); ax.set_xticklabels([f"{o} irradiance" for o in order])
        ax.set_ylabel("mean RMSE (W/m²)")
        ax.set_title("Conditional error by irradiance regime (all 42 pairs)")
        ax.legend(fontsize=8)
        rc.save_figure(fig, "robustness_conditional_regime", data=g2.reset_index().round(4))
        plt.close(fig)

    LOG.info(f"robustness tables -> {rc.OUT/'Robustness'}")


if __name__ == "__main__":
    main()

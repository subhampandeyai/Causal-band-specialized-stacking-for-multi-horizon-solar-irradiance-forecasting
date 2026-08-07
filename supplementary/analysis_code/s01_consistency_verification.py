"""
s01_consistency_verification.py
===============================
RESULT CONSISTENCY VERIFICATION (revision Step 4).

Recomputes the manuscript's reported metrics directly from its own saved
prediction files and checks that they agree. This adds evidence WITHOUT
retraining anything and without changing a single reported value.

Two independent checks:

  A. Internal consistency
     For every one of the 42 (station, horizon) pairs, recompute R2, RMSE and
     MAE from `results/predictions/*.csv` and compare against the values the
     manuscript reports in `results/fame_causal_perstation.csv`.

  B. Aggregate consistency
     Recompute the per-horizon six-model means and compare against
     `results/fame_causal_sixmodel.csv`.

A disagreement would indicate the reported tables and the saved predictions came
from different runs. Agreement is positive evidence that the manuscript's tables
are exactly what its saved predictions produce.

Outputs -> Supplementary_Analysis/Implementation_Audit/
             consistency_perpair.csv/.json
             consistency_summary.csv/.json
           Supplementary_Analysis/Figures/consistency_*.png/.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s01_consistency_verification")
TOL = 5e-4          # the manuscript rounds to 4 decimals


def main():
    LOG.info("=" * 72)
    LOG.info("RESULT CONSISTENCY VERIFICATION (no retraining)")
    LOG.info("=" * 72)

    reported = rc.load_results_table("fame_causal_perstation.csv")
    if reported is None:
        LOG.error("results/fame_causal_perstation.csv not found - cannot verify")
        return
    reported = reported.set_index(["station", "horizon"])

    rows = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        rec = dict(station=s, horizon=h, n_test=len(d))

        for col, tag in [("fame", "fame"), ("unified_xgb", "uxgb")]:
            p = d[col].values
            rec[f"{tag}_r2_recomputed"] = rc.r2(y, p)
            rec[f"{tag}_rmse_W_recomputed"] = rc.rmse(y, p) * rc.SCALE
            rec[f"{tag}_mae_W_recomputed"] = rc.mae(y, p) * rc.SCALE

        if (s, h) in reported.index:
            r = reported.loc[(s, h)]
            for tag, rep_r2, rep_rmse, rep_mae in [
                    ("fame", "fame_r2", "fame_rmse_W", "fame_mae_W"),
                    ("uxgb", "uxgb_r2", "uxgb_rmse_W", "uxgb_mae_W")]:
                for metric, rep_col in [("r2", rep_r2), ("rmse_W", rep_rmse),
                                        ("mae_W", rep_mae)]:
                    if rep_col in r.index:
                        rep = float(r[rep_col])
                        got = rec[f"{tag}_{metric}_recomputed"]
                        rec[f"{tag}_{metric}_reported"] = rep
                        rec[f"{tag}_{metric}_absdiff"] = abs(got - rep)
        rows.append(rec)

    P = pd.DataFrame(rows)
    if P.empty:
        LOG.error("no prediction files found"); return

    # ---- agreement verdicts -------------------------------------------
    diff_cols = [c for c in P.columns if c.endswith("_absdiff")]
    for c in diff_cols:
        base = c[:-8]
        # RMSE/MAE are reported to 2 dp in W/m^2; R2 to 4 dp
        tol = 5e-3 if ("rmse" in c or "mae" in c) else TOL
        P[base + "_agrees"] = P[c] <= tol

    agree_cols = [c for c in P.columns if c.endswith("_agrees")]
    P["all_metrics_agree"] = P[agree_cols].all(axis=1) if agree_cols else False
    rc.save_table(P.round(6), "Implementation_Audit", "consistency_perpair")

    n_pairs = len(P)
    n_ok = int(P["all_metrics_agree"].sum())
    LOG.info(f"per-pair verification: {n_ok}/{n_pairs} pairs reproduce every "
             f"reported metric from the saved predictions")

    summary = []
    for c in diff_cols:
        base = c[:-8]
        tol = 5e-3 if ("rmse" in c or "mae" in c) else TOL
        v = P[c].dropna()
        if v.empty:
            continue
        summary.append(dict(metric=base, n=len(v),
                            max_abs_diff=float(v.max()),
                            mean_abs_diff=float(v.mean()),
                            tolerance=tol,
                            n_within_tolerance=int((v <= tol).sum()),
                            all_agree=bool((v <= tol).all())))
    S = pd.DataFrame(summary)
    if not S.empty:
        rc.save_table(S.round(8), "Implementation_Audit", "consistency_summary")
        LOG.info("\n" + S.round(6).to_string(index=False))

    # ---- aggregate: six-model table -----------------------------------
    six = rc.load_results_table("fame_causal_sixmodel.csv")
    agg_rows = []
    if six is not None:
        six_i = six.set_index("horizon")
        for h in rc.HORIZONS:
            sub = P[P.horizon == h]
            if sub.empty or h not in six_i.index:
                continue
            for tag, rep_col in [("fame", "FAME"), ("uxgb", "Unified_XGB")]:
                col = f"{tag}_r2_recomputed"
                if col not in sub.columns or rep_col not in six_i.columns:
                    continue
                got = float(sub[col].mean()); rep = float(six_i.loc[h, rep_col])
                agg_rows.append(dict(horizon=h, model=rep_col,
                                     reported_mean_r2=rep,
                                     recomputed_mean_r2=got,
                                     abs_diff=abs(got - rep),
                                     agrees=abs(got - rep) <= TOL))
    if agg_rows:
        A = pd.DataFrame(agg_rows)
        rc.save_table(A.round(6), "Implementation_Audit", "consistency_aggregate")
        LOG.info(f"aggregate verification: {int(A.agrees.sum())}/{len(A)} "
                 f"per-horizon means reproduce")

    # ---- figure --------------------------------------------------------
    plt = rc.plot_style()
    if diff_cols:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        show = [c for c in diff_cols if c.endswith("r2_absdiff")]
        for i, c in enumerate(show):
            v = P[c].dropna().values
            if len(v):
                ax.scatter(np.arange(len(v)) + i * 0.15, v, s=16,
                           label=c.replace("_absdiff", ""),
                           color=[rc.CB["blue"], rc.CB["orange"]][i % 2])
        ax.axhline(TOL, ls="--", c=rc.CB["verm"], lw=1,
                   label=f"rounding tolerance ({TOL})")
        ax.set_yscale("log"); ax.set_xlabel("station-horizon pair")
        ax.set_ylabel("|recomputed − reported|")
        ax.set_title("Reported vs recomputed R² from the manuscript's own predictions")
        ax.legend(fontsize=7)
        rc.save_figure(fig, "consistency_r2_absdiff",
                       data=P[["station", "horizon"] + show].round(8))
        plt.close(fig)

    verdict = ("ALL REPORTED METRICS REPRODUCE from the saved predictions"
               if (not S.empty and S.all_agree.all())
               else "SOME METRICS DIFFER - see consistency_summary.csv")
    LOG.info(f"VERDICT: {verdict}")
    rc.save_table(pd.DataFrame([dict(
        check="result consistency", n_pairs=n_pairs, n_pairs_fully_agreeing=n_ok,
        verdict=verdict,
        note="Recomputed directly from results/predictions/*.csv. No model was "
             "retrained and no reported value was modified.")]),
        "Implementation_Audit", "consistency_verdict")


if __name__ == "__main__":
    main()

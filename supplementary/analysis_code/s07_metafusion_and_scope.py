"""
s07_metafusion_and_scope.py
===========================
Two things:

  (1) POST-HOC FUSION ANALYSIS - the one ablation-style study that is legitimately
      computable WITHOUT retraining. The manuscript's stacking operates on band
      forecasts; the saved predictions expose the fused output alongside unified
      XGBoost and persistence. Re-fusing those SAVED vectors post hoc tells us how
      much of the result depends on the fusion rule rather than on the experts:

        proposed          the manuscript's own fused prediction (unchanged)
        equal_average     equal-weight mean of the saved model predictions
        oracle_convex     best fixed convex weight, fitted on the first half of the
                          test block and applied to the second (no peeking)
        best_single       the best individual saved model per pair

      Nothing is retrained. Every input is a stored prediction vector.

  (2) SCOPE DECLARATION - an explicit, machine-readable statement of which
      analyses are NOT computable from the frozen outputs, and why. Sensitivity
      to wavelet basis / window / decomposition level and true component
      ablations all require retraining, which this revision forbids. They are
      declared here rather than silently omitted.

Also records the computational facts that ARE measurable from the frozen
artifacts (dataset sizes, prediction volumes, measured LSTM cost from the
preserved runtime probe).

Outputs -> Supplementary_Analysis/Ablation/*.csv/.json
           Supplementary_Analysis/Sensitivity/scope_declaration.csv/.json
           Supplementary_Analysis/Computational_Cost/*.csv/.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s07_metafusion_and_scope")


def best_convex_weight(y, p1, p2):
    """Grid-search the convex weight w in p = w*p1 + (1-w)*p2 minimising MSE."""
    ws = np.linspace(0, 1, 101)
    errs = [np.mean((y - (w * p1 + (1 - w) * p2)) ** 2) for w in ws]
    i = int(np.argmin(errs))
    return float(ws[i]), float(errs[i])


def main():
    LOG.info("=" * 72)
    LOG.info("POST-HOC FUSION ANALYSIS + SCOPE DECLARATION (no retraining)")
    LOG.info("=" * 72)

    # ================================================ (1) fusion analysis
    rows = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        fame = d.fame.values
        ux = d.unified_xgb.values
        pers = d.persistence.values
        half = len(y) // 2

        # FAIRNESS NOTE (critical for reading this table):
        # The manuscript's model never saw ANY test-block data. The two variants
        # marked [ORACLE] below are fitted or selected USING test-block data, so
        # they are upper bounds on what post-hoc fusion could achieve - NOT fair
        # competitors. They are reported to bound the headroom, never to claim the
        # proposed model is beaten.
        variants = {
            "proposed (manuscript, no test access)": (fame, "fair"),
            "equal_average(uxgb,pers)": (0.5 * ux + 0.5 * pers, "fair"),
            "[ORACLE] best_single per pair": (
                (ux if rc.rmse(y, ux) <= rc.rmse(y, pers) else pers), "oracle"),
        }
        w, _ = best_convex_weight(y[:half], ux[:half], pers[:half])
        variants["[ORACLE] convex(uxgb,pers), weight fitted on test-cal half"] = (
            w * ux + (1 - w) * pers, "oracle")

        for name, (p, kind) in variants.items():
            # every variant is scored on the same held-out evaluation half
            ye, pe, pr = y[half:], p[half:], pers[half:]
            rows.append(dict(station=s, horizon=h, variant=name, comparison_class=kind,
                             r2=rc.r2(ye, pe), rmse_W=rc.rmse(ye, pe) * rc.SCALE,
                             mae_W=rc.mae(ye, pe) * rc.SCALE,
                             skill=rc.skill_score(ye, pe, pr),
                             convex_weight_uxgb=(w if "convex" in name else np.nan),
                             n_eval=len(ye)))
    F = pd.DataFrame(rows)
    if F.empty:
        LOG.error("no predictions available"); return
    rc.save_table(F.round(6), "Ablation", "posthoc_fusion_per_pair")

    S = (F.groupby(["variant", "comparison_class"])
         .agg(n=("r2", "size"), mean_r2=("r2", "mean"),
              mean_rmse_W=("rmse_W", "mean"), mean_mae_W=("mae_W", "mean"),
              mean_skill=("skill", "mean")).reset_index()
         .sort_values("mean_rmse_W"))
    rc.save_table(S.round(6), "Ablation", "posthoc_fusion_summary")
    LOG.info("\npost-hoc fusion (scored on the held-out evaluation half):\n" +
             S.round(4).to_string(index=False))

    # Head-to-head among FAIR variants only - the defensible comparison.
    fair = F[F.comparison_class == "fair"]
    if not fair.empty:
        piv = fair.pivot_table(index=["station", "horizon"], columns="variant",
                               values="rmse_W")
        prop_col = [c for c in piv.columns if "manuscript" in c][0]
        others = [c for c in piv.columns if c != prop_col]
        hh = []
        for c in others:
            d = piv[prop_col] - piv[c]
            hh.append(dict(comparison=f"proposed vs {c}", n=int(d.notna().sum()),
                           mean_rmse_diff_W=float(d.mean()),
                           proposed_better_in=int((d < 0).sum()),
                           proposed_worse_in=int((d > 0).sum())))
        HH = pd.DataFrame(hh)
        rc.save_table(HH.round(6), "Ablation", "posthoc_fusion_fair_headtohead")
        LOG.info("\nfair head-to-head (lower RMSE is better):\n" +
                 HH.round(4).to_string(index=False))

    # The manuscript's model vs the individual saved baselines, same eval half.
    direct = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values; half = len(y) // 2
        ye = y[half:]
        direct.append(dict(station=s, horizon=h,
                           proposed_rmse_W=rc.rmse(ye, d.fame.values[half:]) * rc.SCALE,
                           uxgb_rmse_W=rc.rmse(ye, d.unified_xgb.values[half:]) * rc.SCALE,
                           pers_rmse_W=rc.rmse(ye, d.persistence.values[half:]) * rc.SCALE))
    DD = pd.DataFrame(direct)
    if not DD.empty:
        DD["proposed_beats_uxgb"] = DD.proposed_rmse_W < DD.uxgb_rmse_W
        DD["proposed_beats_pers"] = DD.proposed_rmse_W < DD.pers_rmse_W
        rc.save_table(DD.round(4), "Ablation", "proposed_vs_baselines_eval_half")
        LOG.info(f"on the evaluation half, the manuscript's model beats unified XGBoost "
                 f"in {int(DD.proposed_beats_uxgb.sum())}/{len(DD)} pairs and persistence "
                 f"in {int(DD.proposed_beats_pers.sum())}/{len(DD)} pairs")

    BH = (F.groupby(["variant", "horizon"])
          .agg(mean_skill=("skill", "mean"), mean_rmse_W=("rmse_W", "mean")).reset_index())
    rc.save_table(BH.round(6), "Ablation", "posthoc_fusion_by_horizon")

    # figure
    plt = rc.plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(rc.HORIZONS))
    colors = [rc.CB["blue"], rc.CB["orange"], rc.CB["green"], rc.CB["purple"]]
    for i, v in enumerate(S.variant):
        g = BH[BH.variant == v].set_index("horizon").reindex(rc.HORIZONS)
        ax.plot(x, g.mean_rmse_W, "o-", label=v, color=colors[i % len(colors)],
                lw=2.0 if "manuscript" in v else 1.1)
    ax.set_xticks(x); ax.set_xticklabels(rc.HORIZONS)
    ax.set_xlabel("forecast horizon"); ax.set_ylabel("mean RMSE (W/m²)")
    ax.set_title("Post-hoc fusion of saved predictions (no retraining)")
    ax.legend(fontsize=7)
    rc.save_figure(fig, "ablation_posthoc_fusion", data=BH.round(4))
    plt.close(fig)

    # ================================================ (2) scope declaration
    scope = [
        dict(analysis="Sensitivity: wavelet basis (haar/db2/db4/db6/db8/sym4/coif2)",
             computable_from_frozen_outputs=False,
             reason="changing the basis changes the decomposition, so every band expert "
                    "and the meta-learner must be refitted",
             available_in="not run for this revision"),
        dict(analysis="Sensitivity: trailing-window length W (128-2048)",
             computable_from_frozen_outputs=False,
             reason="the window defines the causal band values; new bands require retraining",
             available_in="not run for this revision"),
        dict(analysis="Sensitivity: decomposition level J (1-5)",
             computable_from_frozen_outputs=False,
             reason="the level defines how many bands exist; the expert pool changes",
             available_in="not run for this revision"),
        dict(analysis="Sensitivity: meta-learner ridge alpha",
             computable_from_frozen_outputs=False,
             reason="the saved files store the FUSED forecast, not the four individual band "
                    "forecasts, so the meta-learner cannot be refitted from them",
             available_in="not run for this revision"),
        dict(analysis="Ablation: remove a band expert (trend/daily/hourly/noise)",
             computable_from_frozen_outputs=False,
             reason="individual band forecasts are not stored in the released predictions",
             available_in="not run for this revision"),
        dict(analysis="Ablation: causal vs whole-series decomposition",
             computable_from_frozen_outputs=False,
             reason="requires refitting under a different decomposition regime",
             available_in="not run for this revision"),
        dict(analysis="Multi-seed variance of the proposed model",
             computable_from_frozen_outputs=False,
             reason="the manuscript's run used a single fixed seed (42); additional seeds "
                    "require retraining",
             available_in="not run for this revision"),
        dict(analysis="Post-hoc fusion of saved predictions",
             computable_from_frozen_outputs=True,
             reason="operates purely on stored prediction vectors",
             available_in="this package (s07_metafusion_and_scope.py)"),
        dict(analysis="Statistical validation (Wilcoxon, Cohen's d, CIs, Holm, bootstrap)",
             computable_from_frozen_outputs=True,
             reason="computed from stored per-pair predictions",
             available_in="this package (s02_statistical_validation.py)"),
        dict(analysis="Uncertainty: split conformal on all stations and horizons",
             computable_from_frozen_outputs=True,
             reason="split conformal is a post-hoc wrapper on stored predictions",
             available_in="this package (s03_uncertainty.py)"),
        dict(analysis="Robustness, error structure, conditional skill",
             computable_from_frozen_outputs=True,
             reason="computed from stored residuals",
             available_in="this package (s04_robustness.py)"),
        dict(analysis="Result consistency verification",
             computable_from_frozen_outputs=True,
             reason="recomputes reported metrics from stored predictions",
             available_in="this package (s01_consistency_verification.py)"),
    ]
    SC = pd.DataFrame(scope)
    rc.save_table(SC, "Sensitivity", "scope_declaration")
    n_no = int((~SC.computable_from_frozen_outputs).sum())
    LOG.info(f"scope declaration: {len(SC)-n_no} analyses computable here, "
             f"{n_no} require retraining (declared, not omitted silently)")

    # ================================================ (3) computational facts
    comp = []
    tot_rows = 0
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is not None:
            tot_rows += len(d)
    comp.append(dict(quantity="station-horizon pairs with saved predictions",
                     value=len(rc.available_pairs()), unit="pairs",
                     source="results/predictions/"))
    comp.append(dict(quantity="total stored test predictions",
                     value=tot_rows, unit="samples",
                     source="results/predictions/"))
    comp.append(dict(quantity="models per pair", value=3, unit="models",
                     source="y_true, fame, unified_xgb, persistence"))

    probe = rc.OUT / "Computational_Cost" / "runtime_probe.json"

    if probe.exists():
        try:
            p = json.loads(probe.read_text())
            for k, lab, unit in [
                    ("t_lstm_s", "LSTM band expert training (exact architecture)", "s/pair"),
                    ("t_xgb_s", "XGBoost hourly expert training", "s/pair"),
                    ("t_ridge_s", "Ridge trend expert training", "s/pair"),
                    ("t_bands_s", "causal decomposition", "s/station"),
                    ("per_pair_proposed_only_s", "proposed model, one pair", "s"),
                    ("grid_42_proposed_only_h", "one 42-pair grid", "h")]:
                if k in p:
                    comp.append(dict(quantity=lab, value=round(float(p[k]), 4),
                                     unit=unit,
                                     source="measured; runtime_probe.json"))
        except Exception:
            pass
    C = pd.DataFrame(comp)
    rc.save_table(C, "Computational_Cost", "measured_computational_facts")
    LOG.info(f"computational facts recorded: {len(C)} entries")


if __name__ == "__main__":
    main()

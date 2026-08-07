"""
s03_uncertainty.py
==================
UNCERTAINTY ANALYSIS (revision Step 4), extended to EVERY station and EVERY
horizon, computed from the manuscript's existing prediction files.

The manuscript reports split-conformal intervals for Station 5 only (Table X).
Split conformal is a post-hoc wrapper on saved predictions, so the identical
procedure can be applied to all 42 station-horizon pairs with no retraining and
without altering the reported Station-5 values.

Computed per (station, horizon, model, nominal level):
  * empirical coverage and mean interval width
  * calibration error (empirical − nominal, in percentage points)
  * over- / under-coverage classification
  * width normalised by the target's standard deviation (comparable across sites)
  * Winkler interval score (proper scoring rule; lower is better)
  * CWC (coverage-width criterion) with the usual penalty for under-coverage

The manuscript's own Station-5 numbers are reproduced alongside as a
cross-check; they are not modified.

Outputs -> Supplementary_Analysis/Uncertainty/*.csv/.json
           Supplementary_Analysis/Figures/uncertainty_*.png/.pdf
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s03_uncertainty")
ALPHAS = [0.10, 0.05]


def winkler(y, lo, hi, alpha):
    """Winkler interval score: width + penalties for exclusions."""
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    w = hi - lo
    below = y < lo
    above = y > hi
    s = w.copy()
    s[below] = w[below] + (2 / alpha) * (lo[below] - y[below])
    s[above] = w[above] + (2 / alpha) * (y[above] - hi[above])
    return float(np.mean(s))


def cwc(coverage, width_norm, nominal, eta=50.0):
    """Coverage-width criterion; penalises under-coverage exponentially."""
    if not np.isfinite(coverage) or not np.isfinite(width_norm):
        return np.nan
    pen = 0.0 if coverage >= nominal else np.exp(-eta * (coverage - nominal))
    return float(width_norm * (1 + pen))


def main():
    LOG.info("=" * 72)
    LOG.info("UNCERTAINTY ANALYSIS - all stations x horizons (no retraining)")
    LOG.info("The manuscript reports Station 5 only; this extends it to all 42 pairs.")
    LOG.info("=" * 72)

    rows = []
    for s, h in rc.available_pairs():
        d = rc.load_pair(s, h)
        if d is None:
            continue
        y = d.y_true.values
        sigma = float(np.std(y)) * rc.SCALE
        for m in rc.MODELS:
            yhat = d[m].values
            for a in ALPHAS:
                res = rc.conformal_interval(y, yhat, a)
                if not np.isfinite(res["coverage"]):
                    continue
                nominal = 1 - a
                half = len(y) // 2
                ey, eh = y[half:], yhat[half:]
                q = res["q"]
                w_score = winkler(ey * rc.SCALE, (eh - q) * rc.SCALE,
                                  (eh + q) * rc.SCALE, a)
                width_W = res["width"] * rc.SCALE
                wn = width_W / sigma if sigma else np.nan
                rows.append(dict(
                    station=s, horizon=h, model=rc.MODEL_LABEL[m],
                    nominal_pct=nominal * 100,
                    empirical_pct=res["coverage"] * 100,
                    width_W_m2=width_W,
                    calibration_error_pp=(res["coverage"] - nominal) * 100,
                    status=("over-covered" if res["coverage"] > nominal
                            else "under-covered" if res["coverage"] < nominal
                            else "exact"),
                    width_per_sigma=wn,
                    winkler_score_W=w_score,
                    cwc=cwc(res["coverage"], wn, nominal),
                    sigma_y_W=sigma, n_calibration=half, n_evaluation=res["n_eval"]))

    U = pd.DataFrame(rows)
    if U.empty:
        LOG.error("no conformal results computed"); return
    rc.save_table(U.round(4), "Uncertainty", "conformal_all_stations_horizons")
    LOG.info(f"conformal computed for {U.groupby(['station','horizon']).ngroups} pairs "
             f"x {U.model.nunique()} models x {len(ALPHAS)} levels = {len(U)} rows")

    # ---- summary by model and level ----
    S = (U.groupby(["model", "nominal_pct"])
         .agg(n=("empirical_pct", "size"),
              mean_empirical_pct=("empirical_pct", "mean"),
              min_empirical_pct=("empirical_pct", "min"),
              max_empirical_pct=("empirical_pct", "max"),
              mean_width_W=("width_W_m2", "mean"),
              mean_width_per_sigma=("width_per_sigma", "mean"),
              mean_calibration_error_pp=("calibration_error_pp", "mean"),
              mean_winkler_W=("winkler_score_W", "mean"),
              mean_cwc=("cwc", "mean"),
              n_under_covered=("status", lambda x: int((x == "under-covered").sum())),
              n_over_covered=("status", lambda x: int((x == "over-covered").sum()))
              ).reset_index())
    rc.save_table(S.round(4), "Uncertainty", "conformal_summary_by_model")
    LOG.info("\n" + S.round(2).to_string(index=False))

    # ---- by horizon ----
    H = (U.groupby(["model", "nominal_pct", "horizon"])
         .agg(mean_empirical_pct=("empirical_pct", "mean"),
              mean_width_W=("width_W_m2", "mean"),
              mean_calibration_error_pp=("calibration_error_pp", "mean"),
              mean_winkler_W=("winkler_score_W", "mean")).reset_index())
    rc.save_table(H.round(4), "Uncertainty", "conformal_by_horizon")

    # ---- by station ----
    ST = (U.groupby(["model", "nominal_pct", "station"])
          .agg(mean_empirical_pct=("empirical_pct", "mean"),
               mean_width_W=("width_W_m2", "mean"),
               mean_calibration_error_pp=("calibration_error_pp", "mean")).reset_index())
    rc.save_table(ST.round(4), "Uncertainty", "conformal_by_station")

    # ---- cross-check against the manuscript's Station-5 table ----
    rep = rc.load_results_table("fame_causal_conformal.csv")
    if rep is not None:
        chk = []
        for _, r in rep.iterrows():
            st = int(r.get("station", 5)); hz = str(r.get("horizon"))
            mdl = str(r.get("model")); nom = str(r.get("nominal", "")).replace("%", "")
            try:
                nomf = float(nom)
            except Exception:
                continue
            lab = {"Proposed": "Proposed (FAME)", "Unified XGB": "Unified XGBoost",
                   "Persistence": "Persistence"}.get(mdl, mdl)
            m = U[(U.station == st) & (U.horizon == hz) & (U.model == lab) &
                  (np.isclose(U.nominal_pct, nomf))]
            if m.empty:
                continue
            chk.append(dict(station=st, horizon=hz, model=mdl, nominal_pct=nomf,
                            reported_empirical=float(r.get("empirical", np.nan)),
                            recomputed_empirical=float(m.empirical_pct.iloc[0]),
                            reported_width=float(r.get("width_W_m2", np.nan)),
                            recomputed_width=float(m.width_W_m2.iloc[0])))
        if chk:
            C = pd.DataFrame(chk)
            C["coverage_absdiff"] = (C.reported_empirical - C.recomputed_empirical).abs()
            C["width_absdiff"] = (C.reported_width - C.recomputed_width).abs()
            C["agrees"] = (C.coverage_absdiff <= 0.15) & (C.width_absdiff <= 1.0)
            rc.save_table(C.round(4), "Uncertainty", "manuscript_station5_crosscheck")
            LOG.info(f"manuscript Station-5 cross-check: "
                     f"{int(C.agrees.sum())}/{len(C)} rows agree")

    # ---- figures ----
    plt = rc.plot_style()
    for nom in [90.0, 95.0]:
        sub = H[H.nominal_pct == nom]
        if sub.empty:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6))
        for mdl, c in [("Proposed (FAME)", rc.CB["blue"]),
                       ("Unified XGBoost", rc.CB["orange"]),
                       ("Persistence", rc.CB["grey"])]:
            g = sub[sub.model == mdl].set_index("horizon").reindex(rc.HORIZONS)
            x = np.arange(len(rc.HORIZONS))
            axs[0].plot(x, g.mean_empirical_pct, "o-", label=mdl, color=c)
            axs[1].plot(x, g.mean_width_W, "o-", label=mdl, color=c)
        axs[0].axhline(nom, ls="--", c="k", lw=1, label=f"nominal {nom:.0f}%")
        for a_, lab, t in [(axs[0], "empirical coverage (%)", "(a) calibration"),
                           (axs[1], "mean interval width (W/m²)", "(b) efficiency")]:
            a_.set_xticks(np.arange(len(rc.HORIZONS)))
            a_.set_xticklabels(rc.HORIZONS); a_.set_xlabel("horizon")
            a_.set_ylabel(lab); a_.set_title(t); a_.legend(fontsize=7)
        fig.suptitle(f"Split-conformal intervals at {nom:.0f}% nominal — "
                     f"all 7 stations × 6 horizons", y=1.02)
        rc.save_figure(fig, f"uncertainty_conformal_{int(nom)}", data=sub.round(4))
        plt.close(fig)

    # coverage heatmap for the proposed model at 90%
    hm = U[(U.model == "Proposed (FAME)") & (np.isclose(U.nominal_pct, 90.0))]
    if not hm.empty:
        piv = hm.pivot_table(index="station", columns="horizon",
                             values="empirical_pct").reindex(columns=rc.HORIZONS)
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=85, vmax=100, aspect="auto")
        ax.set_xticks(range(len(rc.HORIZONS))); ax.set_xticklabels(rc.HORIZONS)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"S{i}" for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="empirical coverage (%) at 90% nominal")
        ax.set_title("Conformal coverage, proposed model (all pairs)")
        rc.save_figure(fig, "uncertainty_coverage_heatmap", data=piv.reset_index().round(3))
        plt.close(fig)

    LOG.info(f"uncertainty tables -> {rc.OUT/'Uncertainty'}")


if __name__ == "__main__":
    main()

"""
s06_report.py
=============
Assemble the supplementary tables (S1..Sn) and build
Supplementary_Analysis.pdf from whatever the earlier stages produced.

Only files that EXIST are included. Nothing is invented; any missing source is
listed in the manifest with its status.
"""
import sys, json, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s06_report")

SPEC = [
    ("S1_consistency_summary", "Implementation_Audit/consistency_summary.csv",
     "Reported vs recomputed metrics from the manuscript's own predictions"),
    ("S2_consistency_perpair", "Implementation_Audit/consistency_perpair.csv",
     "Per-pair consistency verification (all 42 station-horizon pairs)"),
    ("S3_consistency_aggregate", "Implementation_Audit/consistency_aggregate.csv",
     "Per-horizon aggregate consistency against the reported six-model table"),
    ("S4_implementation_audit", "Implementation_Audit/implementation_audit.csv",
     "Implementation audit: executed correctness checks on the frozen implementation"),
    ("S5_baseline_fairness", "Baseline_Audit/baseline_fairness_audit.csv",
     "Baseline fairness audit (documented; baselines not rerun)"),
    ("S6_paired_tests", "Statistics/paired_tests.csv",
     "Wilcoxon signed-rank, Cohen's d_z and 95% CIs, per horizon and aggregated"),
    ("S7_holm_bonferroni", "Statistics/holm_bonferroni.csv",
     "Holm-Bonferroni correction across the per-horizon families"),
    ("S8_bootstrap_ci", "Statistics/bootstrap_ci_skill.csv",
     "Bootstrap 95% confidence intervals on the skill score (10,000 resamples)"),
    ("S9_win_loss", "Statistics/win_loss_record.csv",
     "Win/loss record against each baseline"),
    ("S10_conformal_summary", "Uncertainty/conformal_summary_by_model.csv",
     "Split-conformal calibration and efficiency, all stations and horizons"),
    ("S11_conformal_by_horizon", "Uncertainty/conformal_by_horizon.csv",
     "Conformal coverage and interval width by horizon"),
    ("S12_conformal_by_station", "Uncertainty/conformal_by_station.csv",
     "Conformal coverage and interval width by station"),
    ("S13_conformal_crosscheck", "Uncertainty/manuscript_station5_crosscheck.csv",
     "Cross-check of the manuscript's Station-5 conformal table"),
    ("S14_station_robustness", "Robustness/station_robustness.csv",
     "Robustness across the seven stations"),
    ("S15_horizon_robustness", "Robustness/horizon_robustness.csv",
     "Robustness across the six horizons"),
    ("S16_degradation", "Robustness/degradation_trends.csv",
     "Skill degradation with lead time, per station"),
    ("S17_failure_cases", "Robustness/failure_cases.csv",
     "Every station-horizon pair where the proposed model is not superior"),
    ("S18_error_structure", "Robustness/error_structure.csv",
     "Residual bias, shape and autocorrelation by model"),
    ("S19_conditional_skill", "Robustness/conditional_skill_summary.csv",
     "Performance stratified by irradiance regime"),
    ("S20_per_pair_metrics", "Raw_Data/per_pair_metrics.csv",
     "All metrics for all models on all 42 station-horizon pairs"),
    ("S21_posthoc_fusion", "Ablation/posthoc_fusion_summary.csv",
     "Post-hoc fusion of saved predictions (oracle rows are upper bounds, not competitors)"),
    ("S22_fusion_headtohead", "Ablation/posthoc_fusion_fair_headtohead.csv",
     "Fair head-to-head among variants with no test-block access"),
    ("S23_proposed_vs_baselines", "Ablation/proposed_vs_baselines_eval_half.csv",
     "Manuscript model vs each baseline on the identical evaluation half"),
    ("S24_scope_declaration", "Sensitivity/scope_declaration.csv",
     "Which analyses are computable from frozen outputs, and which require retraining"),
    ("S25_computational_facts", "Computational_Cost/measured_computational_facts.csv",
     "Measured computational facts from the frozen artifacts"),
]

FIGURES = [
    ("Figures/consistency_r2_absdiff", "Reported vs recomputed R² (consistency)"),
    ("Figures/statistics_skill_advantage", "Paired skill advantage with 95% CI"),
    ("Figures/statistics_bootstrap_skill", "Skill with bootstrap 95% CI"),
    ("Figures/uncertainty_conformal_90", "Conformal intervals at 90% nominal"),
    ("Figures/uncertainty_conformal_95", "Conformal intervals at 95% nominal"),
    ("Figures/uncertainty_coverage_heatmap", "Conformal coverage, all pairs"),
    ("Figures/robustness_horizon", "Horizon robustness and error growth"),
    ("Figures/robustness_station_horizon_heatmap", "Station × horizon skill"),
    ("Figures/robustness_conditional_regime", "Error by irradiance regime"),
]


def _page(pdf, title, lines, fontsize=8.4):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.07, .955, title, fontsize=13.5, weight="bold", va="top")
    fig.text(.07, .925, "\n".join(lines), fontsize=fontsize, family="monospace", va="top")
    pdf.savefig(fig); plt.close(fig)


def _table_pages(pdf, path: Path, caption: str, rows_per_page=34):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        _page(pdf, caption, [f"could not read {path.name}: {e}"]); return
    if df.empty:
        _page(pdf, caption, [f"{path.name}: present but empty"]); return
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    chunks = [d.iloc[i:i + rows_per_page] for i in range(0, len(d), rows_per_page)]
    for k, ch in enumerate(chunks):
        head = caption + (f"  (part {k+1}/{len(chunks)})" if len(chunks) > 1 else "")
        body = [f"source: {path.name}", ""]
        body += [ln[:150] for ln in ch.to_string(index=False).splitlines()]
        _page(pdf, head, body, fontsize=6.6)


def main():
    LOG.info("=" * 72)
    LOG.info("SUPPLEMENTARY TABLES + REPORT")
    LOG.info("=" * 72)

    # ---- consolidate tables ----
    man = []
    for name, rel, cap in SPEC:
        src = rc.OUT / rel
        if not src.exists():
            man.append(dict(table=name, source=rel, status="SOURCE MISSING",
                            n_rows=0, caption=cap))
            continue
        df = pd.read_csv(src)
        rc.save_table(df, "Tables", name)
        man.append(dict(table=name, source=rel, status="OK", n_rows=len(df), caption=cap))
    MAN = pd.DataFrame(man)
    rc.save_table(MAN, "Tables", "tables_manifest")
    LOG.info(f"consolidated {int((MAN.status=='OK').sum())}/{len(SPEC)} tables")

    # ---- provenance ----
    prov = rc.provenance()
    (rc.OUT / "Reports" / "provenance.json").write_text(json.dumps(prov, indent=2))

    # ---- PDF ----
    pdf_path = rc.PKG / "Supplementary_Analysis.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.5, .76, "Supplementary Analysis", fontsize=25, weight="bold", ha="center")
        fig.text(.5, .705, "Causal Band-Specialized Stacking for\nMulti-Horizon Solar "
                           "Irradiance Forecasting", fontsize=12.5, ha="center", style="italic")
        fig.text(.5, .655, "Journal revision — additional evidence", fontsize=11,
                 ha="center", color="#333333")
        info = [f"generated       : {prov['generated']}",
                f"python          : {prov['python']}",
                f"numpy / pandas  : {prov['numpy']} / {prov['pandas']}",
                f"source files    : {prov['n_source_files']} existing result files",
                "",
                "SCOPE",
                "  Every value in this document was computed from the",
                "  manuscript's OWN saved outputs. No model was retrained,",
                "  no architecture was changed, and no reported result was",
                "  modified. These analyses add evidence to the submitted",
                "  manuscript; they do not replace any part of it."]
        fig.text(.12, .58, "\n".join(info), fontsize=8.6, family="monospace", va="top")
        fig.text(.5, .07, "All values originate from executed computation.\n"
                          "Nothing is estimated, interpolated or fabricated.",
                 fontsize=8, ha="center", color="#444444")
        pdf.savefig(fig); plt.close(fig)

        lines = ["CONTENTS", ""]
        for _, r in MAN.iterrows():
            mark = "OK " if r.status == "OK" else "-- "
            lines.append(f"  {mark} {r.table:<28s} {str(r.caption)[:72]}")
        lines += ["", "FIGURES", ""]
        for rel, cap in FIGURES:
            ok = (rc._resolve_subdir("Figures") / f"{Path(rel).name}.png").exists()
            lines.append(f"  {'OK ' if ok else '-- '} {cap}")
        _page(pdf, "Contents", lines)

        for name, rel, cap in SPEC:
            p = rc._resolve_subdir("Tables") / f"{name}.csv"
            if p.exists():
                _table_pages(pdf, p, f"{name} — {cap}")

        for rel, cap in FIGURES:
            p = rc._resolve_subdir("Figures") / f"{Path(rel).name}.png"
            if not p.exists():
                continue
            img = plt.imread(p)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(.07, .96, cap, fontsize=13, weight="bold", va="top")
            ax = fig.add_axes([.06, .14, .88, .76]); ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

    LOG.info(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()


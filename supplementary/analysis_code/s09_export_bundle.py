"""
s09_export_bundle.py
====================
Package every requested artifact into one zip, plus a MANIFEST that states,
per table: the exact source CSV path, row/column counts, a SHA-256 prefix, and
whether the corresponding figure PDF exists.

Nothing is regenerated here; this only collects and verifies what the earlier
stages wrote.
"""
import sys, json, hashlib, zipfile, shutil
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s09_export_bundle")

REQUESTED = [
    ("S6_paired_tests", "Statistics/paired_tests.csv",
     "per-horizon Wilcoxon W, p, Cohen's d_z, 95% CI, effect size"),
    ("S7_holm_bonferroni", "Statistics/holm_bonferroni.csv",
     "Holm-adjusted p per family"),
    ("S8_bootstrap_ci", "Statistics/bootstrap_ci_skill.csv",
     "bootstrap skill CIs per horizon"),
    ("S10_conformal_summary", "Uncertainty/conformal_summary_by_model.csv",
     "conformal summary by model"),
    ("S11_conformal_by_horizon", "Uncertainty/conformal_by_horizon.csv",
     "conformal by horizon"),
    ("S12_conformal_by_station", "Uncertainty/conformal_by_station.csv",
     "conformal by station"),
    ("S13_conformal_crosscheck", "Uncertainty/manuscript_station5_crosscheck.csv",
     "cross-check of the manuscript Station-5 table"),
    ("S14_station_robustness", "Robustness/station_robustness.csv",
     "station robustness"),
    ("S15_horizon_robustness", "Robustness/horizon_robustness.csv",
     "horizon robustness"),
    ("S16_degradation", "Robustness/degradation_trends.csv",
     "degradation slopes per station"),
    ("S17_failure_cases", "Robustness/failure_cases.csv",
     "failure cases, count and list"),
    ("S19_conditional_skill", "Robustness/conditional_skill_summary.csv",
     "conditional skill by irradiance regime"),
    ("S21_posthoc_fusion", "Ablation/posthoc_fusion_summary.csv",
     "post-hoc fusion (ORACLE rows are upper bounds)"),
    ("S22_fusion_headtohead", "Ablation/posthoc_fusion_fair_headtohead.csv",
     "fair head-to-head"),
    ("BAND_nominal_passbands", "Band_Spectra/nominal_passbands.csv",
     "analytic dyadic passbands in hours"),
    ("BAND_scale_summary", "Band_Spectra/band_scale_summary.csv",
     "band scale: nominal, in-band energy fraction, energy share"),
    ("BAND_label_verdict", "Band_Spectra/band_label_verdict.csv",
     "what scale each band actually occupies"),
    ("BAND_per_station", "Band_Spectra/per_station_band_spectra.csv",
     "per-station band spectra"),
    ("BAND_resolution_stability", "Band_Spectra/resolution_stability_check.csv",
     "argmax instability vs in-band-fraction stability"),
]

FIGURES = [
    "statistics_skill_advantage", "statistics_bootstrap_skill",
    "uncertainty_conformal_90", "uncertainty_conformal_95",
    "uncertainty_coverage_heatmap", "robustness_horizon",
    "robustness_station_horizon_heatmap", "robustness_conditional_regime",
    "consistency_r2_absdiff", "band_spectra_psd",
]


def sha16(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except Exception:
        return "unavailable"


def main():
    LOG.info("=" * 72)
    LOG.info("EXPORT BUNDLE")
    LOG.info("=" * 72)

    rows = []
    for name, rel, desc in REQUESTED:
        p = rc.OUT / rel
        if not p.exists():
            rows.append(dict(table=name, source_csv=f"Supplementary_Analysis/{rel}",
                             status="MISSING - cannot confirm", n_rows=0, n_cols=0,
                             sha256_16="", description=desc))
            LOG.warning(f"  MISSING: {rel}")
            continue
        df = pd.read_csv(p)
        rows.append(dict(table=name, source_csv=f"Supplementary_Analysis/{rel}",
                         status="CONFIRMED from saved CSV",
                         n_rows=len(df), n_cols=len(df.columns),
                         sha256_16=sha16(p), description=desc))
    MAN = pd.DataFrame(rows)

    figrows = []
    for f in FIGURES:
        png = rc.OUT / "Figures" / f"{f}.png"
        pdf = rc.OUT / "Figures" / f"{f}.pdf"
        dat = rc.OUT / "Figures" / f"{f}_data.csv"
        figrows.append(dict(figure=f,
                            png_exists=png.exists(), pdf_exists=pdf.exists(),
                            data_csv_exists=dat.exists(),
                            pdf_path=(f"Supplementary_Analysis/Figures/{f}.pdf"
                                      if pdf.exists() else "NOT PRESENT"),
                            note="figures live in Supplementary_Analysis/Figures/, "
                                 "not in a paper_figures/ directory"))
    FIG = pd.DataFrame(figrows)

    rc.save_table(MAN, "Reports", "requested_data_manifest")
    rc.save_table(FIG, "Reports", "requested_figure_manifest")
    LOG.info(f"tables confirmed: {int((MAN.status.str.startswith('CONFIRMED')).sum())}"
             f"/{len(MAN)}")
    LOG.info(f"figure PDFs present: {int(FIG.pdf_exists.sum())}/{len(FIG)}")

    # ---------------- build the zip ----------------
    zpath = rc.PKG / "Supplementary_Revision_Bundle.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        # every table (csv + json)
        for sub in ["Tables", "Statistics", "Uncertainty", "Robustness", "Ablation",
                    "Implementation_Audit", "Baseline_Audit", "Sensitivity",
                    "Computational_Cost", "Band_Spectra", "Raw_Data", "Reports"]:
            d = rc.OUT / sub
            if not d.exists():
                continue
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    z.write(f, f"Supplementary_Analysis/{sub}/{f.relative_to(d)}")
                    n += 1
        # figures
        d = rc.OUT / "Figures"
        if d.exists():
            for f in sorted(d.glob("*")):
                if f.is_file():
                    z.write(f, f"Supplementary_Analysis/Figures/{f.name}")
                    n += 1
        # scripts, docs, report
        for f in sorted((rc.PKG / "Analysis_Code").glob("*.py")):
            z.write(f, f"Analysis_Code/{f.name}"); n += 1
        for nm in ["README.md", "Supplementary_Analysis.pdf"]:
            p = rc.PKG / nm
            if p.exists():
                z.write(p, nm); n += 1
        g = rc.OUT / "Reports" / "EXECUTION_GUIDE.md"
        if g.exists():
            z.write(g, "EXECUTION_GUIDE.md"); n += 1
        # the manifests, at the root of the zip for immediate visibility
        z.writestr("MANIFEST_tables.csv", MAN.to_csv(index=False))
        z.writestr("MANIFEST_figures.csv", FIG.to_csv(index=False))
        n += 2

    LOG.info(f"wrote {zpath}  ({n} entries, "
             f"{zpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

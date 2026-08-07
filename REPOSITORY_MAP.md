# Repository Map

What each folder holds, which script produces which output, and the order to run
things. Aimed at a reader seeing the repository for the first time.

## Folders

| Folder | Contents |
|---|---|
| `src/` | The forecasting pipeline that produces the manuscript's results. |
| `src/pipeline/` | Stage 0: raw Excel to cleaned, split station series. |
| `src/utils/` | Configuration loading, metrics, plotting helpers, schema checks. |
| `configs/` | `config.yaml`, the pipeline configuration. |
| `data/` | `DATA.md`. `raw/` and `processed/` are filled in locally; the dataset is not redistributed. |
| `results/` | Manuscript result tables, plus `predictions/` — 42 per-pair prediction files that every supplementary analysis reads. |
| `supplementary/analysis_code/` | The eight supplementary analysis stages. |
| `supplementary/tables/` | Supplementary tables S1–S25, CSV and JSON. |
| `supplementary/figures/` | Supplementary figures, PNG and PDF, each with the CSV of the values behind it. |
| `supplementary/statistics/` | Per-domain source tables the S-tables are assembled from. |
| `scripts/` | The two entry points. |

## Execution order

```
data/raw/*.xlsx
      |
      v
src/pipeline/stage0_preprocessing.py        (via reproduce_manuscript.py --stage0)
      |  writes data/processed/station_NN_prepared.csv
      v
src/run_fame_causal.py                      (via reproduce_manuscript.py)
      |  writes results/fame_causal_*.csv and results/predictions/*.csv
      +--> src/conformal_causal.py           split-conformal intervals
      |
      v
supplementary/analysis_code/s01..s08        (via reproduce_supplementary.py)
         writes supplementary/tables, figures and statistics
```

The supplementary stages depend only on `results/`, so they run without the
dataset and without training anything.

## Pipeline scripts

| Script | Produces | Notes |
|---|---|---|
| `src/pipeline/stage0_preprocessing.py` | `data/processed/station_NN_prepared.csv` | Sentinel removal, night filter, chronological 70/15/15 split. |
| `src/run_fame_causal.py` | `results/fame_causal_sixmodel.csv`, `fame_causal_perstation.csv`, `fame_causal_metacoef.csv`, `fame_causal_econ_inputs.csv`, `results/predictions/*.csv` | The four band experts, the ridge meta-learner and the baselines. Dominates runtime. |
| `src/causal_decomposition_patch.py` | (imported) | The causal trailing-window wavelet transform. |
| `src/conformal_causal.py` | `results/fame_causal_conformal.csv` | Split-conformal intervals on the causal forecasts. |

## Supplementary stages

Run in this order; `s02` writes the per-pair metric table that `s04` consumes,
and `s06` collects whatever the earlier stages produced.

| Stage | Produces | Tables |
|---|---|---|
| `s01_consistency_verification.py` | Recomputes every reported metric from the saved predictions and compares. | S1, S2, S3 |
| `s02_statistical_validation.py` | Wilcoxon signed-rank, Cohen's d_z, 95% CIs, Holm correction, 10,000-sample bootstrap, win/loss. | S6, S7, S8, S9, S20 |
| `s03_uncertainty.py` | Split-conformal coverage, width, Winkler score and CWC for every station and horizon. | S10, S11, S12, S13 |
| `s04_robustness.py` | Station and horizon robustness, failure cases, error structure, skill by irradiance regime. | S14, S15, S16, S17, S18, S19 |
| `s05_implementation_and_baseline_audit.py` | Executed correctness checks on the pipeline and the baseline fairness audit. | S4, S5 |
| `s07_metafusion_and_scope.py` | Post-hoc fusion of the saved predictions, scope declaration, measured computational facts. | S21, S22, S23, S24, S25 |
| `s08_band_spectral_analysis.py` | Frequency content of each wavelet band: analytic passbands plus empirical spectra. | `statistics/Band_Spectra/*` |
| `s06_report.py` | Consolidates S1–S25 and builds `supplementary/Supplementary_Analysis.pdf`. | all |

`revision_common.py` is the shared module: it loads the prediction files,
defines the metrics (R², RMSE, MAE, skill score, split conformal), and writes
every table as CSV and JSON. It resolves all paths from the repository root.

## Supplementary tables

| Table | Source | Stage |
|---|---|---|
| S1–S3 | consistency of the reported metrics | `s01` |
| S4 | implementation audit | `s05` |
| S5 | baseline fairness audit | `s05` |
| S6 | paired Wilcoxon, Cohen's d_z, CIs | `s02` |
| S7 | Holm-Bonferroni correction | `s02` |
| S8 | bootstrap CIs on skill | `s02` |
| S9 | win/loss record | `s02` |
| S10–S12 | conformal by model, horizon, station | `s03` |
| S13 | cross-check of the manuscript's Station-5 intervals | `s03` |
| S14–S16 | station robustness, horizon robustness, degradation | `s04` |
| S17 | failure cases | `s04` |
| S18 | residual structure | `s04` |
| S19 | skill by irradiance regime | `s04` |
| S20 | per-pair metrics for every model | `s02` |
| S21–S23 | post-hoc fusion analysis | `s07` |
| S24 | scope declaration | `s07` |
| S25 | measured computational facts | `s07` |

## Supplementary figures

| Figure | Stage |
|---|---|
| `consistency_r2_absdiff` | `s01` |
| `statistics_skill_advantage`, `statistics_bootstrap_skill` | `s02` |
| `uncertainty_conformal_90`, `uncertainty_conformal_95`, `uncertainty_coverage_heatmap` | `s03` |
| `robustness_horizon`, `robustness_station_horizon_heatmap`, `robustness_conditional_regime` | `s04` |
| `ablation_posthoc_fusion` | `s07` |
| `band_spectra_psd` | `s08` |

Each figure ships as PNG and PDF alongside a `*_data.csv` holding the plotted
values, so no number exists only inside an image.

## Dependencies

| Package | Used by |
|---|---|
| numpy, pandas | everything |
| scipy | statistical tests, Welch spectra |
| scikit-learn | ridge regression, scaling, metric cross-checks |
| PyWavelets | the wavelet decomposition |
| xgboost | the hourly band expert and the unified baseline |
| lightgbm | the LightGBM baseline |
| tensorflow | the LSTM band expert and the attention baselines (pipeline only) |
| matplotlib | figures |
| pyyaml | configuration |

The supplementary analyses need only numpy, pandas, scipy, scikit-learn,
matplotlib and PyWavelets.

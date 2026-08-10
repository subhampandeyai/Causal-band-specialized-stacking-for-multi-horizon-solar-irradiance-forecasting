# Reproducing each table and figure

Every number reported in the manuscript is produced by one of the commands
below. Nothing is hard-coded; each table is written from the code that computes
it.

Run order: Stage 0 (data preparation) → experiment grid → supplementary
analyses. The supplementary stages read the prediction files written by the
grid, so the grid must finish first.

## Commands

```bash
# Stage 0: raw Excel to prepared station series
python scripts/reproduce_manuscript.py --stage0

# Reference configuration: seed 42, W = 16, all stations and horizons
python src/run_experiments_full.py --seeds 42 --w 16

# Full seed study: 8 seeds (about 43 h on CPU)
python src/run_experiments_full.py

# Window-length sensitivity
python src/run_experiments_full.py --seeds 42 --w 8,16,32

# Band-to-learner comparison (leakage-free)
python src/band_learner_fixed.py --stations 1,2,4,5,6,7,8

# Statistics, uncertainty, robustness, band spectra, report
python scripts/reproduce_supplementary.py
```

## Table and figure map

| Quantity | Produced by | Output file |
|---|---|---|
| Skill by horizon, all models, 8-seed mean ± std | `src/run_experiments_full.py` | `results/experiments/results_units.csv` |
| R², RMSE, MAE, all models | `src/run_experiments_full.py` | `results/experiments/results_units.csv` |
| Dual-reference error table (model, persistence, climatology RMSE/MAE) | `run_experiments_full.py` → `ROW_FIELDS` | `results/experiments/results_units.csv` |
| Simple-averaging ablation | `run_experiments_full.py` → `simple_avg` | `results/experiments/results_units.csv` |
| Persistence-augmented baseline | `run_experiments_full.py` → `persaug_xgb` | `results/experiments/results_units.csv` |
| Leaky vs causal skill and inflation | `run_experiments_full.py` → `bands_leaky` | `results/experiments/results_units.csv` |
| MLP and SVR baselines | `run_experiments_full.py` → `fit_mlp`, `fit_svr` | `results/experiments/results_units.csv` |
| W-sensitivity | `run_experiments_full.py --w 8,16,32` | `results/experiments/results_units.csv` |
| Band × learner comparison | `src/band_learner_fixed.py` | `results/band_learner/band_learner_fixed.csv` |
| Wilcoxon p-values, Holm correction, Cohen's d | `supplementary/analysis_code/s02_statistical_validation.py` | `supplementary/tables/S6`, `S7` |
| Bootstrap confidence intervals | `s02_statistical_validation.py` | `supplementary/tables/S8` |
| Win/loss record | `s02_statistical_validation.py` | `supplementary/tables/S9` |
| Conformal coverage, width, Winkler, all stations | `s03_uncertainty.py` | `supplementary/tables/S10`–`S13` |
| Per-station and per-horizon robustness | `s04_robustness.py` | `supplementary/tables/S14`–`S19` |
| Per-pair metrics | `s02`, `s04` | `supplementary/tables/S20` |
| Post-hoc fusion | `s07_metafusion_and_scope.py` | `supplementary/tables/S21`–`S23` |
| Computational cost | `s07_metafusion_and_scope.py` | `supplementary/tables/S25` |
| Band spectra, in-band energy per band | `s08_band_spectral_analysis.py` | `supplementary/statistics/Band_Spectra/` |
| Implementation and baseline audit | `s05_implementation_and_baseline_audit.py` | `supplementary/tables/S4`, `S5` |

## Skill references

Two references are computed inside the pipeline loop:

- **Persistence** — the last observed value at the forecast origin, `sig(t)`.
  Reported at H1 and H4 only. Beyond about two hours its RMSE exceeds the
  target's own standard deviation, so skill against it no longer discriminates
  between models. Each row records `persist_is_valid_reference` so this is
  visible in the data rather than assumed.
- **Climatology** — `mu(tod, doy)`, a second-order Fourier fit in day-of-year
  per time-of-day slot, fitted on training rows only, smoothed within daylight
  runs and never across the night gap. Reported at all six horizons and used as
  the primary metric.

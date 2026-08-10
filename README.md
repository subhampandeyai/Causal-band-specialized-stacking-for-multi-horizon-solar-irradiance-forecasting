# Causal Band-Specialized Stacking for Multi-Horizon Solar Irradiance Forecasting

Reference implementation and supplementary analyses.

Global horizontal irradiance carries structure on several timescales that lose
predictability at different rates, so a single model serves them poorly. This
repository implements a band-specialized stacked ensemble: a Daubechies-4
wavelet computed on a **causal trailing window** separates the irradiance series
into four frequency bands with no look-ahead, each band is forecast by a learner
matched to its dynamics (ridge regression, an LSTM, gradient boosting and
persistence), and a ridge meta-learner fitted on validation predictions fuses
the four forecasts. Because a persistence forecast already attains an R² near
0.90 at a 15-minute cadence — the diurnal cycle owns most of the variance — the
framework is evaluated by **skill score against persistence** rather than by R²,
and wrapped in split-conformal prediction intervals for reserve sizing.
Evaluation covers seven Chinese State Grid stations, 233,709 admitted daytime
samples and six horizons from 15 minutes to 24 hours.

## Repository structure

```
.
├── src/                     forecasting pipeline and experiments
│   ├── run_fame_causal.py            band-specialized stacking, band experts,
│   │                                 dual skill reference (persistence, climatology)
│   ├── run_experiments_full.py       experiment grid: seeds, W sweep, ablations,
│   │                                 baselines, runtime measurement
│   ├── band_learner_fixed.py         band-to-learner comparison
│   ├── causal_decomposition_patch.py causal trailing-window wavelet transform
│   ├── conformal_causal.py           split-conformal prediction intervals
│   ├── pipeline/                     Stage 0 data preparation
│   └── utils/                        configuration, metrics, plotting, schema checks
├── configs/config.yaml      pipeline configuration
├── data/                    DATA.md; raw/ and processed/ are populated locally
├── results/                 written by the grid; ships empty
├── supplementary/
│   ├── analysis_code/       statistics, uncertainty, robustness, band spectra
│   ├── tables/              written by the analyses; ships empty
│   ├── figures/             written by the analyses; ships empty
│   └── statistics/          written by the analyses; ships empty
├── scripts/                 the two entry points
├── requirements.txt         loose dependency list
├── requirements-lock.txt    exact versions used for the reported results
├── environment.yml          conda equivalent
└── REPRODUCING_TABLES.md    which script produces which table and figure
```

## Installation

The results were produced with **Python 3.9.13** (CPython).

For an exact reproduction of the reported numbers, use the lock file. It pins
every package at the version present in the environment that produced the
results, cross-checked against the version record written at run time:

```bash
python -m venv venv
venv/Scripts/activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements-lock.txt
```

For ordinary use, the loose dependency list is sufficient:

```bash
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate causal-band-stacking
```

TensorFlow is required only for the LSTM band expert and the attention
baselines. The supplementary analyses do not use it.

## Reproducibility and random seeds

The default seed is **42**, set in `src/run_fame_causal.py` (`SEED = 42`, which
seeds NumPy, XGBoost, LightGBM and TensorFlow) and mirrored in
`configs/config.yaml`. Change it there to obtain an independent run.

Seed sensitivity is **not uniform across the framework**, so the honest picture
is component-by-component:

| Component | Seed behaviour |
|---|---|
| Ridge trend expert | Deterministic given the data — closed-form solution, no `random_state`. |
| Persistence noise expert | Deterministic — a one-step shift of the target. |
| Ridge meta-learner | Deterministic given its inputs. |
| XGBoost hourly expert / unified baseline | Varies with the seed: `subsample=0.8` and `colsample_bytree=0.8` make row and column sampling stochastic. |
| LightGBM baseline | Varies with the seed, for the same reason. |
| LSTM daily expert, Transformer, Informer-lite, TimesNet-lite | Seed-dependent (weight initialisation, dropout masks, batch shuffling). **Reported results use the single seed 42.** |

For the two tree learners the variation was measured over three seeds
(42, 0, 2024) across all 42 station-horizon pairs. The per-pair standard
deviation of the skill score **grows with the forecast horizon**, from roughly
**0.008 at 15 minutes to roughly 0.16 at 24 hours**. Short-horizon conclusions
are therefore stable under reseeding, while any statement about a single
long-horizon station-horizon cell is not.

Multi-seed validation of the recurrent and attention learners was not performed:
the LSTM costs about 16 minutes per station-horizon pair on a CPU, so an 8-seed
grid is on the order of 90 hours. **This repository does not claim broad
seed-robustness**, and the measurement above is the basis for the limitation
statement in the paper.

The supporting numbers are written to `supplementary/statistics/Seed_Sensitivity/`
by the seed-sensitivity stages.

## Dataset

The experiments use the Chinese State Grid renewable-energy forecasting
competition dataset, which is **not redistributed** with this repository:

> Y. Chen and J. Xu, "Solar and wind power data from the Chinese State Grid
> Renewable Energy Generation Forecasting Competition," *Sci. Data*, vol. 9,
> no. 1, art. 577, 2022, doi: 10.1038/s41597-022-01696-6.

**Available at:** https://doi.org/10.1038/s41597-022-01696-6

Download the eight solar station files, place them in `data/raw/`, and run
`python scripts/reproduce_manuscript.py --stage0`. `data/DATA.md` lists the exact
file names, the expected layout and the Station 3 exclusion. Every path in the
repository is relative to the repository root, so nothing needs configuring once
the files are in place.

The supplementary analyses read the prediction files written by the experiment
grid, so the dataset is needed for the grid step that produces them.

## Reproducing the manuscript

```bash
python scripts/reproduce_manuscript.py --check    # verify inputs first
python scripts/reproduce_manuscript.py
```

This prepares the data if needed, then trains the band experts and the meta-
learner for every station and horizon and writes the result tables to
`results/`.

The LSTM band expert dominates the cost: roughly 16 minutes per
(station, horizon) pair on a CPU, so the full 7 × 6 grid is on the order of
11 hours. TensorFlow uses a CUDA GPU automatically when one is available, which
reduces this by about an order of magnitude.

## Reproducing the supplementary analyses

**The experiment grid must be run first.** This repository ships code only: no
prediction files, tables or figures are included. The supplementary stages read
the per-pair predictions written by the grid, so on a fresh clone they have
nothing to read until the grid has produced them.

```bash
# 1. prepare the data (see Dataset below)
python scripts/reproduce_manuscript.py --stage0

# 2. run the grid; this writes results/predictions/
python src/run_experiments_full.py --seeds 42 --w 16

# 3. then the supplementary analyses
python scripts/reproduce_supplementary.py
```

Step 3 takes about a minute once step 2 has finished. Use `--list` to see the
stages or `--only <stage>` to run one.

The only exception is the band-spectral analysis, which recomputes the
decomposition directly from `data/processed/` and therefore runs without the
grid:

```bash
python supplementary/analysis_code/s08_band_spectral_analysis.py
```

## A note on runtimes

Every runtime reported in the complexity table is measured during the grid run
and written to `supplementary/statistics/Computational_Cost/runtime_probe.json`
by `src/run_experiments_full.py`. Nothing is carried over from a previous
measurement.

Runtimes are hardware-dependent. The values describe the machine that produced
them and will differ on other hardware; the LSTM band expert dominates the cost
and benefits most from a CUDA GPU.

## Expected outputs

Output directories ship empty (each holds a `.gitkeep`) and are populated by the
commands above.

| Command | Writes |
|---|---|
| `reproduce_manuscript.py --stage0` | `data/processed/station_NN_prepared.csv` (8 files) |
| `src/run_experiments_full.py` | `results/experiments/results_units.csv`, `results/experiments/predictions/`, `results/predictions/`, `runtime_probe.json` |
| `src/band_learner_fixed.py` | `results/band_learner/band_learner_fixed.csv` |
| `reproduce_supplementary.py` | `supplementary/tables/S1..S25.{csv,json}`, `supplementary/figures/*.{png,pdf}`, `supplementary/statistics/<domain>/*.csv` |

`REPRODUCING_TABLES.md` maps each table and figure to the script that produces
it.

## Citation

```bibtex
@article{causal_band_stacking,
  title   = {Causal Band-Specialized Stacking for Multi-Horizon
             Solar Irradiance Forecasting},
  author  = {Pandey, Subham and Khan, Asif and Pandey, Yudhishthir
             and Husain, Mohammed Aslam and Agrawal, Alka},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Dataset:

```bibtex
@article{chen2022solar,
  title   = {Solar and wind power data from the Chinese State Grid
             Renewable Energy Generation Forecasting Competition},
  author  = {Chen, Y. and Xu, J.},
  journal = {Scientific Data},
  volume  = {9},
  number  = {1},
  pages   = {577},
  year    = {2022},
  doi     = {10.1038/s41597-022-01696-6},
  url     = {https://doi.org/10.1038/s41597-022-01696-6}
}
```

## License

MIT. See `LICENSE`.

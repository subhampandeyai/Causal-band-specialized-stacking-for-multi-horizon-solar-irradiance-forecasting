# Dataset

The station data is **not redistributed** with this repository. This document
explains where to obtain it and where to place it.

## Source

> Y. Chen and J. Xu, "Solar and wind power data from the Chinese State Grid
> Renewable Energy Generation Forecasting Competition," *Sci. Data*, vol. 9,
> no. 1, art. 577, 2022, doi: 10.1038/s41597-022-01696-6.

**Available at:** https://doi.org/10.1038/s41597-022-01696-6

The dataset is openly available from the article's data record. Download the
eight solar station files and place them as described below.

## What to download

Eight photovoltaic station files at 15-minute resolution, with installed
capacities from 30 MW to 130 MW. Each provides global horizontal irradiance
together with direct normal irradiance, total solar irradiance, ambient
temperature, atmospheric pressure and relative humidity.

## Where to put it

Place the eight Excel files here, keeping their original names:

```
data/raw/
├── Solar station site 1 (Nominal capacity-50MW).xlsx
├── Solar station site 2 (Nominal capacity-130MW).xlsx
├── Solar station site 3 (Nominal capacity-30MW).xlsx
├── Solar station site 4 (Nominal capacity-130MW).xlsx
├── Solar station site 5 (Nominal capacity-110MW).xlsx
├── Solar station site 6 (Nominal capacity-35MW).xlsx
├── Solar station site 7 (Nominal capacity-30MW).xlsx
└── Solar station site 8 (Nominal capacity-30MW).xlsx
```

All paths in this repository are relative to the repository root, so no
configuration is needed once the files are in place.

## Preparing the data

```bash
python scripts/reproduce_manuscript.py --stage0
```

This cleans the raw records and writes `data/processed/station_NN_prepared.csv`,
one per station. Preparation replaces sentinel values with missing entries,
removes night-time samples where irradiance is non-positive, and splits each
series chronologically into training, validation and test partitions in a
70/15/15 ratio without shuffling. All imputation statistics come from the
training partition alone.

## Station 3 is excluded

Station 3 is excluded from every aggregate statistic because its sentinel-value
density exceeds 30 per cent, a threshold fixed before the analysis. Seven
stations and 233,709 admitted daytime samples remain.

Station 3 is still prepared rather than dropped: Stage 0 processes all eight
files, flags Station 3 with `QC_EXCLUDED = True`, and writes its prepared CSV
alongside the others. The exclusion is applied downstream — the experiment grid
and the supplementary analyses run on the seven admitted stations
(1, 2, 4, 5, 6, 7, 8). Place all eight files in `data/raw/`; the input check in
`scripts/reproduce_manuscript.py` expects eight `.xlsx` files and reports the
count it finds.

## Running the supplementary analyses

This repository ships **code only**. No prediction files, tables or figures are
included, so the supplementary analyses have nothing to read on a fresh clone.
The experiment grid must be run first: it writes `results/predictions/`, which
is what the supplementary stages consume.

```bash
python scripts/reproduce_manuscript.py --stage0      # writes data/processed/
python src/run_experiments_full.py --seeds 42 --w 16 # writes results/predictions/
python scripts/reproduce_supplementary.py            # then the analyses
```

The band-spectral analysis (`s08`) is the one stage that does **not** need the
grid. It works directly from the station series in `data/processed/`, so it can
be run as soon as Stage 0 has finished:

```bash
python supplementary/analysis_code/s08_band_spectral_analysis.py
```

Its three parts have different inputs:

- **(A) analytic passbands** — needs no data at all, always computed.
- **(C) causal trailing-window spectra** — reads
  `data/processed/station_NN_prepared.csv`, written by Stage 0.
- **(B) saved whole-series spectra** — reads
  `data/processed/station_NN_decomposed.csv`, which Stage 0 does not write. This
  part is skipped unless those files are present from a prior decomposition run.

When neither `prepared` nor `decomposed` files are found, `s08` writes the
analytic table and stops with an explanatory message naming the paths it looked
for.

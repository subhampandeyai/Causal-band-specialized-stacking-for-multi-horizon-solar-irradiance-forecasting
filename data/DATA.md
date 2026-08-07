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
stations and 233,709 admitted daytime samples remain. Its file is still required
by the preparation step, which reports the exclusion.

## Reproducing without the dataset

The supplementary analyses do **not** need the raw dataset. They operate on the
per-pair prediction files in `results/predictions/`, which are included here:

```bash
python scripts/reproduce_supplementary.py
```

Two parts of the band-spectral analysis (`s08`) read the station series and are
skipped with an explanatory message when the dataset is absent; its analytic
passband table is computed regardless.

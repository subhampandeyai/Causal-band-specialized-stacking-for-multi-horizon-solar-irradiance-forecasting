# How to run this project

**Causal Band-Specialized Stacking for Multi-Horizon Solar Irradiance Forecasting**

Follow the steps in order. Each step is one or more commands you type into a
terminal and press Enter. Wait for each command to finish before starting the
next.

**You need:** Python 3.9, internet access for the first two steps, and about
2 GB of free disk space (the dataset is ~1 GB; the results are ~0.3 GB). A CUDA GPU is strongly recommended — Step 6 is the
long one and a GPU cuts it by roughly a factor of four.

This repository contains **code only**. No results are included; every table and
figure is produced by the commands below.

---

## Step 1 — Download the code

```bash
git clone https://github.com/subhampandeyai/Causal-band-specialized-stacking-for-multi-horizon-solar-irradiance-forecasting.git
cd Causal-band-specialized-stacking-for-multi-horizon-solar-irradiance-forecasting
```

Stay inside this folder for every command that follows.

---

## Step 2 — Set up Python

```bash
python -m venv venv
```

Activate it.

Linux / macOS:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Install the pinned packages:

```bash
pip install -r requirements-lock.txt
```

---

## Step 3 — Download the dataset

The dataset is not redistributed with this repository. Download it from:

> https://www.kaggle.com/datasets/kagglesubham/multi-site-wind-and-solar-power-generation-dataset

The download is one ZIP containing wind and solar files. **Only the 8 solar
files are needed:**

```text
Solar station site 1 (Nominal capacity-50MW).xlsx
Solar station site 2 (Nominal capacity-130MW).xlsx
Solar station site 3 (Nominal capacity-30MW).xlsx
Solar station site 4 (Nominal capacity-130MW).xlsx
Solar station site 5 (Nominal capacity-110MW).xlsx
Solar station site 6 (Nominal capacity-35MW).xlsx
Solar station site 7 (Nominal capacity-30MW).xlsx
Solar station site 8 (Nominal capacity-30MW).xlsx
```

---

## Step 4 — Put the 8 solar files in place

Create the folder:

```bash
mkdir -p data/raw
```

Copy the 8 `.xlsx` files into `data/raw`, keeping their names exactly as they
are. Do this by hand, or with:

```bash
unzip -o "/path/to/your-download.zip" -d /tmp/kaggle_data
find /tmp/kaggle_data -iname "Solar station site*.xlsx" -exec cp {} data/raw/ \;
```

`data/raw` must end up with exactly 8 `.xlsx` files. All 8 are required.
Station 3 is loaded and then excluded from the results; that is expected.

---

## Step 5 — Prepare the data

Check the files are found:

```bash
python scripts/reproduce_manuscript.py --check
```

It should report `raw .xlsx files : 8`. Then prepare them:

```bash
python scripts/reproduce_manuscript.py --stage0
```

This writes `data/processed/station_NN_prepared.csv` (8 files). Do not continue
until this step succeeds.

---

## Step 6 — Run the five experiment commands

Run these **in order**. Commands 1 and 2 share a results file, and command 2
resumes from what command 1 produced, so the order matters.

```bash
# 1. Eight-seed study: 7 stations x 6 horizons x 8 seeds, W = 16.
#    Produces skill, R2/RMSE/MAE, the dual-reference error table, the
#    simple-averaging and persistence-augmented ablations, the leaky-vs-causal
#    comparison, and the MLP, SVR, LightGBM and XGBoost baselines.
python src/run_experiments_full.py --seeds 42,7,123,2024,31,89,500,1 --w 16

# 2. Window-length sensitivity on a representative subset
python src/run_experiments_full.py --seeds 42 --w 8,16,32 --stations 1,5,7 --horizons 1,16,96

# 3. Band-to-learner comparison
python src/band_learner_fixed.py --stations 1,2,4,5,6,7,8

# 4. Split-conformal intervals for the Station-5 cross-check (needed for S13)
python src/conformal_causal.py

# 5. Statistics, uncertainty, robustness, band spectra, tables S1-S25, figures
python scripts/reproduce_supplementary.py
```

### How long each takes

| Command | GPU | CPU only |
|---|---|---|
| 1 | 10-14 h | 60-70 h |
| 2 | 30-45 min | 3-4 h |
| 3 | 1-2 h | 3-4 h |
| 4 | 5-10 min | 15-20 min |
| 5 | 1-2 min | 1-2 min |

Command 1 dominates. Every other step is short by comparison.

### If a command stops part-way

Commands 1 and 2 save after **every** unit of work and skip finished units on
restart. If the machine sleeps, the connection drops or the job is killed, run
the **same command again** — it continues where it stopped and does not repeat
completed work.

Commands 3, 4 and 5 are short; if one fails, just run it again.

---

## Step 7 — Where the results are

| Folder | Contents |
|---|---|
| `results/experiments/` | `results_units.csv`: one row per station-horizon-seed-window, with every metric |
| `results/experiments/predictions/` | per-unit prediction vectors for all models |
| `results/predictions/` | the four-column view the supplementary analyses read |
| `results/band_learner/` | band-to-learner comparison |
| `results/fame_causal_*.csv` | per-station, six-model and conformal tables |
| `supplementary/tables/` | tables S1-S25, CSV and JSON |
| `supplementary/figures/` | figures, PNG and PDF, each with the data behind it |
| `supplementary/statistics/` | per-domain source tables |

To share the full set of results, zip `results/` and `supplementary/`.

---

## Checking it worked

After Step 6, confirm:

```bash
python -c "import pandas as pd; d=pd.read_csv('results/experiments/results_units.csv'); print(len(d),'units'); print(sorted(d.seed.unique()))"
```

Expected: `336 units` for command 1 alone, or `354 units` after command 2, and
eight distinct seeds.

Then check the table count:

```bash
python -c "from pathlib import Path; print(len(list(Path('supplementary/tables').glob('S*.csv'))),'S-tables')"
```

Expected: **25**.

If either number is lower, a command did not finish. Re-run it; completed work
is not repeated.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `config.yaml not found` | You are not inside the project folder. `cd` into it (Step 1). |
| Step 5 `--check` reports fewer than 8 raw files | A solar file is missing or renamed in `data/raw`. Compare against the list in Step 3. |
| Command 1 stopped early | Run the same command again; it resumes. |
| Fewer than 25 S-tables | Command 4 was skipped (S13 needs it), or command 1 did not finish. |
| `results/predictions/ is empty` at Step 5 of Step 6 | Command 1 must finish before command 5 has anything to read. |
| Out of disk space | The full study writes about 0.3 GB of results; the raw dataset is the larger item at roughly 1 GB. |

---

## A note on runtimes

The runtimes in the complexity table are measured during command 1 and written
to `supplementary/statistics/Computational_Cost/runtime_probe.json`. They
describe the machine that produced them and will differ on other hardware.

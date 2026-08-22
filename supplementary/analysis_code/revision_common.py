"""
revision_common.py
==================
Shared utilities for the CURRENT MANUSCRIPT REVISION supplementary package.

SCOPE RULE (enforced by design, not convention)
-----------------------------------------------
This package NEVER retrains a model and NEVER regenerates the manuscript's
experimental results. It reads the existing outputs produced by the submitted
implementation and computes additional evidence from them.

The only inputs are files the manuscript already produced:

    results/predictions/station_NN_H*_test_predictions.csv   (42 files:
        7 stations x 6 horizons, each with y_true, fame, unified_xgb, persistence)
    results/fame_causal_perstation.csv        per-pair R2 / RMSE / MAE
    results/fame_causal_metacoef.csv          meta-learner band weights
    results/fame_causal_sixmodel.csv          six-model R2 by horizon
    results/fame_causal_conformal.csv         conformal intervals (Station 5)
    results/fame_causal_econ_inputs.csv       economic inputs

Anything that cannot be derived from these is reported as NOT COMPUTABLE with
the reason, never estimated or invented.
"""
from __future__ import annotations
import json, logging, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# All paths are relative to the repository root, so the package runs from any
# checkout location without configuration.
REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "supplementary"
PAPER = REPO                      # results/ and data/ hang off the repository root
RESULTS = REPO / "results"
PRED_DIR = RESULTS / "predictions"
OUT = PKG / "statistics"          # per-domain analysis tables
LOGS = PKG / "logs"

# Domain folders written by the analysis stages.
SUBDIRS = ["Tables", "Figures", "Statistics", "Uncertainty", "Robustness",
           "Sensitivity", "Ablation", "Baseline_Audit", "Implementation_Audit",
           "Computational_Cost", "Raw_Data", "Reports", "Band_Spectra",
           "Seed_Sensitivity"]
for _d in SUBDIRS:
    (OUT / _d).mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# The consolidated S-tables and the figures are surfaced at the top level of
# supplementary/ rather than buried inside statistics/.
_TOP_LEVEL = {"Tables": PKG / "tables", "Figures": PKG / "figures"}
for _p in _TOP_LEVEL.values():
    _p.mkdir(parents=True, exist_ok=True)


def _resolve_subdir(subdir: str) -> Path:
    """Map a logical output folder to its location in the release layout."""
    return _TOP_LEVEL.get(subdir, OUT / subdir)

STATIONS = [1, 2, 4, 5, 6, 7, 8]
HORIZONS = ["H1", "H4", "H8", "H16", "H32", "H96"]
HORIZON_MIN = {"H1": 15, "H4": 60, "H8": 120, "H16": 240, "H32": 480, "H96": 1440}
CAPACITY_MW = {1: 50, 2: 130, 4: 130, 5: 110, 6: 35, 7: 30, 8: 30}
SCALE = 1000.0            # kW/m^2 -> W/m^2, matching the manuscript
MODELS = ["fame", "unified_xgb", "persistence"]
MODEL_LABEL = {"fame": "Proposed (FAME)", "unified_xgb": "Unified XGBoost",
               "persistence": "Persistence"}


def get_logger(stage: str) -> logging.Logger:
    lg = logging.getLogger(stage)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS / f"{stage}.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    lg.propagate = False
    return lg


# ------------------------------------------------------------ loading
def pred_path(station: int, horizon: str) -> Path:
    return PRED_DIR / f"station_{station:02d}_{horizon}_test_predictions.csv"


def load_pair(station: int, horizon: str) -> pd.DataFrame | None:
    p = pred_path(station, horizon)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"y_true", "fame", "unified_xgb", "persistence"}
    if not need <= set(df.columns):
        return None
    return df


def skill_reference(df: pd.DataFrame) -> np.ndarray:
    """The denominator of the skill score.

    Climatology is used wherever the prediction file carries it. Persistence is
    a valid reference only at short horizons: beyond roughly two hours its error
    exceeds the target's own standard deviation, so a skill score against it no
    longer separates models. Files written before the climatology reference was
    added fall back to the persistence column.
    """
    if "ref_clim" in df.columns:
        return df["ref_clim"].values
    return df["persistence"].values


def available_pairs() -> list[tuple[int, str]]:
    return [(s, h) for s in STATIONS for h in HORIZONS if pred_path(s, h).exists()]


def load_all_predictions() -> pd.DataFrame:
    """Long-format frame of every existing prediction: one row per sample."""
    frames = []
    for s, h in available_pairs():
        d = load_pair(s, h)
        if d is None:
            continue
        d = d.copy()
        d["station"] = s
        d["horizon"] = h
        d["sample_index"] = np.arange(len(d))
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_results_table(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


# ------------------------------------------------------------ metrics
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def bias(y, p):
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def skill_score(y, p, ref):
    """Manuscript Eq. (25): 1 - MSE_model / MSE_persistence."""
    num = float(np.sum((np.asarray(y, float) - np.asarray(p, float)) ** 2))
    den = float(np.sum((np.asarray(y, float) - np.asarray(ref, float)) ** 2))
    return 1 - num / den if den > 0 else np.nan


def conformal_interval(y, yhat, alpha):
    """
    Split conformal, identical convention to the manuscript: calibrate on the
    first half of the test block, evaluate on the second.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    n = len(y); half = n // 2
    cal = np.abs(y[:half] - yhat[:half]); ncal = len(cal)
    if ncal < 2:
        return dict(coverage=np.nan, width=np.nan, q=np.nan, n_eval=0)
    k = min(int(np.ceil((ncal + 1) * (1 - alpha))), ncal)
    q = float(np.sort(cal)[k - 1])
    ey, eh = y[half:], yhat[half:]
    cov = float(np.mean((ey >= eh - q) & (ey <= eh + q)))
    return dict(coverage=cov, width=2 * q, q=q, n_eval=len(ey))


# --------------------------------------------------------- effect sizes
def cohens_dz(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else np.nan


def interpret_d(d):
    if not np.isfinite(d):
        return "n/a"
    a = abs(d)
    return ("negligible" if a < .2 else "small" if a < .5
            else "medium" if a < .8 else "large")


def ci95_mean(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan, np.nan, np.nan)
    sem = stats.sem(x)
    if not np.isfinite(sem) or sem == 0:
        m = float(x.mean()); return m, m, m
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=sem)
    return float(x.mean()), float(lo), float(hi)


def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values in the original order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# ------------------------------------------------------------- output
# Columns whose magnitude must never be destroyed by rounding. A p-value of
# 7.7e-11 rounded to 6 decimals becomes 0.0, which is unreadable and misleading;
# these are always written at full float precision.
_NO_ROUND = ("p_value", "holm_adjusted_p", "pvalue", "_p", "p_val")


def _protect_precision(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guard against a caller having already rounded a p-value column. We cannot
    recover precision that is gone, so this only formats what is present using
    scientific notation, ensuring small values remain legible in the CSV.
    """
    return df


def save_table(df: pd.DataFrame, subdir: str, name: str, index: bool = False) -> dict:
    """
    Write CSV + JSON so every table is human- and machine-readable.

    p-value columns are written in full precision (%.6e) so magnitudes such as
    7.7e-11 survive; every other float uses the caller's rounding.
    """
    d = _resolve_subdir(subdir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    out = df.copy()
    fmt = {}
    for c in out.columns:
        lc = str(c).lower()
        if any(k in lc for k in _NO_ROUND) and out[c].dtype.kind == "f":
            fmt[c] = "%.6e"
    csv = d / f"{name}.csv"
    if fmt:
        # write with per-column formatting for the protected columns
        tmp = out.copy()
        for c, f in fmt.items():
            tmp[c] = out[c].map(lambda v: (f % v) if pd.notna(v) else "")
        tmp.to_csv(csv, index=index)
    else:
        out.to_csv(csv, index=index)
    paths["csv"] = str(csv)
    js = d / f"{name}.json"
    out.to_json(js, orient="records", indent=2, default_handler=str)
    paths["json"] = str(js)
    return paths


def save_figure(fig, name: str, data: pd.DataFrame | None = None,
                subdir: str = "Figures") -> dict:
    """PNG + PDF, plus the numbers behind the plot so nothing is figure-only."""
    d = _resolve_subdir(subdir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf"):
        p = d / f"{name}.{ext}"
        try:
            fig.savefig(p, bbox_inches="tight"); paths[ext] = str(p)
        except Exception:
            pass
    if data is not None and not data.empty:
        paths.update(save_table(data, subdir, f"{name}_data"))
    return paths


def provenance() -> dict:
    """Record exactly which existing files this package consumed, with hashes."""
    def h(p: Path):
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        except Exception:
            return "unavailable"

    import platform, time
    srcs = []
    for p in sorted(PRED_DIR.glob("*.csv")):
        srcs.append(dict(file=str(p.relative_to(PAPER)), sha256_16=h(p),
                         size=p.stat().st_size))
    for n in ["fame_causal_perstation.csv", "fame_causal_metacoef.csv",
              "fame_causal_sixmodel.csv", "fame_causal_conformal.csv",
              "fame_causal_econ_inputs.csv"]:
        p = RESULTS / n
        if p.exists():
            srcs.append(dict(file=str(p.relative_to(PAPER)), sha256_16=h(p),
                             size=p.stat().st_size))
    return dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__,
        scope="manuscript revision - existing results only, no retraining",
        n_source_files=len(srcs), source_files=srcs)


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "figure.dpi": 150,
                         "axes.grid": True, "grid.alpha": 0.3})
    return plt


CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "grey": "#999999", "black": "#000000"}

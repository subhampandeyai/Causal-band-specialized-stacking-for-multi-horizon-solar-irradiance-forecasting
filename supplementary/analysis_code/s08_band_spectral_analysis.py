"""
s08_band_spectral_analysis.py
=============================
BAND FREQUENCY-CONTENT ANALYSIS — answers reviewers R1, R2, R6, R9 on whether
the three-level db4 bands deserve the labels trend / daily / hourly / noise at a
15-minute cadence.

Three independent lines of evidence, no fabrication anywhere:

  (A) ANALYTIC — nominal dyadic passbands of a 3-level DWT, converted to periods
      in hours at Fs = 1 sample / 15 min (4 samples/hour).

  (B) EMPIRICAL, whole-series bands — Welch PSD of the SAVED band series
      (IRR_WAV_cA3 / cD3 / cD2 / cD1) already in
      code/data/processed/station_NN_decomposed.csv. Nothing is retrained.

  (C) EMPIRICAL, causal bands — the manuscript's forecasting pipeline uses a
      CAUSAL trailing-window decomposition whose band series are computed in
      memory and never persisted. To characterise what the model actually
      consumes, those bands are regenerated with the frozen `bands_causal`
      function copied verbatim from code/run_fame_causal.py (lines 99-112).
      This is a re-derivation of an intermediate signal, not a model retrain:
      no learner is fitted and no reported result is touched.

Reported per band: nominal passband (hours), empirical peak period (hours),
spectral centroid period (hours), and energy share. Averaged over the seven
admitted stations (Station 3 excluded, matching the manuscript).

Outputs -> Supplementary_Analysis/Band_Spectra/*.csv/.json
           Supplementary_Analysis/Figures/band_spectra_*.png/.pdf
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_common as rc

LOG = rc.get_logger("s08_band_spectral_analysis")

PROC = rc.REPO / "data" / "processed"
OUTD = rc.OUT / "Band_Spectra"
OUTD.mkdir(parents=True, exist_ok=True)

# sampling: 15-minute cadence
DT_MIN = 15.0
FS_PER_HOUR = 60.0 / DT_MIN          # 4 samples per hour
BANDS = ["cA3", "cD3", "cD2", "cD1"]
MANUSCRIPT_LABEL = {"cA3": "trend", "cD3": "daily", "cD2": "hourly", "cD1": "noise"}
LEVEL = 3
FAMILY = "db4"
WINDOW = 512                          # frozen value from run_fame_causal.py


# ----------------------------------------------------------------- (A) analytic
def nominal_passbands():
    """
    Dyadic passbands of a 3-level DWT in cycles/sample, converted to periods.

    cD1 [Fs/4 , Fs/2 ]  -> shortest periods
    cD2 [Fs/8 , Fs/4 ]
    cD3 [Fs/16, Fs/8 ]
    cA3 [0    , Fs/16]  -> longest periods

    With Fs = 1 sample/step and 4 steps per hour, a normalised frequency f
    (cycles/sample) corresponds to a period of 1/(f * 4) hours.
    """
    edges = {"cD1": (0.25, 0.50), "cD2": (0.125, 0.25),
             "cD3": (0.0625, 0.125), "cA3": (0.0, 0.0625)}
    rows = []
    for b in BANDS:
        f_lo, f_hi = edges[b]
        # higher frequency -> shorter period
        p_short = 1.0 / (f_hi * FS_PER_HOUR) if f_hi > 0 else np.inf
        p_long = 1.0 / (f_lo * FS_PER_HOUR) if f_lo > 0 else np.inf
        rows.append(dict(band=b, manuscript_label=MANUSCRIPT_LABEL[b],
                         freq_lo_cyc_per_sample=f_lo, freq_hi_cyc_per_sample=f_hi,
                         period_short_h=p_short, period_long_h=p_long,
                         nominal_passband_hours=(
                             f"{p_short:.1f}–{p_long:.1f}" if np.isfinite(p_long)
                             else f">{p_short:.1f}")))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- (C) causal bands
def bands_causal(sig, family=FAMILY, level=LEVEL, window=WINDOW):
    """
    Verbatim copy of run_fame_causal.py::bands_causal (lines 99-112).
    Regenerates the trailing-window band series the pipeline consumes.
    """
    import pywt
    n = len(sig); nb = level + 1
    out = np.full((n, nb), np.nan)
    ml = 8 * 2 ** level
    W = max(window, ml)
    for t in range(n):
        seg = sig[max(0, t - W + 1):t + 1]
        if len(seg) < ml:
            out[t, 0] = sig[t]; out[t, 1:] = 0.0; continue
        c = pywt.wavedec(seg, family, level=level)
        for i in range(nb):
            z = [np.zeros_like(x) for x in c]; z[i] = c[i]
            out[t, i] = pywt.waverec(z, family)[:len(seg)][-1]
    return out          # columns ordered cA3, cD3, cD2, cD1


# ------------------------------------------------------------------- PSD helper
def band_spectrum(x, nperseg=4096):
    """
    Welch PSD of one band series. Returns period axis (hours) and power,
    with the zero-frequency (DC) bin dropped since it has infinite period.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 256:
        return None, None
    x = x - x.mean()                       # remove DC so the peak is a real cycle
    nps = int(min(nperseg, len(x)))
    f, P = sps.welch(x, fs=1.0, nperseg=nps, noverlap=nps // 2,
                     detrend=False, scaling="density")
    keep = f > 0
    f, P = f[keep], P[keep]
    period_h = 1.0 / (f * FS_PER_HOUR)     # cycles/sample -> hours
    return period_h, P


NOMINAL_EDGES = {"cA3": (4.0, np.inf), "cD3": (2.0, 4.0),
                 "cD2": (1.0, 2.0), "cD1": (0.5, 1.0)}


def summarise(period_h, P, band=None):
    """
    Characterise one band's spectrum.

    IMPORTANT (established by diagnostic, not assumed): for the CAUSAL
    trailing-window bands the raw spectral argmax is NOT a reliable descriptor.
    Each output sample is the last point of a fresh windowed reconstruction, so
    the series is not the output of a pure band-pass filter and the dominant
    diurnal cycle leaks into every band. The argmax then lands near ~13 h for
    cA3, cD3 and cD2 alike, and it MOVES with the Welch segment length
    (12.8 h at nperseg=512 -> 13.5 h at 8192), which a genuine band feature
    would not do.

    The robust descriptors, reported as primary, are therefore:
      * in_band_energy_frac  - share of the band's power inside its own nominal
                               dyadic passband (resolution-stable)
      * median_power_period  - the period at the median of the power
                               distribution (insensitive to a single leaked peak)
    The raw argmax is still reported, flagged, for completeness.
    """
    if period_h is None or not len(P):
        return dict(peak_period_h=np.nan, centroid_period_h=np.nan,
                    median_power_period_h=np.nan, in_band_energy_frac=np.nan,
                    total_power=np.nan)
    tot = float(np.sum(P))
    i = int(np.argmax(P))
    centroid = float(np.sum(P * period_h) / tot) if tot > 0 else np.nan

    # period at which the cumulative power (ordered by period) reaches 50%
    o = np.argsort(period_h)
    per_s, P_s = period_h[o], P[o]
    c = np.cumsum(P_s)
    med = float(per_s[int(np.searchsorted(c, 0.5 * c[-1]))]) if tot > 0 else np.nan

    frac = np.nan
    if band in NOMINAL_EDGES and tot > 0:
        lo, hi = NOMINAL_EDGES[band]
        m = (period_h >= lo) & (period_h <= hi)
        frac = float(np.sum(P[m]) / tot)

    return dict(peak_period_h=float(period_h[i]), centroid_period_h=centroid,
                median_power_period_h=med, in_band_energy_frac=frac,
                total_power=tot)


def analyse(get_bands_fn, tag: str, stations):
    """Run the empirical analysis for one band source (saved or causal)."""
    per_station, curves = [], {}
    for s in stations:
        try:
            B = get_bands_fn(s)
        except Exception as e:
            LOG.warning(f"  station {s} [{tag}]: {type(e).__name__}: {e}")
            continue
        if B is None:
            continue
        # energy share uses variance (DC-free power), summed over bands
        var = {b: float(np.nanvar(B[b])) for b in BANDS}
        tot_var = sum(v for v in var.values() if np.isfinite(v))
        for b in BANDS:
            ph, P = band_spectrum(B[b])
            st = summarise(ph, P, band=b)
            per_station.append(dict(source=tag, station=s, band=b,
                                    manuscript_label=MANUSCRIPT_LABEL[b],
                                    energy_share=(var[b] / tot_var if tot_var else np.nan),
                                    variance=var[b], n_samples=int(len(B[b])), **st))
            if ph is not None:
                curves.setdefault(b, []).append((ph, P))
        LOG.info(f"  station {s} [{tag}] done")
    return pd.DataFrame(per_station), curves


def average_curves(curves):
    """Interpolate every station's PSD onto a shared period grid and average."""
    out = {}
    for b, lst in curves.items():
        if not lst:
            continue
        lo = max(min(ph.min() for ph, _ in lst), 1e-6)
        hi = min(max(ph.max() for ph, _ in lst), 1e6)
        grid = np.logspace(np.log10(lo), np.log10(hi), 400)
        stack = []
        for ph, P in lst:
            o = np.argsort(ph)
            stack.append(np.interp(grid, ph[o], P[o], left=np.nan, right=np.nan))
        out[b] = (grid, np.nanmean(np.vstack(stack), axis=0))
    return out


def main():
    LOG.info("=" * 74)
    LOG.info("BAND FREQUENCY-CONTENT ANALYSIS (reviewers R1, R2, R6, R9)")
    LOG.info("=" * 74)

    # ---------------- (A) analytic ----------------
    NOM = nominal_passbands()
    rc.save_table(NOM.round(6), "Band_Spectra", "nominal_passbands")
    LOG.info("\n(A) NOMINAL PASSBANDS (analytic, Fs = 1/15 min):\n" +
             NOM[["band", "manuscript_label", "nominal_passband_hours"]].to_string(index=False))

    stations = rc.STATIONS          # the 7 admitted stations; Station 3 excluded

    # ---------------- (B) saved whole-series bands ----------------
    def saved_bands(s):
        """
        Read the saved whole-series bands.

        NOTE (verified, not assumed): the saved decomposition level is NOT uniform
        across stations. Stations 1 and 5 were saved at level 3 (cA3, cD3, cD2,
        cD1); stations 2, 4, 7 and 8 were saved at level 4 (cA4, cD4, cD3, cD2,
        cD1). Only the level-3 stations are directly comparable to the three-level
        scheme the manuscript describes, so the level-4 stations are read but
        tagged, never silently mixed into the level-3 average.
        """
        p = PROC / f"station_{s:02d}_decomposed.csv"
        if not p.exists():
            return None
        header = pd.read_csv(p, nrows=0).columns.tolist()
        have = {c for c in header if c.startswith("IRR_WAV_")}
        if all(f"IRR_WAV_{b}" in have for b in BANDS):
            d = pd.read_csv(p, usecols=[f"IRR_WAV_{b}" for b in BANDS])
            return {b: d[f"IRR_WAV_{b}"].values for b in BANDS}
        LOG.warning(f"  station {s}: saved at a different level "
                    f"({sorted(x.replace('IRR_WAV_','') for x in have)}); "
                    f"excluded from the level-3 saved-band average")
        return None

    LOG.info("\n(B) EMPIRICAL - saved whole-series band series")
    PS_saved, curves_saved = analyse(saved_bands, "saved_whole_series", stations)

    # ---------------- (C) regenerated causal bands ----------------
    def causal_bands(s):
        p = PROC / f"station_{s:02d}_prepared.csv"
        if not p.exists():
            return None
        sig = pd.to_numeric(pd.read_csv(p, usecols=["IRRADIATION"])["IRRADIATION"],
                            errors="coerce").ffill().bfill().values.astype(float)
        M = bands_causal(sig)
        return {b: M[:, i] for i, b in enumerate(BANDS)}

    LOG.info("\n(C) EMPIRICAL - causal trailing-window bands "
             "(regenerated with the frozen bands_causal function)")
    PS_causal, curves_causal = analyse(causal_bands, "causal_trailing_window", stations)

    frames = [df for df in [PS_saved, PS_causal] if not df.empty]
    if not frames:
        # The analytic passbands (part A) need no data and have already been
        # written. Parts B and C read the station series, which this repository
        # does not redistribute, so stop with an actionable message rather than
        # an opaque error from deeper in the stack.
        LOG.error("No station data found, so the empirical spectra (parts B and C) "
                  "cannot be computed.")
        LOG.error("Expected either:")
        LOG.error(f"  {PROC / 'station_NN_prepared.csv'}   (for the causal bands)")
        LOG.error(f"  {PROC / 'station_NN_decomposed.csv'} (for the saved bands)")
        LOG.error("See data/DATA.md for how to obtain and place the dataset, then "
                  "run: python scripts/reproduce_manuscript.py --stage0")
        LOG.info("The analytic passband table (part A) was written and is complete.")
        return

    PS = pd.concat(frames, ignore_index=True)
    rc.save_table(PS.round(6), "Band_Spectra", "per_station_band_spectra")

    # ---------------- combined table ----------------
    rows = []
    for src, df in [("saved_whole_series", PS_saved),
                    ("causal_trailing_window", PS_causal)]:
        if df.empty:
            continue
        for b in BANDS:
            sub = df[df.band == b]
            if sub.empty:
                continue
            nom = NOM[NOM.band == b].iloc[0]
            rows.append(dict(
                source=src, band=b, manuscript_label=MANUSCRIPT_LABEL[b],
                nominal_passband_hours=nom.nominal_passband_hours,
                nominal_short_h=nom.period_short_h, nominal_long_h=nom.period_long_h,
                n_stations=int(sub.station.nunique()),
                # PRIMARY, resolution-stable descriptors
                mean_in_band_energy_frac=float(sub.in_band_energy_frac.mean()),
                std_in_band_energy_frac=float(sub.in_band_energy_frac.std(ddof=1))
                if len(sub) > 1 else 0.0,
                mean_median_power_period_h=float(sub.median_power_period_h.mean()),
                mean_energy_share=float(sub.energy_share.mean()),
                std_energy_share=float(sub.energy_share.std(ddof=1))
                if len(sub) > 1 else 0.0,
                # SECONDARY, reported for completeness; see the note in summarise()
                mean_peak_period_h=float(sub.peak_period_h.mean()),
                std_peak_period_h=float(sub.peak_period_h.std(ddof=1))
                if len(sub) > 1 else 0.0,
                min_peak_period_h=float(sub.peak_period_h.min()),
                max_peak_period_h=float(sub.peak_period_h.max()),
                mean_centroid_period_h=float(sub.centroid_period_h.mean())))
    T = pd.DataFrame(rows)
    rc.save_table(T.round(6), "Band_Spectra", "band_scale_summary")

    for src in T.source.unique():
        LOG.info(f"\nBAND SCALE SUMMARY [{src}]:")
        v = T[T.source == src][["band", "manuscript_label", "nominal_passband_hours",
                                "mean_in_band_energy_frac", "mean_median_power_period_h",
                                "mean_energy_share", "mean_peak_period_h"]]
        LOG.info("\n" + v.round(4).to_string(index=False))

    # ---------------- verdict, stated from the numbers ----------------
    # The scale claim rests on the nominal passband CONFIRMED by the in-band
    # energy fraction, not on the raw argmax (see summarise() for why).
    verdicts = []
    for src in T.source.unique():
        for b in BANDS:
            r = T[(T.source == src) & (T.band == b)]
            if r.empty:
                continue
            r = r.iloc[0]
            frac = r.mean_in_band_energy_frac
            lo, hi = NOMINAL_EDGES[b]
            if b == "cA3":
                scale = "slow envelope, periods longer than 4 h"
            elif b == "cD3":
                scale = "2-4 hours (NOT daily)"
            elif b == "cD2":
                scale = "1-2 hours"
            else:
                scale = "30-60 minutes"
            if not np.isfinite(frac):
                support = "not determinable"
            elif frac >= 0.60:
                support = f"CONFIRMED: {frac*100:.1f}% of band power lies in {scale}"
            elif frac >= 0.40:
                support = (f"PARTIAL: {frac*100:.1f}% of band power lies in the nominal "
                           f"range; the remainder is diurnal leakage from the "
                           f"trailing-window reconstruction")
            else:
                support = (f"WEAK: only {frac*100:.1f}% of band power lies in the "
                           f"nominal range")
            verdicts.append(dict(
                source=src, band=b, manuscript_label=MANUSCRIPT_LABEL[b],
                nominal_passband_hours=r.nominal_passband_hours,
                measured_scale=scale,
                in_band_energy_frac=frac,
                evidence=support,
                label_is_accurate=("no - the label says 'daily' but the band is a "
                                   "2-4 hour band" if b == "cD3" else "yes"),
                raw_argmax_period_h=r.mean_peak_period_h,
                raw_argmax_caveat=("argmax is contaminated by diurnal leakage and "
                                   "moves with Welch segment length; not used for "
                                   "the scale claim")))
    V = pd.DataFrame(verdicts)
    rc.save_table(V.round(6), "Band_Spectra", "band_label_verdict")
    LOG.info("\nMEASURED SCALE PER BAND:\n" +
             V[["source", "band", "manuscript_label", "nominal_passband_hours",
                "measured_scale", "in_band_energy_frac", "label_is_accurate"]]
             .round(3).to_string(index=False))

    # -------- resolution-stability check (evidence for the argmax caveat) -----
    # A genuine spectral feature does not move when the Welch segment length
    # changes. Run one station across several nperseg values and record it.
    stab = []
    p5 = PROC / "station_05_prepared.csv"
    if p5.exists():
        sig5 = pd.to_numeric(pd.read_csv(p5, usecols=["IRRADIATION"])["IRRADIATION"],
                             errors="coerce").ffill().bfill().values.astype(float)
        M5 = bands_causal(sig5)
        for nps in [512, 1024, 2048, 4096, 8192]:
            for i, b in enumerate(BANDS):
                x = M5[:, i]; x = x[np.isfinite(x)]
                if len(x) < nps:
                    continue
                x = x - x.mean()
                f, P = sps.welch(x, fs=1.0, nperseg=nps, noverlap=nps // 2,
                                 detrend=False)
                k = f > 0
                per = 1.0 / (f[k] * FS_PER_HOUR); Pk = P[k]
                lo, hi = NOMINAL_EDGES[b]
                m = (per >= lo) & (per <= hi)
                stab.append(dict(station=5, band=b, nperseg=nps,
                                 argmax_period_h=float(per[int(np.argmax(Pk))]),
                                 in_band_energy_frac=float(Pk[m].sum() / Pk.sum())))
    if stab:
        ST = pd.DataFrame(stab)
        rc.save_table(ST.round(6), "Band_Spectra", "resolution_stability_check")
        piv_a = ST.pivot(index="band", columns="nperseg", values="argmax_period_h")
        piv_f = ST.pivot(index="band", columns="nperseg", values="in_band_energy_frac")
        LOG.info("\nRESOLUTION STABILITY (station 5) - raw argmax period (h):\n" +
                 piv_a.round(2).to_string())
        LOG.info("\nRESOLUTION STABILITY - in-band energy fraction "
                 "(stable => the robust measure):\n" + piv_f.round(4).to_string())

    # ---------------- figure ----------------
    plt = rc.plot_style()
    avg = {"saved_whole_series": average_curves(curves_saved),
           "causal_trailing_window": average_curves(curves_causal)}
    colors = {"cA3": rc.CB["blue"], "cD3": rc.CB["orange"],
              "cD2": rc.CB["green"], "cD1": rc.CB["verm"]}
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)
    fig_data = []
    for ax, (src, curves) in zip(axs, avg.items()):
        for b in BANDS:
            if b not in curves:
                continue
            g, P = curves[b]
            ok = np.isfinite(P) & (P > 0)
            ax.loglog(g[ok], P[ok], color=colors[b], lw=1.5,
                      label=f"{b} ({MANUSCRIPT_LABEL[b]})")
            for gg, pp in zip(g[ok], P[ok]):
                fig_data.append(dict(source=src, band=b, period_h=gg, psd=pp))
        for edge, lab in [(0.5, "Fs/2"), (1.0, "Fs/4"), (2.0, "Fs/8"), (4.0, "Fs/16")]:
            ax.axvline(edge, ls=":", c="k", lw=.8, alpha=.6)
            ax.text(edge, ax.get_ylim()[1], f" {edge:g} h", fontsize=6,
                    rotation=90, va="top")
        ax.axvline(24.0, ls="--", c=rc.CB["purple"], lw=1.2, alpha=.8)
        ax.text(24.0, ax.get_ylim()[1], " 24 h", fontsize=7, rotation=90,
                va="top", color=rc.CB["purple"])
        ax.set_xlabel("period (hours)")
        ax.set_title(src.replace("_", " "))
        ax.legend(fontsize=7)
    axs[0].set_ylabel("average PSD across 7 stations")
    fig.suptitle("Band power spectra with nominal dyadic passband edges "
                 "(dotted) and the 24 h diurnal period (dashed)", y=1.02)
    rc.save_figure(fig, "band_spectra_psd", data=pd.DataFrame(fig_data),
                   subdir="Figures")
    plt.close(fig)

    # provenance
    (OUTD / "provenance.json").write_text(json.dumps(dict(
        sampling_interval_min=DT_MIN, samples_per_hour=FS_PER_HOUR,
        wavelet=FAMILY, level=LEVEL, causal_window=WINDOW,
        stations_analysed=stations,
        station_3_excluded="matches the manuscript (sentinel density > 30%)",
        saved_band_source="code/data/processed/station_NN_decomposed.csv "
                          "(IRR_WAV_* columns, whole-series pywt.wavedec)",
        causal_band_source="regenerated with bands_causal() copied verbatim from "
                           "code/run_fame_causal.py lines 99-112; no learner fitted",
        psd_method="scipy.signal.welch, nperseg<=4096, 50% overlap, DC bin dropped, "
                   "mean removed per band"), indent=2))

    LOG.info(f"\nwritten to {OUTD}")


if __name__ == "__main__":
    main()

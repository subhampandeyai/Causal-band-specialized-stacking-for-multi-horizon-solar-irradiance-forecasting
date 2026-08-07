"""
causal_decomposition_patch.py
=============================
Drop-in CAUSAL replacement for decompose_wavelet() in
pipeline/stage1_decomposition.py.

WHY: the existing decompose_wavelet() runs pywt.wavedec on the FULL series
(train+val+test). The db4 filter at time t reaches into future samples, so the
reconstructed band at a training/test point is contaminated by future
irradiance -- the same future the model forecasts. Measured leak: +0.0785 R2
at S5/H8. This version removes that leak.

HOW IT WORKS (strictly causal):
  For each time t, the band value uses ONLY signal[:t+1]. We compute it with an
  expanding-then-sliding window: decompose the window ending at t, take the LAST
  reconstructed sample of each band. No sample at t ever sees signal[t+1:].

SPEED: naive per-t decomposition is O(n^2). We use a sliding window of fixed
length W (default 512) so cost is O(n*W) -- about 1-2 min per station. A window
of 512 samples (~5.3 days at 15-min) fully captures level-3 db4 support and the
diurnal cycle, so band values are stable.

INTEGRATION (two edits in pipeline/stage1_decomposition.py):
  1. Paste this function in, and rename the old one to decompose_wavelet_LEAKY
     (keep it -- you need BOTH to produce the comparative leaky-vs-causal table).
  2. In process_station(), call BOTH and store both results so the paper can
     report the gap. See run_both_example() at the bottom.
"""
import numpy as np
import pywt


def decompose_wavelet_causal(signal, family="db4", level=3, window=512):
    """
    Strictly causal wavelet band reconstruction.

    Returns {component_name: array} identical in shape/keys to the original
    decompose_wavelet, but every value at index t depends only on signal[:t+1].

    Parameters
    ----------
    signal : 1D np.ndarray   (the IRRADIATION series, full length)
    family : str             wavelet family (db4)
    level  : int             decomposition level (3)
    window : int             sliding window length (>= 8*2**level recommended)
    """
    n = len(signal)
    nb = level + 1
    out = np.full((n, nb), np.nan, dtype=float)

    # Minimum samples before a level-`level` transform is valid
    min_len = 2 ** (level + 1)
    # Effective window: large enough to avoid boundary effects, capped by `window`
    W = max(window, 4 * 2 ** level)

    for t in range(n):
        lo = max(0, t - W + 1)
        seg = signal[lo:t + 1]
        if len(seg) < min_len:
            # not enough history yet -- fall back to the raw value in cA, zeros else
            out[t, 0] = signal[t]
            out[t, 1:] = 0.0
            continue

        # decompose ONLY the causal window
        coeffs = pywt.wavedec(seg, family, level=level)
        for i in range(nb):
            z = [np.zeros_like(c) for c in coeffs]
            z[i] = coeffs[i]
            rec = pywt.waverec(z, family)
            # take the LAST sample (= time t), the only causal point we trust
            out[t, i] = rec[:len(seg)][-1]

    # name the bands EXACTLY like the original so downstream code is unchanged
    components = {}
    for i in range(nb):
        if i == 0:
            name = f"IRR_WAV_cA{level}"
        else:
            name = f"IRR_WAV_cD{level - i + 1}"
        components[name] = out[:, i]

    # reconstruction error on the causal bands (will be higher than leaky -- expected)
    recon = sum(components.values())
    recon_error = float(np.sqrt(np.nanmean((signal - recon) ** 2)))
    return components, recon_error


# ----------------------------------------------------------------------
# Example: how to call BOTH in process_station() to get the comparison
# ----------------------------------------------------------------------
def run_both_example():
    """
    Pseudocode for the edit inside process_station(), after `signal` is loaded:

        # --- LEAKY (existing) ---
        comp_leaky, err_leaky = decompose_wavelet(signal, wav_family, level)
        m_leaky, _ = evaluate_decomposition(df, base_features, comp_leaky,
                                            train_idx, val_idx)

        # --- CAUSAL (new) ---
        comp_causal, err_causal = decompose_wavelet_causal(signal, wav_family, level)
        m_causal, _ = evaluate_decomposition(df, base_features, comp_causal,
                                             train_idx, val_idx)

        print(f"  LEAKY  R2 = {m_leaky['r2']:.4f}")
        print(f"  CAUSAL R2 = {m_causal['r2']:.4f}")
        print(f"  LEAK INFLATION = {m_leaky['r2'] - m_causal['r2']:+.4f}")

        # Inject the CAUSAL bands into df (these go to stage2/stage3):
        for col_name, col_vals in comp_causal.items():
            df[col_name] = col_vals
    """
    pass

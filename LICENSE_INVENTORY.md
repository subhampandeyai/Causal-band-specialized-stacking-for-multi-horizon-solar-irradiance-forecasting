# License Inventory

An inventory of the licenses of this repository's dependencies and of any
third-party material it redistributes.

> **This is information for the authors to review, not a legal clearance.**
> License compatibility is a legal judgement. The table below records what each
> package's installed metadata declares; it does not certify that the repository
> is free of licensing obligations. Anything marked *needs confirmation* should
> be checked against the upstream project, and if in doubt with someone
> qualified to advise.

Repository license: **MIT** (see `LICENSE`).

## Direct dependencies

License strings were read from the installed package metadata in the environment
that produced the results, not from documentation or memory.

| Package | Version | Declared license | Permissive? | Notes |
|---|---|---|---|---|
| numpy | 2.0.2 | BSD | yes | |
| pandas | 2.3.3 | BSD | yes | |
| scipy | 1.13.1 | BSD | yes | |
| scikit-learn | 1.6.1 | BSD | yes | |
| xgboost | 2.1.4 | Apache 2.0 | yes | Patent grant; retain the notice if redistributing binaries. |
| lightgbm | 4.6.0 | MIT | yes | |
| PyWavelets | 1.6.0 | MIT | yes | |
| tensorflow | 2.20.0 | Apache 2.0 | yes | As above. |
| keras | 3.10.0 | Apache 2.0 | yes | |
| matplotlib | 3.9.4 | PSF-based (matplotlib license) | yes | BSD-compatible; a custom PSF-style license. |
| PyYAML | 6.0.3 | MIT | yes | |
| openpyxl | 3.1.5 | MIT | yes | |
| seaborn | 0.13.2 | BSD | yes | Not imported by the released code. |
| requests | 2.32.5 | Apache 2.0 | yes | Not imported by the released code. |
| kagglehub | 0.3.13 | Apache 2.0 | yes | Not imported by the released code. |
| ngboost | 0.5.10 | Apache 2.0 | yes | Not imported by the released code. |
| vmdpy | 0.2 | MIT | yes | Not imported by the released code. |
| **tqdm** | 4.67.3 | **MPL-2.0 AND MIT** | mostly | MPL-2.0 is a weak copyleft. It imposes obligations only on modified MPL files, and this repository neither modifies nor redistributes tqdm. **Needs confirmation** only if tqdm source is ever vendored. |
| **EMD-signal** | 1.9.0 | **UNKNOWN in metadata** | unclear | The installed metadata declares no license. Upstream (PyEMD) is generally distributed under Apache 2.0, but this was **not** verifiable from the installed package. **Needs confirmation** from the upstream project. Not imported by the released code. |
| **joblib** | 1.5.3 | **UNKNOWN in metadata** | unclear | Metadata declares no license classifier. Upstream is BSD-3-Clause, but this was **not** verifiable from the installed package. **Needs confirmation.** Pulled in transitively by scikit-learn. |

All licenses that could be read are permissive (BSD, MIT, Apache 2.0, PSF) and
are, on their face, compatible with releasing this repository under MIT. The two
UNKNOWN entries are metadata gaps rather than known conflicts, and neither
package is imported by the released code.

## Redistributed third-party material

| Item | Status |
|---|---|
| Vendored third-party source files | **None.** A scan for copyright headers, SPDX identifiers and BSD/Apache notice text across every `.py` in the release found no third-party source. |
| Third-party binaries | **None.** |
| Dataset | **Not redistributed.** See below. |

## Dataset

The Chinese State Grid dataset is **cited and linked, never redistributed**:

> Y. Chen and J. Xu, "Solar and wind power data from the Chinese State Grid
> Renewable Energy Generation Forecasting Competition," *Sci. Data*, vol. 9,
> no. 1, art. 577, 2022, doi: 10.1038/s41597-022-01696-6.
>
> Available at: https://doi.org/10.1038/s41597-022-01696-6

Verified at release time:

| Path | Contents |
|---|---|
| `data/raw/` | **0 files** — user-supplied |
| `data/processed/` | **0 files** — generated locally by Stage 0 |

`data/DATA.md` documents the source, the required file names and the placement,
and `.gitignore` excludes both directories so the data cannot be committed by
accident.

`results/predictions/` contains 42 CSV files. These are **model outputs produced
by the code in this repository**, not extracts of the source dataset: each holds
the observed target alongside three model predictions for one station-horizon
pair. They are the authors' own output and are released under the repository
license.

## Items needing manual confirmation

1. **EMD-signal 1.9.0** — no license in the installed metadata. Confirm upstream
   before relying on it. It is not imported by the released code and could be
   removed from `requirements.txt` if unused.
2. **joblib 1.5.3** — no license classifier in the installed metadata. Widely
   BSD-3-Clause upstream; confirm if that matters for your distribution.
3. **tqdm 4.67.3** — dual MPL-2.0 / MIT. No obligation arises from importing it,
   but confirm before vendoring any tqdm source.
4. **Institutional policy** — check whether your institution or funder requires a
   specific license, an author list, or a data-availability statement in the
   repository rather than only in the paper.

## Method

- Licenses were read with `importlib.metadata` from the environment that produced
  the results, preferring the license classifier over the free-text field.
- Redistributed third-party source was searched for by scanning every `.py` in
  the release for copyright headers, SPDX identifiers and standard notice text.
- Dataset directories were checked to be empty.

Re-run the inventory after changing dependencies; the versions above correspond
to `requirements-lock.txt`.

# Seed-sensitivity note

**Scope.** Measured on the frozen pipeline logic, at **Station 5, H1**, with seeds **[42, 0, 2024]**. Every model was refit on the identical data, features, chronological split and hyperparameters; only the seed changed. **The LSTM was not retrained** — see the cost note below.

Elapsed for this check: **12 s**.

## Measured spread across three seeds

| Model | max abs Δ skill | max abs Δ RMSE (W/m²) | max abs Δ prediction | deterministic given data? |
|---|---|---|---|---|
| LightGBM | 1.047e-02 | 5.783e-01 | 1.204e-01 | **no** |
| Unified XGBoost | 2.285e-03 | 1.250e-01 | 1.162e-01 | **no** |
| Persistence | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |
| Ridge (trend expert) | 0.000e+00 | 0.000e+00 | 0.000e+00 | **yes** |

## Which components consume a seed

Read directly from `code/run_fame_causal.py`:

- line 92: numpy global seed — `np.random.seed(SEED)`
- line 151: sklearn/xgboost/lightgbm seed — `random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0)`
- line 160: TensorFlow graph-level seed — `tf.random.set_seed(SEED)`
- line 184: TensorFlow graph-level seed — `tf.random.set_seed(SEED)`
- line 275: sklearn/xgboost/lightgbm seed — `subsample=0.8, colsample_bytree=0.8, random_state=SEED,`

### Per model

| Component | Stochastic? | Why |
|---|---|---|
| Ridge (trend expert) | **No** | `Ridge(alpha=1.0)` takes no `random_state`; the solution is the closed-form ridge estimator |
| Persistence (noise expert) | **No** | a deterministic one-step shift of the target |
| Ridge meta-learner | **No** | same closed form, fitted on validation predictions |
| Unified XGBoost | **Yes, nominally** | `random_state=SEED` with `subsample=0.8`, `colsample_bytree=0.8`, so row/column sampling is seeded |
| LightGBM | **Yes, nominally** | same construction |
| **LSTM (daily expert)** | **Yes** | `tf.random.set_seed(SEED)`; weight initialisation, dropout masks and batch shuffling are all stochastic |
| Transformer / Informer-lite / TimesNet-lite | **Yes** | same, via `tf.random.set_seed(SEED)` |

## Result

The tree models are **not** exactly seed-invariant — subsampling is seeded, so different seeds draw different rows and columns. The measured spread is reported above and is the honest figure to quote.

Exactly invariant (max |Δ| = 0 across all three seeds): Persistence, Ridge (trend expert).

## The LSTM, and why the full grid was not run

The LSTM daily-band expert is the only component whose cost makes a multi-seed grid expensive. Measured on this machine (CPU-only, no CUDA GPU): **966.0 s per (station, horizon) pair**.

A full 8-seed grid over 7 stations × 6 horizons would therefore cost
**966.0 s × 42 pairs × 8 seeds ≈ 90.2 hours (3.8 days)** of continuous computation, which is why it was not run here.

That figure is a measurement from `Further_Computation/Supplementary_Analysis/Computational_Cost/runtime_probe.json`, not an estimate.

**Consequence for the manuscript.** The seed limitation is confined to the LSTM band expert and the three deep baselines. The trend expert, the persistence expert and the meta-learner are deterministic given the data, and the tree models vary only by the amount measured above. Any statement about seed robustness should be scoped to the recurrent and attention components rather than to the framework as a whole.

## Provenance

- Pipeline logic copied verbatim from `code/run_fame_causal.py` (`bands_causal`, `build_features`, `make_xy`, and the exact learner hyperparameters).
- The frozen root was read only; nothing outside `Supplementary_File/` was written.
- Raw per-seed metrics: `Supplementary_Analysis/Seed_Sensitivity/per_seed_metrics.csv`.

# BEQ-GRU: short-term PM2.5 forecasting with emphasis on extreme events

Code for the paper

> Jayasinghe JMVAB, Tilakaratne CD. *Bidirectional enhanced quantile gated recurrent unit for
> short-term forecasting of PM2.5 with emphasis on extreme events in Sri Lanka.* PLOS ONE (under review).

The code forecasts hourly PM2.5 24 and 48 hours ahead at two Sri Lankan monitoring stations
(Colombo – Battaramulla, and Kandy). It compares a vector autoregression, Random Forest, XGBoost and
recurrent networks with the proposed **BEQ-GRU**: a bidirectional GRU encoder followed by residual GRU
blocks, trained with a quantile (pinball) loss at τ = 0.95 so that pollution peaks are not smoothed away.

## Repository layout

```
config.py                  every setting used in the paper (paths, split, window, hyperparameters)
pm25/
  data.py                  loading, 60% missingness rule, chronological split, robust scaling, windows
  impute.py                SAITS imputation fitted on the training partition only
  var_model.py             VAR baseline: STL, unit-root, Johansen, Granger, lag order, diagnostics
  trees.py                 Random Forest and XGBoost on 504 hourly lags
  networks.py              LSTM, GRU, Bi-LSTM, Bi-GRU, attention variants, BEQ-GRU, quantile loss
  training.py              fitting with early stopping and forecasting the scored test hours
  selection.py             greedy forward predictor selection on the validation set
  metrics.py               RMSE, MAE, confusion counts, detection rate with 95% Wilson interval
  explain.py               SHAP (GradientExplainer) importance
  plots.py                 figures as 300-dpi TIFF (PLOS format)
scripts/01_impute_saits.py ... 09_evaluate.py   one script per step (see below)
tests/test_pipeline.py     fast checks on synthetic data
data/README.md             where to put the CEA files (not included)
```

## Paper ↔ code

| Paper | Script | Output |
|---|---|---|
| Section 2.2, imputation | `01_impute_saits.py` | `data/final_df_*.csv`, missingness table |
| Section 2.4.1 and 3.2, VAR (Table 3, Fig 2) | `02_var_baseline.py` | VAR forecasts and all test tables |
| Section 2.4.2 and 3.3, RF and XGBoost (Tables 4–5) | `03_tree_models.py` | forecasts, grouped importances |
| Section 2.4.3 and 3.4, recurrent benchmarks (Tables 6–7) | `04_deep_benchmarks.py` | forecasts |
| Sections 2.5–2.6 and 3.5, BEQ-GRU (Tables 8–10, Fig 3) | `05_beqgru.py` | forecasts, trained models |
| Section 2.5.4, Bayesian tuning | `06_tune.py` | best units, dropout, learning rate |
| Section 2.4.3, predictor selection | `07_feature_selection.py` | `results/selected_features.json` |
| Section 3.6, SHAP (Fig 6) | `08_shap.py` | importance tables, Fig 6 |
| Sections 2.9–2.10 and 3, evaluation (Tables 3–11, Figs 1–4) | `09_evaluate.py` | `point_accuracy.csv`, `detection.csv`, figures |

## Forecasting protocol (Section 2.10)

* 10,968 hourly observations per station, split chronologically into training (8,776 h),
  validation (1,096 h) and test (1,096 h) partitions, with no shuffling.
* Every model uses the preceding 504 hours (21 days) as input.
* Test forecasts are issued from non-overlapping origins: every 24 h for the 24-h models (44 forecasts)
  and every 48 h for the 48-h models (22 forecasts). Both horizons are scored on the same
  **1,056 test hours**; the last 40 test hours cannot hold a full 48-h forecast and are not scored.
* Scalers are fitted on the training partition only. SAITS is fitted on the training partition only.
* An exceedance is an hour above 50 µg/m³. Detection = TP / (TP + FN), reported with a 95% Wilson interval.
* The VAR is evaluated at a single origin (the final 48 h of the record) and is a qualitative reference.

## The BEQ-GRU (Sections 2.5–2.6)

```
input (504 h × p) → Bi-GRU encoder → dropout
                  → residual GRU block 1 → residual GRU block 2 → residual GRU block 3
                  → GRU read-out → dropout → dense layer (24 or 48 outputs)
loss: quantile (pinball) loss, τ = 0.95      optimiser: Nadam      batch size: 64
```

Each residual block is a GRU layer with dropout (0.2) whose output is added to its input; the skip
path is projected by a dense layer when the widths differ. In the last block, the final GRU state is
added to every time step of the skip path before the GRU read-out. The tuned settings of the final
models are in `config.TUNED` (Colombo BEQ-GRU: 224 units, dropout 0.5, learning rate 0.001 at 24 h;
128 units, dropout 0.4, learning rate 0.0005 at 48 h).

## Installation

Python 3.11 is recommended (the paper used 3.11.9). On Windows, in a terminal:

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # optional: fast checks on synthetic data
```

The imputation step uses PyTorch, so give it its own environment:

```bash
python -m venv .venv-impute
.venv-impute\Scripts\activate
pip install -r requirements-impute.txt
```

## Running

Put the CEA files in `data/` (see `data/README.md`), then run the steps in order. Each script
accepts `--station Colombo|Kandy|all` and `--horizon 24|48|all`.

```bash
python scripts/01_impute_saits.py            # skip if data/final_df_*.csv already exist
python scripts/02_var_baseline.py
python scripts/03_tree_models.py
python scripts/04_deep_benchmarks.py --untuned
python scripts/07_feature_selection.py       # optional: predictor subsets for the final models
python scripts/04_deep_benchmarks.py --station Kandy --model Bi-GRU   # tuned final model (Bi-GRU-tuned)
python scripts/05_beqgru.py
python scripts/08_shap.py
python scripts/09_evaluate.py
```

`06_tune.py` repeats the Bayesian search; its result is already stored in `config.TUNED`.
All outputs are written to `results/`. A GPU shortens training considerably.

## Reproducibility

Random seeds are fixed (`config.SEED = 100`), but recurrent-network training on GPUs is not bit-for-bit
deterministic, so retrained models can differ slightly from the published values. The exceedance
counts and metrics in the paper can be recomputed exactly from saved predictions with `09_evaluate.py`.

## Data availability

The CEA data cannot be redistributed and are not included. They can be requested from the Central
Environmental Authority of Sri Lanka, as described in the paper's Data Availability Statement.

## Citation and licence

Please cite the paper and this code (see `CITATION.cff`). The code is released under the MIT licence.

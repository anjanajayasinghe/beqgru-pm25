"""Settings for every experiment in the paper.

Edit the file paths in STATIONS if your CSV files have different names.
All other values reproduce the configuration reported in the manuscript
(PONE-D-26-42835) and in the authors' thesis.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------------------------------- data
STATIONS = {
    # "raw": hourly CEA export BEFORE imputation (only needed for scripts/01_impute_saits.py)
    # "imputed": gap-free hourly series used for modelling (output of scripts/01_impute_saits.py)
    "Colombo": {"raw": DATA_DIR / "raw_battaramulla.csv", "imputed": DATA_DIR / "final_df_battaramulla.csv"},
    "Kandy": {"raw": DATA_DIR / "raw_kandy.csv", "imputed": DATA_DIR / "final_df_kandy1.csv"},
}
DATETIME_COL = "Period Start Time"      # parsed and used as the index
DROP_COLS = ["Period End Time"]         # removed if present
TARGET = "PM2.5 Conc"                   # response variable (µg/m³)

N_HOURS = 10_968                        # 1 Oct 2022 – 31 Dec 2023
N_TRAIN, N_VAL, N_TEST = 8_776, 1_096, 1_096   # chronological 80/10/10 split (Section 2.10)
MAX_MISSING = 0.60                      # variables with more missing hours are dropped (Section 2.2)

# --------------------------------------------------------------------------- forecasting protocol
LOOKBACK = 504                          # 21 days of hourly input (Section 2.3)
HORIZONS = (24, 48)
N_SCORED = 1_056                        # 44 x 24 = 22 x 48 scored test hours (Section 2.10)
TRAIN_STRIDE = None                     # None = one training window every H hours (as in the original code)
THRESHOLD = 50.0                        # exceedance threshold, µg/m³ (Section 2.9)
SEED = 100

# --------------------------------------------------------------------------- SAITS imputation (Section 2.2)
SAITS_SEQ_LEN = 457                     # hours per imputation sample
SAITS_PARAMS = {
    "Colombo": dict(n_layers=2, d_model=256, d_ffn=128, n_heads=8, d_k=64, d_v=32, dropout=0.2, epochs=200),
    "Kandy": dict(n_layers=3, d_model=512, d_ffn=128, n_heads=8, d_k=64, d_v=32, dropout=0.2, epochs=200),
}

# --------------------------------------------------------------------------- VAR baseline (Section 2.4.1)
STL_PERIOD = 24
GRANGER_MAXLAG = 15
VAR_MAXLAGS = 60
VAR_LAG = {"Colombo": 26, "Kandy": 28}  # lag orders reported in the paper; set to None to re-select
TREND_DEGREE = 2                        # polynomial used to extrapolate the STL trend

# --------------------------------------------------------------------------- tree models (Section 2.4.2)
TREE_LOOKBACK = 504
TREE_EXCLUDE = ["rain"]                 # columns whose name contains these strings are removed (rainfall)
CV_SPLITS = 5                           # time-ordered folds inside the training partition
RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", "log2"],
}
XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_child_weight": [1, 2, 3],
    "subsample": [0.7, 0.8, 1.0],
}

# --------------------------------------------------------------------------- deep learning (Sections 2.4.3, 2.5, 2.6)
TAU = 0.95                              # quantile level of the BEQ-GRU loss
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10                           # early stopping on validation loss
N_RES_BLOCKS = 3                        # residual GRU blocks in the BEQ-GRU (Table 1)
BLOCK_DROPOUT = 0.2                     # dropout inside each residual block

# Default (untuned) settings of the networks, as used for the benchmark tables
DEFAULTS = {
    "LSTM": dict(units=128, dropout=0.2),
    "GRU": dict(units=128, dropout=0.2),
    "Bi-LSTM": dict(units=128, dropout=0.2),
    "Bi-GRU": dict(units=128, dropout=0.2),
    "MHA-Bi-LSTM": dict(units=128, dropout=0.2),
    "MHA-Bi-GRU": dict(units=128, dropout=0.2),
    "BEQ-GRU": dict(units=128, block_units=64, dropout=0.1),
}
# Bayesian-optimised settings of the final models (thesis Section 5.4.4)
TUNED = {
    ("Colombo", "BEQ-GRU", 24): dict(units=224, dropout=0.5, learning_rate=1e-3),
    ("Colombo", "BEQ-GRU", 48): dict(units=128, dropout=0.4, learning_rate=5e-4),
    ("Kandy", "Bi-GRU", 24): dict(units=512, dropout=0.3, learning_rate=1e-3),
    ("Kandy", "Bi-GRU", 48): dict(units=256, dropout=0.5, learning_rate=1e-3),
}
TUNER_TRIALS = 50

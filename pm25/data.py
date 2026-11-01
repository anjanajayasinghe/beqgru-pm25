"""Loading, cleaning, splitting, scaling and windowing of the hourly station data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import config as C


def load_station(path, drop_sparse: bool = True) -> pd.DataFrame:
    """Read one station CSV, index it by time and keep numeric columns only."""
    df = pd.read_csv(path)
    if C.DATETIME_COL in df.columns:
        df[C.DATETIME_COL] = pd.to_datetime(df[C.DATETIME_COL])
        df = df.set_index(C.DATETIME_COL)
    df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    if drop_sparse:
        df, _ = drop_sparse_columns(df)
    return df


def drop_sparse_columns(df: pd.DataFrame, max_missing: float = C.MAX_MISSING):
    """Drop variables with more than `max_missing` missing hours (Section 2.2)."""
    share = df.isna().mean()
    dropped = [c for c in df.columns if share[c] > max_missing and c != C.TARGET]
    return df.drop(columns=dropped), dropped


def clip_negative(df: pd.DataFrame) -> pd.DataFrame:
    """Set implausible negative readings to zero (CEA guidance, Section 2.2)."""
    return df.clip(lower=0)


def partitions(n: int = C.N_HOURS) -> dict[str, slice]:
    """Chronological training, validation and test partitions (Section 2.10)."""
    if n != C.N_TRAIN + C.N_VAL + C.N_TEST:
        raise ValueError(f"expected {C.N_TRAIN + C.N_VAL + C.N_TEST} hourly rows, found {n}")
    return {
        "train": slice(0, C.N_TRAIN),
        "val": slice(C.N_TRAIN, C.N_TRAIN + C.N_VAL),
        "test": slice(C.N_TRAIN + C.N_VAL, n),
    }


def forecast_origins(part: str, horizon: int, lookback: int = C.LOOKBACK, stride: int | None = None) -> np.ndarray:
    """Hours at which forecasts are issued.

    Each origin `o` uses inputs o-lookback ... o-1 and forecasts hours o ... o+horizon-1.
    Validation and test origins are non-overlapping (every `horizon` hours). In the test
    partition only the first 1,056 hours are scored, so both horizons are evaluated on the
    same hours: 44 origins at 24 h and 22 origins at 48 h.
    """
    stride = stride or horizon
    if part == "train":
        return np.arange(lookback, C.N_TRAIN - horizon + 1, stride)
    if part == "val":
        return np.arange(C.N_TRAIN, C.N_TRAIN + C.N_VAL - horizon + 1, horizon)
    if part == "test":
        start = C.N_TRAIN + C.N_VAL
        return np.arange(start, start + C.N_SCORED, horizon)
    raise ValueError(part)


class Scaled:
    """Robust (median/IQR) scaling fitted on the training partition only."""

    def __init__(self, df: pd.DataFrame, features: list[str] | None = None):
        self.features = features or list(df.columns)
        tr = partitions(len(df))["train"]
        self.x_scaler = RobustScaler().fit(df[self.features].iloc[tr])
        self.y_scaler = RobustScaler().fit(df[[C.TARGET]].iloc[tr])
        self.X = self.x_scaler.transform(df[self.features]).astype("float32")
        self.y = self.y_scaler.transform(df[[C.TARGET]]).ravel().astype("float32")
        self.y_raw = df[C.TARGET].to_numpy()
        self.index = df.index

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        shape = np.shape(y_scaled)
        return self.y_scaler.inverse_transform(np.reshape(y_scaled, (-1, 1))).reshape(shape)


def windows(X: np.ndarray, y: np.ndarray, origins: np.ndarray, horizon: int, lookback: int = C.LOOKBACK):
    """Stack input windows (n, lookback, p) and target blocks (n, horizon)."""
    Xw = np.stack([X[o - lookback:o] for o in origins])
    yw = np.stack([y[o:o + horizon] for o in origins])
    return Xw, yw


def scored_frame(index, origins, horizon, y_true, y_pred) -> pd.DataFrame:
    """Long table of the scored test hours: datetime, y_true, y_pred (µg/m³)."""
    hours = np.concatenate([np.arange(o, o + horizon) for o in origins])
    return pd.DataFrame({"datetime": np.asarray(index)[hours], "y_true": np.ravel(y_true), "y_pred": np.ravel(y_pred)})

"""SAITS imputation fitted on the training partition only (Section 2.2).

Runs in its own environment (requirements-impute.txt) because PyPOTS uses PyTorch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _to_samples(values: np.ndarray, seq_len: int) -> np.ndarray:
    """Cut a (T, p) array into (n, seq_len, p) samples, padding the end with NaN."""
    T, p = values.shape
    n = int(np.ceil(T / seq_len))
    padded = np.full((n * seq_len, p), np.nan, dtype="float32")
    padded[:T] = values
    return padded.reshape(n, seq_len, p)


def saits_impute(df: pd.DataFrame, station: str, seq_len: int = C.SAITS_SEQ_LEN, device=None) -> pd.DataFrame:
    """Fit SAITS on the training hours and fill every gap in the full record."""
    from pypots.imputation import SAITS  # imported here so the rest of the code does not need PyTorch

    n_train = C.N_TRAIN
    mean = df.iloc[:n_train].mean()
    std = df.iloc[:n_train].std().replace(0, 1)
    z = ((df - mean) / std).to_numpy(dtype="float32")

    params = dict(C.SAITS_PARAMS[station])
    epochs = params.pop("epochs")
    model = SAITS(n_steps=seq_len, n_features=z.shape[1], epochs=epochs, device=device, **params)
    model.fit({"X": _to_samples(z[:n_train], seq_len)})

    out = model.predict({"X": _to_samples(z, seq_len)})
    filled = out["imputation"] if isinstance(out, dict) else out
    filled = filled.reshape(-1, z.shape[1])[: len(df)]
    filled = np.where(np.isnan(z), filled, z)          # keep every observed value
    result = pd.DataFrame(filled, index=df.index, columns=df.columns) * std + mean
    return result.clip(lower=0)                        # negative readings set to zero

"""Point-accuracy and exceedance-detection metrics (Section 2.9)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config as C


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def wilson(k: int, n: int, z: float = 1.959964):
    """95% Wilson score interval for a proportion k/n, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (centre - half)), min(100.0, 100 * (centre + half))


def confusion(y, yhat, c: float = C.THRESHOLD):
    """Counts of the hourly exceedance classification (y > c versus y_hat > c)."""
    obs, fc = np.asarray(y) > c, np.asarray(yhat) > c
    return {"TP": int(np.sum(obs & fc)), "FN": int(np.sum(obs & ~fc)),
            "FP": int(np.sum(~obs & fc)), "TN": int(np.sum(~obs & ~fc))}


def summary(frame: pd.DataFrame, c: float = C.THRESHOLD) -> dict:
    """RMSE, MAE, confusion counts and detection rate with its Wilson interval."""
    y, yhat = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
    cm = confusion(y, yhat, c)
    n_exc = cm["TP"] + cm["FN"]
    lo, hi = wilson(cm["TP"], n_exc)
    return {"RMSE": round(rmse(y, yhat), 2), "MAE": round(mae(y, yhat), 2), **cm,
            "Detection (%)": round(100 * cm["TP"] / n_exc, 2) if n_exc else float("nan"),
            "95% CI": f"[{lo:.1f}, {hi:.1f}]", "Hours": len(frame)}

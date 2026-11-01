"""SHAP feature importance for the best network at each station (Section 3.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_importance(model, X_background, X_explain, feature_names, nsamples=200) -> pd.Series:
    """Mean |SHAP| per input variable, averaged over windows, time steps and lead times.

    Values refer to the scaled target, so they rank predictors within one model
    but are not comparable in magnitude across models.
    """
    import shap

    explainer = shap.GradientExplainer(model, X_background)
    values = explainer.shap_values(X_explain, nsamples=nsamples)
    arr = np.abs(np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values))
    # arr: (windows, time steps, variables, lead times)
    imp = arr.mean(axis=(0, 1, 3)) if arr.ndim == 4 else arr.mean(axis=(0, 1))
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)

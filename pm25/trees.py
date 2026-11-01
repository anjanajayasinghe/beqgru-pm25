"""Random Forest and XGBoost benchmarks on lagged features (Section 2.4.2)."""
from __future__ import annotations

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import config as C
from pm25 import data as D


def tree_features(df: pd.DataFrame) -> list[str]:
    """All columns except those matching TREE_EXCLUDE (rainfall)."""
    return [c for c in df.columns if not any(p in c.lower() for p in C.TREE_EXCLUDE)]


def lagged(S: D.Scaled, origins, horizon, lookback=C.TREE_LOOKBACK):
    """Flatten each input window into one row of lagged features."""
    X, y = D.windows(S.X, S.y, origins, horizon, lookback)
    return X.reshape(len(X), -1), y


def estimator(kind: str):
    if kind == "RF":
        return RandomForestRegressor(random_state=C.SEED, n_jobs=-1), C.RF_GRID
    if kind == "XGBoost":
        return xgb.XGBRegressor(random_state=C.SEED, n_jobs=-1), C.XGB_GRID
    raise ValueError(kind)


def fit_and_forecast(kind, df, horizon, grid=None, features=None, verbose=0):
    """Grid-search the model on the training partition and forecast the scored test hours."""
    features = features or tree_features(df)
    S = D.Scaled(df, features)
    stride = C.TRAIN_STRIDE or horizon
    Xtr, ytr = lagged(S, D.forecast_origins("train", horizon, C.TREE_LOOKBACK, stride), horizon)
    o_te = D.forecast_origins("test", horizon, C.TREE_LOOKBACK)
    Xte, yte = lagged(S, o_te, horizon)
    model, default_grid = estimator(kind)
    search = GridSearchCV(model, grid if grid is not None else default_grid, cv=TimeSeriesSplit(C.CV_SPLITS),
                          scoring="neg_mean_squared_error", n_jobs=1, verbose=verbose)
    search.fit(Xtr, ytr)
    y_pred = S.inverse_y(search.predict(Xte))
    frame = D.scored_frame(S.index, o_te, horizon, S.inverse_y(yte), y_pred)
    return search, frame, features


def grouped_importance(search, features, lookback=C.TREE_LOOKBACK) -> pd.Series:
    """Gini (RF) or gain (XGBoost) importance summed over the lags of each variable."""
    imp = search.best_estimator_.feature_importances_.reshape(lookback, len(features)).sum(axis=0)
    return pd.Series(imp, index=features).sort_values(ascending=False)

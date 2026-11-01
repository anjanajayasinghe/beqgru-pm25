"""Fit a network on the training windows and forecast the scored test hours."""
from __future__ import annotations

import json

import keras

import config as C
from pm25 import data as D
from pm25 import networks as N


def prepare(df, horizon, features=None):
    """Scaled arrays and train/validation/test windows for one station and horizon."""
    S = D.Scaled(df, features)
    stride = C.TRAIN_STRIDE or horizon
    o_tr = D.forecast_origins("train", horizon, stride=stride)
    o_va = D.forecast_origins("val", horizon)
    o_te = D.forecast_origins("test", horizon)
    return S, {
        "train": D.windows(S.X, S.y, o_tr, horizon),
        "val": D.windows(S.X, S.y, o_va, horizon),
        "test": D.windows(S.X, S.y, o_te, horizon),
        "test_origins": o_te,
    }


def fit_and_forecast(name, df, station, horizon, features=None, tuned=True, epochs=C.EPOCHS, verbose=0):
    """Train one model and return (model, scored test-hour table in µg/m³)."""
    keras.utils.set_random_seed(C.SEED)
    S, W = prepare(df, horizon, features)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = W["train"], W["val"], W["test"]
    model = N.build(name, Xtr.shape[1:], horizon, **N.settings(station, name, horizon, tuned))
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=C.BATCH_SIZE, verbose=verbose,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                                       restore_best_weights=True)])
    y_pred = S.inverse_y(model.predict(Xte, verbose=0))
    y_true = S.inverse_y(yte)
    return model, D.scored_frame(S.index, W["test_origins"], horizon, y_true, y_pred)


def selected_features(station, name, horizon):
    """Predictor subset chosen by scripts/07_feature_selection.py, if it was kept."""
    path = C.RESULTS_DIR / "selected_features.json"
    if not path.exists():
        return None
    sel = json.loads(path.read_text()).get(f"{station}|{name}|{horizon}")
    return sel["features"] if sel and sel.get("keep") else None


def save_predictions(frame, station, name, horizon):
    out = C.RESULTS_DIR / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{station}_{name}_{horizon}h.csv", index=False)
    return out / f"{station}_{name}_{horizon}h.csv"

"""Fast checks of the protocol, metrics and models on synthetic data (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from pm25 import data as D  # noqa: E402
from pm25 import metrics as M  # noqa: E402


def synthetic(p=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(C.N_HOURS)
    pm = 25 + 10 * np.sin(2 * np.pi * t / 24) + rng.gamma(2, 6, C.N_HOURS)
    cols = {C.TARGET: pm}
    for j in range(p - 1):
        cols[f"X{j}"] = 0.5 * pm + rng.normal(0, 5, C.N_HOURS)
    idx = pd.date_range("2022-10-01", periods=C.N_HOURS, freq="h", name=C.DATETIME_COL)
    return pd.DataFrame(cols, index=idx)


def test_protocol_counts():
    for h, n in ((24, 44), (48, 22)):
        o = D.forecast_origins("test", h)
        assert len(o) == n and len(o) * h == C.N_SCORED == 1056
        assert o[0] == C.N_TRAIN + C.N_VAL
        tr = D.forecast_origins("train", h)
        assert tr.min() >= C.LOOKBACK and tr.max() + h <= C.N_TRAIN       # training targets stay in training
        va = D.forecast_origins("val", h)
        assert va.min() >= C.N_TRAIN and va.max() + h <= C.N_TRAIN + C.N_VAL


def test_wilson_matches_table9():
    assert [round(x, 1) for x in M.wilson(30, 41)] == [58.1, 84.3]
    assert [round(x, 1) for x in M.wilson(20, 28)] == [52.9, 84.7]
    assert [round(x, 1) for x in M.wilson(0, 41)] == [0.0, 8.6]


def test_confusion_and_summary():
    f = pd.DataFrame({"y_true": [10, 60, 70, 20], "y_pred": [55, 65, 30, 10]})
    s = M.summary(f)
    assert (s["TP"], s["FN"], s["FP"], s["TN"]) == (1, 1, 1, 1)


def test_quantile_loss():
    from pm25.networks import quantile_loss
    loss = quantile_loss(0.95)
    under = float(np.asarray(loss(np.array([[1.0]]), np.array([[0.0]]))).ravel()[0])
    over = float(np.asarray(loss(np.array([[0.0]]), np.array([[1.0]]))).ravel()[0])
    assert under == pytest.approx(0.95) and over == pytest.approx(0.05)    # 19:1 asymmetry


def test_beqgru_structure():
    from pm25.networks import beq_gru
    m = beq_gru((C.LOOKBACK, 4), 24, units=16, block_units=8)
    assert m.output_shape == (None, 24)
    names = [l.__class__.__name__ for l in m.layers]
    assert names.count("Add") == C.N_RES_BLOCKS and "Bidirectional" in names


def test_scaler_uses_training_partition_only():
    df = synthetic()
    df.iloc[C.N_TRAIN:, 0] += 1000                     # shift outside training must not move the scaler
    S = D.Scaled(df)
    assert abs(S.y_scaler.center_[0] - df[C.TARGET].iloc[:C.N_TRAIN].median()) < 1e-6


@pytest.mark.parametrize("name", ["GRU", "BEQ-GRU"])
def test_network_end_to_end(name):
    from pm25 import training as T
    df = synthetic()
    _, frame = T.fit_and_forecast(name, df, "Colombo", 48, epochs=1)
    assert len(frame) == C.N_SCORED and frame["y_pred"].notna().all()


def test_tree_end_to_end():
    from pm25 import trees
    df = synthetic()
    _, frame, feats = trees.fit_and_forecast("RF", df, 48, grid={"n_estimators": [10], "max_depth": [5]})
    assert len(frame) == C.N_SCORED and feats == list(df.columns)


def test_var_end_to_end(monkeypatch):
    from pm25 import var_model as V
    monkeypatch.setitem(C.VAR_LAG, "Colombo", 3)
    frame, tests = V.run(synthetic(), "Colombo")
    assert len(frame) == 48 and np.isfinite(frame["y_pred"]).all()
    assert {"unit_roots", "johansen", "diagnostics"} <= set(tests)

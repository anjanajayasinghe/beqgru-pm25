"""Greedy forward wrapper for predictor selection on the validation set (Section 2.4.3)."""
from __future__ import annotations

import keras
import numpy as np

import config as C
from pm25 import networks as N
from pm25 import training as T


def forward_selection(df, station, name, horizon, epochs=50, patience=5, verbose=1):
    """Add one predictor at a time while the validation loss decreases.

    Past PM2.5 is always kept as an input. Returns the selected predictors and
    the validation loss of each accepted step.
    """
    candidates = [c for c in df.columns if c != C.TARGET]
    selected, best, history = [], np.inf, []
    while candidates:
        scores = {}
        for cand in candidates:
            keras.utils.set_random_seed(C.SEED)
            feats = [C.TARGET] + selected + [cand]
            _, W = T.prepare(df, horizon, feats)
            (Xtr, ytr), (Xva, yva) = W["train"], W["val"]
            model = N.build(name, Xtr.shape[1:], horizon, **N.settings(station, name, horizon))
            model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=C.BATCH_SIZE, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                               restore_best_weights=True)])
            scores[cand] = model.evaluate(Xva, yva, verbose=0)[0]
        cand = min(scores, key=scores.get)
        if scores[cand] >= best:
            break
        best = scores[cand]
        selected.append(cand)
        candidates.remove(cand)
        history.append((cand, float(best)))
        if verbose:
            print(f"  + {cand}: validation loss {best:.5f}")
    return selected, history

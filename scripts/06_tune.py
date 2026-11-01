"""Step 6 – Bayesian hyperparameter search with Keras Tuner (Section 2.5.4).

Searches units, dropout and learning rate on the validation loss and prints the best
values. Copy them into config.TUNED to use them in steps 4 and 5.
"""
import _path  # noqa: F401

import keras
import keras_tuner as kt

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import networks as N
from pm25 import training as T

p = cli.parser(__doc__, ["BEQ-GRU", "Bi-GRU"])
p.add_argument("--trials", type=int, default=C.TUNER_TRIALS)
p.add_argument("--epochs", type=int, default=C.EPOCHS)
args = p.parse_args()
stations, horizons, models = cli.expand(args, ["BEQ-GRU", "Bi-GRU"])

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    for h in horizons:
        for name in models:
            _, W = T.prepare(df, h, T.selected_features(st, name, h))
            (Xtr, ytr), (Xva, yva) = W["train"], W["val"]
            max_units = 512 if name == "Bi-GRU" else 256

            def build(hp, name=name, shape=Xtr.shape[1:], h=h, max_units=max_units):
                return N.build(name, shape, h,
                               units=hp.Int("units", 32, max_units, step=32),
                               dropout=hp.Float("dropout", 0.1, 0.5, step=0.1),
                               learning_rate=hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5]))

            tuner = kt.BayesianOptimization(build, objective="val_loss", max_trials=args.trials, seed=C.SEED,
                                            directory=str(C.RESULTS_DIR / "tuner"), project_name=f"{st}_{name}_{h}h")
            tuner.search(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epochs, batch_size=C.BATCH_SIZE, verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE),
                                    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
            best = tuner.get_best_hyperparameters(1)[0].values
            print(f'("{st}", "{name}", {h}): dict(units={best["units"]}, dropout={best["dropout"]:.1f}, '
                  f'learning_rate={best["learning_rate"]}),')

"""Step 4 – Benchmark recurrent networks (Section 2.4.3, Tables 6–7).

Trains LSTM, GRU, Bi-LSTM, Bi-GRU, MHA-Bi-LSTM and MHA-Bi-GRU with the log-cosh loss.
The Kandy Bi-GRU uses its tuned settings (config.TUNED) unless --untuned is given, and, if
available, the predictor subset from step 7; its predictions are then saved as Bi-GRU-tuned. Output: results/predictions/<station>_<model>_<H>h.csv
"""
import _path  # noqa: F401

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import metrics as M
from pm25 import training as T

BENCH = ["LSTM", "GRU", "Bi-LSTM", "Bi-GRU", "MHA-Bi-LSTM", "MHA-Bi-GRU"]
p = cli.parser(__doc__, BENCH)
p.add_argument("--epochs", type=int, default=C.EPOCHS)
p.add_argument("--untuned", action="store_true", help="ignore config.TUNED (benchmark settings only)")
args = p.parse_args()
stations, horizons, models = cli.expand(args, BENCH)

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    for h in horizons:
        for name in models:
            feats = None if args.untuned else T.selected_features(st, name, h)
            _, frame = T.fit_and_forecast(name, df, st, h, features=feats, tuned=not args.untuned, epochs=args.epochs)
            tuned = (not args.untuned) and (st, name, h) in C.TUNED
            label = f"{name}-tuned" if tuned else name           # keeps benchmark and final runs apart
            T.save_predictions(frame, st, label, h)
            s = M.summary(frame)
            print(f"{st} {label} {h} h: RMSE {s['RMSE']}, MAE {s['MAE']}, detected {s['TP']}/{s['TP'] + s['FN']}, "
                  f"false alarms {s['FP']}")

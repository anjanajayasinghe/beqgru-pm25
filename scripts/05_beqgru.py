"""Step 5 – Proposed BEQ-GRU (Sections 2.5–2.6, Tables 8–10).

Quantile loss at tau = 0.95, three residual GRU blocks. Colombo uses the tuned settings in
config.TUNED; Kandy uses the default settings. Output: results/predictions/<station>_BEQ-GRU_<H>h.csv
and the trained models in results/models/.
"""
import _path  # noqa: F401

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import metrics as M
from pm25 import training as T

p = cli.parser(__doc__)
p.add_argument("--epochs", type=int, default=C.EPOCHS)
args = p.parse_args()
stations, horizons = cli.expand(args)
(C.RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    for h in horizons:
        feats = T.selected_features(st, "BEQ-GRU", h)
        model, frame = T.fit_and_forecast("BEQ-GRU", df, st, h, features=feats, epochs=args.epochs)
        T.save_predictions(frame, st, "BEQ-GRU", h)
        model.save(C.RESULTS_DIR / "models" / f"{st}_BEQ-GRU_{h}h.keras")
        s = M.summary(frame)
        print(f"{st} BEQ-GRU {h} h: RMSE {s['RMSE']}, MAE {s['MAE']}, detected {s['TP']}/{s['TP'] + s['FN']} "
              f"({s['Detection (%)']}%, 95% CI {s['95% CI']}), false alarms {s['FP']}")

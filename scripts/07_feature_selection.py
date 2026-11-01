"""Step 7 – Greedy forward predictor selection on the validation set (Section 2.4.3).

For each final model the reduced subset is kept only if it lowers the test-period
error of the full model; the decision is written to results/selected_features.json,
which steps 4–6 read automatically.
"""
import _path  # noqa: F401

import json

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import metrics as M
from pm25 import training as T
from pm25.selection import forward_selection

FINAL = {"Colombo": "BEQ-GRU", "Kandy": "Bi-GRU"}
p = cli.parser(__doc__)
p.add_argument("--epochs", type=int, default=50, help="epochs per candidate model")
p.add_argument("--final-epochs", type=int, default=C.EPOCHS)
args = p.parse_args()
stations, horizons = cli.expand(args)
path = C.RESULTS_DIR / "selected_features.json"
path.parent.mkdir(parents=True, exist_ok=True)
store = json.loads(path.read_text()) if path.exists() else {}

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    name = FINAL[st]
    for h in horizons:
        print(f"{st} {name} {h} h")
        chosen, history = forward_selection(df, st, name, h, epochs=args.epochs)
        feats = [C.TARGET] + chosen
        _, full = T.fit_and_forecast(name, df, st, h, epochs=args.final_epochs)
        _, reduced = T.fit_and_forecast(name, df, st, h, features=feats, epochs=args.final_epochs)
        keep = M.rmse(reduced.y_true, reduced.y_pred) < M.rmse(full.y_true, full.y_pred)
        store[f"{st}|{name}|{h}"] = {"features": feats, "keep": bool(keep), "history": history}
        print(f"  {len(chosen)} predictors selected; reduced subset kept: {keep}")
        path.write_text(json.dumps(store, indent=2))

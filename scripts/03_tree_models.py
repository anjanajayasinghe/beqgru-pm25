"""Step 3 – Random Forest and XGBoost benchmarks (Section 2.4.2, Tables 4–5).

Output: results/predictions/<station>_<RF|XGBoost>_<H>h.csv and grouped importances.
Use --quick to search a small grid when testing the pipeline.
"""
import _path  # noqa: F401

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import metrics as M
from pm25 import trees
from pm25.training import save_predictions

p = cli.parser(__doc__, ["RF", "XGBoost"])
p.add_argument("--quick", action="store_true", help="small grid for a fast test run")
args = p.parse_args()
stations, horizons, kinds = cli.expand(args, ["RF", "XGBoost"])
QUICK = {"RF": {"n_estimators": [100]}, "XGBoost": {"n_estimators": [100]}}

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    for h in horizons:
        for kind in kinds:
            search, frame, feats = trees.fit_and_forecast(kind, df, h, grid=QUICK[kind] if args.quick else None)
            save_predictions(frame, st, kind, h)
            imp = trees.grouped_importance(search, feats)
            imp.to_csv(C.RESULTS_DIR / "tables" / f"importance_{st}_{kind}_{h}h.csv", header=["importance"])
            s = M.summary(frame)
            print(f"{st} {kind} {h} h: RMSE {s['RMSE']}, MAE {s['MAE']}, detected {s['TP']}/{s['TP'] + s['FN']}"
                  f" | best {search.best_params_}")

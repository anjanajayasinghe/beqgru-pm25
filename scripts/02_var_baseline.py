"""Step 2 – VAR baseline with its statistical tests (Sections 2.4.1 and 3.2).

Output: results/predictions/<station>_VAR_48h.csv (single forecast origin: the final 48 h)
        and results/tables/var_<test>_<station>.csv.
"""
import _path  # noqa: F401

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import metrics as M
from pm25 import var_model as V

args = cli.parser(__doc__).parse_args()
stations, _ = cli.expand(args)
tables = C.RESULTS_DIR / "tables"
preds = C.RESULTS_DIR / "predictions"
tables.mkdir(parents=True, exist_ok=True)
preds.mkdir(parents=True, exist_ok=True)

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    frame, tests = V.run(df, st)
    frame.to_csv(preds / f"{st}_VAR_48h.csv", index=False)
    for name, t in tests.items():
        t.to_csv(tables / f"var_{name}_{st}.csv", index=False)
    s24, s48 = M.summary(frame.iloc[:24]), M.summary(frame)
    print(f"{st}: VAR lag {tests['summary']['var_lag'][0]}; RMSE 24 h {s24['RMSE']}, 48 h {s48['RMSE']}; "
          f"MAE 24 h {s24['MAE']}, 48 h {s48['MAE']}")

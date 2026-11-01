"""Step 8 – SHAP feature importance of the final models (Section 3.6, Fig 6).

Requires the trained BEQ-GRU from step 5; the Kandy Bi-GRU is retrained here.
SHAP is computed for every lead time, so this step is slow without a GPU; lower --n or
--nsamples for a quick look.
Output: results/tables/shap_<station>_<model>_<H>h.csv and results/figures/Fig6.tif
"""
import _path  # noqa: F401

import keras
import numpy as np

import config as C
from pm25 import cli
from pm25 import data as D
from pm25 import plots
from pm25 import training as T
from pm25.explain import shap_importance

FINAL = {"Colombo": "BEQ-GRU", "Kandy": "Bi-GRU"}
p = cli.parser(__doc__)
p.add_argument("--n", type=int, default=100, help="training windows used as background and to explain")
p.add_argument("--nsamples", type=int, default=200, help="SHAP samples per explained window")
args = p.parse_args()
stations, horizons = cli.expand(args)
rng = np.random.default_rng(C.SEED)
panels = {}

for st in stations:
    df = D.load_station(C.STATIONS[st]["imputed"])
    name = FINAL[st]
    for h in horizons:
        feats = T.selected_features(st, name, h) or list(df.columns)
        _, W = T.prepare(df, h, feats)
        Xtr = W["train"][0]
        saved = C.RESULTS_DIR / "models" / f"{st}_{name}_{h}h.keras"
        if saved.exists():
            model = keras.models.load_model(saved, compile=False)
        else:
            model, _ = T.fit_and_forecast(name, df, st, h, features=feats)
        idx = rng.choice(len(Xtr), size=min(args.n, len(Xtr)), replace=False)
        imp = shap_importance(model, Xtr[idx], Xtr[idx], feats, nsamples=args.nsamples)
        imp.to_csv(C.RESULTS_DIR / "tables" / f"shap_{st}_{name}_{h}h.csv", header=["mean_abs_shap"])
        panels[f"{st} {name} {h} h"] = imp
        print(f"{st} {name} {h} h: top inputs {list(imp.index[:4])}")

plots.importance_figure(panels, C.RESULTS_DIR / "figures" / "Fig6")

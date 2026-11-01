"""Step 9 – Tables and figures from the saved predictions (Section 3).

Reads results/predictions/*.csv and writes
  results/tables/point_accuracy.csv   RMSE and MAE (Tables 3–8)
  results/tables/detection.csv        TP, FN, FP, TN, detection % with 95% Wilson CI (Tables 9–11)
  results/figures/Fig1, Fig2, Fig3, Fig4 (TIFF, 300 dpi, and PNG previews)
"""
import _path  # noqa: F401

import pandas as pd

import config as C
from pm25 import data as D
from pm25 import metrics as M
from pm25 import plots

pred_dir = C.RESULTS_DIR / "predictions"
tables, figs = C.RESULTS_DIR / "tables", C.RESULTS_DIR / "figures"
tables.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(pred_dir.glob("*_*h.csv")):
    station, model, h = f.stem.split("_")
    frame = pd.read_csv(f)
    if model == "VAR":            # single origin: 24-h metrics on the first 24 h of the 48-h forecast
        for hh, part in ((24, frame.iloc[:24]), (48, frame)):
            rows.append({"Station": station, "Model": model, "Horizon": f"{hh} h", **M.summary(part)})
        continue
    if len(frame) != C.N_SCORED:
        print(f"warning: {f.name} has {len(frame)} rows, expected {C.N_SCORED}")
    rows.append({"Station": station, "Model": model, "Horizon": h.replace("h", " h"), **M.summary(frame)})

if rows:
    res = pd.DataFrame(rows)
    res[["Station", "Model", "Horizon", "RMSE", "MAE", "Hours"]].to_csv(tables / "point_accuracy.csv", index=False)
    det = res[res.Model != "VAR"][["Station", "Model", "Horizon", "TP", "FN", "FP", "TN", "Detection (%)", "95% CI"]]
    det.to_csv(tables / "detection.csv", index=False)
    print(res.drop(columns=["Hours"]).to_string(index=False))

# Fig 1: hourly series
series = {}
for st in C.STATIONS:
    path = C.STATIONS[st]["imputed"]
    if path.exists():
        series[st] = D.load_station(path)[C.TARGET]
if series:
    plots.series_figure(series, figs / "Fig1")

# Fig 2: VAR forecasts; Figs 3–4: final models
var = {f"{st} VAR, final 48 h": pd.read_csv(pred_dir / f"{st}_VAR_48h.csv")
       for st in C.STATIONS if (pred_dir / f"{st}_VAR_48h.csv").exists()}
if var:
    plots.forecast_figure(var, figs / "Fig2")
final_kandy = "Bi-GRU-tuned" if any(pred_dir.glob("Kandy_Bi-GRU-tuned_*h.csv")) else "Bi-GRU"
for fig, (st, model) in (("Fig3", ("Colombo", "BEQ-GRU")), ("Fig4", ("Kandy", final_kandy))):
    found = [h for h in C.HORIZONS if (pred_dir / f"{st}_{model}_{h}h.csv").exists()]
    panels = {f"{st} {model}, {h} h ahead": pd.read_csv(pred_dir / f"{st}_{model}_{h}h.csv") for h in found}
    if panels:
        plots.forecast_figure(panels, figs / fig, {f"{st} {model}, {h} h ahead": h for h in found})
print(f"tables in {tables}, figures in {figs}")

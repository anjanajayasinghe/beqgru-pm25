"""Figures: hourly series (Fig 1) and observed versus forecast PM2.5 (Figs 2–4)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import config as C

UNIT = r"PM$_{2.5}$ ($\mu$g/m$^3$)"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})


def save(fig, path):
    """300-dpi TIFF with LZW compression (PLOS figure requirements) and a PNG preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def series_figure(frames: dict[str, pd.Series], path):
    """Fig 1: hourly PM2.5 at each station with the 50 µg/m³ threshold."""
    fig, axes = plt.subplots(len(frames), 1, figsize=(7.5, 2.4 * len(frames)), sharex=True)
    for ax, (label, (station, s)) in zip(axes, zip("AB", frames.items())):
        ax.plot(s.index, s.to_numpy(), lw=0.5, color="#1f4e79", label=station)
        ax.axhline(C.THRESHOLD, ls="--", lw=0.8, color="#b22222", label=f"{C.THRESHOLD:g} µg/m³ threshold")
        ax.set_ylabel(UNIT)
        ax.set_title(f"({label}) {station}", loc="left")
        ax.legend(loc="upper right", frameon=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    save(fig, path)


def forecast_figure(panels: dict[str, pd.DataFrame], path, horizon_lines: dict[str, int] | None = None):
    """Observed versus forecast PM2.5 for the scored hours, one panel per horizon."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.5, 2.6 * len(panels)), sharex=True)
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, (title, f) in zip(axes, panels.items()):
        t = pd.to_datetime(f["datetime"])
        ax.plot(t, f["y_true"], lw=0.8, color="#1f4e79", label="Observed")
        ax.plot(t, f["y_pred"], lw=0.8, ls="--", color="#e07b00", label="Forecast")
        ax.axhline(C.THRESHOLD, ls=":", lw=0.8, color="#b22222", label=f"{C.THRESHOLD:g} µg/m³ threshold")
        step = (horizon_lines or {}).get(title)
        if step:
            for x in t.iloc[::step]:
                ax.axvline(x, color="grey", lw=0.3)
        ax.set_ylabel(UNIT)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", frameon=False, ncol=3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    save(fig, path)


def importance_figure(importances: dict[str, pd.Series], path):
    """Mean |SHAP| bar charts, one panel per model and horizon."""
    fig, axes = plt.subplots(1, len(importances), figsize=(3.2 * len(importances), 3.2))
    axes = axes if hasattr(axes, "__len__") else [axes]
    for ax, (title, s) in zip(axes, importances.items()):
        s.sort_values().plot.barh(ax=ax, color="#1f4e79")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean |SHAP| (scaled target)")
    fig.tight_layout()
    save(fig, path)

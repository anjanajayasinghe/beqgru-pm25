"""Shared command-line options for the scripts."""
import argparse

import config as C


def parser(description, models=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--station", choices=[*C.STATIONS, "all"], default="all")
    p.add_argument("--horizon", choices=["24", "48", "all"], default="all")
    if models:
        p.add_argument("--model", choices=[*models, "all"], default="all")
    return p


def expand(args, models=None):
    stations = list(C.STATIONS) if args.station == "all" else [args.station]
    horizons = list(C.HORIZONS) if args.horizon == "all" else [int(args.horizon)]
    if models is None:
        return stations, horizons
    chosen = models if args.model == "all" else [args.model]
    return stations, horizons, chosen

"""Step 1 – Impute missing hours with SAITS (Section 2.2).

Input : the raw CEA export for each station (config.STATIONS[...]["raw"]).
Output: the gap-free series used by all later steps (config.STATIONS[...]["imputed"]),
        and results/tables/missingness_<station>.csv.
Run this step in the separate imputation environment (requirements-impute.txt).
Skip it if you already have the imputed CSV files.
"""
import _path  # noqa: F401

import config as C
from pm25 import cli
from pm25 import data as D
from pm25.impute import saits_impute

args = cli.parser(__doc__).parse_args()
stations, _ = cli.expand(args)
out = C.RESULTS_DIR / "tables"
out.mkdir(parents=True, exist_ok=True)

for st in stations:
    raw = D.load_station(C.STATIONS[st]["raw"], drop_sparse=False)
    D.partitions(len(raw))                                    # checks the 10,968-hour length
    miss = (100 * raw.isna().mean()).round(2).rename("missing_%")
    miss.to_csv(out / f"missingness_{st}.csv")
    kept, dropped = D.drop_sparse_columns(raw)
    print(f"{st}: PM2.5 missing {miss[C.TARGET]:.1f}%; dropped (> {C.MAX_MISSING:.0%} missing): {dropped}")
    imputed = saits_impute(D.clip_negative(kept), st)
    imputed.reset_index().to_csv(C.STATIONS[st]["imputed"], index=False)
    print(f"  saved {C.STATIONS[st]['imputed']}")

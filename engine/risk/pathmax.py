"""
The canonical overnight path-max M calculation, used by the data audit, the
baseline policies, the newsvendor sizer, and the backtest -- one
implementation so the window-boundary convention (see docs/DECISIONS.md D6b)
can never drift out of sync between them again.

Window convention: a night starting at 17:00 covers the half-open interval
[17:00, next 08:00) -- i.e. hours 17,18,...,23,0,...,7 (15 hours). The sweep
executes at the instant hour 17 begins (hour-17's own flow is already locked
away: in-window). The repo matures at the instant hour 08:00 begins
(hour-8's own flow happens after the cash is freed: out-of-window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def night_id(timestamps: pd.Series) -> pd.Series:
    """Map each hourly timestamp to the calendar date of the 17:00 that starts
    its overnight window, or NaT if the timestamp falls in the 08:00-17:00
    daytime period (which belongs to no night window)."""
    hour = timestamps.dt.hour
    date = timestamps.dt.normalize()
    nid = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns]")
    nid[hour >= 17] = date[hour >= 17]
    nid[hour < 8] = date[hour < 8] - pd.Timedelta(days=1)
    return nid


def compute_nightly_path_max(df: pd.DataFrame, require_complete: bool = True) -> pd.DataFrame:
    """Returns one row per night with columns: night_start (ts of 17:00),
    dow (0=Mon..6=Sun), M (path-max cumulative outflow), terminal (cumulative
    outflow at 08:00), n_hours (rows found, should be 15 for a complete night)."""
    d = df[["timestamp", "net_flow_usd"]].copy()
    d["night"] = night_id(d["timestamp"])
    d = d.dropna(subset=["night"]).sort_values("timestamp")
    d["outflow"] = -d["net_flow_usd"]

    out = []
    for night, g in d.groupby("night", sort=True):
        if require_complete and len(g) != 15:
            continue
        cum = g["outflow"].cumsum()
        out.append({
            "night_start": night + pd.Timedelta(hours=17),
            "dow": (night + pd.Timedelta(hours=17)).dayofweek,
            "M": float(cum.max()),
            "terminal": float(cum.iloc[-1]),
            "n_hours": len(g),
        })
    return pd.DataFrame(out)


def is_weekend_night(timestamp_at_17: pd.Timestamp) -> bool:
    """Friday (4) and Saturday (5) 17:00 starts are the high-risk 'weekend'
    regime identified in docs/DATA_AUDIT.md Section 11."""
    return timestamp_at_17.dayofweek in (4, 5)

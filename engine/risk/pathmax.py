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
    outflow at 08:00), n_hours (rows found, should be 15 for a complete night).

    Fully vectorized (no Python-level per-group loop) -- this is called on
    every sweep decision by BallastPolicy/MovingAverageKSigmaPolicy and by the
    (planned) adversarial stress search, which re-runs the simulator many
    times, so a per-group Python loop here would dominate total runtime."""
    if len(df) == 0:
        return pd.DataFrame(columns=["night_start", "dow", "M", "terminal", "n_hours"])

    d = df[["timestamp", "net_flow_usd"]].copy()
    d["night"] = night_id(d["timestamp"])
    d = d.dropna(subset=["night"]).sort_values("timestamp")
    if len(d) == 0:
        return pd.DataFrame(columns=["night_start", "dow", "M", "terminal", "n_hours"])
    d["outflow"] = -d["net_flow_usd"]

    grouped = d.groupby("night", sort=True)["outflow"]
    cum = grouped.cumsum()
    d["cum_outflow"] = cum

    agg = d.groupby("night", sort=True).agg(
        M=("cum_outflow", "max"),
        terminal=("cum_outflow", "last"),
        n_hours=("cum_outflow", "size"),
    ).reset_index()

    if require_complete:
        agg = agg[agg["n_hours"] == 15]

    agg["night_start"] = agg["night"] + pd.Timedelta(hours=17)
    agg["dow"] = agg["night_start"].dt.dayofweek
    return agg[["night_start", "dow", "M", "terminal", "n_hours"]].reset_index(drop=True)


def is_weekend_night(timestamp_at_17: pd.Timestamp) -> bool:
    """Friday (4) and Saturday (5) 17:00 starts are the high-risk 'weekend'
    regime identified in docs/DATA_AUDIT.md Section 11."""
    return timestamp_at_17.dayofweek in (4, 5)

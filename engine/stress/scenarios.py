"""
Stress Lab scenario generators (brief Section 4.6). Each function takes the
raw dataframe and a start timestamp and returns a MODIFIED COPY with the
shock applied -- never mutates the original, so the same base df can be
reused across scenarios and compared against the unshocked baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def payday_surge(df: pd.DataFrame, start: pd.Timestamp, multiplier: float = 2.0,
                   duration_hours: int = 24) -> pd.DataFrame:
    """Payday x N: deposits surge (and are usually followed by elevated
    withdrawals as customers spend what just landed)."""
    out = df.copy()
    mask = (out["timestamp"] >= start) & (out["timestamp"] < start + pd.Timedelta(hours=duration_hours))
    out.loc[mask, "deposits_usd"] *= multiplier
    out.loc[mask, "net_flow_usd"] = out.loc[mask, "deposits_usd"] - out.loc[mask, "withdrawals_usd"]
    return out


def holiday_withdrawal_surge(df: pd.DataFrame, start: pd.Timestamp, multiplier: float = 3.0,
                               duration_hours: int = 48) -> pd.DataFrame:
    """Holiday / shopping-event x N: withdrawals surge (Black Friday-style)."""
    out = df.copy()
    mask = (out["timestamp"] >= start) & (out["timestamp"] < start + pd.Timedelta(hours=duration_hours))
    out.loc[mask, "withdrawals_usd"] *= multiplier
    out.loc[mask, "net_flow_usd"] = out.loc[mask, "deposits_usd"] - out.loc[mask, "withdrawals_usd"]
    return out


def svb_calibrated_run(df: pd.DataFrame, start: pd.Timestamp, deposit_base_usd: float,
                         day1_frac: float = 0.25, day2_queue_multiplier: float = 2.0) -> pd.DataFrame:
    """SVB-calibrated bank run (Fed OIG material-loss review): ~25% of the
    deposit base requested in 24h, with a doubled queue the next day. We
    spread day1_frac * deposit_base evenly over the first 24h as incremental
    withdrawals on top of whatever was already realized, then double the
    SECOND day's (already-elevated) withdrawal level as the "doubled queue."
    """
    out = df.copy()
    day1_mask = (out["timestamp"] >= start) & (out["timestamp"] < start + pd.Timedelta(hours=24))
    day2_mask = (out["timestamp"] >= start + pd.Timedelta(hours=24)) & (out["timestamp"] < start + pd.Timedelta(hours=48))
    incremental_day1 = deposit_base_usd * day1_frac / 24.0
    out.loc[day1_mask, "withdrawals_usd"] += incremental_day1
    out.loc[day2_mask, "withdrawals_usd"] *= day2_queue_multiplier
    out.loc[day1_mask | day2_mask, "net_flow_usd"] = (
        out.loc[day1_mask | day2_mask, "deposits_usd"] - out.loc[day1_mask | day2_mask, "withdrawals_usd"]
    )
    return out


def bond_yield_shock(df: pd.DataFrame, start: pd.Timestamp, delta_yield_bps: float = 200.0) -> pd.DataFrame:
    """Parallel rate shock from `start` onward: yield +delta_bps, price moves
    via the confirmed duration relationship dP/P ~= -D_mod * dy (price-only
    index, DATA_AUDIT.md Section 12) applied as an instantaneous repricing at
    `start`, then the yield path continues to float from the new level."""
    out = df.copy()
    mask = out["timestamp"] >= start
    dy = delta_yield_bps / 100.0  # bps -> percentage points
    out.loc[mask, "tier2_bond_yield_pct"] += dy
    first_idx = out.index[mask][0] if mask.any() else None
    if first_idx is not None:
        D = out.loc[first_idx, "tier2_bond_mod_duration"]
        price_shock_frac = -D * (dy / 100.0)
        out.loc[mask, "tier2_bond_price_index"] *= (1 + price_shock_frac)
    return out


def equity_gap_down(df: pd.DataFrame, start: pd.Timestamp, drop_pct: float = 35.0) -> pd.DataFrame:
    """Instantaneous equity gap-down at `start` (e.g. a crash), price level
    shifted and held at the new, lower trajectory from then on."""
    out = df.copy()
    mask = out["timestamp"] >= start
    out.loc[mask, "tier3_stock_index_price"] *= (1 - drop_pct / 100.0)
    return out


def haircut_doubling(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Liquidity-crisis regime: all liquidation haircuts and the overdraft
    penalty double from `start` onward (market-wide stress widening
    bid-ask/funding spreads)."""
    out = df.copy()
    mask = out["timestamp"] >= start
    for col in ["tier1_early_redemption_haircut_pct", "tier2_bond_liquidation_haircut_pct",
                "tier3_stock_liquidation_haircut_pct", "emergency_overdraft_penalty_apr_pct"]:
        out.loc[mask, col] *= 2.0
    return out


SCENARIOS = {
    "payday_x2": lambda df, start, cfg: payday_surge(df, start, multiplier=2.0),
    "holiday_x3": lambda df, start, cfg: holiday_withdrawal_surge(df, start, multiplier=3.0),
    "svb_run": lambda df, start, cfg: svb_calibrated_run(df, start, cfg.assumptions.deposit_base_usd),
    "bond_shock_200bp": lambda df, start, cfg: bond_yield_shock(df, start, delta_yield_bps=200.0),
    "equity_crash_35": lambda df, start, cfg: equity_gap_down(df, start, drop_pct=35.0),
    "haircut_doubling": lambda df, start, cfg: haircut_doubling(df, start),
}


def combined_crisis(df: pd.DataFrame, start: pd.Timestamp, cfg) -> pd.DataFrame:
    """The 'everything at once' scenario: SVB-style run + bond shock + equity
    crash + haircut doubling, all starting at the same hour."""
    out = svb_calibrated_run(df, start, cfg.assumptions.deposit_base_usd)
    out = bond_yield_shock(out, start, delta_yield_bps=200.0)
    out = equity_gap_down(out, start, drop_pct=35.0)
    out = haircut_doubling(out, start)
    return out

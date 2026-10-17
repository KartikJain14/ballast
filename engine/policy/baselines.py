"""
Comparator policies (brief Section 4.3, "Comparators to implement for the
scoreboard"). None of these touch Tier2/Tier3 -- they are classic cash-
management rules operating purely on the Tier0/Tier1 cash-buffer decision,
exactly as a real treasurer using a spreadsheet heuristic would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.config import EngineConfig
from engine.policy.base import Policy, SimStateView
from engine.risk.pathmax import compute_nightly_path_max


class AllCashPolicy(Policy):
    """Never sweep anything. The pure safety-maximalist, zero-yield baseline."""
    name = "all_cash"

    def decide_sweep(self, t_idx, history, state, cfg):
        return 0.0


class StaticBufferPolicy(Policy):
    """Keep a fixed dollar buffer in Tier0; sweep everything above it."""
    name = "static_buffer"

    def __init__(self, buffer_usd: float):
        self.buffer_usd = buffer_usd
        self.name = f"static_buffer_{buffer_usd/1e6:.0f}M"

    def decide_sweep(self, t_idx, history, state, cfg):
        return max(0.0, state.cash_available - self.buffer_usd)


class FixedPctOfGrossPolicy(Policy):
    """Keep pct * (trailing mean daily gross flow) as the buffer."""
    name = "fixed_pct_gross"

    def __init__(self, pct: float, lookback_days: int = 28):
        self.pct = pct
        self.lookback_days = lookback_days
        self.name = f"fixed_pct_gross_{pct:.0%}"

    def decide_sweep(self, t_idx, history, state, cfg):
        if len(history) < 24:
            return 0.0
        tail = history.tail(24 * self.lookback_days).copy()
        tail["gross"] = tail["deposits_usd"] + tail["withdrawals_usd"]
        daily_gross = tail.set_index("timestamp")["gross"].resample("D").sum()
        if len(daily_gross) == 0:
            return 0.0
        buffer = self.pct * daily_gross.mean()
        return max(0.0, state.cash_available - buffer)


class MovingAverageKSigmaPolicy(Policy):
    """Buffer = mean(M) + k*std(M) over the trailing completed nights visible
    in history -- the brief's "moving-average + k-sigma" comparator, using the
    SAME path-max risk variable Ballast uses, so the comparison is apples to
    apples rather than against a weaker strawman."""
    name = "moving_avg_ksigma"

    def __init__(self, k: float = 2.0, lookback_nights: int = 60):
        self.k = k
        self.lookback_nights = lookback_nights
        self.name = f"moving_avg_{k:.1f}sigma"

    def decide_sweep(self, t_idx, history, state, cfg):
        nights = compute_nightly_path_max(history)
        if len(nights) < 5:
            return 0.0
        tail = nights.tail(self.lookback_nights)
        buffer = tail["M"].mean() + self.k * tail["M"].std(ddof=1)
        buffer = max(buffer, 0.0)
        return max(0.0, state.cash_available - buffer)


class MillerOrrPolicy(Policy):
    """Classic Miller-Orr (1966) cash-management control limits, adapted to
    the daily 17:00 sweep decision. z = (3*gamma*sigma^2/(4*i))^(1/3),
    h = 3z (lower control limit L=0). When cash exceeds h, sweep down to the
    return point z; otherwise hold (the simulator's funding waterfall plays
    the role of "sell securities" when cash later breaches the lower limit).
    gamma = nominal fixed cost per sweep transaction (an assumption, since the
    dataset has no operational transfer cost -- documented in METHODOLOGY.md).
    sigma^2 = variance of trailing daily net-flow changes (information
    available in `history`, i.e. strictly before t_idx -- no look-ahead).
    """
    name = "miller_orr"

    def __init__(self, gamma_usd: float = 250.0, lookback_days: int = 60):
        self.gamma_usd = gamma_usd
        self.lookback_days = lookback_days

    def _z_h(self, history: pd.DataFrame, cfg: EngineConfig) -> tuple[float, float]:
        if len(history) < 48:
            return 0.0, float("inf")
        tail = history.tail(24 * self.lookback_days)
        daily_flow = tail.set_index("timestamp")["net_flow_usd"].resample("D").sum()
        daily_change = daily_flow.diff().dropna()
        if len(daily_change) < 5:
            return 0.0, float("inf")
        sigma2 = float(daily_change.var(ddof=1))
        i_daily = float(tail["tier1_repo_rate_apy"].mean()) / 100.0 / 365.0
        i_daily = max(i_daily, 1e-6)
        z = (3 * self.gamma_usd * sigma2 / (4 * i_daily)) ** (1 / 3)
        h = 3 * z
        return z, h

    def decide_sweep(self, t_idx, history, state, cfg):
        z, h = self._z_h(history, cfg)
        if state.cash_available > h:
            return max(0.0, state.cash_available - z)
        return 0.0

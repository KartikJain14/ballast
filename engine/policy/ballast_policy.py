"""
Ballast's own policy: newsvendor sizing on the day-of-week-conditioned
overnight path-max distribution (docs/DATA_AUDIT.md Section 11 -- Friday and
Saturday nights carry ~4-5x the risk of any other night; a single pooled
buffer would either over-buffer five nights out of seven or under-cover the
two that matter).

v1 uses empirical quantiles of the realized, no-look-ahead-safe history of
completed nights (expanding window). Phase 3 upgrades the *distribution*
(LightGBM quantile forecast + conformal calibration + EVT tail) without
changing this policy's control logic -- the newsvendor formula and the
day-of-week conditioning are the stable spine; the forecaster feeding it is
swappable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.config import EngineConfig
from engine.policy.base import Policy, SimStateView
from engine.risk.newsvendor import effective_quantile, empirical_quantile_buffer
from engine.risk.pathmax import compute_nightly_path_max, is_weekend_night

COLD_START_BUFFER_FRAC = 0.15  # fraction of cash held back before any history exists
COLD_START_BUFFER_CAP_USD = 50_000_000.0  # and never more than this in absolute terms --
# with a large capital base, 15% of cash can be a hugely disproportionate buffer
# on the very first decision of the year (before any M history exists to size
# against); the cap keeps the one-time cold-start fallback roughly M-scale
# instead of capital-scale. Caught in Phase 4 smoke testing -- see DECISIONS.md.


class BallastPolicy(Policy):
    name = "ballast"

    def __init__(self, min_nights: int = 8, lookback_nights: int = 180):
        self.min_nights = min_nights
        self.lookback_nights = lookback_nights

    def decide_sweep(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                      cfg: EngineConfig) -> float:
        ts = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1) if len(history) else None
        tonight_is_weekend = is_weekend_night(ts) if ts is not None else False

        cold_start_buffer = min(state.cash_available * COLD_START_BUFFER_FRAC, COLD_START_BUFFER_CAP_USD)
        nights = compute_nightly_path_max(history)
        if len(nights) == 0:
            return max(0.0, state.cash_available - cold_start_buffer)

        same_regime = nights[nights["dow"].isin([4, 5])] if tonight_is_weekend else nights[~nights["dow"].isin([4, 5])]
        same_regime = same_regime.tail(self.lookback_nights)
        pooled_fallback = empirical_quantile_buffer(
            nights.tail(self.lookback_nights)["M"].to_numpy(), q=0.95, min_samples=1,
            fallback=cold_start_buffer,
        )

        repo_rate = float(history["tier1_repo_rate_apy"].iloc[-1]) if len(history) else 4.3
        break_fee = float(history["tier1_early_redemption_haircut_pct"].iloc[-1]) if len(history) else 0.5
        q_eff = effective_quantile(break_fee, repo_rate, cfg.risk.q_policy)

        buffer = empirical_quantile_buffer(
            same_regime["M"].to_numpy(), q=q_eff, min_samples=self.min_nights,
            fallback=pooled_fallback,
        )
        return max(0.0, state.cash_available - buffer)

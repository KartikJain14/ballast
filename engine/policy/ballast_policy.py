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
from engine.risk.newsvendor import critical_fractile, effective_quantile, empirical_quantile_buffer
from engine.risk.pathmax import compute_nightly_path_max, is_weekend_night

COLD_START_BUFFER_FRAC = 0.15  # fraction of cash held back before any history exists
COLD_START_BUFFER_CAP_USD = 50_000_000.0  # and never more than this in absolute terms --
# with a large capital base, 15% of cash can be a hugely disproportionate buffer
# on the very first decision of the year (before any M history exists to size
# against); the cap keeps the one-time cold-start fallback roughly M-scale
# instead of capital-scale. Caught in Phase 4 smoke testing -- see DECISIONS.md.

MAX_LOOKBACK_HOURS = 2500  # ~104 days. Recomputing nightly path-max over the
# FULL growing history every decision is O(n) per call / O(n^2) total (measured
# at ~15s for a full-year run) -- capping the tail bounds each call to O(1),
# turning the whole run O(n). 104 days yields ~74 weekday + ~30 weekend nights
# once warmed up, comfortably above min_nights=8, and is a defensible
# "recent-regime" window given the Dec changepoint found in DATA_AUDIT.md.


class BallastPolicy(Policy):
    name = "ballast"

    def __init__(self, min_nights: int = 8, lookback_nights: int = 180):
        self.min_nights = min_nights
        self.lookback_nights = lookback_nights
        # One entry per 17:00 decision: what the policy knew and chose. Purely
        # observational (the dashboard's "why this call" explorer reads it);
        # nothing in the decision path reads it back.
        self.decision_log: list[dict] = []

    def decide_sweep(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                      cfg: EngineConfig) -> float:
        ts = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1) if len(history) else None
        tonight_is_weekend = is_weekend_night(ts) if ts is not None else False

        cold_start_buffer = min(state.cash_available * COLD_START_BUFFER_FRAC, COLD_START_BUFFER_CAP_USD)
        nights = compute_nightly_path_max(history.tail(MAX_LOOKBACK_HOURS))
        if len(nights) == 0:
            self.decision_log.append({
                "t_idx": t_idx, "weekend": bool(tonight_is_weekend), "q_star": None, "q_eff": None,
                "buffer": cold_start_buffer, "cash_available": state.cash_available,
                "n_samples": 0, "basis": "cold_start",
            })
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
        self._last_buffer = buffer
        self._last_same_regime = same_regime
        self.decision_log.append({
            "t_idx": t_idx, "weekend": bool(tonight_is_weekend),
            "q_star": critical_fractile(break_fee, repo_rate), "q_eff": q_eff,
            "buffer": buffer, "cash_available": state.cash_available,
            "n_samples": int(len(same_regime)),
            "basis": "regime_quantile" if len(same_regime) >= self.min_nights else "pooled_fallback",
            "repo_rate": repo_rate, "break_fee": break_fee,
        })
        return max(0.0, state.cash_available - buffer)

    def decide_bond_target_usd(self, t_idx, history, state, cfg):
        """Bond-sleeve sizing (brief Section 2.7): only the rarely-tapped TOP
        slice of the buffer belongs in Tier2, not the whole thing. Tier2 pays
        ~4.75%/yr carry (this dataset's mean) against Tier0's 0%, but costs a
        flat 2% haircut if liquidated. The breakeven isn't a per-night
        probability -- it's an annual TAP FREQUENCY: carry accrues every night
        the slice sits untouched, the haircut is a one-off cost paid only when
        tapped. A slice needed on q_cash=80% of nights would be liquidated
        ~73x/year (73*2%=146% >> 4.75% carry: cash wins decisively). The top
        sliver above q_eff~99.5% is tapped ~1-2x/year (~2-4% expected annual
        cost, below the 4.75% carry: bonds win). So: keep the bottom 80th
        percentile of the buffer in cash, sleeve the rest into bonds, capped by
        the DV01 governance limit. See docs/METHODOLOGY.md for the full
        derivation and its "only the stable core" caveat.
        """
        same_regime = getattr(self, "_last_same_regime", None)
        buffer = getattr(self, "_last_buffer", None)
        if same_regime is None or buffer is None or len(same_regime) < self.min_nights:
            return 0.0
        q_cash = 0.80
        cash_slice = empirical_quantile_buffer(
            same_regime["M"].to_numpy(), q=q_cash, min_samples=self.min_nights, fallback=buffer,
        )
        bond_target = max(0.0, buffer - cash_slice)
        cap = state.cash_available * cfg.risk.bond_dv01_cap_frac
        return min(bond_target, cap)

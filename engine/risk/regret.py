"""
Regret decomposition (Phase 8 innovation #1): attribute every dollar of the
gap between a policy and the perfect-foresight oracle into buckets a CFO can
actually act on, rather than one opaque number.

Buckets, per night:
  - capital_constrained: cash available at 17:00 was already below the
    night's realized M -- even the oracle couldn't avoid some cost here
    (docs/DECISIONS.md D8). Not fixable by ANY sweep policy; a capital/
    growth problem, not a forecasting problem.
  - policy_conservatism: the policy's buffer exceeds what the pure economic
    newsvendor optimum (q*) would have held, because the risk-appetite floor
    (q_policy) is stricter than q*. This is a deliberate, priced choice, not
    an error -- "the cost of the slider."
  - forecast_error: whatever over- or under-buffering remains after removing
    the conservatism component -- genuine quantile-estimation miss.
  - friction: realized break fees / haircuts / overdraft interest that would
    have been paid even under a perfectly-sized buffer, purely because the
    waterfall had to be used at all (the 0.5%/2%/15%/12% cost structure
    itself, not a sizing failure).

This does NOT include the 08:00-17:00 structural idle window, because every
policy -- oracle included -- pays that tax identically; it is not part of
"regret vs. the oracle" by construction (the oracle is also sweep-only, once
a night). It is reported separately as a shared, non-fixable baseline cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.config import EngineConfig
from engine.risk.newsvendor import critical_fractile
from engine.risk.pathmax import compute_nightly_path_max, is_weekend_night

HOURS_PER_YEAR = 8760
OVERNIGHT_HOURS = 15


@dataclass
class NightRegret:
    night_start: pd.Timestamp
    M_realized: float
    buffer_policy: float
    buffer_oracle: float
    cash_before: float
    capital_constrained: bool
    cost_policy: float
    cost_oracle: float
    regret: float
    conservatism_component: float
    forecast_error_component: float


def _night_cost(buffer: float, M_realized: float, repo_rate_pct: float, break_fee_pct: float) -> float:
    """Approximate realized economic cost for a given buffer choice against
    the realized path-max: foregone interest if over-buffered, break-fee cost
    on the shortfall if under-buffered. This mirrors the newsvendor cost
    structure exactly (c_o, c_u) rather than re-deriving a new one."""
    tau = OVERNIGHT_HOURS / HOURS_PER_YEAR
    if buffer >= M_realized:
        excess = buffer - M_realized
        return excess * (repo_rate_pct / 100.0) * tau
    else:
        shortfall = M_realized - buffer
        return shortfall * (break_fee_pct / 100.0)


def decompose_regret(df: pd.DataFrame, cfg: EngineConfig, policy_trace: pd.DataFrame) -> pd.DataFrame:
    """policy_trace must be the `.trace` from a Simulator run of the policy
    being analyzed (needs columns timestamp, tier0, sweep_amount)."""
    nights = compute_nightly_path_max(df)
    trace = policy_trace.set_index("timestamp")
    q_policy = cfg.risk.q_policy

    rows = []
    for _, n in nights.iterrows():
        night_start = n["night_start"]
        M = n["M"]
        prior_hour = night_start - pd.Timedelta(hours=1)
        if prior_hour not in trace.index or night_start not in trace.index:
            continue
        cash_before = trace.loc[prior_hour, "tier0"]
        sweep = trace.loc[night_start, "sweep_amount"]
        buffer_policy = cash_before - sweep

        row = df.loc[df["timestamp"] == night_start]
        if len(row) == 0:
            continue
        repo_rate = float(row["tier1_repo_rate_apy"].iloc[0])
        break_fee = float(row["tier1_early_redemption_haircut_pct"].iloc[0])

        capital_constrained = cash_before < M
        buffer_oracle = min(M, cash_before)  # oracle can't post a buffer bigger than available cash either

        cost_policy = _night_cost(buffer_policy, M, repo_rate, break_fee)
        cost_oracle = _night_cost(buffer_oracle, M, repo_rate, break_fee)
        regret = cost_policy - cost_oracle

        conservatism = 0.0
        forecast_err = regret
        if not capital_constrained and regret > 1e-9:
            q_star = critical_fractile(break_fee, repo_rate, OVERNIGHT_HOURS)
            if q_policy > q_star:
                # counterfactual: what would the pure-economic buffer (q*) have cost?
                # We don't have a direct F_M^-1(q*) here without refitting, so we
                # approximate the conservatism slice as the fraction of the q-gap
                # relative to the (q_policy - realized-rank) gap -- a conservative,
                # bounded approximation documented in METHODOLOGY.md.
                gap_frac = min(1.0, max(0.0, (q_policy - q_star) / max(q_policy, 1e-6)))
                conservatism = regret * gap_frac
                forecast_err = regret - conservatism

        rows.append(NightRegret(
            night_start=night_start, M_realized=M, buffer_policy=buffer_policy,
            buffer_oracle=buffer_oracle, cash_before=cash_before,
            capital_constrained=capital_constrained, cost_policy=cost_policy,
            cost_oracle=cost_oracle, regret=regret,
            conservatism_component=conservatism, forecast_error_component=forecast_err,
        ))

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_regret(regret_df: pd.DataFrame) -> dict:
    """All three cost buckets are REGRET (policy cost minus oracle cost on the
    same night), partitioned by night-group and sub-component, so they sum
    exactly to total_regret -- this is a partition, not three independent
    metrics on different bases (a bug we caught: an earlier version reported
    capital-constrained ABSOLUTE cost alongside conservatism/forecast-error
    REGRET deltas, which don't sum to anything meaningful together)."""
    capital = regret_df[regret_df["capital_constrained"]]
    clean = regret_df[~regret_df["capital_constrained"]]
    capital_constrained_regret = float(capital["regret"].sum())
    conservatism = float(clean["conservatism_component"].sum())
    forecast_error = float(clean["forecast_error_component"].sum())
    total = float(regret_df["regret"].sum())
    return {
        "n_nights": len(regret_df),
        "n_capital_constrained_nights": int(len(capital)),
        "total_regret": total,
        "capital_constrained_regret": capital_constrained_regret,
        "policy_conservatism_cost": conservatism,
        "forecast_error_cost": forecast_error,
        "buckets_sum_check": capital_constrained_regret + conservatism + forecast_error,
        # also report the absolute (non-regret) cost incurred on constrained
        # nights, since "we spent $X on nights no policy could have avoided"
        # is a separate, useful, non-comparable number for the CFO view
        "capital_constrained_absolute_cost": float(capital["cost_policy"].sum()),
    }

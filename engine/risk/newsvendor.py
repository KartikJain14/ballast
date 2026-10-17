"""
Closed-form newsvendor sizing for the overnight buffer (brief Section 2.3).

B* = F_M^-1(q*),  q* = c_u / (c_u + c_o)

  c_u = underage cost per dollar of shortfall = cost of the cheapest
        policy-sanctioned fallback (Tier1 early-break haircut; overdraft is
        excluded from this calc even though it's ex-post cheaper, because it
        is barred from the waterfall ahead of Tier1 by policy -- D6c).
  c_o = overage cost per dollar of buffer held idle overnight = foregone
        repo interest for the 15h window = r_repo * tau, tau = 15/8760.

The policy overlay B = F_M^-1(max(q*, q_policy)) lets the treasurer's risk
slider (q_policy) dominate the economically "optimal" quantile when it's
stricter -- a deliberate governance floor, not a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760
OVERNIGHT_HOURS = 15


def critical_fractile(break_fee_pct: float, repo_rate_apy_pct: float,
                        overnight_hours: int = OVERNIGHT_HOURS) -> float:
    c_u = break_fee_pct / 100.0
    tau = overnight_hours / HOURS_PER_YEAR
    c_o = (repo_rate_apy_pct / 100.0) * tau
    return c_u / (c_u + c_o)


def effective_quantile(break_fee_pct: float, repo_rate_apy_pct: float, q_policy: float,
                         overnight_hours: int = OVERNIGHT_HOURS) -> float:
    q_star = critical_fractile(break_fee_pct, repo_rate_apy_pct, overnight_hours)
    return max(q_star, q_policy)


def empirical_quantile_buffer(M_samples: np.ndarray, q: float, min_samples: int = 8,
                                fallback: float | None = None) -> float:
    """F_M^-1(q) from empirical samples. Falls back to `fallback` (or 0.0) when
    there isn't enough history yet -- this only matters in the first few weeks
    of a cold-started book, see engine/policy/ballast_policy.py."""
    clean = M_samples[~np.isnan(M_samples)]
    if len(clean) < min_samples:
        return fallback if fallback is not None else 0.0
    return float(np.quantile(clean, q, method="higher"))

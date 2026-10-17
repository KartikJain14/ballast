"""
Tier economics: accrual, mark-to-market, and liquidation cost functions.

Each function takes the *realized* market-data row (rates, haircuts, prices)
for the hour being settled, plus the dollar amount involved, and returns a
dollar outcome. These are pure functions so they're trivially unit-testable
and reusable by both the simulator (ex-post settlement) and the optimizer
(ex-ante scenario costing).
"""
from __future__ import annotations

import pandas as pd

from engine.config import AccrualConvention, HaircutBase

HOURS_PER_YEAR = 8760


def repo_overnight_interest(amount: float, repo_rate_apy_pct: float, hours_locked: int,
                             convention: AccrualConvention) -> float:
    """Interest earned on `amount` swept into Tier 1 repo for `hours_locked` hours."""
    r = repo_rate_apy_pct / 100.0
    if convention == AccrualConvention.PRO_RATA_15H:
        return amount * r * (hours_locked / HOURS_PER_YEAR)
    elif convention == AccrualConvention.ACT_360_ONE_DAY:
        days = max(1, round(hours_locked / 24))
        return amount * r * (days / 360.0)
    raise ValueError(convention)


def repo_early_break_cost(amount: float, break_haircut_pct: float,
                            haircut_base: HaircutBase, whole_position: float | None = None) -> float:
    """Cost of breaking `amount` out of a repo position before maturity."""
    base = amount if haircut_base == HaircutBase.AMOUNT_WITHDRAWN else (whole_position or amount)
    return base * (break_haircut_pct / 100.0)


def bond_price_return(mod_duration: float, delta_yield_pct: float) -> float:
    """Fractional price return from a modified-duration approximation: dP/P ~= -D*dy."""
    dy = delta_yield_pct / 100.0
    return -mod_duration * dy


def bond_carry(amount: float, bond_yield_pct: float, hours_held: int) -> float:
    r = bond_yield_pct / 100.0
    return amount * r * (hours_held / HOURS_PER_YEAR)


def bond_liquidation_cost(amount: float, haircut_pct: float, haircut_base: HaircutBase,
                            whole_position: float | None = None) -> float:
    base = amount if haircut_base == HaircutBase.AMOUNT_WITHDRAWN else (whole_position or amount)
    return base * (haircut_pct / 100.0)


def equity_liquidation_cost(amount: float, haircut_pct: float, haircut_base: HaircutBase,
                              whole_position: float | None = None) -> float:
    base = amount if haircut_base == HaircutBase.AMOUNT_WITHDRAWN else (whole_position or amount)
    return base * (haircut_pct / 100.0)


def overdraft_cost(amount: float, penalty_apr_pct: float, hours: int) -> float:
    """Pro-rata overdraft interest for drawing `amount` for `hours` hours."""
    r = penalty_apr_pct / 100.0
    return amount * r * (hours / HOURS_PER_YEAR)


def break_vs_overdraft_effective_apr(break_fee_pct: float, hours_remaining: int) -> float:
    """The APR-equivalent of paying a flat break fee to cover a shortfall for
    `hours_remaining` hours -- this is the number that makes the 'overdraft trap'
    (brief Section 2, point 6) concrete: a 0.5% flat fee over a short window is an
    enormous annualized rate."""
    if hours_remaining <= 0:
        return float("inf")
    return break_fee_pct * (HOURS_PER_YEAR / hours_remaining)

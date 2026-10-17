"""
Policy interface. A policy only ever sees `HistoryView.as_of(t_idx - 1)` for
decisions it makes "for" hour t_idx (the no-look-ahead invariant lives in
engine.data.HistoryView; the simulator enforces it structurally by never
passing the policy anything else).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from engine.config import EngineConfig


@dataclass
class SimStateView:
    """Read-only snapshot of simulator state passed to a policy at decision time."""
    t_idx: int
    cash_available: float          # TIER0 balance right now (before this hour's own flow)
    tier_balances: dict[str, float]
    repo_locked_amount: float      # Tier1 principal currently locked (0 if none)
    overdraft_drawn: float


class Policy(ABC):
    name: str = "base"

    @abstractmethod
    def decide_sweep(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                      cfg: EngineConfig) -> float:
        """Return the dollar amount to sweep from TIER0 into TIER1 at this 17:00
        decision hour. `history` is already sliced to as_of(t_idx - 1) by the
        simulator -- it physically cannot contain hour t_idx's own row."""
        raise NotImplementedError

    def decide_bond_target_usd(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                                 cfg: EngineConfig) -> float:
        """Dollar target for the Tier2 bond sleeve balance after today's 17:00
        rebalance. Default: 0 (baselines don't use bonds). The simulator only
        BUYS up toward this target (banded, to avoid round-tripping the 2%
        liquidation haircut on voluntary rebalancing) -- it never voluntarily
        sells down to it; selling only ever happens via the funding waterfall
        when cash is actually needed."""
        return 0.0

    def decide_equity_target_frac(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                                    cfg: EngineConfig) -> float:
        """Fraction of strategic surplus allocated to Tier3 equities. Default: 0."""
        return 0.0

"""
OraclePolicy: perfect-foresight upper bound for the scoreboard (brief Section
4.3's "oracle"). This is the ONE policy in the codebase that is deliberately
allowed to see the future -- it exists purely to quantify the best any
policy could possibly do on realized flows, so Ballast's regret vs. the
oracle is measurable. It is never deployable and is clearly labeled as such
everywhere it appears (dashboard, docs, scoreboard).

It sweeps exactly `cash_available - M_tonight_exact` every night, where
M_tonight_exact is the REALIZED path-max for that specific night (computed
from the full dataset, not `history`). This achieves zero breaches and zero
break fees by construction, maximizing interest income -- the textbook
perfect-foresight LP brief Section 4.3 asks for, in closed form (since with
no forecast uncertainty, the newsvendor collapses to "buffer = exact need").
"""
from __future__ import annotations

import pandas as pd

from engine.config import EngineConfig
from engine.policy.base import Policy, SimStateView
from engine.risk.pathmax import compute_nightly_path_max


class OraclePolicy(Policy):
    name = "oracle"

    def __init__(self, df_full: pd.DataFrame):
        nights = compute_nightly_path_max(df_full)
        self._M_by_night_start = dict(zip(nights["night_start"], nights["M"]))

    def decide_sweep(self, t_idx: int, history: pd.DataFrame, state: SimStateView,
                      cfg: EngineConfig) -> float:
        # The current decision hour's timestamp = one hour after the last row
        # in `history` (history is as_of(t_idx - 1)); for an oracle this is
        # the ONE place in the whole codebase where "cheating" is intentional
        # and documented, not a bug.
        if len(history) == 0:
            return 0.0
        tonight = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        M_tonight = self._M_by_night_start.get(tonight, 0.0)
        return max(0.0, state.cash_available - M_tonight)

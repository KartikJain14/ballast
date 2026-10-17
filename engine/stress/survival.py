"""
Survival horizon + reverse stress search (brief Section 4.6).

survival_horizon_hours: given a scenario-shocked dataframe, a policy, and a
finite overdraft line cap, runs the simulator with `stop_on_default=True` and
reports the hour the waterfall first fails to fund a withdrawal (None if it
survives the whole scenario window).

reverse_stress_multiplier: binary-searches the smallest withdrawal multiplier
(applied from the scenario start onward) that causes a default within the
scenario window -- "the smallest outflow multiplier that exhausts the
waterfall," per the brief.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.config import EngineConfig
from engine.policy.base import Policy
from engine.simulator.engine import Simulator


@dataclass
class SurvivalResult:
    survived: bool
    survival_hours: int | None
    survival_timestamp: pd.Timestamp | None
    final_summary: dict


def survival_horizon(df: pd.DataFrame, cfg: EngineConfig, policy: Policy,
                       overdraft_line_cap_usd: float) -> SurvivalResult:
    cfg2 = EngineConfig(mode=cfg.mode, tiers=cfg.tiers, risk=cfg.risk, assumptions=cfg.assumptions, seed=cfg.seed)
    cfg2.risk.overdraft_line_cap_usd = overdraft_line_cap_usd
    sim = Simulator(df, cfg2)
    res = sim.run(policy, stop_on_default=True)
    sh = res.summary.get("survival_hours")
    ts = df["timestamp"].iloc[sh] if sh is not None else None
    return SurvivalResult(survived=(sh is None), survival_hours=sh, survival_timestamp=ts,
                            final_summary=res.summary)


def reverse_stress_multiplier(df: pd.DataFrame, cfg: EngineConfig, policy: Policy,
                                 shock_fn, start: pd.Timestamp, overdraft_line_cap_usd: float,
                                 lo: float = 1.0, hi: float = 20.0, tol: float = 0.05,
                                 max_iter: int = 12) -> dict:
    """Binary search over an outflow multiplier `m` for `shock_fn(df, start, m)
    -> shocked_df`. Returns the smallest m (within tol) that causes a default,
    or reports that even `hi` survives."""
    def defaults_at(m: float) -> bool:
        shocked = shock_fn(df, start, m)
        result = survival_horizon(shocked, cfg, policy, overdraft_line_cap_usd)
        return not result.survived

    if not defaults_at(hi):
        return {"found": False, "message": f"Survives even at {hi}x -- waterfall is robust to this shock shape."}
    if defaults_at(lo):
        return {"found": True, "multiplier": lo, "message": f"Already defaults at the baseline multiplier {lo}x."}

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if defaults_at(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return {"found": True, "multiplier": round(hi, 3)}

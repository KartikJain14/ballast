"""
make precompute -> python scripts/precompute.py

Bakes the full-year replay, regret decomposition, and a set of canned stress
scenarios into reports/dashboard_cache.json so the Streamlit app loads
instantly (brief Section 5: "precompute and cache the full-year replay so the
demo is instant and never fails live; keep live recompute for the slider and
Stress Lab"). The slider and ad-hoc Stress Lab runs still recompute live
(both take <2s thanks to the vectorized pathmax -- see docs/DECISIONS.md D2/D10).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import EngineConfig  # noqa: E402
from engine.data import load_raw, DEFAULT_CSV_PATH  # noqa: E402
from engine.simulator.engine import Simulator  # noqa: E402
from engine.policy.baselines import AllCashPolicy, StaticBufferPolicy, FixedPctOfGrossPolicy, MovingAverageKSigmaPolicy, MillerOrrPolicy  # noqa: E402
from engine.policy.ballast_policy import BallastPolicy  # noqa: E402
from engine.policy.oracle import OraclePolicy  # noqa: E402
from engine.risk.regret import decompose_regret, summarize_regret
from engine.risk.pathmax import compute_nightly_path_max
from engine.stress.scenarios import SCENARIOS, combined_crisis
from engine.stress.survival import survival_horizon, reverse_stress_multiplier


def main():
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    sim = Simulator(df, cfg)

    cache: dict = {}

    # --- scoreboard ---
    policies = {
        "all_cash": AllCashPolicy(),
        "static_buffer_38M": StaticBufferPolicy(38_000_000),
        "fixed_pct_gross_50%": FixedPctOfGrossPolicy(0.5),
        "moving_avg_2.0sigma": MovingAverageKSigmaPolicy(k=2.0),
        "miller_orr": MillerOrrPolicy(),
        "ballast": BallastPolicy(),
        "oracle": OraclePolicy(df),
    }
    results = {}
    ballast_trace = None
    for name, pol in policies.items():
        res = sim.run(pol)
        results[name] = res.summary
        if name == "ballast":
            ballast_trace = res.trace
    cache["scoreboard"] = results

    # --- full hourly trace for the fan chart (Ballast only -- others on demand) ---
    cache["ballast_trace"] = ballast_trace.assign(
        timestamp=ballast_trace["timestamp"].astype(str)
    ).to_dict(orient="records")

    # --- nightly path-max series (for the fan chart background distribution) ---
    nights = compute_nightly_path_max(df)
    cache["nightly_M"] = nights.assign(
        night_start=nights["night_start"].astype(str)
    ).to_dict(orient="records")

    # --- regret decomposition ---
    rdf = decompose_regret(df, cfg, ballast_trace)
    cache["regret_summary"] = summarize_regret(rdf)

    # --- cost-of-confidence curve: foregone interest vs quantile, 90%-99.99% ---
    import numpy as np
    from engine.risk.newsvendor import empirical_quantile_buffer
    qs = np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.9999, 19)])
    repo_rate = df["tier1_repo_rate_apy"].mean()
    tau = 15 / 8760
    curve = []
    weekday_M = nights[~nights["dow"].isin([4, 5])]["M"].to_numpy()
    weekend_M = nights[nights["dow"].isin([4, 5])]["M"].to_numpy()
    for q in qs:
        b_wd = empirical_quantile_buffer(weekday_M, q, min_samples=1)
        b_we = empirical_quantile_buffer(weekend_M, q, min_samples=1)
        avg_buffer = (b_wd * 5 + b_we * 2) / 7
        annual_cost = avg_buffer * (repo_rate / 100.0) * (15 * 365 / 8760)
        curve.append({"q": float(q), "avg_buffer": float(avg_buffer), "annual_foregone_cost": float(annual_cost)})
    cache["cost_of_confidence_curve"] = curve

    # --- canned stress scenarios ---
    CAP = 1_000_000_000.0  # survives the unshocked year -- see docs/DECISIONS.md
    stress_results = {}
    start = pd.Timestamp("2026-06-15 09:00:00")
    for name, fn in SCENARIOS.items():
        shocked = fn(df, start, cfg)
        res = survival_horizon(shocked, cfg, BallastPolicy(), overdraft_line_cap_usd=CAP)
        stress_results[name] = {
            "survived": res.survived,
            "survival_hours": res.survival_hours,
            "survival_timestamp": str(res.survival_timestamp) if res.survival_timestamp is not None else None,
        }
    combined = combined_crisis(df, start, cfg)
    res = survival_horizon(combined, cfg, BallastPolicy(), overdraft_line_cap_usd=CAP)
    stress_results["combined_crisis"] = {
        "survived": res.survived, "survival_hours": res.survival_hours,
        "survival_timestamp": str(res.survival_timestamp) if res.survival_timestamp is not None else None,
    }
    cache["stress_scenarios"] = stress_results
    cache["stress_overdraft_cap"] = CAP
    cache["stress_start"] = str(start)

    out_path = ROOT / "reports" / "dashboard_cache.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""
The analysis pipeline run per uploaded (or default) dataset. Runs in a
background thread; writes staged progress into a shared job dict so the
frontend can poll real progress rather than a fabricated animation timed to
guess at duration. Every number produced here is computed from the actual
CSV, including a user-uploaded one -- nothing is hardcoded to the sample
dataset's year or dates (so it generalizes to new data, not just 2026).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import load_raw, validate, DEFAULT_CSV_PATH  # noqa: E402
from engine.config import EngineConfig  # noqa: E402
from engine.simulator.engine import Simulator  # noqa: E402
from engine.policy.baselines import AllCashPolicy, StaticBufferPolicy, MovingAverageKSigmaPolicy  # noqa: E402
from engine.policy.ballast_policy import BallastPolicy  # noqa: E402
from engine.policy.oracle import OraclePolicy  # noqa: E402
from engine.risk.pathmax import compute_nightly_path_max, is_weekend_night  # noqa: E402
from engine.risk.regret import decompose_regret, summarize_regret  # noqa: E402
from engine.risk.newsvendor import empirical_quantile_buffer, critical_fractile  # noqa: E402
from engine.stress.scenarios import SCENARIOS  # noqa: E402
from engine.stress.survival import survival_horizon  # noqa: E402

STAGES = [
    ("validate", "Validating uploaded data"),
    ("flows", "Analyzing deposit and withdrawal patterns"),
    ("risk", "Computing overnight risk distribution"),
    ("sim", "Running policy simulations"),
    ("stress", "Testing stress scenarios"),
    ("done", "Finalizing report"),
]


def _set(job: dict, stage: str, percent: float, **extra):
    job["stage"] = stage
    job["stage_label"] = dict(STAGES).get(stage, stage)
    job["percent"] = percent
    job.update(extra)


def classify_spike(ts: pd.Timestamp, net_flow: float) -> str:
    """Generic, data-driven classification -- computed from calendar
    position alone, not a hardcoded date list, so it generalizes to any
    uploaded year's data, not just the sample dataset."""
    day = ts.day
    dow = ts.dayofweek
    is_month_end = (ts + pd.Timedelta(days=3)).month != ts.month
    if net_flow > 0 and day in (1, 2, 15, 16):
        return "Likely payday inflow"
    if net_flow < 0 and dow in (4, 5):
        return "Weekend spending pattern"
    if net_flow < 0 and is_month_end:
        return "Month-end outflow"
    if net_flow > 0 and is_month_end:
        return "Month-end inflow"
    return "Unclassified anomaly"


def run_pipeline(csv_path: Path, job: dict):
    try:
        _set(job, "validate", 2)
        df = load_raw(csv_path)
        report = validate(df)
        job["validation"] = {
            "n_rows": report.n_rows, "is_continuous_hourly": report.is_continuous_hourly,
            "warnings": report.warnings, "errors": report.errors,
        }
        if not report.ok:
            job["status"] = "error"
            job["error"] = f"Validation failed: {report.errors}"
            return

        _set(job, "flows", 10)
        df["hour"] = df["timestamp"].dt.hour
        df["dow"] = df["timestamp"].dt.dayofweek
        mean_dep = float(df["deposits_usd"].mean())
        mean_wd = float(df["withdrawals_usd"].mean())
        cum_net = df["net_flow_usd"].cumsum()
        # daily series for the headline flow chart (keeps payload small on long datasets)
        daily = df.set_index("timestamp")[["deposits_usd", "withdrawals_usd", "net_flow_usd"]].resample("D").sum()
        daily_cum_net = daily["net_flow_usd"].cumsum()

        thresh = df["net_flow_usd"].abs().quantile(0.99)
        spikes_df = df.loc[df["net_flow_usd"].abs() >= thresh, ["timestamp", "net_flow_usd"]].copy()
        spikes_df = spikes_df.reindex(spikes_df["net_flow_usd"].abs().sort_values(ascending=False).index).head(14)
        spikes = [
            {"ts": str(r.timestamp), "net_flow": float(r.net_flow_usd),
             "label": classify_spike(r.timestamp, r.net_flow_usd)}
            for r in spikes_df.itertuples()
        ]
        cause_counts = pd.Series([s["label"] for s in spikes]).value_counts().to_dict()

        job["flows"] = {
            "mean_deposits_per_hour": mean_dep, "mean_withdrawals_per_hour": mean_wd,
            "pct_hours_net_outflow": float((df["net_flow_usd"] < 0).mean() * 100),
            "cumulative_net_flow_final": float(cum_net.iloc[-1]),
            "cumulative_net_flow_min": float(cum_net.min()),
            "daily_timestamps": [str(x) for x in daily.index],
            "daily_deposits": daily["deposits_usd"].round(0).tolist(),
            "daily_withdrawals": daily["withdrawals_usd"].round(0).tolist(),
            "daily_cum_net": daily_cum_net.round(0).tolist(),
            "spikes": spikes,
            "spike_cause_counts": cause_counts,
        }

        _set(job, "risk", 28)
        nights = compute_nightly_path_max(df)
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_stats = nights.groupby("dow")["M"].agg(["mean", "median", "count"])
        by_dow = {dow_names[i]: {"mean": float(r["mean"]), "median": float(r["median"]), "count": int(r["count"])}
                   for i, r in dow_stats.iterrows()}
        job["risk"] = {
            "n_nights": len(nights),
            "M_mean": float(nights["M"].mean()), "M_p50": float(nights["M"].median()),
            "M_p90": float(nights["M"].quantile(0.90)), "M_p99": float(nights["M"].quantile(0.99)),
            "M_max": float(nights["M"].max()),
            "by_dow": by_dow,
            "histogram": np.histogram(nights["M"], bins=30)[0].tolist(),
            "histogram_edges": np.histogram(nights["M"], bins=30)[1].round(0).tolist(),
        }

        _set(job, "sim", 40)
        cfg = EngineConfig.full_4tier()
        try:
            is_bundled_sample = Path(csv_path).resolve() == DEFAULT_CSV_PATH.resolve()
        except OSError:
            is_bundled_sample = False
        if is_bundled_sample:
            # Use the SAME $2B figure documented in docs/DECISIONS.md D6d for the
            # bundled sample -- it was arrived at deliberately (large enough that
            # policy quality, not capital adequacy, drives the scoreboard, while
            # still leaving the book under real pressure by year-end) and the
            # live dashboard's numbers must match docs/RESULTS.md for the SAME
            # dataset, or they'd look inconsistent/wrong when compared side by side.
            capital = cfg.assumptions.opening_cash_usd  # EngineConfig default: $2,000,000,000
        else:
            # For an arbitrary uploaded dataset we have no such pre-derived figure,
            # so scale capital to the dataset's OWN empirical worst-case overnight
            # risk (M): a 40x margin over the single worst observed night keeps the
            # book solvent without over-capitalizing to the point where no policy
            # difference is visible (same reasoning as D6d, generalized).
            capital = float(nights["M"].max()) * 40
        cfg.assumptions.opening_cash_usd = capital
        cfg.assumptions.deposit_base_usd = capital
        sim = Simulator(df, cfg)

        # quick adaptive grid for the static baseline, centered on the data's own risk scale
        m_p95 = float(nights["M"].quantile(0.95))
        grid = sorted(set([m_p95 * f for f in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0)]))
        best_buf, best_cost = grid[0], float("inf")
        for g in grid:
            r = sim.run(StaticBufferPolicy(g))
            if r.summary["total_cost_of_liquidity"] < best_cost:
                best_buf, best_cost = g, r.summary["total_cost_of_liquidity"]

        policies = {
            "all_cash": AllCashPolicy(),
            f"static_${best_buf/1e6:.0f}M": StaticBufferPolicy(best_buf),
            "moving_avg_2sigma": MovingAverageKSigmaPolicy(k=2.0),
            "ballast": BallastPolicy(),
            "oracle": OraclePolicy(df),
        }
        scoreboard = {}
        ballast_trace = None
        for name, pol in policies.items():
            r = sim.run(pol)
            scoreboard[name] = r.summary
            if name == "ballast":
                ballast_trace = r.trace
        job["scoreboard"] = scoreboard
        job["best_static_buffer"] = best_buf

        regret_df = decompose_regret(df, cfg, ballast_trace)
        job["regret"] = summarize_regret(regret_df)

        _set(job, "stress", 75)
        CAP = cfg.assumptions.opening_cash_usd * 0.5
        start = df["timestamp"].iloc[int(len(df) * 0.45)].replace(hour=9, minute=0)
        stress_results = {}
        key_scenarios = ["svb_run", "holiday_x3", "payday_x2", "bond_shock_200bp"]
        for name in key_scenarios:
            fn = SCENARIOS[name]
            shocked = fn(df, start, cfg)
            res = survival_horizon(shocked, cfg, BallastPolicy(), overdraft_line_cap_usd=CAP)
            stress_results[name] = {
                "survived": res.survived, "survival_timestamp": str(res.survival_timestamp) if res.survival_timestamp else None,
            }
        job["stress"] = stress_results
        job["stress_cap"] = CAP

        _set(job, "done", 100)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{e}\n{traceback.format_exc()}"

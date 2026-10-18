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

# Bump whenever the shape of the pipeline output changes, so results cached
# on disk by an older build are recomputed instead of served without the new
# fields (api/main.py keys its precompute cache on this).
PIPELINE_VERSION = 2

STAGES = [
    ("validate", "Validating uploaded data"),
    ("flows", "Analyzing deposit and withdrawal patterns"),
    ("risk", "Computing overnight risk distribution"),
    ("sim", "Running policy simulations"),
    ("stress", "Testing stress scenarios"),
    ("audit", "Running integrity checks"),
    ("done", "Finalizing report"),
]

OVERNIGHT_HOURS = 15


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
            "n_expected": report.n_expected, "n_duplicate_timestamps": report.n_duplicate_timestamps,
            "n_missing_hours": report.n_missing_hours,
            "net_flow_ok": bool(report.net_flow_matches_deposits_minus_withdrawals),
            "max_net_flow_mismatch": report.max_net_flow_mismatch,
            "n_missing_values": int(df.isna().sum().sum()),
            "date_start": str(df["timestamp"].iloc[0]), "date_end": str(df["timestamp"].iloc[-1]),
            "n_sweep_windows": int(df["repo_sweep_window_open"].astype(bool).sum()),
            "n_maturities": int(df["repo_sweep_mature"].astype(bool).sum()),
            "sweep_hours_ok": bool((df.loc[df["repo_sweep_window_open"].astype(bool), "timestamp"].dt.hour == 17).all()),
            "maturity_hours_ok": bool((df.loc[df["repo_sweep_mature"].astype(bool), "timestamp"].dt.hour == 8).all()),
            "rates": {c: v["mean"] for c, v in report.rate_unit_findings.items()},
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
        # Cost-of-confidence curve: for the risk-appetite slider, computed ONCE
        # here so the frontend can interpolate it client-side instantly on every
        # drag, with zero network round-trips per slider movement.
        repo_rate_mean = float(df["tier1_repo_rate_apy"].mean())
        break_fee_mean = float(df["tier1_early_redemption_haircut_pct"].mean())
        q_star = critical_fractile(break_fee_mean, repo_rate_mean)
        weekday_M = nights[~nights["dow"].isin([4, 5])]["M"].to_numpy()
        weekend_M = nights[nights["dow"].isin([4, 5])]["M"].to_numpy()
        qs = np.concatenate([np.linspace(0.90, 0.99, 19), np.linspace(0.991, 0.9999, 19)])
        tau = 15 / 8760
        curve = []
        for q in qs:
            b_wd = empirical_quantile_buffer(weekday_M, float(q), min_samples=1)
            b_we = empirical_quantile_buffer(weekend_M, float(q), min_samples=1)
            avg_buffer = (b_wd * 5 + b_we * 2) / 7
            annual_cost = avg_buffer * (repo_rate_mean / 100.0) * (15 * 365 / 8760)
            curve.append({"q": float(q), "buffer_weekday": float(b_wd), "buffer_weekend": float(b_we),
                          "annual_foregone_cost": float(annual_cost)})

        job["risk"] = {
            "n_nights": len(nights),
            "M_mean": float(nights["M"].mean()), "M_p50": float(nights["M"].median()),
            "M_p90": float(nights["M"].quantile(0.90)), "M_p99": float(nights["M"].quantile(0.99)),
            "M_max": float(nights["M"].max()),
            "by_dow": by_dow,
            "histogram": np.histogram(nights["M"], bins=30)[0].tolist(),
            "histogram_edges": np.histogram(nights["M"], bins=30)[1].round(0).tolist(),
            "q_star": q_star,
            "q_policy": EngineConfig.full_4tier().risk.q_policy,
            "cost_curve": curve,
            # Raw samples behind the reserve calculator, which re-derives
            # buffers client-side for any rate / fee / coverage the user types.
            "M_weekday": sorted(round(float(x)) for x in weekday_M),
            "M_weekend": sorted(round(float(x)) for x in weekend_M),
            "repo_rate_mean": repo_rate_mean, "break_fee_mean": break_fee_mean,
            "overdraft_apr_mean": float(df["emergency_overdraft_penalty_apr_pct"].mean()),
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
        traces = {}
        for name, pol in policies.items():
            r = sim.run(pol)
            scoreboard[name] = r.summary
            traces[name] = r.trace
        ballast_trace = traces["ballast"]
        job["net_income_daily"] = _net_income_daily(traces)
        job["scoreboard"] = scoreboard
        job["best_static_buffer"] = best_buf
        job["capital"] = capital

        regret_df = decompose_regret(df, cfg, ballast_trace)
        job["regret"] = summarize_regret(regret_df)
        decisions = _decision_log(df, ballast_trace, regret_df, policies["ballast"].decision_log)
        job["decisions"] = decisions
        job["calibration"] = _calibration(decisions, cfg.risk.q_policy)
        static_name = next(n for n in policies if n.startswith("static_"))
        job["monthly"] = _monthly_pnl(traces["ballast"], traces[static_name], decisions)

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
        job["stress_start"] = str(start)

        _set(job, "audit", 90)
        job["integrity"] = _integrity_checks(df, cfg, ballast_trace, scoreboard["ballast"], report)

        _set(job, "done", 100)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{e}\n{traceback.format_exc()}"


def _net_income_daily(traces: dict) -> dict:
    """Cumulative net income (interest earned minus every fee, haircut and
    penalty) per policy, sampled at each day's last hour -- daily resolution
    keeps the polled payload small."""
    out, dates = {}, None
    for name, tr in traces.items():
        daily = tr.set_index("timestamp")[["cumulative_accrual_income", "cumulative_fees_paid"]].resample("D").last()
        if dates is None:
            dates = [str(d.date()) for d in daily.index]
        out[name] = (daily["cumulative_accrual_income"] - daily["cumulative_fees_paid"]).round(0).tolist()
    return {"dates": dates or [], "series": out}


def _decision_log(df: pd.DataFrame, trace: pd.DataFrame, regret_df: pd.DataFrame, log: list[dict]) -> list[dict]:
    """One row per night: what Ballast knew at 17:00, what it did, and what
    the night then did to it. Outcome classes keep the book's structural
    cash drain (capital-constrained nights, where even a perfect forecaster
    could not have covered the drain) separate from genuine policy misses."""
    if len(regret_df) == 0:
        return []
    by_t = {e["t_idx"]: e for e in log}
    ts_to_idx = {ts: i for i, ts in enumerate(trace["timestamp"])}
    net = df["net_flow_usd"].to_numpy()
    cum_fees = trace["cumulative_fees_paid"].to_numpy()
    wf_t1 = trace["waterfall_tier1"].to_numpy()
    wf_od = trace["waterfall_overdraft"].to_numpy()
    od = trace["overdraft_drawn"].to_numpy()
    sweep = trace["sweep_amount"].to_numpy()
    rows = []
    for n in regret_df.itertuples():
        t = ts_to_idx.get(n.night_start)
        if t is None or t == 0:
            continue
        end = min(t + OVERNIGHT_HOURS, len(trace))
        e = by_t.get(t, {})
        swept = float(sweep[t])
        buffer = float(n.buffer_policy)
        M = float(n.M_realized)
        if n.capital_constrained:
            outcome = "constrained"
        elif buffer >= M:
            outcome = "covered"
        else:
            outcome = "short"
        rate = e.get("repo_rate", float(df["tier1_repo_rate_apy"].iloc[t]))
        path = -np.cumsum(net[t:end])
        rows.append({
            "date": str(n.night_start.date()), "dow": int(n.night_start.dayofweek),
            "weekend": bool(e.get("weekend", n.night_start.dayofweek in (4, 5))),
            "cash": round(float(n.cash_before)), "buffer": round(buffer), "swept": round(swept),
            "M": round(M), "outcome": outcome,
            "in_overdraft": bool(od[t - 1] > 1e-6),
            "q_star": _r5(e.get("q_star")), "q_eff": _r5(e.get("q_eff")),
            "n_samples": e.get("n_samples", 0), "basis": e.get("basis", "unknown"),
            "repo_rate": round(rate, 4),
            "break_fee": e.get("break_fee", float(df["tier1_early_redemption_haircut_pct"].iloc[t])),
            "interest": round(swept * rate / 100.0 * OVERNIGHT_HOURS / 8760),
            "repo_broken": round(float(wf_t1[t:end].sum())),
            "overdraft_used": round(float(wf_od[t:end].sum())),
            "fees": round(float(cum_fees[end - 1] - cum_fees[t - 1])),
            # Cumulative outflow hour by hour through the night, in $K to keep
            # the payload small (364 nights x 15 hours).
            "path_k": [round(float(x) / 1000) for x in path],
        })
    return rows


def _r5(x):
    return None if x is None else round(float(x), 5)


def _calibration(decisions: list[dict], q_policy: float) -> dict:
    """Out-of-sample check of the coverage promise: on nights where the book
    had the cash to post the buffer it chose, how often did that buffer
    actually cover the night's worst drain?"""
    out = {"target": q_policy}
    for key, sel in (("weekday", False), ("weekend", True)):
        eligible = [d for d in decisions if d["weekend"] == sel and d["outcome"] != "constrained"
                    and d["basis"] == "regime_quantile"]
        covered = sum(1 for d in eligible if d["outcome"] == "covered")
        out[key] = {"n": len(eligible), "covered": covered,
                    "rate": covered / len(eligible) if eligible else None}
    all_e = out["weekday"]["n"] + out["weekend"]["n"]
    all_c = out["weekday"]["covered"] + out["weekend"]["covered"]
    out["all"] = {"n": all_e, "covered": all_c, "rate": all_c / all_e if all_e else None}
    out["n_constrained"] = sum(1 for d in decisions if d["outcome"] == "constrained")
    out["n_warmup"] = sum(1 for d in decisions if d["basis"] != "regime_quantile")
    return out


def _monthly_pnl(ballast: pd.DataFrame, static: pd.DataFrame, decisions: list[dict]) -> list[dict]:
    def monthly(tr):
        m = tr.set_index("timestamp")[["cumulative_accrual_income", "cumulative_fees_paid"]].resample("MS").last()
        return m.diff().fillna(m.iloc[:1])
    b, st = monthly(ballast), monthly(static)
    dec = pd.DataFrame(decisions)
    if len(dec):
        dec["month"] = pd.to_datetime(dec["date"]).dt.strftime("%Y-%m")
    rows = []
    for ts in b.index:
        key = ts.strftime("%Y-%m")
        md = dec[dec["month"] == key] if len(dec) else dec
        b_net = float(b.loc[ts, "cumulative_accrual_income"] - b.loc[ts, "cumulative_fees_paid"])
        s_net = float(st.loc[ts, "cumulative_accrual_income"] - st.loc[ts, "cumulative_fees_paid"])
        rows.append({
            "month": key,
            "interest": round(float(b.loc[ts, "cumulative_accrual_income"])),
            "fees": round(float(b.loc[ts, "cumulative_fees_paid"])),
            "net": round(b_net), "static_net": round(s_net), "vs_static": round(b_net - s_net),
            "nights": int(len(md)),
            "nights_swept": int((md["swept"] > 0).sum()) if len(md) else 0,
            "nights_short": int((md["outcome"] == "short").sum()) if len(md) else 0,
            "avg_swept": round(float(md["swept"].mean())) if len(md) else 0,
        })
    return rows


def _integrity_checks(df: pd.DataFrame, cfg: EngineConfig, trace: pd.DataFrame, summary: dict, report) -> list[dict]:
    """Invariants re-proven on every run, on this data -- not unit tests that
    ran once on the sample file."""
    checks = []
    total = trace["tier0"] + trace["tier1"] + trace["tier2"] + trace["tier3"] - trace["overdraft_drawn"]
    expected = (cfg.assumptions.opening_cash_usd + df["net_flow_usd"].iloc[:len(trace)].cumsum().to_numpy()
                + trace["cumulative_accrual_income"] - trace["cumulative_fees_paid"])
    drift = float((total - expected).abs().max())
    checks.append({"name": "Every dollar accounted for",
                   "detail": f"Cash + all tiers − overdraft equals opening capital + net flows + income − fees "
                             f"at every one of {len(trace):,} hours (max drift ${drift:,.2f}).",
                   "passed": bool(drift < 1.0)})

    swept = trace[trace["sweep_amount"] > 1e-6]
    off_hour = int((swept["timestamp"].dt.hour != 17).sum())
    checks.append({"name": "Sweeps only at 17:00",
                   "detail": f"{len(swept):,} sweeps executed; {off_hour} outside the 17:00 window.",
                   "passed": off_hour == 0})

    d1 = trace["tier1"].diff()
    early = trace[(d1 < -1.0) & (trace["timestamp"].dt.hour != 8)]
    unlogged = int((early["waterfall_tier1"] <= 1e-6).sum())
    checks.append({"name": "Repo released only at 08:00",
                   "detail": f"{len(early):,} early repo draws, every one a logged emergency break "
                             f"({unlogged} unexplained).",
                   "passed": unlogged == 0})

    # No look-ahead: decisions made on a truncated file (the future physically
    # absent) must match the full-file run hour for hour.
    cut = min(len(df), 24 * 75 + 18)
    short = Simulator(df.iloc[:cut].reset_index(drop=True), cfg).run(BallastPolicy()).trace
    mismatch = float(np.abs(short["sweep_amount"].to_numpy() - trace["sweep_amount"].iloc[:cut].to_numpy()).max())
    n_dec = int((short["timestamp"].dt.hour == 17).sum())
    checks.append({"name": "No look-ahead",
                   "detail": f"Re-ran the first {n_dec} decisions with every later row deleted from the file: "
                             f"identical sweeps (max difference ${mismatch:,.2f}).",
                   "passed": bool(mismatch < 1.0)})

    checks.append({"name": "Every payout funded",
                   "detail": f"{summary['payout_success_rate'] * 100:.2f}% of withdrawals paid on time through the "
                             f"funding waterfall.",
                   "passed": bool(summary["payout_success_rate"] >= 0.999999)})

    checks.append({"name": "Input data is clean",
                   "detail": f"{report.n_rows:,} hourly rows, {report.n_missing_hours} missing, "
                             f"{report.n_duplicate_timestamps} duplicated; net flow = deposits − withdrawals on every row "
                             f"(to within ${report.max_net_flow_mismatch:.2f} rounding).",
                   "passed": bool(report.is_continuous_hourly and report.max_net_flow_mismatch <= 0.011)})
    return checks


def _deposit_freeze(df: pd.DataFrame, start: pd.Timestamp, duration_hours: int) -> pd.DataFrame:
    out = df.copy()
    mask = (out["timestamp"] >= start) & (out["timestamp"] < start + pd.Timedelta(hours=duration_hours))
    out.loc[mask, "deposits_usd"] = 0.0
    out.loc[mask, "net_flow_usd"] = out.loc[mask, "deposits_usd"] - out.loc[mask, "withdrawals_usd"]
    return out


STRESS_SCENARIOS = ("withdrawal_surge", "deposit_freeze", "bank_run", "payday_surge")


def run_custom_stress(csv_path: Path, capital: float, scenario: str, start: str,
                      multiplier: float, duration_hours: int) -> dict:
    """Stress Lab: one user-specified shock, replayed through the full
    waterfall against a finite emergency line (half the opening capital,
    the same line the preset scenarios use)."""
    from engine.stress.scenarios import payday_surge, holiday_withdrawal_surge, svb_calibrated_run

    df = load_raw(csv_path)
    cfg = EngineConfig.full_4tier()
    cfg.assumptions.opening_cash_usd = capital
    cfg.assumptions.deposit_base_usd = capital
    cap = capital * 0.5
    start_ts = pd.Timestamp(start).floor("h")
    start_ts = min(max(start_ts, df["timestamp"].iloc[24 * 8]), df["timestamp"].iloc[-48])
    duration_hours = int(max(1, min(duration_hours, 24 * 14)))

    if scenario == "withdrawal_surge":
        shocked = holiday_withdrawal_surge(df, start_ts, multiplier=multiplier, duration_hours=duration_hours)
        label = f"Withdrawals ×{multiplier:g} for {duration_hours}h"
    elif scenario == "deposit_freeze":
        shocked = _deposit_freeze(df, start_ts, duration_hours)
        label = f"Deposits stop for {duration_hours}h"
    elif scenario == "bank_run":
        shocked = svb_calibrated_run(df, start_ts, capital, day1_frac=multiplier / 100.0)
        label = f"Bank run: {multiplier:g}% of deposits pulled in 24h, doubled queue the next day"
    elif scenario == "payday_surge":
        shocked = payday_surge(df, start_ts, multiplier=multiplier, duration_hours=duration_hours)
        label = f"Deposits ×{multiplier:g} for {duration_hours}h"
    else:
        raise ValueError(f"Unknown scenario '{scenario}'")

    res = survival_horizon(shocked, cfg, BallastPolicy(), overdraft_line_cap_usd=cap)

    # Hour-by-hour window around the shock for the chart, shocked vs. unshocked.
    # Truncating the file at the window's end is safe: no decision before it
    # can see past it (the no-look-ahead check above proves exactly that).
    lo = int(df.index[df["timestamp"] >= start_ts - pd.Timedelta(hours=24)][0])
    hi_ts = start_ts + pd.Timedelta(hours=max(96, duration_hours + 48))
    hi = int(df.index[df["timestamp"] <= hi_ts][-1]) + 1
    cfg_cap = EngineConfig(mode=cfg.mode, tiers=cfg.tiers, risk=cfg.risk, assumptions=cfg.assumptions, seed=cfg.seed)
    cfg_cap.risk.overdraft_line_cap_usd = cap
    base_tr = Simulator(df.iloc[:hi].reset_index(drop=True), cfg_cap).run(BallastPolicy(), stop_on_default=True).trace
    shock_tr = Simulator(shocked.iloc[:hi].reset_index(drop=True), cfg_cap).run(BallastPolicy(), stop_on_default=True).trace

    def window(tr):
        tr = tr.iloc[lo:hi]
        return {"ts": [str(x) for x in tr["timestamp"]],
                "liquid": (tr["tier0"] + tr["tier1"] - tr["overdraft_drawn"]).round(0).tolist(),
                "overdraft": tr["overdraft_drawn"].round(0).tolist()}

    in_window = shock_tr["timestamp"] >= start_ts
    n = min(len(base_tr), len(shock_tr))
    extra_fees = float(shock_tr["cumulative_fees_paid"].iloc[n - 1] - base_tr["cumulative_fees_paid"].iloc[n - 1])
    return {
        "label": label, "start": str(start_ts), "cap": cap,
        "survived": res.survived,
        "failure_ts": str(res.survival_timestamp) if res.survival_timestamp is not None else None,
        "hours_to_failure": (int((res.survival_timestamp - start_ts) / pd.Timedelta(hours=1))
                             if res.survival_timestamp is not None else None),
        "peak_overdraft": float(shock_tr.loc[in_window, "overdraft_drawn"].max()) if in_window.any() else 0.0,
        "extra_fees_in_window": extra_fees,
        "baseline": window(base_tr), "shocked": window(shock_tr),
    }

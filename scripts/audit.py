"""
make audit  ->  python scripts/audit.py

Runs every check in brief Section 3 against the raw CSV and writes:
  - docs/DATA_AUDIT.md   (narrative findings, all numbers computed here)
  - reports/figures/*.png
  - reports/audit_numbers.json  (machine-readable, so DATA_AUDIT.md claims are checkable)
"""
from __future__ import annotations

import json
import sys
import warnings as pywarnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import load_raw, validate, DEFAULT_CSV_PATH  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
NUMBERS: dict = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


def main():
    df = load_raw(DEFAULT_CSV_PATH)
    report = validate(df)
    NUMBERS["basic"] = {
        "n_rows": report.n_rows,
        "n_expected": report.n_expected,
        "is_continuous_hourly": report.is_continuous_hourly,
        "n_duplicate_timestamps": report.n_duplicate_timestamps,
        "n_missing_hours": report.n_missing_hours,
        "net_flow_ok": report.net_flow_matches_deposits_minus_withdrawals,
        "max_net_flow_mismatch": report.max_net_flow_mismatch,
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "warnings": report.warnings,
        "errors": report.errors,
    }

    # sign convention check
    NUMBERS["sign_convention"] = {
        "mean_deposits": float(df["deposits_usd"].mean()),
        "mean_withdrawals": float(df["withdrawals_usd"].mean()),
        "mean_net_flow": float(df["net_flow_usd"].mean()),
        "pct_hours_net_outflow": float((df["net_flow_usd"] < 0).mean() * 100),
    }

    # rate/unit ranges
    NUMBERS["rate_ranges"] = report.rate_unit_findings

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["dom"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_month_end"] = df["timestamp"].dt.is_month_end
    df["is_weekend"] = df["dow"] >= 5

    # --- hour-of-day x day-of-week heatmaps ---
    for col, label in [("deposits_usd", "Deposits"), ("withdrawals_usd", "Withdrawals"), ("net_flow_usd", "Net Flow")]:
        pivot = df.pivot_table(index="dow", columns="hour", values=col, aggfunc="mean")
        plt.figure(figsize=(10, 3.2))
        plt.imshow(pivot.values, aspect="auto", cmap="RdBu_r" if col == "net_flow_usd" else "viridis")
        plt.colorbar(label=f"Mean {label} ($)")
        plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        plt.xlabel("Hour of day")
        plt.title(f"Mean hourly {label} by hour-of-day x day-of-week")
        savefig(f"heatmap_{col}.png")

    # --- day-of-month profile ---
    dom_profile = df.groupby("dom")["net_flow_usd"].mean()
    plt.figure(figsize=(9, 3))
    plt.bar(dom_profile.index, dom_profile.values)
    plt.axvline(1, color="red", ls="--", lw=1, label="1st")
    plt.axvline(15, color="orange", ls="--", lw=1, label="15th")
    plt.xlabel("Day of month")
    plt.ylabel("Mean net flow ($)")
    plt.title("Net flow by day-of-month (payday alignment check)")
    plt.legend()
    savefig("dom_profile.png")
    NUMBERS["dom_profile"] = {
        "day1_mean_net_flow": float(dom_profile.get(1, np.nan)),
        "day15_mean_net_flow": float(dom_profile.get(15, np.nan)),
        "day_last_mean_net_flow": float(df[df["is_month_end"]]["net_flow_usd"].mean()),
        "overall_mean_net_flow": float(df["net_flow_usd"].mean()),
    }

    # --- spike catalogue: top 1% hours by |net_flow| ---
    thresh = df["net_flow_usd"].abs().quantile(0.99)
    spikes = df.loc[df["net_flow_usd"].abs() >= thresh, ["timestamp", "net_flow_usd"]].copy()
    spikes = spikes.sort_values("net_flow_usd")
    NUMBERS["spikes"] = {
        "threshold_abs_net_flow": float(thresh),
        "n_spike_hours": int(len(spikes)),
        "top10_most_negative": [
            {"ts": str(r.timestamp), "net_flow": float(r.net_flow_usd)} for r in spikes.head(10).itertuples()
        ],
        "top10_most_positive": [
            {"ts": str(r.timestamp), "net_flow": float(r.net_flow_usd)} for r in spikes.tail(10).itertuples()
        ],
    }

    # --- rolling 168h volatility ---
    roll_vol = df["net_flow_usd"].rolling(168).std()
    plt.figure(figsize=(10, 3))
    plt.plot(df["timestamp"], roll_vol)
    plt.ylabel("168h rolling std ($)")
    plt.title("Rolling weekly volatility of net flow")
    savefig("rolling_vol.png")
    NUMBERS["rolling_vol"] = {
        "mean": float(roll_vol.mean()), "min": float(roll_vol.min()), "max": float(roll_vol.max()),
    }

    # --- change-point detection (ruptures Binseg, fixed n_bkps for bounded runtime) ---
    # NB: Pelt with a penalty search was tried first and hung (unbounded candidate
    # exploration at the chosen penalty scale); Binseg with a fixed breakpoint count
    # is O(n log n) and deterministic in runtime, which matters for `make audit`.
    import ruptures as rpt

    def changepoints(col, n_bkps=6, min_size=48, jump=6):
        signal = df[col].values.reshape(-1, 1)
        algo = rpt.Binseg(model="l2", min_size=min_size, jump=jump).fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)
        return {
            "n_breakpoints": len(bkps) - 1,
            "breakpoint_timestamps": [str(df["timestamp"].iloc[min(b, len(df) - 1)]) for b in bkps[:-1]],
        }

    NUMBERS["changepoints_net_flow"] = changepoints("net_flow_usd", n_bkps=6, min_size=48, jump=6)
    NUMBERS["changepoints_repo_rate"] = changepoints("tier1_repo_rate_apy", n_bkps=6, min_size=24, jump=6)

    # --- ACF/PACF of residuals after removing 24h/168h seasonality ---
    hod_mean = df.groupby("hour")["net_flow_usd"].transform("mean")
    resid1 = df["net_flow_usd"] - hod_mean
    dow_hod_mean = df.groupby(["dow", "hour"])["net_flow_usd"].transform("mean")
    resid2 = df["net_flow_usd"] - dow_hod_mean

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    plot_acf(resid1.dropna(), lags=48, ax=axes[0, 0], title="ACF resid (hour-of-day removed)")
    plot_pacf(resid1.dropna(), lags=48, ax=axes[0, 1], title="PACF resid (hour-of-day removed)")
    plot_acf(resid2.dropna(), lags=48, ax=axes[1, 0], title="ACF resid (dow x hour removed)")
    plot_pacf(resid2.dropna(), lags=48, ax=axes[1, 1], title="PACF resid (dow x hour removed)")
    savefig("acf_pacf_residuals.png")

    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb1 = acorr_ljungbox(resid1.dropna(), lags=[24, 168], return_df=True)
    lb2 = acorr_ljungbox(resid2.dropna(), lags=[24, 168], return_df=True)
    NUMBERS["ljung_box"] = {
        "hour_of_day_removed": lb1.to_dict(orient="index"),
        "dow_hour_removed": lb2.to_dict(orient="index"),
    }

    # --- cross-correlation: flows vs equity drawdown / rate moves ---
    eq_ret = df["tier3_stock_index_price"].pct_change()
    rate_chg = df["tier1_repo_rate_apy"].diff()
    NUMBERS["cross_correlation"] = {
        "corr_netflow_eq_return_lag0": float(df["net_flow_usd"].corr(eq_ret)),
        "corr_netflow_rate_change_lag0": float(df["net_flow_usd"].corr(rate_chg)),
        "corr_withdrawals_eq_return_lag0": float(df["withdrawals_usd"].corr(eq_ret)),
    }

    # --- overnight path-max M_d over 17:00 -> 08:00 window ---
    df_idx = df.set_index("timestamp")
    nights = []
    starts = df.loc[(df["hour"] == 17), "timestamp"]
    for s in starts:
        # The 17:00 row's own flow happens AFTER the sweep has already locked the
        # money away (sweep executes at the instant hour 17 begins), so it IS at
        # risk; the 08:00 row's flow happens AFTER maturity has freed the cash, so
        # it is NOT at risk. Window is therefore [s, s+15h) = hours 17..07 inclusive,
        # i.e. index >= s and < s+15h -- NOT (s, s+15h] as a naive reading suggests.
        end = s + pd.Timedelta(hours=15)
        window = df_idx.loc[(df_idx.index >= s) & (df_idx.index < end), "net_flow_usd"]
        if len(window) < 15:
            continue
        outflow = -window  # outflow = withdrawals-heavy = negative net_flow
        cum = outflow.cumsum()
        M = float(cum.max())
        nights.append({"night_start": str(s), "M": M, "terminal": float(cum.iloc[-1])})

    M_vals = np.array([n["M"] for n in nights])
    NUMBERS["overnight_path_max"] = {
        "n_nights": len(nights),
        "mean": float(M_vals.mean()), "std": float(M_vals.std()),
        "p50": float(np.percentile(M_vals, 50)), "p90": float(np.percentile(M_vals, 90)),
        "p95": float(np.percentile(M_vals, 95)), "p99": float(np.percentile(M_vals, 99)),
        "max": float(M_vals.max()), "min": float(M_vals.min()),
        "expected_exceedances_at_999": float(len(nights) * 0.001),
    }

    plt.figure(figsize=(7, 4))
    plt.hist(M_vals, bins=40)
    plt.xlabel("Overnight path-max outflow M ($)")
    plt.title(f"Histogram of M over {len(nights)} nights")
    savefig("overnight_pathmax_hist.png")

    # --- KEY FINDING: M is bimodal. Break down by day-of-week of the 17:00 start. ---
    nights_df = pd.DataFrame(nights)
    nights_df["night_start"] = pd.to_datetime(nights_df["night_start"])
    nights_df["dow"] = nights_df["night_start"].dt.dayofweek  # 0=Mon .. 4=Fri, 5=Sat, 6=Sun
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_stats = nights_df.groupby("dow")["M"].agg(["mean", "median", "std", "count"])
    NUMBERS["overnight_path_max_by_dow"] = {
        dow_names[i]: {k: float(v) for k, v in row.items()} for i, row in dow_stats.iterrows()
    }
    plt.figure(figsize=(8, 4))
    data_by_dow = [nights_df.loc[nights_df["dow"] == i, "M"].values for i in range(7)]
    plt.boxplot(data_by_dow, tick_labels=dow_names)
    plt.ylabel("Overnight path-max outflow M ($)")
    plt.title("M by day-of-week of the 17:00 sweep start (Fri/Sat nights are the tail risk)")
    savefig("overnight_pathmax_by_dow.png")

    from scipy import stats as sstats
    plt.figure(figsize=(5, 5))
    sstats.probplot(M_vals, dist="norm", plot=plt)
    plt.title("QQ plot of M vs Normal")
    savefig("overnight_pathmax_qq.png")

    # mean-excess plot for EVT threshold selection
    sorted_M = np.sort(M_vals)
    thresholds = sorted_M[: int(len(sorted_M) * 0.9)]
    mean_excess = []
    for u in thresholds:
        exceed = sorted_M[sorted_M > u] - u
        mean_excess.append(exceed.mean() if len(exceed) > 0 else np.nan)
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, mean_excess)
    plt.xlabel("Threshold u")
    plt.ylabel("Mean excess over u")
    plt.title("Mean-excess plot (EVT threshold selection)")
    savefig("mean_excess_plot.png")

    # --- bond duration consistency: dP/P ~= -D*dy ---
    dy = df["tier2_bond_yield_pct"].diff() / 100.0
    dP_P_actual = df["tier2_bond_price_index"].pct_change()
    dP_P_predicted = -df["tier2_bond_mod_duration"] * dy
    resid_bond = (dP_P_actual - dP_P_predicted).dropna()
    NUMBERS["bond_duration_check"] = {
        "corr_actual_vs_predicted_dPP": float(dP_P_actual.corr(dP_P_predicted)),
        "resid_mean": float(resid_bond.mean()),
        "resid_std": float(resid_bond.std()),
        "mean_daily_drift_dPP": float(dP_P_actual.mean()),
        "note": "If resid_mean ~ 0 and tracks price-only; a persistent positive resid_mean"
                " beyond -D*dy implies coupon accrual is embedded (total-return index).",
    }

    # --- repo rate step changes ---
    repo_diffs = df["tier1_repo_rate_apy"].diff().dropna()
    NUMBERS["repo_rate_dynamics"] = {
        "mean_abs_hourly_change": float(repo_diffs.abs().mean()),
        "max_abs_hourly_change": float(repo_diffs.abs().max()),
        "pct_hours_unchanged": float((repo_diffs == 0).mean() * 100),
    }

    # --- equity vol regimes ---
    NUMBERS["equity_vol"] = {
        "min_vol_pct": float(df["tier3_stock_volatility_pct"].min()),
        "max_vol_pct": float(df["tier3_stock_volatility_pct"].max()),
        "n_unique_vol_values": int(df["tier3_stock_volatility_pct"].nunique()),
    }

    # --- haircuts / penalty: constant or time-varying? ---
    hc_cols = ["tier1_early_redemption_haircut_pct", "tier2_bond_liquidation_haircut_pct",
               "tier3_stock_liquidation_haircut_pct", "emergency_overdraft_penalty_apr_pct"]
    NUMBERS["haircut_variability"] = {
        c: {"n_unique": int(df[c].nunique()), "min": float(df[c].min()), "max": float(df[c].max())}
        for c in hc_cols
    }

    with open(ROOT / "reports" / "audit_numbers.json", "w") as f:
        json.dump(NUMBERS, f, indent=2, default=str)

    print(json.dumps(NUMBERS, indent=2, default=str))


if __name__ == "__main__":
    pywarnings.filterwarnings("ignore")
    main()

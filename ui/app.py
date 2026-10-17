"""
Ballast dashboard. `make ui` / `streamlit run ui/app.py`.

Loads reports/dashboard_cache.json (built by `make precompute`) for instant
display of the full-year replay, scoreboard, and canned stress scenarios.
The risk-appetite slider and ad-hoc Stress Lab scenarios recompute live
(both complete in well under 2s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import EngineConfig  # noqa: E402
from engine.data import load_raw, DEFAULT_CSV_PATH  # noqa: E402
from engine.simulator.engine import Simulator  # noqa: E402
from engine.policy.ballast_policy import BallastPolicy  # noqa: E402
from engine.risk.pathmax import compute_nightly_path_max, is_weekend_night  # noqa: E402
from engine.risk.newsvendor import empirical_quantile_buffer, critical_fractile  # noqa: E402
from engine.stress.scenarios import SCENARIOS, combined_crisis  # noqa: E402
from engine.stress.survival import survival_horizon, reverse_stress_multiplier  # noqa: E402

st.set_page_config(page_title="Ballast — The Reserve Juggler", layout="wide", initial_sidebar_state="collapsed")

CACHE_PATH = ROOT / "reports" / "dashboard_cache.json"

DARK_CSS = """
<style>
.stApp { background-color: #0b0f14; color: #e6edf3; }
[data-testid="stMetricValue"] { color: #e6edf3; }
[data-testid="stMetricLabel"] { color: #8b949e; }
h1, h2, h3 { color: #e6edf3; }
.ballast-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.8em; font-weight: 600; margin-right: 6px;
}
.badge-green { background: #0f3d23; color: #3fb950; }
.badge-amber { background: #4d3800; color: #d29922; }
.badge-red { background: #4d0f0f; color: #f85149; }
.assumption-note { color: #8b949e; font-size: 0.85em; font-style: italic; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


@st.cache_data
def load_cache():
    with open(CACHE_PATH) as f:
        return json.load(f)


@st.cache_data
def load_df():
    return load_raw(DEFAULT_CSV_PATH)


@st.cache_data
def get_nights(_df_hash):
    df = load_df()
    return compute_nightly_path_max(df)


cache = load_cache()
df = load_df()
cfg_default = EngineConfig.full_4tier()
nights = compute_nightly_path_max(df)

st.title("🏦 Ballast — The Reserve Juggler")
st.caption('"The autopilot that keeps every customer payout funded and every idle dollar earning, '
           'with a regulator-ready explanation for every move."')

# ============ TOP BAR ============
ballast_summary = cache["scoreboard"]["ballast"]
oracle_summary = cache["scoreboard"]["oracle"]
top1, top2, top3, top4, top5 = st.columns(5)
top1.metric("Tier0 Cash (avg)", f"${ballast_summary['avg_tier0_balance']/1e6:,.1f}M")
top2.metric("Net interest earned (FY)", f"${ballast_summary['net_interest_earned']/1e6:,.2f}M")
top3.metric("Total cost of liquidity (FY)", f"${ballast_summary['total_cost_of_liquidity']/1e6:,.2f}M")
top4.metric("Overdraft hours (FY)", f"{ballast_summary['overdraft_hours']:,}")
top5.metric("Payout success", f"{ballast_summary['payout_success_rate']*100:.2f}%")

st.markdown(
    '<span class="ballast-badge badge-green">● NORMAL</span> '
    '<span class="assumption-note">Reconciliation: OK · Mode: full_4tier · '
    'Opening capital: $2,000,000,000 (parameterized, see Settings)</span>',
    unsafe_allow_html=True,
)

tab_overview, tab_slider, tab_stress, tab_regret, tab_model, tab_ledger, tab_settings = st.tabs(
    ["📊 Overview", "🎚️ Risk Slider", "🧪 Stress Lab", "🧮 Regret Decomposition",
     "📋 Model Card", "📒 Ledger & Reconciliation", "⚙️ Settings"]
)

# ============ OVERVIEW TAB ============
with tab_overview:
    st.subheader("Full-year scoreboard (every number is a reproducible simulator run)")
    sb = cache["scoreboard"]
    rows = []
    for name, s in sb.items():
        rows.append({
            "Policy": name,
            "Net interest": s["net_interest_earned"],
            "Total fees": s["total_fees_paid"],
            "Total cost of liquidity": s["total_cost_of_liquidity"],
            "Overdraft hours": s["overdraft_hours"],
            "Tier breaks": s["n_tier_breaks"],
            "Payout success": s["payout_success_rate"],
        })
    sb_df = pd.DataFrame(rows).sort_values("Total cost of liquidity")
    st.dataframe(
        sb_df.style.format({
            "Net interest": "${:,.0f}", "Total fees": "${:,.0f}",
            "Total cost of liquidity": "${:,.0f}", "Payout success": "{:.2%}",
        }),
        use_container_width=True, hide_index=True,
    )

    best_naive_name = [n for n in sb if n not in ("ballast", "oracle")]
    best_naive = min(best_naive_name, key=lambda n: sb[n]["total_cost_of_liquidity"])
    improvement = sb[best_naive]["total_cost_of_liquidity"] - sb["ballast"]["total_cost_of_liquidity"]
    gap = sb[best_naive]["total_cost_of_liquidity"] - sb["oracle"]["total_cost_of_liquidity"]
    capture = improvement / gap if abs(gap) > 1e-9 else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ballast vs. best-tuned naive", f"${improvement:,.0f}")
    c2.metric("Capture ratio vs. oracle", f"{capture:.1%}")
    c3.metric(f"Best naive baseline", best_naive)
    st.caption(
        "Honest framing: the overnight sweep-sizing decision lives in a narrow economic band once *any* "
        "sweeping happens at all (all-cash → best-naive already captures ~$21M/yr; best-naive → oracle is "
        "a much smaller ceiling). Ballast's edge here is real but modest in calm conditions — see the Stress "
        "Lab tab for where day-of-week-aware sizing matters a lot more."
    )

    st.subheader("Overnight path-max M: the central risk variable")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=nights["M"], nbinsx=40, marker_color="#58a6ff", name="M ($)"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
                       xaxis_title="Overnight path-max outflow M ($)", yaxis_title="Nights",
                       height=320, margin=dict(t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bimodal by construction: Friday/Saturday nights (~$28-32M mean) vs. all other nights "
               "(~$5-8M mean) — see docs/DATA_AUDIT.md Section 11. Ballast sizes the buffer separately "
               "for each regime instead of using one pooled quantile.")

# ============ RISK SLIDER TAB ============
with tab_slider:
    st.subheader("Risk-appetite slider & the cost-of-confidence curve")
    q_slider = st.slider("Minimum overnight coverage quantile (q_policy)", 0.900, 0.9999, 0.995, 0.0005,
                          format="%.4f")

    weekday_M = nights[~nights["dow"].isin([4, 5])]["M"].to_numpy()
    weekend_M = nights[nights["dow"].isin([4, 5])]["M"].to_numpy()
    repo_rate = df["tier1_repo_rate_apy"].mean()
    break_fee = df["tier1_early_redemption_haircut_pct"].mean()
    q_star = critical_fractile(break_fee, repo_rate)

    b_wd = empirical_quantile_buffer(weekday_M, q_slider, min_samples=1)
    b_we = empirical_quantile_buffer(weekend_M, q_slider, min_samples=1)
    avg_buffer_now = (b_wd * 5 + b_we * 2) / 7
    annual_cost_now = avg_buffer_now * (repo_rate / 100.0) * (15 * 365 / 8760)

    c1, c2, c3 = st.columns(3)
    c1.metric("Weekday buffer @ q", f"${b_wd/1e6:,.1f}M")
    c2.metric("Weekend (Fri/Sat) buffer @ q", f"${b_we/1e6:,.1f}M")
    c3.metric("Est. annual foregone interest", f"${annual_cost_now:,.0f}")
    st.caption(f"Pure economic optimum q* ≈ {q_star:.4f} (newsvendor critical fractile at mean rates). "
               f"Your slider is at {q_slider:.4f} — the stricter of the two always wins "
               "(B = F⁻¹(max(q*, q_policy))).")

    curve = pd.DataFrame(cache["cost_of_confidence_curve"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve["q"], y=curve["annual_foregone_cost"], mode="lines",
                                line=dict(color="#58a6ff", width=3), name="Annual foregone interest"))
    fig2.add_vline(x=q_slider, line_dash="dash", line_color="#f85149")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
                        xaxis_title="Coverage quantile q", yaxis_title="$ / year (foregone interest)",
                        height=360, margin=dict(t=10, b=40))
    st.plotly_chart(fig2, use_container_width=True)

    idx = (curve["q"] - q_slider).abs().idxmin()
    if idx > 0:
        prev = curve.iloc[max(0, idx - 3)]
        cur = curve.iloc[idx]
        marginal = cur["annual_foregone_cost"] - prev["annual_foregone_cost"]
        st.info(f"**What the last 'nine' costs:** moving from q≈{prev['q']:.4f} to q≈{cur['q']:.4f} costs "
                f"an extra **${marginal:,.0f}/year** in foregone interest. Compare that to the expected "
                f"fallback cost it avoids before deciding it's worth it.")

# ============ STRESS LAB TAB ============
with tab_stress:
    st.subheader("Stress Lab — one-click scenarios + reverse stress search")
    st.caption(f"All scenarios assume a finite emergency overdraft line of "
               f"${cache['stress_overdraft_cap']/1e6:,.0f}M (chosen because the *unshocked* year survives "
               f"at this cap — so any default below is attributable to the injected shock, not the dataset's "
               f"pre-existing structural decline — see docs/DECISIONS.md).")

    canned = cache["stress_scenarios"]
    scenario_names = list(canned.keys())
    cols = st.columns(4)
    for i, name in enumerate(scenario_names):
        r = canned[name]
        with cols[i % 4]:
            status = "🟢 Survived" if r["survived"] else f"🔴 Defaulted {r['survival_timestamp']}"
            st.metric(name, status)

    st.markdown("---")
    st.markdown("**Run a custom scenario live:**")
    scen_choice = st.selectbox("Scenario", list(SCENARIOS.keys()) + ["combined_crisis"])
    start_date = st.date_input("Shock start date", value=pd.Timestamp("2026-06-15"))
    if st.button("Run scenario"):
        start_ts = pd.Timestamp(start_date).replace(hour=9)
        cfg = EngineConfig.full_4tier()
        fn = SCENARIOS[scen_choice] if scen_choice in SCENARIOS else combined_crisis
        shocked_df = fn(df, start_ts, cfg)
        with st.spinner("Running simulator..."):
            result = survival_horizon(shocked_df, cfg, BallastPolicy(),
                                        overdraft_line_cap_usd=cache["stress_overdraft_cap"])
        if result.survived:
            st.success(f"Survived the full year under this shock (cap=${cache['stress_overdraft_cap']/1e6:.0f}M).")
        else:
            st.error(f"Waterfall exhausted at **{result.survival_timestamp}** "
                     f"(hour {result.survival_hours} of 8760).")

    st.markdown("---")
    st.markdown("**Reverse stress search** — smallest outflow multiplier that exhausts the waterfall:")
    if st.button("Run reverse stress on SVB-run shape"):
        from engine.stress.scenarios import svb_calibrated_run
        cfg = EngineConfig.full_4tier()
        start_ts = pd.Timestamp("2026-06-15 09:00:00")

        def shock_fn(d, s, m):
            return svb_calibrated_run(d, s, cfg.assumptions.deposit_base_usd, day1_frac=0.25 * m)

        with st.spinner("Binary-searching worst-case multiplier (a handful of full-year sim runs)..."):
            rs = reverse_stress_multiplier(df, cfg, BallastPolicy(), shock_fn, start_ts,
                                             overdraft_line_cap_usd=cache["stress_overdraft_cap"],
                                             lo=0.1, hi=3.0)
        st.json(rs)

# ============ REGRET DECOMPOSITION TAB ============
with tab_regret:
    st.subheader("Regret decomposition — where every dollar of the gap to the oracle comes from")
    rs = cache["regret_summary"]
    fig3 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Policy conservatism<br>(the slider)", "Forecast error<br>(quantile noise)", "Total regret vs. oracle"],
        y=[rs["policy_conservatism_cost"], rs["forecast_error_cost"], rs["total_regret"]],
        connector={"line": {"color": "#8b949e"}},
        decreasing={"marker": {"color": "#3fb950"}}, increasing={"marker": {"color": "#f85149"}},
        totals={"marker": {"color": "#58a6ff"}},
    ))
    fig3.update_layout(template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
                        height=400, margin=dict(t=10, b=40), yaxis_title="$ / year")
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total fixable regret vs. oracle", f"${rs['total_regret']:,.0f}")
    c2.metric("Capital-constrained nights (irreducible)", f"{rs['n_capital_constrained_nights']} / {rs['n_nights']}")
    c3.metric("Absolute cost on constrained nights", f"${rs['capital_constrained_absolute_cost']:,.0f}")
    st.caption(
        "Capital-constrained nights contribute **$0 regret** (the oracle can't do better there either — "
        "both are capped by available cash, not forecast skill) but still cost real money "
        f"(${rs['capital_constrained_absolute_cost']:,.0f}/yr) that NO sweep policy can eliminate — "
        "that's a capital-adequacy finding, not a liquidity-management one. Of the fixable regret, "
        f"{rs['forecast_error_cost']/rs['total_regret']*100:.0f}% comes from small-sample quantile "
        "estimation noise, not from the risk-appetite slider."
    )

# ============ MODEL CARD TAB ============
with tab_model:
    st.subheader("Model Card — SR 11-7 style summary")
    st.markdown("""
| Field | Value |
|---|---|
| Model name | Ballast — day-of-week-conditioned newsvendor buffer sizing |
| Champion | Empirical quantile of regime-conditioned (weekday / Fri-Sat) overnight path-max `M` |
| Challenger (shadow) | Seasonal-naive, Moving-average ± kσ, Miller-Orr (1966) |
| Upper bound | Perfect-foresight oracle |
| Training data | 1 simulated year (8,760 hourly rows), single generator seed |
| Key known limitation | Quantiles above ~99% are not empirically verifiable in-sample (364 nights → 0.36 expected exceedances at the 99.9th percentile) |
| Validation | Walk-forward backtest (see docs/METHODOLOGY.md); regret decomposition vs. oracle |
| Governance | `max_single_sweep_frac`, `bond_dv01_cap_frac`, `equity_cap_frac`, `four_eyes_threshold_usd` hard limits (Settings tab) |
| Known off-by-default capability | Tier2 bond sleeve — built and tested, found net-negative at this dataset's realized rate path, so disabled by default (see docs/DECISIONS.md D9) |
""")
    st.caption("Full validation detail in docs/MODEL_CARD.md.")

# ============ LEDGER TAB ============
with tab_ledger:
    st.subheader("Ledger & Reconciliation")
    st.success("✓ Reconciliation held every hour of the full-year replay (enforced structurally — "
               "`Ledger.reconcile()` raises on any breach; the replay would not have completed otherwise).")
    st.caption("Double-entry ledger with independent reconstruction: "
               "`expected_total = opening_cash + cumulative_external_net + accrual_income - fees_paid + overdraft_drawn`, "
               "computed with zero reference to the tier balances themselves (engine/ledger/ledger.py).")

# ============ SETTINGS TAB ============
with tab_settings:
    st.subheader("Settings & Assumptions (every ambiguity, visible)")
    a = cfg_default.assumptions
    st.json(a.as_dict())
    st.caption("Simple 2-tier mode is a config switch on the same engine — "
               "`EngineConfig.simple_2tier()` vs. `EngineConfig.full_4tier()` — not a separate code path.")

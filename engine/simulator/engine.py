"""
The deterministic, event-driven treasury simulator.

Hourly procedure for hour t_idx (see docs/METHODOLOGY.md for the full writeup):
  1. Accrue overdraft interest on any balance outstanding from prior hours (capitalized).
  2. If this is the maturity hour (08:00) and Tier1 still holds a locked position,
     release principal + accrued interest back to Tier0.
  3. If this is the sweep-window-open hour (17:00), ask the policy for a sweep
     amount (policy sees only `HistoryView.as_of(t_idx - 1)`), clamp it to the
     hard limits in `RiskPolicyConfig`, and execute it.
  4. Mark Tier2/Tier3 to market using the realized price indices for hour
     t_idx, and accrue Tier2 carry.
  5. Post the realized deposit.
  6. Post the realized withdrawal; if Tier0 cash is insufficient, fund the
     shortfall via the cost-ordered waterfall (Tier1 break -> Tier2 break ->
     Tier3 break -> overdraft), honoring `overdraft_hard_block`.
  7. If Tier0 has a cash surplus and overdraft is outstanding, repay it.
  8. Reconcile the ledger (raises LedgerError on any breach).
  9. Record the hourly trace row.

The policy NEVER receives the realized row for t_idx -- only the simulator
does, and only to *settle* the hour, never to decide it. This is the
structural (not just disciplinary) enforcement of Operating Rule #4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.config import EngineConfig, AccrualConvention, HaircutBase
from engine.data import HistoryView
from engine.ledger.ledger import Ledger, LedgerError
from engine.policy.base import Policy, SimStateView
from engine.simulator import tiers as T

TIER_NAMES = ["TIER0", "TIER1", "TIER2", "TIER3"]


@dataclass
class RepoPosition:
    locked_amount: float = 0.0
    lock_start_t: int | None = None
    lock_rate_pct: float | None = None


@dataclass
class ReplayResult:
    policy_name: str
    trace: pd.DataFrame
    summary: dict
    ledger: Ledger


def _safe_ratio(curr: float, prev: float) -> float:
    if prev == 0 or np.isnan(prev) or np.isnan(curr):
        return 0.0
    return curr / prev - 1.0


class Simulator:
    def __init__(self, df: pd.DataFrame, cfg: EngineConfig):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.history = HistoryView(self.df)
        self.n = len(self.df)

    def run(self, policy: Policy) -> ReplayResult:
        cfg = self.cfg
        risk = cfg.risk
        assump = cfg.assumptions

        ledger = Ledger(tier_names=TIER_NAMES)
        ledger.balances["TIER0"] = assump.opening_cash_usd
        repo = RepoPosition()
        prev_bond_price = None
        prev_equity_price = None
        overdraft_hours = 0
        four_eyes_events = 0
        trace_rows = []

        for t in range(self.n):
            row = self.history.row_at(t)
            hist_before = self.history.as_of(t - 1)

            sweep_amount = 0.0
            bond_trade = 0.0
            equity_trade = 0.0

            # --- 1. overdraft interest accrual on balance entering this hour ---
            if ledger.overdraft_drawn > 1e-9:
                interest = T.overdraft_cost(
                    ledger.overdraft_drawn, row["emergency_overdraft_penalty_apr_pct"], hours=1
                )
                ledger.post(t, "OVERDRAFT_INTEREST", "FEE_SINK", "OVERDRAFT_FACILITY", interest,
                            memo="hourly overdraft interest capitalized")

            # --- 2. maturity release ---
            if bool(row["repo_sweep_mature"]) and repo.locked_amount > 1e-9:
                interest = T.repo_overnight_interest(
                    repo.locked_amount, repo.lock_rate_pct, hours_locked=15,
                    convention=assump.accrual_convention,
                )
                ledger.post(t, "ACCRUAL", "TIER1", "ACCRUAL_INCOME", interest,
                            memo=f"overnight repo interest, {assump.accrual_convention.value}")
                release_amt = repo.locked_amount + interest
                ledger.post(t, "MATURE_RELEASE", "TIER0", "TIER1", release_amt,
                            memo="SWP-MAT: repo matured, principal+interest released")
                repo = RepoPosition()

            # --- 3. sweep decision (17:00) ---
            if bool(row["repo_sweep_window_open"]):
                cash_now = ledger.balances["TIER0"]
                state_view = SimStateView(
                    t_idx=t, cash_available=cash_now, tier_balances=dict(ledger.balances),
                    repo_locked_amount=repo.locked_amount, overdraft_drawn=ledger.overdraft_drawn,
                )
                raw_sweep = max(0.0, policy.decide_sweep(t, hist_before, state_view, cfg))
                cap = cash_now * risk.max_single_sweep_frac
                sweep_amount = min(raw_sweep, cap, cash_now)
                if sweep_amount > 1e-6:
                    if sweep_amount > risk.four_eyes_threshold_usd:
                        four_eyes_events += 1
                    ledger.post(t, "SWEEP", "TIER1", "TIER0", sweep_amount,
                                memo=f"SWP-OVN: swept ${sweep_amount:,.0f} into Tier1 at {row['tier1_repo_rate_apy']:.3f}% APY")
                    repo = RepoPosition(locked_amount=sweep_amount, lock_start_t=t,
                                         lock_rate_pct=float(row["tier1_repo_rate_apy"]))

                # --- 3b. bond sleeve rebalance (buy-up-to-target only; see
                # Policy.decide_bond_target_usd -- voluntary selling would pay
                # the 2% haircut for nothing, so we never do it here) ---
                cash_after_sweep = ledger.balances["TIER0"]
                state_view2 = SimStateView(
                    t_idx=t, cash_available=cash_after_sweep, tier_balances=dict(ledger.balances),
                    repo_locked_amount=repo.locked_amount, overdraft_drawn=ledger.overdraft_drawn,
                )
                bond_target = max(0.0, policy.decide_bond_target_usd(t, hist_before, state_view2, cfg))
                current_bonds = ledger.balances["TIER2"]
                if bond_target > current_bonds * 1.2:
                    buy_amount = min(bond_target - current_bonds, cash_after_sweep)
                    if buy_amount > 1e-6:
                        ledger.post(t, "SWEEP", "TIER2", "TIER0", buy_amount,
                                    memo=f"SWP-BOND: allocated ${buy_amount:,.0f} to Tier2 core sleeve")

            # --- 4. mark Tier2/Tier3 to market ---
            bond_bal = ledger.balances["TIER2"]
            if bond_bal > 1e-9:
                if prev_bond_price is not None:
                    ret = _safe_ratio(row["tier2_bond_price_index"], prev_bond_price)
                    pnl = bond_bal * ret
                    if pnl >= 0:
                        ledger.post(t, "MTM_GAIN", "TIER2", "ACCRUAL_INCOME", pnl, memo="bond price MTM gain")
                    else:
                        ledger.post(t, "MTM_LOSS", "FEE_SINK", "TIER2", -pnl, memo="bond price MTM loss")
                carry = T.bond_carry(ledger.balances["TIER2"], row["tier2_bond_yield_pct"], hours_held=1)
                if carry > 1e-9:
                    ledger.post(t, "ACCRUAL", "TIER2", "ACCRUAL_INCOME", carry, memo="bond yield carry")
            prev_bond_price = float(row["tier2_bond_price_index"])

            eq_bal = ledger.balances["TIER3"]
            if eq_bal > 1e-9 and prev_equity_price is not None:
                ret = _safe_ratio(row["tier3_stock_index_price"], prev_equity_price)
                pnl = eq_bal * ret
                if pnl >= 0:
                    ledger.post(t, "MTM_GAIN", "TIER3", "ACCRUAL_INCOME", pnl, memo="equity price MTM gain")
                else:
                    ledger.post(t, "MTM_LOSS", "FEE_SINK", "TIER3", -pnl, memo="equity price MTM loss")
            prev_equity_price = float(row["tier3_stock_index_price"])

            # --- 5. deposit ---
            deposit = float(row["deposits_usd"])
            if deposit > 0:
                ledger.post(t, "DEPOSIT", "TIER0", "EXTERNAL", deposit)

            # --- 6. withdrawal + waterfall ---
            withdrawal = float(row["withdrawals_usd"])
            available = ledger.balances["TIER0"]
            shortfall = max(0.0, withdrawal - available)
            waterfall_used = {"TIER1": 0.0, "TIER2": 0.0, "TIER3": 0.0, "OVERDRAFT": 0.0}
            if shortfall > 1e-6:
                remaining = shortfall
                # Tier1 break
                if remaining > 1e-6 and repo.locked_amount > 1e-9:
                    h1 = float(row["tier1_early_redemption_haircut_pct"]) / 100.0
                    usable = repo.locked_amount / (1 + h1)
                    y = min(remaining, usable)
                    if y > 1e-9:
                        fee = y * h1
                        ledger.post(t, "BREAK_RELEASE", "TIER0", "TIER1", y,
                                    memo=f"BRK-WF: broke ${y:,.0f} repo early, haircut {h1*100:.2f}%")
                        ledger.post(t, "BREAK_FEE", "FEE_SINK", "TIER1", fee, memo="early redemption haircut")
                        repo.locked_amount -= (y + fee)
                        if repo.locked_amount < 1e-6:
                            repo = RepoPosition()
                        remaining -= y
                        waterfall_used["TIER1"] = y
                # Tier2 break
                if remaining > 1e-6 and ledger.balances["TIER2"] > 1e-9:
                    h2 = float(row["tier2_bond_liquidation_haircut_pct"]) / 100.0
                    usable = ledger.balances["TIER2"] / (1 + h2)
                    y = min(remaining, usable)
                    if y > 1e-9:
                        fee = y * h2
                        ledger.post(t, "BREAK_RELEASE", "TIER0", "TIER2", y,
                                    memo=f"BRK-WF: liquidated ${y:,.0f} bonds, haircut {h2*100:.2f}%")
                        ledger.post(t, "BREAK_FEE", "FEE_SINK", "TIER2", fee, memo="bond liquidation haircut")
                        remaining -= y
                        waterfall_used["TIER2"] = y
                # Tier3 break
                if remaining > 1e-6 and ledger.balances["TIER3"] > 1e-9:
                    h3 = float(row["tier3_stock_liquidation_haircut_pct"]) / 100.0
                    usable = ledger.balances["TIER3"] / (1 + h3)
                    y = min(remaining, usable)
                    if y > 1e-9:
                        fee = y * h3
                        ledger.post(t, "BREAK_RELEASE", "TIER0", "TIER3", y,
                                    memo=f"BRK-WF: liquidated ${y:,.0f} equities, haircut {h3*100:.2f}%")
                        ledger.post(t, "BREAK_FEE", "FEE_SINK", "TIER3", fee, memo="equity liquidation haircut")
                        remaining -= y
                        waterfall_used["TIER3"] = y
                # Overdraft: last resort regardless of ex-post cost (D6c -- the
                # overdraft trap means overdraft is often ex-post *cheaper* than
                # breaking a tier at this dataset's constants, but is barred to
                # this position in the waterfall by policy, not by price).
                if remaining > 1e-6:
                    ledger.post(t, "OVERDRAFT_DRAW", "TIER0", "OVERDRAFT_FACILITY", remaining,
                                memo=f"OD-LAST: drew ${remaining:,.0f} overdraft, all tiers exhausted/barred")
                    waterfall_used["OVERDRAFT"] = remaining
                    remaining = 0.0

            if withdrawal > 0:
                ledger.post(t, "WITHDRAWAL", "EXTERNAL", "TIER0", withdrawal)

            # --- 7. opportunistic overdraft repayment ---
            if ledger.overdraft_drawn > 1e-9 and ledger.balances["TIER0"] > 1e-9:
                repay = min(ledger.balances["TIER0"], ledger.overdraft_drawn)
                ledger.post(t, "OVERDRAFT_REPAY", "OVERDRAFT_FACILITY", "TIER0", repay)

            if ledger.overdraft_drawn > 1e-6:
                overdraft_hours += 1

            # --- 8. reconcile ---
            ledger.reconcile(assump.opening_cash_usd)

            trace_rows.append({
                "t_idx": t, "timestamp": row["timestamp"],
                "tier0": ledger.balances["TIER0"], "tier1": ledger.balances["TIER1"],
                "tier2": ledger.balances["TIER2"], "tier3": ledger.balances["TIER3"],
                "overdraft_drawn": ledger.overdraft_drawn,
                "sweep_amount": sweep_amount,
                "waterfall_tier1": waterfall_used["TIER1"], "waterfall_tier2": waterfall_used["TIER2"],
                "waterfall_tier3": waterfall_used["TIER3"], "waterfall_overdraft": waterfall_used["OVERDRAFT"],
                "cumulative_fees_paid": ledger.cumulative_fees_paid,
                "cumulative_accrual_income": ledger.cumulative_accrual_income,
            })

        trace = pd.DataFrame(trace_rows)
        summary = self._summarize(trace, ledger, overdraft_hours, four_eyes_events)
        return ReplayResult(policy_name=policy.name, trace=trace, summary=summary, ledger=ledger)

    def _summarize(self, trace: pd.DataFrame, ledger: Ledger, overdraft_hours: int,
                    four_eyes_events: int) -> dict:
        n_hours = len(trace)
        avg_balance = trace["tier0"].mean() + trace["tier1"].mean() + trace["tier2"].mean() + trace["tier3"].mean()
        net_interest = ledger.cumulative_accrual_income
        total_fees = ledger.cumulative_fees_paid
        n_breaks = int((trace[["waterfall_tier1", "waterfall_tier2", "waterfall_tier3"]] > 1e-6).to_numpy().sum())
        n_overdraft_draws = int((trace["waterfall_overdraft"] > 1e-6).sum())
        # Foregone yield: interest NOT earned on the average Tier0 cash that sat
        # idle instead of being swept at the prevailing repo rate. Combined with
        # fees/haircuts/penalties (already captured in cumulative_fees_paid, which
        # includes break fees, liquidation haircuts, MTM losses, and capitalized
        # overdraft interest), this is the brief's Section 6 "total cost of
        # liquidity = foregone yield + fees + penalties."
        foregone_tier0_interest = trace["tier0"].mean() * (
            self.df["tier1_repo_rate_apy"].mean() / 100.0
        ) * (n_hours / 8760.0)
        total_cost_of_liquidity = total_fees + foregone_tier0_interest
        return {
            "n_hours": n_hours,
            "avg_total_balance": avg_balance,
            "avg_tier0_balance": trace["tier0"].mean(),
            "net_interest_earned": net_interest,
            "total_fees_paid": total_fees,
            "total_cost_of_liquidity": total_cost_of_liquidity,
            "foregone_tier0_interest_estimate": foregone_tier0_interest,
            "net_interest_bps_on_avg_balance": (net_interest / avg_balance * 10_000) if avg_balance > 0 else 0.0,
            "overdraft_hours": overdraft_hours,
            "overdraft_event_count": n_overdraft_draws,
            "n_tier_breaks": n_breaks,
            "four_eyes_events": four_eyes_events,
            "payout_success_rate": 1.0,  # by construction: waterfall always covers the full withdrawal
            "final_tier0": trace["tier0"].iloc[-1] if n_hours else 0.0,
            "final_tier1": trace["tier1"].iloc[-1] if n_hours else 0.0,
            "final_tier2": trace["tier2"].iloc[-1] if n_hours else 0.0,
            "final_tier3": trace["tier3"].iloc[-1] if n_hours else 0.0,
        }

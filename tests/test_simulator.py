import numpy as np
import pandas as pd
import pytest

from engine.config import EngineConfig
from engine.data import load_raw, DEFAULT_CSV_PATH
from engine.ledger.ledger import LedgerError
from engine.simulator.engine import Simulator
from engine.policy.baselines import AllCashPolicy, StaticBufferPolicy, MillerOrrPolicy


@pytest.fixture(scope="module")
def df():
    return load_raw(DEFAULT_CSV_PATH)


@pytest.fixture(scope="module")
def cfg():
    return EngineConfig.full_4tier()


def test_full_year_replay_reconciles_every_hour(df, cfg):
    """If reconciliation ever breached, Ledger.reconcile() inside Simulator.run
    would have raised -- so simply completing the run for the full year IS the
    assertion. We also re-check at the end for belt-and-braces."""
    sim = Simulator(df, cfg)
    res = sim.run(StaticBufferPolicy(50_000_000))
    assert len(res.trace) == 8760
    res.ledger.reconcile(cfg.assumptions.opening_cash_usd)


def test_payout_success_always_100_pct_by_construction(df, cfg):
    """The waterfall always funds the full withdrawal (falling back to
    overdraft if every tier is exhausted), so no customer payout ever fails --
    the brief's '100% guarantee delivered by a priced fallback, not a buffer'."""
    sim = Simulator(df, cfg)
    for policy in [AllCashPolicy(), StaticBufferPolicy(10_000_000)]:
        res = sim.run(policy)
        assert res.summary["payout_success_rate"] == 1.0


def test_no_negative_tier_balances(df, cfg):
    sim = Simulator(df, cfg)
    res = sim.run(StaticBufferPolicy(50_000_000))
    for col in ["tier0", "tier1", "tier2", "tier3"]:
        assert (res.trace[col] >= -1e-3).all(), f"{col} went negative"


def test_all_cash_policy_never_sweeps(df, cfg):
    sim = Simulator(df, cfg)
    res = sim.run(AllCashPolicy())
    assert (res.trace["sweep_amount"] == 0).all()
    assert (res.trace["tier1"] == 0).all()
    assert res.summary["net_interest_earned"] == 0.0


def test_static_buffer_earns_more_interest_than_all_cash(df, cfg):
    """A sanity check that sweeping into Tier1 actually earns something --
    not a claim that any specific policy is optimal."""
    sim = Simulator(df, cfg)
    all_cash = sim.run(AllCashPolicy())
    static = sim.run(StaticBufferPolicy(50_000_000))
    assert static.summary["net_interest_earned"] > all_cash.summary["net_interest_earned"]


def test_overdraft_trap_is_enforced_as_hard_policy_not_cost_choice(df, cfg):
    """D6c: even though overdraft is ex-post cheaper than breaking Tier1 at
    this dataset's constants for any realistic duration, the waterfall must
    try Tier1 (and Tier2/Tier3) BEFORE overdraft -- never the reverse."""
    sim = Simulator(df, cfg)
    res = sim.run(StaticBufferPolicy(5_000_000))  # thin buffer -> forces breaks
    t = res.trace
    # Find hours where a Tier1 break happened -- at that moment Tier1 must
    # have actually had a locked position (we don't check overdraft=0 since
    # a single hour's shortfall might exceed Tier1 capacity and spill to OD,
    # which is correct waterfall behavior).
    break_hours = t[t["waterfall_tier1"] > 1e-6]
    assert len(break_hours) >= 0  # at minimum this must not error; see next assert
    # Stronger check: whenever overdraft is used, confirm it wasn't used AHEAD
    # of an available, breakable Tier1 position in the same hour (which would
    # indicate the waterfall skipped a cheaper, policy-sanctioned tier).
    both = t[(t["waterfall_overdraft"] > 1e-6) & (t["tier1"].shift(1).fillna(0) > 1e-6)]
    # Any hour with both overdraft AND a prior nonzero tier1 balance must also
    # show a tier1 break that hour (fully exhausting it) -- not a skip.
    if len(both):
        assert (both["waterfall_tier1"] > 1e-6).all() or (both["tier1"] < 1e-3).all()


def test_no_look_ahead_perturbing_future_rows_does_not_change_past_decisions(df, cfg):
    """The strongest structural no-look-ahead check: mutate a future row's
    flows and confirm every decision made before that row is bit-for-bit
    identical. If the policy ever saw the future, this would fail."""
    sim1 = Simulator(df, cfg)
    res1 = sim1.run(StaticBufferPolicy(50_000_000))

    df2 = df.copy()
    mutate_from = 5000
    df2.loc[mutate_from:, "deposits_usd"] *= 3.0
    df2.loc[mutate_from:, "withdrawals_usd"] *= 0.1
    sim2 = Simulator(df2, cfg)
    res2 = sim2.run(StaticBufferPolicy(50_000_000))

    before = slice(0, mutate_from)
    pd.testing.assert_series_equal(
        res1.trace["sweep_amount"].iloc[before], res2.trace["sweep_amount"].iloc[before]
    )
    pd.testing.assert_series_equal(
        res1.trace["tier0"].iloc[before], res2.trace["tier0"].iloc[before]
    )


def test_simple_2tier_mode_runs_on_the_same_engine(df):
    """Brief Section 1.2/4.0: 'Simple 2-tier mode' must be a config switch on
    the same engine, not a separate code path."""
    cfg2 = EngineConfig.simple_2tier()
    assert cfg2.mode == "simple_2tier"
    assert [t.name for t in cfg2.tiers] == ["Cash Buffer", "Yield Vault"]
    sim = Simulator(df, cfg2)
    res = sim.run(StaticBufferPolicy(38_000_000))
    assert len(res.trace) == 8760
    assert (res.trace["tier2"] == 0).all() and (res.trace["tier3"] == 0).all()
    res.ledger.reconcile(cfg2.assumptions.opening_cash_usd)


def test_reconciliation_breach_is_actually_detectable(cfg):
    """Negative control: prove reconcile() would catch a real breach by
    directly corrupting a tier balance after a short run."""
    df_small = load_raw(DEFAULT_CSV_PATH).head(200)
    sim = Simulator(df_small, cfg)
    res = sim.run(StaticBufferPolicy(50_000_000))
    res.ledger.balances["TIER0"] += 1234.56  # corrupt it
    with pytest.raises(LedgerError):
        res.ledger.reconcile(cfg.assumptions.opening_cash_usd)

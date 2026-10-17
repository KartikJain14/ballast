import pandas as pd

from engine.config import EngineConfig
from engine.data import load_raw, DEFAULT_CSV_PATH
from engine.policy.ballast_policy import BallastPolicy
from engine.stress.scenarios import payday_surge, holiday_withdrawal_surge, svb_calibrated_run, bond_yield_shock
from engine.stress.survival import survival_horizon


def test_payday_surge_only_affects_window():
    df = load_raw(DEFAULT_CSV_PATH)
    start = pd.Timestamp("2026-06-01 09:00:00")
    shocked = payday_surge(df, start, multiplier=2.0, duration_hours=24)
    mask = (df["timestamp"] >= start) & (df["timestamp"] < start + pd.Timedelta(hours=24))
    assert (shocked.loc[mask, "deposits_usd"].values == df.loc[mask, "deposits_usd"].values * 2.0).all()
    assert (shocked.loc[~mask, "deposits_usd"].values == df.loc[~mask, "deposits_usd"].values).all()


def test_svb_run_removes_real_deposit_base_fraction():
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    start = pd.Timestamp("2026-06-15 09:00:00")
    shocked = svb_calibrated_run(df, start, cfg.assumptions.deposit_base_usd, day1_frac=0.25)
    extra = (shocked["withdrawals_usd"] - df["withdrawals_usd"]).sum()
    assert extra > 0.9 * cfg.assumptions.deposit_base_usd * 0.25  # day1 + doubled day2 queue


def test_bond_shock_moves_price_via_duration():
    df = load_raw(DEFAULT_CSV_PATH)
    start = pd.Timestamp("2026-06-15 09:00:00")
    shocked = bond_yield_shock(df, start, delta_yield_bps=200.0)
    row = shocked[shocked["timestamp"] == start].iloc[0]
    orig_row = df[df["timestamp"] == start].iloc[0]
    assert row["tier2_bond_yield_pct"] == orig_row["tier2_bond_yield_pct"] + 2.0
    assert row["tier2_bond_price_index"] < orig_row["tier2_bond_price_index"]  # price falls as yield rises


def test_unshocked_year_survives_at_1b_cap():
    """The chosen Stress Lab overdraft cap must let the UNSHOCKED year survive,
    so any default in a scenario is attributable to the injected shock."""
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    result = survival_horizon(df, cfg, BallastPolicy(), overdraft_line_cap_usd=1_000_000_000.0)
    assert result.survived


def test_capped_overdraft_can_produce_a_default():
    """Negative control: confirm the cap mechanism actually works (a small
    enough cap WILL cause a default), so the 'survives at $1B' result above
    is meaningful and not just an unreachable code path."""
    df = load_raw(DEFAULT_CSV_PATH)
    cfg = EngineConfig.full_4tier()
    result = survival_horizon(df, cfg, BallastPolicy(), overdraft_line_cap_usd=50_000_000.0)
    assert not result.survived
    assert result.survival_hours is not None

import pytest

from engine.config import AccrualConvention, HaircutBase
from engine.simulator.tiers import (
    repo_overnight_interest, repo_early_break_cost, break_vs_overdraft_effective_apr,
    bond_price_return, overdraft_cost,
)


def test_break_fee_effective_apr_matches_brief_section2():
    # 0.5% flat break fee over a 15h overnight window ~= 292% APR (brief point 6)
    apr_15h = break_vs_overdraft_effective_apr(0.5, 15)
    assert apr_15h == pytest.approx(292.0, abs=0.5)
    # over 1 hour ~= 4380% APR
    apr_1h = break_vs_overdraft_effective_apr(0.5, 1)
    assert apr_1h == pytest.approx(4380.0, abs=1.0)


def test_repo_accrual_conventions_differ_by_known_factor():
    amount = 1_000_000.0
    rate = 4.3  # pct
    pro_rata = repo_overnight_interest(amount, rate, 15, AccrualConvention.PRO_RATA_15H)
    act360 = repo_overnight_interest(amount, rate, 15, AccrualConvention.ACT_360_ONE_DAY)
    # act360 treats any overnight placement as "one day" = 24h equivalent at 360-day count
    # ratio should be close to (1/360) / (15/8760) = 1.62333...
    ratio = act360 / pro_rata
    assert ratio == pytest.approx(8760 / (360 * 15), rel=1e-6)


def test_repo_early_break_cost_amount_vs_whole_position():
    cost_amt = repo_early_break_cost(100_000, 0.5, HaircutBase.AMOUNT_WITHDRAWN)
    assert cost_amt == pytest.approx(500.0)
    cost_whole = repo_early_break_cost(100_000, 0.5, HaircutBase.WHOLE_POSITION, whole_position=1_000_000)
    assert cost_whole == pytest.approx(5000.0)


def test_bond_price_return_duration_approx():
    # +100bp move with D_mod=1.9 -> price change ~ -1.9%
    ret = bond_price_return(mod_duration=1.9, delta_yield_pct=1.0)
    assert ret == pytest.approx(-0.019, abs=1e-6)
    ret2 = bond_price_return(mod_duration=1.9, delta_yield_pct=2.0)
    assert ret2 == pytest.approx(-0.038, abs=1e-6)


def test_overdraft_cost_scales_with_hours():
    c1 = overdraft_cost(1_000_000, 12.0, 1)
    c2 = overdraft_cost(1_000_000, 12.0, 2)
    assert c2 == pytest.approx(2 * c1)

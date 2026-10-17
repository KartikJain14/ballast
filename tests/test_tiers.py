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


def test_overdraft_trap_is_real_at_dataset_parameters():
    """THE central reason the brief demands overdraft be a hard policy
    constraint rather than a cost term: at this dataset's constant parameters
    (12% APR overdraft, 0.5/2.0/15.0% flat break fees), overdraft is cheaper
    than breaking ANY tier for any realistic (<365h) shortfall duration,
    including every overnight (15h) gap. A pure cost-minimizing optimizer
    given these numbers would never break a tier and would always draw
    overdraft -- exactly the degenerate policy brief Section 2 point 6 warns
    against. Verified here so this isn't an assertion without evidence."""
    overnight_hours = 15
    od_cost_per_dollar = 0.12 * overnight_hours / 8760  # ~0.0205%
    repo_break_pct = 0.5 / 100
    bond_break_pct = 2.0 / 100
    equity_break_pct = 15.0 / 100
    assert od_cost_per_dollar < repo_break_pct
    assert od_cost_per_dollar < bond_break_pct
    assert od_cost_per_dollar < equity_break_pct
    # repo breakeven is exactly 365h (15.2 days) at these dataset constants
    breakeven_hours = (0.5 / 12.0) * 8760
    assert breakeven_hours == pytest.approx(365.0)

import pytest
from hypothesis import given, strategies as st, settings

from engine.ledger.ledger import Ledger, LedgerError


def test_deposit_increases_tier0_and_reconciles():
    led = Ledger(tier_names=["TIER0", "TIER1"])
    led.post(0, "DEPOSIT", "TIER0", "EXTERNAL", 1000.0)
    assert led.balances["TIER0"] == 1000.0
    led.reconcile(opening_cash=0.0)


def test_withdrawal_decreases_tier0_and_reconciles():
    led = Ledger(tier_names=["TIER0", "TIER1"])
    led.post(0, "DEPOSIT", "TIER0", "EXTERNAL", 1000.0)
    led.post(1, "WITHDRAWAL", "EXTERNAL", "TIER0", 400.0)
    assert led.balances["TIER0"] == 600.0
    led.reconcile(opening_cash=0.0)


def test_sweep_moves_money_between_tiers_net_zero_externally():
    led = Ledger(tier_names=["TIER0", "TIER1"])
    led.post(0, "DEPOSIT", "TIER0", "EXTERNAL", 1000.0)
    led.post(0, "SWEEP", "TIER1", "TIER0", 600.0)
    assert led.balances["TIER0"] == 400.0
    assert led.balances["TIER1"] == 600.0
    assert led.total_tier_balance() == 1000.0
    led.reconcile(opening_cash=0.0)


def test_overdraft_draw_is_recorded_and_tracked_separately():
    led = Ledger(tier_names=["TIER0"])
    led.post(0, "WITHDRAWAL", "EXTERNAL", "TIER0", 100.0)  # would go negative without OD
    led.post(0, "OVERDRAFT_DRAW", "TIER0", "OVERDRAFT_FACILITY", 100.0)
    assert led.balances["TIER0"] == 0.0
    assert led.overdraft_drawn == 100.0
    led.reconcile(opening_cash=0.0)


def test_accrual_income_is_value_created_from_outside_the_tier_system():
    led = Ledger(tier_names=["TIER0", "TIER1"])
    led.post(0, "DEPOSIT", "TIER0", "EXTERNAL", 600.0)
    led.post(0, "SWEEP", "TIER1", "TIER0", 600.0)
    led.post(1, "ACCRUAL", "TIER1", "ACCRUAL_INCOME", 5.0, memo="overnight repo interest credited")
    assert led.balances["TIER1"] == 605.0
    assert led.cumulative_accrual_income == 5.0
    # reconciliation must account for the 5.0 of income automatically
    led.reconcile(opening_cash=0.0)


def test_negative_amount_rejected():
    led = Ledger(tier_names=["TIER0"])
    with pytest.raises(LedgerError):
        led.post(0, "WITHDRAWAL", "EXTERNAL", "TIER0", -5.0)


def test_unknown_account_rejected():
    led = Ledger(tier_names=["TIER0"])
    with pytest.raises(LedgerError):
        led.post(0, "SWEEP", "NOT_A_TIER", "TIER0", 10.0)


@settings(max_examples=200, deadline=None)
@given(
    flows=st.lists(
        st.tuples(st.sampled_from(["DEPOSIT", "WITHDRAWAL"]), st.floats(min_value=0, max_value=1e6, allow_nan=False)),
        min_size=1, max_size=50,
    )
)
def test_money_conservation_under_random_deposit_withdrawal_sequences(flows):
    """Property: after any sequence of deposits/withdrawals (allowing overdraft
    to absorb any shortfall), total tier balance + (-overdraft_drawn) always equals
    cumulative external net flow. No money is created or destroyed."""
    led = Ledger(tier_names=["TIER0"])
    for i, (kind, amt) in enumerate(flows):
        if kind == "DEPOSIT":
            led.post(i, "DEPOSIT", "TIER0", "EXTERNAL", amt)
        else:
            available = led.balances["TIER0"]
            if amt <= available:
                led.post(i, "WITHDRAWAL", "EXTERNAL", "TIER0", amt)
            else:
                # pay what we can from cash, draw overdraft for the rest -- never let
                # the tier balance go negative without a recorded overdraft draw.
                shortfall = amt - available
                if available > 0:
                    led.post(i, "WITHDRAWAL", "EXTERNAL", "TIER0", available)
                led.post(i, "OVERDRAFT_DRAW", "TIER0", "OVERDRAFT_FACILITY", shortfall)
                led.post(i, "WITHDRAWAL", "EXTERNAL", "TIER0", shortfall)
        assert led.balances["TIER0"] >= -1e-6, "tier balance went negative without a recorded overdraft draw"
    led.reconcile(opening_cash=0.0)

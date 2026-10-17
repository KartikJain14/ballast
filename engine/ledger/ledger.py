"""
Double-entry ledger with an hourly reconciliation invariant.

Every movement of money is recorded as a journal entry between two accounts.
Tier accounts (TIER0, TIER1, ...) hold real balances. EXTERNAL and
OVERDRAFT_FACILITY are virtual counterparty accounts (customers, and the
emergency credit line) that are not part of "money we hold" but whose
cumulative flow lets us reconstruct the expected total independently of the
tier balances -- that independent reconstruction is what `reconcile()` checks.
This mirrors exactly the invariant whose violation sank Synapse/Evolve
(ledger vs. bank-held-funds mismatch).
"""
from __future__ import annotations

from dataclasses import dataclass, field

VIRTUAL_ACCOUNTS = {"EXTERNAL", "OVERDRAFT_FACILITY", "ACCRUAL_INCOME", "FEE_SINK"}

FEE_KINDS = {"FEE", "BREAK_FEE", "HAIRCUT", "MTM_LOSS", "OVERDRAFT_INTEREST"}


class LedgerError(RuntimeError):
    pass


@dataclass
class JournalEntry:
    t_idx: int
    kind: str              # "DEPOSIT", "WITHDRAWAL", "SWEEP", "BREAK_RELEASE", "BREAK_FEE", "OVERDRAFT_DRAW",
                            # "OVERDRAFT_REPAY", "OVERDRAFT_INTEREST", "ACCRUAL", "MTM_GAIN", "MTM_LOSS", "FEE"
    debit_account: str      # account that increases
    credit_account: str     # account that decreases
    amount: float
    memo: str = ""


@dataclass
class Ledger:
    tier_names: list[str]
    balances: dict[str, float] = field(default_factory=dict)
    overdraft_drawn: float = 0.0
    cumulative_external_net: float = 0.0  # cumulative (deposits - withdrawals) ever posted
    cumulative_fees_paid: float = 0.0     # cumulative value destroyed by fees/haircuts/penalties
    cumulative_accrual_income: float = 0.0  # cumulative interest/yield credited from outside the tier system
    entries: list[JournalEntry] = field(default_factory=list)

    def __post_init__(self):
        for name in self.tier_names:
            self.balances.setdefault(name, 0.0)

    def post(self, t_idx: int, kind: str, debit_account: str, credit_account: str,
              amount: float, memo: str = "") -> None:
        """Debit increases debit_account's balance; credit decreases credit_account's.
        For virtual accounts (EXTERNAL, OVERDRAFT_FACILITY) we don't track a balance,
        but we do track the cumulative flow needed for independent reconciliation."""
        if amount < -1e-6:
            raise LedgerError(f"Negative journal amount {amount} ({kind})")
        amount = max(amount, 0.0)
        self.entries.append(JournalEntry(t_idx, kind, debit_account, credit_account, amount, memo))

        if debit_account not in VIRTUAL_ACCOUNTS:
            if debit_account not in self.balances:
                raise LedgerError(f"Unknown account {debit_account}")
            self.balances[debit_account] += amount
        if credit_account not in VIRTUAL_ACCOUNTS:
            if credit_account not in self.balances:
                raise LedgerError(f"Unknown account {credit_account}")
            self.balances[credit_account] -= amount

        # External account credited (money leaves the bank to a customer, i.e.
        # a withdrawal) or debited (a deposit arrives) adjusts cumulative_external_net.
        if credit_account == "EXTERNAL":
            self.cumulative_external_net += amount   # a deposit: external gives money
        if debit_account == "EXTERNAL":
            self.cumulative_external_net -= amount   # a withdrawal: external takes money

        if credit_account == "OVERDRAFT_FACILITY":
            self.overdraft_drawn += amount
        if debit_account == "OVERDRAFT_FACILITY":
            self.overdraft_drawn -= amount

        if credit_account == "ACCRUAL_INCOME":
            self.cumulative_accrual_income += amount

        if kind in FEE_KINDS:
            self.cumulative_fees_paid += amount

    def total_tier_balance(self) -> float:
        return sum(self.balances.values())

    def net_position(self) -> float:
        """Total tier balances minus overdraft drawn -- the bank's net liquidity position."""
        return self.total_tier_balance() - self.overdraft_drawn

    def expected_total(self, opening_cash: float) -> float:
        """Independent reconstruction of what the tier total *should* be, built only
        from cumulative external flow, accrual income, fees paid, and overdraft drawn --
        with zero reference to the tier balances themselves."""
        return (opening_cash + self.cumulative_external_net + self.cumulative_accrual_income
                - self.cumulative_fees_paid + self.overdraft_drawn)

    def reconcile(self, opening_cash: float, tol: float = 1e-3) -> bool:
        actual = self.total_tier_balance()
        expected = self.expected_total(opening_cash)
        if abs(actual - expected) > tol:
            raise LedgerError(
                f"RECONCILIATION BREACH: ledger total {actual:,.6f} != expected {expected:,.6f} "
                f"(diff {actual - expected:,.6f})"
            )
        return True

    def snapshot(self) -> dict:
        return {
            "balances": dict(self.balances),
            "overdraft_drawn": self.overdraft_drawn,
            "total_tier_balance": self.total_tier_balance(),
            "net_position": self.net_position(),
            "cumulative_fees_paid": self.cumulative_fees_paid,
        }

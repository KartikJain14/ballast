"""
Central configuration for Ballast.

Every spec ambiguity called out in the brief (Section 3 / Section 9) is resolved
here as a *named parameter with a default*, not buried in code. The dashboard's
Settings tab renders this object so every assumption is visible on screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Literal


class AccrualConvention(str, Enum):
    """How overnight repo interest is accrued over the 17:00->08:00 window.

    PRO_RATA_15H: rate * (15/8760) -- interest accrues for the actual 15 wall-clock
        hours the cash is locked up. This is the economically "correct" convention
        for a true overnight sweep that happens to span 15 hours.
    ACT_360_ONE_DAY: rate * (1/360) -- the money-market Act/360 convention that
        treats any overnight placement as "one day" regardless of actual hours.
        This is how real O/N repo and O/N deposits are conventionally quoted and
        booked, including over weekends (3-day carry on a Friday placement).
    The two conventions differ by a factor of 360/(15*24) = 1.6x on interest
    earned per sweep. We default to PRO_RATA_15H (conservative: it assigns less
    income to the 15h window than ACT_360 would) and report both.
    """
    PRO_RATA_15H = "pro_rata_15h"
    ACT_360_ONE_DAY = "act_360_one_day"


class HaircutBase(str, Enum):
    """What a liquidation haircut / early-break fee is charged against.

    AMOUNT_WITHDRAWN: fee = haircut_pct * (dollars actually pulled back).
    WHOLE_POSITION: fee = haircut_pct * (entire position in that tier), even if
        only part of it was needed -- i.e. breaking a repo ticket or liquidating
        a bond lot is "all or nothing" operationally.
    We default to AMOUNT_WITHDRAWN (more favorable to the institution, and the
    dataset's haircut columns read naturally as per-dollar rates), and flag this
    as a conservative-vs-aggressive choice in METHODOLOGY.md.
    """
    AMOUNT_WITHDRAWN = "amount_withdrawn"
    WHOLE_POSITION = "whole_position"


class BondIndexConvention(str, Enum):
    """Whether tier2_bond_price_index is a price-only index or a total-return index.

    PRICE_ONLY: the index tracks clean price only; yield/coupon accrual must be
        added separately when valuing a bond position. This matches
        `dP/P ~= -D_mod * dy` holding exactly against the index level.
    TOTAL_RETURN: the index already embeds coupon reinvestment; adding yield
        accrual on top would double-count carry.
    Determined empirically in the data audit (Section 3.11) by checking whether
    day-to-day index changes are explained by -D_mod*dy alone (price-only) or
    drift upward beyond that (total-return).
    """
    PRICE_ONLY = "price_only"
    TOTAL_RETURN = "total_return"


@dataclass
class TierConfig:
    """One liquidity tier as a configuration record (not a code path)."""
    tier_id: int
    name: str
    base_apy: float  # used only when the CSV has no floating-rate column for this tier
    liquid_intraday: bool  # can be pulled same-hour with zero notice/fee
    entry_hour: int | None  # hour-of-day at which money can enter this tier (None = anytime)
    maturity_hour: int | None  # hour-of-day at which this tier releases cash (None = anytime)
    early_break_fee_col: str | None  # CSV column name for the time-varying penalty, if any
    uses_duration_risk: bool = False
    dv01_cap_frac: float | None = None  # max fraction of core balance exposed to this tier
    cap_frac: float | None = None  # hard allocation cap (e.g. equities strategic sleeve)


# ----- The two deployment modes required by the brief (Section 1.2 / 4.0) -----

SIMPLE_2TIER_TIERS: list[TierConfig] = [
    TierConfig(tier_id=0, name="Cash Buffer", base_apy=0.0, liquid_intraday=True,
               entry_hour=None, maturity_hour=None, early_break_fee_col=None),
    TierConfig(tier_id=1, name="Yield Vault", base_apy=0.05, liquid_intraday=False,
               entry_hour=17, maturity_hour=8, early_break_fee_col="tier1_early_redemption_haircut_pct"),
]

FULL_4TIER_TIERS: list[TierConfig] = [
    TierConfig(tier_id=0, name="Cash Buffer (Tier 0)", base_apy=0.0, liquid_intraday=True,
               entry_hour=None, maturity_hour=None, early_break_fee_col=None),
    TierConfig(tier_id=1, name="Overnight Repo (Tier 1)", base_apy=0.043, liquid_intraday=False,
               entry_hour=17, maturity_hour=8, early_break_fee_col="tier1_early_redemption_haircut_pct"),
    TierConfig(tier_id=2, name="2Y Sovereign Bonds (Tier 2)", base_apy=0.048, liquid_intraday=True,
               entry_hour=None, maturity_hour=None, early_break_fee_col="tier2_bond_liquidation_haircut_pct",
               uses_duration_risk=True, dv01_cap_frac=0.5),
    TierConfig(tier_id=3, name="Equity Index (Tier 3)", base_apy=0.10, liquid_intraday=True,
               entry_hour=None, maturity_hour=None, early_break_fee_col="tier3_stock_liquidation_haircut_pct",
               cap_frac=0.10),
]

DeploymentMode = Literal["simple_2tier", "full_4tier"]


@dataclass
class RiskPolicyConfig:
    """Treasurer-facing risk appetite knobs. Exposed as the dashboard slider."""
    q_policy: float = 0.995          # minimum overnight coverage quantile (the "slider")
    cvar_beta: float = 0.975         # CVaR confidence level for the second line of defense
    overdraft_hard_block: bool = True  # if True, overdraft is a constraint, not just a cost
    # Hard limit: max frac of EOD cash swept into Tier1. NOT meant to be the
    # thing that actually sizes the buffer -- that's the newsvendor quantile
    # (engine/risk/newsvendor.py). It exists purely as a sanity backstop
    # against a broken forecaster recommending an absurd sweep. At this
    # dataset's scale ($2B capital vs. $10-47M of nightly path-max risk --
    # DATA_AUDIT.md Section 11), the risk-correct buffer is routinely <5% (and
    # in the first weeks of the year, when cash is largest relative to M,
    # <2%) of cash. We first set this to 0.60 (caught: forced ~40% of cash
    # idle every night, total_cost_of_liquidity swamped by the cap, not the
    # newsvendor logic). Raised to 0.98 -- still found binding on 200/364
    # nights when regret-decomposing the Oracle against ITSELF (which should
    # be exactly zero; it showed ~$239K of spurious "regret" purely from the
    # 2% floor forcing extra buffer early in the year). Raised to 0.999: a
    # true sanity backstop (bars literally sweeping 100%) without distorting
    # the economics at this capital scale. See docs/DECISIONS.md D10.
    max_single_sweep_frac: float = 0.999
    # Hard limit: max frac of cash eligible for the Tier2 bond sleeve. Default
    # is 0.0 -- Ballast does NOT use bonds by default. We built the sleeve
    # (engine/policy/ballast_policy.py::decide_bond_target_usd) and tested it
    # at dv01_cap in {0.05, 0.10, 0.25, 0.50}: net profit was HIGHEST at 0.0
    # and fell monotonically as allocation grew (realized bond-yield volatility
    # this year, 3.76-5.92%, generates mark-to-market losses that exceed the
    # ~0.45%/yr carry pickup over repo -- the brief's own Section 2 point 7
    # flags bonds as "barely clear the bar"; our empirical test agrees they
    # don't clear it at all for this realized path). The sleeve is kept fully
    # functional and configurable (Settings tab / Stress Lab) so a treasurer
    # can opt in and see the live tradeoff, rather than silently deleting a
    # brief-mandated capability -- see docs/DECISIONS.md.
    bond_dv01_cap_frac: float = 0.0
    equity_cap_frac: float = 0.10        # hard limit: max frac of strategic surplus in Tier 3
    four_eyes_threshold_usd: float = 2_000_000.0  # sweeps above this need a second approval
    # None = unlimited (the main scoreboard's assumption: the emergency credit
    # line always has room, so "100% payout success" is a structural property
    # of the waterfall -- brief Section 2 point 5). The Stress Lab sets this to
    # a finite value to compute a genuine survival horizon: once a shortfall
    # can't be funded even by the capped overdraft line, that is the first
    # real payment failure, and the hour it occurs IS the survival horizon.
    overdraft_line_cap_usd: float | None = None


@dataclass
class DataAssumptions:
    """Every ambiguity in Section 3/9, resolved and made visible in the UI."""
    accrual_convention: AccrualConvention = AccrualConvention.PRO_RATA_15H
    haircut_base: HaircutBase = HaircutBase.AMOUNT_WITHDRAWN
    bond_index_convention: BondIndexConvention = BondIndexConvention.PRICE_ONLY
    event_ordering: str = (
        "Decisions at the start of hour t use only rows with timestamp <= t-1h; "
        "the row for hour t (including its repo_sweep_window_open/mature flags) "
        "is realized *during* hour t and is not visible when the hour-t decision is made."
    )
    # Unknown in the dataset; parameterized, see DATA_AUDIT.md Section 16. The
    # dataset's cumulative net flow drifts from 0 to -$2.53B over the year (a
    # genuine structural finding, not noise -- 51.4% of hours are net-outflow
    # and the mean is persistently negative). $20M (one hour of gross flow)
    # is nowhere near enough capital to be solvent under any policy; it was
    # our first guess and produced a degenerate "always in overdraft"
    # simulation that said nothing about policy quality. $2B leaves the book
    # under real, visible pressure by year-end (the structural decline is a
    # genuine feature of this data, not hidden by over-capitalizing) while
    # still being large enough that policy choice -- not capital adequacy --
    # is what differentiates the scoreboard.
    opening_cash_usd: float = 2_000_000_000.0
    deposit_base_usd: float = 2_000_000_000.0  # scale used to report bps / per-$100M figures
    timezone_note: str = "Timestamps are treated as naive local wall-clock hours, no DST adjustment found in data (see DATA_AUDIT.md)."

    def as_dict(self) -> dict:
        d = asdict(self)
        d["accrual_convention"] = self.accrual_convention.value
        d["haircut_base"] = self.haircut_base.value
        d["bond_index_convention"] = self.bond_index_convention.value
        return d


@dataclass
class EngineConfig:
    mode: DeploymentMode = "full_4tier"
    tiers: list[TierConfig] = field(default_factory=lambda: list(FULL_4TIER_TIERS))
    risk: RiskPolicyConfig = field(default_factory=RiskPolicyConfig)
    assumptions: DataAssumptions = field(default_factory=DataAssumptions)
    seed: int = 42

    @staticmethod
    def simple_2tier(seed: int = 42) -> "EngineConfig":
        return EngineConfig(mode="simple_2tier", tiers=list(SIMPLE_2TIER_TIERS), seed=seed)

    @staticmethod
    def full_4tier(seed: int = 42) -> "EngineConfig":
        return EngineConfig(mode="full_4tier", tiers=list(FULL_4TIER_TIERS), seed=seed)

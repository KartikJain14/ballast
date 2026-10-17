# Roadmap

Production path, in order of actual deployment risk:

1. **Shadow mode on a real `camt.052` intraday feed.** Replace
   `HistoryView`'s CSV backing with a live adapter implementing the same
   `as_of(t)` contract; run Ballast alongside the existing process with zero
   execution authority, logging what it *would* have done.
2. **Human-approved execution.** Sweep recommendations surface to a
   treasurer for one-click approval, with the existing four-eyes threshold
   (`RiskPolicyConfig.four_eyes_threshold_usd`) already wired for anything
   above a configurable size.
3. **Autonomous execution within limits.** Remove the approval step once
   shadow-mode agreement with the treasurer's manual decisions is
   demonstrated over at least one full seasonal cycle (to see a real
   Friday/Saturday pattern, a real month-end, a real holiday).
4. **Multi-entity / multi-currency.** Per-currency `F_M`, an FX-aware
   waterfall ordering (break the cheapest currency's tier first, not
   necessarily the home-currency one).
5. **Intraday repo.** The dataset's 17:00→08:00 lock reflects the
   traditional tri-party repo unwind cycle; BNY's 2025 intraday tri-party
   pilots suggest this constraint is eroding. A configurable "intraday repo"
   tier with a shorter/continuous entry window is a natural extension —
   the tier-as-config-record architecture (`engine/config.py::TierConfig`)
   already supports adding it without a new code path.
6. **Interest on reserve balances / collateralized intraday credit.** The
   Fed's actual payment-system-risk policy (zero-cost collateralized
   daylight overdraft, 50bp uncollateralized, a penalty rate only on
   *overnight* overdraft) is richer than this build's flat 12% APR
   assumption. Modeling that properly would likely further discourage
   uncollateralized overdraft reliance and strengthen the case for the
   hard-barred waterfall ordering already in place.
7. **Full forecasting stack upgrade.** LightGBM quantile regression +
   conformal calibration (CQR/ACI) + EVT tail, explicitly deferred in v1
   because the entire addressable ceiling for forecasting improvement was
   measured at ~$600K/year (`docs/METHODOLOGY.md` Section 7) — worth
   building once the governance/stress-testing foundation is live, not
   before.
8. **Real LCR / BCBS 248 intraday liquidity monitoring tools.** Scoped out
   of this build's time budget; the ledger reconciliation and hard-limit
   framework it would sit on top of are already in place.
9. **Social-media-accelerated-run early warning.** Cookson et al. find
   Twitter attention over the prior 4 hours predicts bank-run losses at
   hourly frequency — a natural addition to the (currently simple)
   early-warning signal set once a real social-listening feed exists.

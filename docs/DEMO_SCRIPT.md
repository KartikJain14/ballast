# Demo Script (5 minutes) + Q&A Prep

Backup if live demo fails: `ui/story.html` (static, animated, opens in any
browser, no server) plus screenshots of `docs/RESULTS.md`.

## The arc

**1. Hook (30s).** Liquidity risk moves at hourly speed now, not quarterly.
SVB: ~$42B of withdrawal requests in a single day in March 2023 — roughly a
quarter of the bank's deposits (Fed OIG material-loss review).

**2. The problem, in this dataset's own numbers (30s).** A 0.5% early-break
fee on a 15-hour overnight repo loan is a **292% annualized rate**
(`break_vs_overdraft_effective_apr(0.5, 15)` — `tests/test_tiers.py` proves
this). Hold too much cash, you bleed interest; break the vault to cover a
shortfall, you pay a rate that would be criminal outside a bank.

**3. Live replay (90s).** Open `ui/app.py`'s Overview tab (or `ui/story.html`
if live demo risk is a concern). Show the scoreboard: all-cash vs. the
best-tuned static buffer vs. Ballast vs. the oracle. Point at the day-of-week
bar chart — Friday/Saturday nights carry 4-5x the risk of any other night,
found in the data, not assumed, and that's *why* Ballast conditions its
buffer on day-of-week instead of using one number.

**4. The slider (45s).** Risk Slider tab. Drag `q_policy` from 0.995 toward
0.9999. Watch the buffer grow and the cost-of-confidence curve move. State
the marginal cost of the last "nine" in dollars/year, live, from the curve.

**5. Stress (60s).** Stress Lab tab. Click the SVB-run scenario. Show the
waterfall firing tier by tier, and that even with a finite ($1B) emergency
credit line, show which scenarios survive and which don't — and be honest
that the SVB-run and holiday-surge scenarios *do* eventually cause a default,
months after the acute shock, because the permanent deposit-base reduction
compounds with this dataset's existing structural decline. That's a feature
of transparent modeling, not a embarrassing bug to hide.

**6. Scoreboard (30s).** Restate: Ballast beats the best-tuned static
baseline by a real, modest, reproducible margin in calm markets; the bigger
value is in the Stress Lab. Capture ratio vs. oracle from `docs/RESULTS.md`.

**7. Trust (15s).** Model Card tab: known limitations stated, not hidden.
Ledger tab: reconciliation held every hour, structurally enforced. Settings:
every assumption (accrual convention, haircut base, bond-index convention,
opening capital) is a named, visible parameter, not a hardcoded guess.
Simple 2-tier mode: `EngineConfig.simple_2tier()` is the *same engine*, a
config switch — `tests/test_simulator.py` runs it.

## 15 hard questions

1. **Why does your headline improvement look so small (0.14-0.7%)?**
   Because once a bank sweeps at all, most of the available value is already
   captured — the remaining band between a well-tuned static buffer and a
   theoretical perfect-foresight oracle is inherently narrow at this
   dataset's rates (a 15-hour lock at ~4%). We checked this explicitly
   (`docs/METHODOLOGY.md` Section 7) rather than inflate the number. The
   bigger value shows up under stress, where day-of-week-aware sizing
   diverges much more from a static number.

2. **Did you tune the static baseline to be weak so Ballast looks good?**
   No — we grid-searched it (`scripts/replay.py::best_static_buffer`) and
   report the *actual optimum* ($37-38M), not a round-number strawman.
   Ballast beats the searched optimum, not an easy target.

3. **Why doesn't Ballast use the bond sleeve / equities by default?**
   We built and tested the bond sleeve across 5 allocation levels; net
   profit was highest at 0% allocation and fell monotonically as allocation
   grew, because this year's realized bond-yield volatility (3.76%-5.92%)
   generates mark-to-market losses exceeding the carry pickup. Equities are
   never used for operational liquidity — real banks generally can't, and
   the brief frames them as capital-funded strategic surplus only.

4. **How do you know your buffer sizing isn't overfit to this one dataset
   year?** We condition on day-of-week, which generalizes to any year by
   construction (Friday is Friday regardless of which calendar year).
   We do NOT hardcode specific dates (holidays, paydays) from this year —
   the day-of-month payday effect and any future holiday features would be
   computed from the calendar, not memorized from 2026's specific dates.

5. **What happens above the 99% quantile — is that buffer size actually
   validated?** No, and we say so explicitly. 364 nights give an expected
   0.36 exceedances at the true 99.9th percentile — anything above ~99% is
   model-based extrapolation, not empirically checkable in-sample. See the
   Model Card.

6. **Why is overdraft unlimited in the main scoreboard — doesn't that make
   "100% payout success" meaningless?** It makes it a *structural* property
   of the waterfall under the brief's stated assumption (a priced fallback,
   not a probability-one buffer). We deliberately test the alternative — a
   finite credit line — in the Stress Lab, where real defaults do occur.

7. **Your oracle has nonzero overdraft hours — is your oracle broken?**
   No: we proved it isn't. 66 of 364 nights are genuinely capital-constrained
   (cash on hand is below that night's worst-case need even before any sweep
   decision) — the oracle has zero regret on those nights specifically
   (verified in `tests/test_oracle.py` and `tests/test_regret.py`) because
   no policy, however smart, can avoid the shortfall with the cash actually
   available.

8. **What's the single biggest thing you'd build next?** A proper LightGBM
   quantile forecaster with conformal calibration, to shrink the
   forecast-error component of the regret decomposition (currently ~99% of
   the fixable $590K gap to the oracle) — explicitly scoped out of v1 because
   the entire addressable ceiling for forecasting improvement is bounded at
   ~$600K/year, and that money was better spent on the Stress Lab and the
   bond-sleeve investigation (which, even though it came back negative, is a
   real finding a bank would want to know before deploying it).

9. **Found any bugs in your own system while building it?** Yes, and we
   document them rather than hide them: an off-by-one in the overnight
   window boundary (understated risk by ~16-22%), and twice — at 0.60 and
   again at 0.98 — a "safety" sweep-fraction cap that silently became the
   *binding* constraint instead of the risk-based sizing logic it was meant
   to backstop. The second one was caught by our own regret-decomposition
   sanity check (oracle-vs-itself should be exactly zero; it wasn't).
   `docs/DECISIONS.md`.

10. **How is the data's realism checked — is this just a toy generator?**
    We audited it rigorously (`docs/DATA_AUDIT.md`): confirmed sign
    conventions, confirmed the bond index is price-only via a 0.9998-
    correlation duration regression, found the haircut columns are constant
    (not time-varying as the brief suggested) and said so, and found a real
    structural net-outflow trend the brief doesn't mention.

11. **What's your event-ordering/no-look-ahead guarantee, concretely?**
    `HistoryView.as_of(t-1)` physically cannot return rows beyond t-1 — it's
    an `iloc` slice bound, not a discipline. `tests/test_simulator.py`
    proves it by mutating future rows and checking all prior decisions are
    bit-for-bit unchanged.

12. **Why $2B of opening capital — isn't that arbitrary?** Yes, explicitly —
    the dataset gives flows, not a balance sheet, so this is a stated
    assumption, not a hidden one. We chose it because $20M (literally one
    hour of gross flow) produced a degenerate always-overdrawn simulation
    that measured capital adequacy, not policy quality; $2B is large enough
    to measure policy quality while still leaving the book under real,
    visible pressure by year-end (the structural decline isn't hidden by
    over-capitalizing either).

13. **Does Ballast work in "Simple 2-tier mode" (the original brief's
    framing)?** Yes — `EngineConfig.simple_2tier()` is the same simulator,
    same policy code, just a different tier config list. Tested.

14. **What's the regulatory story — LCR, intraday liquidity monitoring?**
    Scoped as a roadmap item (`docs/ROADMAP.md`) given the time budget;
    the governance primitives that would feed it (ledger reconciliation,
    hard limits, four-eyes threshold tracking) are implemented now.

15. **What would you tell a real bank CRO about deploying this?** Shadow
    mode first, against the camt.052 intraday feed, with the champion/
    challenger baselines running in parallel for at least one full seasonal
    cycle before any autonomous execution — and watch the capital-adequacy
    signal (cumulative net flow trend), not just the nightly buffer, because
    this system cannot and should not try to fix a shrinking deposit base.

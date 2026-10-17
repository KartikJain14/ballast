# Decision Log

Each entry: the decision, alternatives considered, and the evidence. Newest
at the bottom within each phase.

---

## Phase 1 — Scaffold, ingest, data audit

**D1. Stack choice: Python engine + Streamlit UI (not React/FastAPI split).**
Alternatives: a Next.js + FastAPI/WebSocket frontend would look more
"enterprise," but costs a day of plumbing (auth-free local app, so the main
win is pure visual polish) that is better spent on honest backtests,
governance, and the three required innovations. Streamlit, themed carefully
(dark mode, custom CSS, Plotly charts, no default widgets left un-styled),
gets us 90% of the "Bloomberg console" feel for a fraction of the time. A
thin FastAPI layer is still built (`api/`) so the engine is consumable
outside Streamlit (and as the integration story for `camt.052`/`pain.001`
adapters), keeping the door open to swap frontends later.

**D2. `ruptures.Pelt` with a penalty search hangs on this dataset; switched to
`ruptures.Binseg` with a fixed `n_bkps`.**
Evidence: `Pelt(model="rbf").fit()` completed in 1.3s, but
`.predict(pen=5e13)` did not return within 100s (confirmed via isolated,
unbuffered, timed sub-process tests — see shell history). The rbf cost
model's penalty scale interacted badly with 8,760 candidate points and
caused pathological exploration. `Binseg(model="l2").predict(n_bkps=6)` is
O(n log n), returns in <0.5s, and gives interpretable, bounded output. This
is used in `scripts/audit.py` for both the net-flow and repo-rate
change-point checks. Lesson: *never ship an unbounded-runtime call in a
`make` target without a timeout or a bounded alternative* — `make audit`
must finish in a demo-reasonable time every single time.

**D3. Bond index convention resolved empirically, not assumed.**
The brief flags "bond-index double counting" as a trap and says to determine
price-only vs. total-return from the data. Regressing
`tier2_bond_price_index` returns against `-D_mod·Δy` gives correlation
0.9998 and a mean residual of ~1.2e-7 — a price-only index's returns are
fully explained by the duration approximation with no drift; a total-return
index would show persistent positive residual (coupon accrual not captured
by duration alone). Confirmed **price-only**. See `docs/DATA_AUDIT.md`
Section 12.

**D4. Haircut/penalty columns are constant in this dataset, contradicting the
brief's description of them as "time series." Data wins (Operating Rule #2).**
All four haircut/penalty columns have exactly one unique value across all
8,760 rows. We still read every one of them **per-row** in
`engine/simulator/tiers.py` (no code path assumes constancy), so the engine
is correct if a future dataset has genuinely time-varying haircuts — but we
report honestly that in *this* dataset there is no time-variation to
exploit, rather than manufacturing a dynamic haircut story the data doesn't
support.

**D5. Day-of-week-conditional risk, not a single pooled distribution, for the
overnight path-max `M`.**
Audit finding: `M`'s histogram is sharply bimodal — Friday-night and
Saturday-night overnight windows have ~5x the mean path-max outflow of any
other night ($25.5M vs. $4.5–6.7M), and it is not explained by day-of-month
(payday). A single newsvendor/LP buffer fit to the pooled distribution of
`M` across all nights would either over-buffer Sun–Thu (burning foregone
interest for ~250+ nights/year) or under-cover Fri/Sat. **Decision: all
risk-sizing (newsvendor quantile, EVT tail, scenario LP) is conditioned on an
`is_fri_or_sat_night` feature**, computed at minimum as two separate
empirical/EVT distributions (weekend-night vs. weeknight), with the LP/MPC
layer free to learn a continuous day-of-week effect if the feature set
supports it. This is the single most consequential modeling decision from
the data audit — see `docs/DATA_AUDIT.md` Section 11.

**D6. Net-flow identity test tolerance set to $0.01, not stricter.**
`net_flow_usd - (deposits_usd - withdrawals_usd)` has a max absolute value of
exactly $0.01 across 8,760 rows (floating point representation of the CSV's
stored decimals, not a real discrepancy). Rounding both sides to cents before
comparing in `engine/data.py::validate` avoids a flaky boundary-equality test
while still catching a *real* identity violation (anything beyond one cent of
rounding slop) as a validation error.

**D6a. Verified the 17:00->08:00 window is exactly 15h on every single night,
including Fri/Sat, before trusting the weekend finding (D5).**
Before building the simulator's maturity rule on the assumption that
`repo_sweep_window_open`/`repo_sweep_mature` always mean "17:00 today ->
08:00 tomorrow," we checked explicitly: `repo_sweep_window_open` is True at
exactly 17:00 on all 365 days (one per day-of-week, roughly evenly split 52
or 53 per dow), `repo_sweep_mature` is True at exactly 08:00 on all 365 days,
and **every** open-to-next-mature gap across the entire year is exactly 15.0
hours with zero exceptions — including Friday-open-to-Saturday-mature and
Saturday-open-to-Sunday-mature. There is **no multi-day weekend repo lock**
in this dataset's flag design (unlike the real-world Act/360 weekend-carry
convention the brief mentions in Section 3.12, which affects *interest
accrual* day-count, not the *lock duration* here). This confirms the D5
Friday/Saturday finding is genuine customer-flow signal, not a window-sizing
artifact from incorrectly splitting one long weekend lock into two slices.

**D6b. Fixed an off-by-one in the overnight window boundary that understated
`M` by ~16-22%.**
Caught via advisor review before Phase 2's simulator could encode the same
mistake in its maturity rule. The sweep executes at the instant hour 17
begins, so the hour-17 row's own flow happens *after* the money is locked —
it must be counted as at-risk. The repo matures at the instant hour 08:00
begins, so the hour-8 row's own flow happens *after* the cash is freed — it
must be excluded. The correct window is the half-open interval
`[17:00, next 08:00)`, i.e. `timestamp >= s and timestamp < s + 15h`. The
first version of `scripts/audit.py` used `(s, s+15h]` (excluded hour-17,
included hour-8) — the reverse convention. The error understated mean `M` by
16% ($10.85M → corrected $12.98M) and understated max `M` by 22% ($38.1M →
corrected $46.7M); the day-of-week weekend effect (D5) survives and in fact
sharpens — Saturday-night, not a Friday/Saturday tie, is the single worst
night ($31.7M mean vs. $27.8M for Friday). `docs/DATA_AUDIT.md` Section 11
carries the corrected numbers; `engine/simulator/engine.py`'s maturity logic
(Phase 2) uses the same half-open convention, with a unit test asserting the
boundary hours land on the correct side.

**D6c. Confirmed the overdraft trap is real at this dataset's actual
parameters, and fixed the simulator's funding waterfall order accordingly.**
With the dataset's constant values (12% overdraft APR; 0.5/2.0/15.0% flat
break fees for Tier1/2/3 — Section 3 of DATA_AUDIT.md), the breakeven
duration where a flat break fee stops being worse than pro-rata overdraft is
**exactly 365.0 hours (15.2 days) for Tier1**, 1,460h for Tier2, and 10,950h
for Tier3. Every overnight gap (15h) is far below all three thresholds, so
**overdraft at 12% APR is cheaper than breaking any tier, by roughly 24x in
the Tier1 case** (0.0205% vs. 0.5%). A pure ex-post cost-minimizer would
therefore never break a tier and would always draw overdraft — precisely the
degenerate behavior brief Section 2 point 6 warns a naive optimizer will
learn. **Decision:** the simulator's funding waterfall is cost-ordered among
Tier0→Tier1→Tier2→Tier3 breaks *before* overdraft is ever touched, as a hard
policy rule (`RiskPolicyConfig.overdraft_hard_block=True`), not by sorting on
realized ex-post cost. This is deliberately *not* the cost-minimal choice on
paper; the justification (documented on-screen, not hidden) is that real
overdraft facilities carry uninternalized non-price consequences the dataset
doesn't charge for — regulatory attention, correspondent-bank relationship
risk, potential credit-line cancellation — exactly the gap between "the
model's price" and "the real cost" that SR 11-7 governance exists to catch.
The Stress Lab includes a "naive cost-minimizer" toggle that disables this
constraint specifically to *show* the trap springing (overdraft usage
replaces all tier breaks), as a deliberate teaching moment, not a hidden
failure mode.

**D6d. Raised `opening_cash_usd`/`deposit_base_usd` from $20M to $2B after
the first simulator smoke test produced a degenerate result.**
The dataset's cumulative net flow declines nearly monotonically to -$2.53B
over the year (Section 14 of DATA_AUDIT.md). With opening capital of $20M
(one hour of gross flow -- our first literal reading of the brief's "$20M/
hour" framing as if it were a balance, not a flow), the `all_cash` baseline
spent **98.5% of all hours in overdraft** with the overdraft balance
snowballing past $2.7B. That result is a statement about capital adequacy,
not policy quality, and would have made every policy comparison in the
scoreboard meaningless (everyone drowns identically). Raised to $2B: enough
that policy choice, not capital adequacy, drives the scoreboard, while still
leaving the book under genuine pressure by year-end so the structural decline
isn't hidden. This is exactly the kind of thing Operating Rule #1 ("never
fabricate a flattering number") cuts both ways on -- don't pick a number so
large the real finding disappears, either.

**D9. Built the Tier2 bond sleeve, tested it rigorously, and found it
net-destroys value at this dataset's realized rate path -- so it's off by
default, not forced on.**
Following the brief's own barbell design (Section 2.7: core/stable balance in
bonds, DV01-capped), we implemented `decide_bond_target_usd` -- a sleeve
sized from the breakeven logic "carry accrues every untouched night, the 2%
haircut is a one-off cost only when tapped, so only the rarely-tapped TOP
slice of the buffer (above the ~80th percentile) belongs in bonds, not the
whole thing" -- and wired it into the simulator (buy-up-to-target only,
never voluntarily sell, to avoid paying the haircut for nothing). Swept the
DV01 cap across {0%, 5%, 10%, 25%, 50%} of eligible cash on the full year:
net profit (net interest minus all fees/haircuts/MTM losses) was **highest
at 0% and fell monotonically as allocation grew** ($14.70M -> $14.61M ->
$14.53M -> $14.17M -> $13.93M). The realized `tier2_bond_yield_pct` path this
year ranges 3.76%-5.92% -- real yield volatility that generates genuine
mark-to-market losses (via the confirmed price-only duration relationship,
D7a/Section 12 of DATA_AUDIT.md) exceeding the ~0.45%/yr carry pickup over
repo. This directly validates the brief's own Section 2 point 7 caution that
bonds "barely clear the bar for operational money" -- our empirical test says
they don't clear it at all for this realized path, and Operating Rule #1
(never fabricate a flattering number) means we report that rather than
forcing bonds into the default policy because the textbook says they should
pay more. `bond_dv01_cap_frac` defaults to 0.0; the sleeve stays fully
functional and configurable (Settings / Stress Lab) so a treasurer can opt in
and see the live tradeoff rather than the capability being silently removed.

**D8. Discovered that even the perfect-foresight Oracle shows overnight
overdraft on 66 of 364 nights -- not a bug, but a capital-adequacy finding
that's now central to the regret story.**
Debugging why `OraclePolicy` (which sweeps exactly `cash_available -
M_tonight_exact`, the realized path-max, and should mathematically achieve
zero breach whenever it has enough cash to post that exact buffer) still
showed ~527 hourly overnight overdraft events led to checking, for every one
of the 364 nights, whether `cash_available_at_17:00 < M_tonight`. Result:
**66 nights are genuinely capital-constrained** (Tier0 cash is less than that
night's realized worst-case need before the sweep decision is even made --
consistent with the -$2.53B structural annual decline from D6d eroding the
$2B starting capital by Q4), and on **zero** of the remaining ~298 nights did
the oracle ever breach. This is a clean, booleans-don't-lie correctness proof
of the simulator and the oracle (`tests/test_oracle.py`), and it reframes
part of the product story: **some overdraft in this dataset is a capital
problem, not a forecasting or policy problem**, and no sweep policy --
however smart -- can fix a night where the cash simply isn't there yet. This
feeds directly into the regret-decomposition innovation (Phase 8): regret vs.
the oracle must be split into forecast error / policy conservatism /
structural (08:00-17:00 idle) / friction / **capital-constrained (irreducible)**
buckets, or the product would implicitly overclaim what better forecasting
can achieve.

**D7. Ledger accrual modeled via a virtual `ACCRUAL_INCOME` account, not a
manual reconciliation parameter.**
First draft of `Ledger.reconcile()` took a `cumulative_accruals` argument the
caller had to supply by hand — brittle, and easy to drift out of sync with
what was actually posted. Redesigned so interest/yield credits are journaled
as `post(..., debit_account=<tier>, credit_account="ACCRUAL_INCOME", ...)`,
exactly like `EXTERNAL` and `OVERDRAFT_FACILITY` are already virtual
counterparty accounts. `reconcile()` now reconstructs the expected tier total
entirely from cumulative journal flow (external net, accrual income, fees
paid, overdraft drawn) with **zero reference to the tier balances
themselves** — which is the actual point of a reconciliation check (an
independent reconstruction, not a tautology).

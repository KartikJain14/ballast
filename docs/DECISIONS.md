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

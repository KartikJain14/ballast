# Ballast — The Reserve Juggler

*"The autopilot that keeps every customer payout funded and every idle dollar earning, with a regulator-ready explanation for every move."*

Ballast is a cost-aware, risk-calibrated treasury engine for a digital bank
moving ~$20M/hour in deposits and withdrawals. It decides, every hour, how
much cash to keep instantly available versus sweep into yield-bearing tiers —
sized from the data's own realized risk, not a guess — and funds every
customer payout through a priced, cost-ordered fallback waterfall rather than
pretending a finite buffer can give probability-one coverage.

**Every number in this README, the dashboard, and the docs is computed by
running the code against `data/treasury_4tier_dataset.csv`. Nothing here is
illustrative.** Reproduce any claim with the commands below.

## Quickstart

On macOS/Linux (or Windows with `make` on PATH):

```bash
make setup        # pip install -r requirements.txt
make test         # 52 tests: no-look-ahead, ledger reconciliation, oracle correctness, ...
make audit        # regenerates docs/DATA_AUDIT.md's numbers -> reports/audit_numbers.json
make replay       # full-year scoreboard for all policies -> docs/RESULTS.md
make precompute   # bakes the dashboard cache -> reports/dashboard_cache.json
make ui           # streamlit run ui/app.py  (the technical console)
make demo         # test + audit + replay + precompute + launch the UI, one command
```

**On a stock Windows install** (`make` is not on PATH by default — unlike
every Unix/macOS system), use the identical cross-platform runner instead —
every target name and meaning is the same:

```powershell
python run.py setup
python run.py test
python run.py demo        # same as `make demo`
```

Also included: `ui/story.html` — a self-contained, animated, plain-language
walkthrough of the whole system for a non-technical audience, built from the
same computed numbers (open it directly in a browser, no server needed).

## The headline result (from `docs/RESULTS.md`, regenerate with `make replay`)

| Policy | Total cost of liquidity (FY) |
|---|---|
| Never sweep (all-cash) | $41.3M |
| Best-tuned static daily cushion (grid-searched, not guessed) | $20.38M |
| **Ballast** | **$20.35M** |
| Theoretical perfect-foresight upper bound (oracle) | $19.75M |

Ballast beats the best-tuned static baseline, honestly and reproducibly, by a
**real but modest** margin in calm market conditions — we checked the
economics explicitly (`docs/DECISIONS.md`) rather than assume a bigger win.
The actual value of the day-of-week-aware sizing shows up in the Stress Lab:
see "Why the margin looks small" below.

## Architecture

```
data/                   treasury_4tier_dataset.csv (the one input)
engine/
  data.py               ingest, validation, no-look-ahead HistoryView
  config.py             every spec ambiguity as a named, visible parameter
  ledger/                double-entry ledger, hourly reconciliation invariant
  simulator/             event-driven engine, tier economics, cost-ordered waterfall
  policy/                baselines (all-cash, static, Miller-Orr, ...), Ballast, Oracle
  risk/                  newsvendor sizing, path-max, regret decomposition
  stress/                scenario generators, survival horizon, reverse stress search
scripts/
  audit.py               make audit
  replay.py              make replay
  precompute.py          make precompute
ui/
  app.py                 Streamlit technical console
  story.html             animated plain-language walkthrough
docs/                    DATA_AUDIT, METHODOLOGY, DECISIONS, MODEL_CARD, RESULTS, DEMO_SCRIPT, ROADMAP
tests/                   51 tests: property-based ledger tests, no-look-ahead mutation test,
                         oracle correctness, regret-bucket consistency, stress scenarios
```

## The three findings that shaped the design

1. **Friday and Saturday overnight windows carry ~4-5x the cash-drain risk of
   any other night** — not a payday effect, a pure weekend-spending effect.
   A single pooled buffer would either waste money five nights a week or
   under-cover the two that matter. Ballast sizes the buffer separately per
   regime. See `docs/DATA_AUDIT.md` Section 11.
2. **The overdraft trap is real at this dataset's numbers.** A flat 0.5%
   repo break fee is *cheaper* at a 12%/yr overdraft rate only beyond ~365
   hours (15 days) of continuous draw — meaning for any realistic overnight
   gap, a pure cost-minimizer would always prefer overdraft over breaking a
   tier. Ballast treats overdraft as a hard, last-resort policy constraint,
   not a cost to be optimized away. See `docs/DECISIONS.md` D6c.
3. **Even a perfect-foresight oracle can't avoid overdraft on 66 of 364
   nights** — the book is genuinely capital-constrained by Q4 (a -$2.53B
   structural annual net-outflow trend in the dataset), not a forecasting
   failure. We built a regret decomposition specifically so this doesn't get
   misattributed to "the model wasn't good enough." See `docs/DECISIONS.md` D8.

## Why the margin looks small (and where it doesn't)

Once a bank sweeps *at all*, the overwhelming majority of available value is
already captured by any competent policy (all-cash → best-static ≈
$21M/year). The remaining band between a well-tuned static buffer and a
perfect oracle is inherently narrow (~$600K/year) — that's the economics of
a 15-hour overnight lock at ~4% rates, not a weak result. Ballast closes a
real slice of that narrow band in calm conditions, and the Stress Lab shows
where the same day-of-week logic matters far more: a doubled payday, a 3x
holiday surge, and an SVB-calibrated run were all replayed against a finite
emergency credit line, with results in `docs/RESULTS.md` and the dashboard's
Stress Lab tab.

## Honesty notes

- Three innovations beyond the brief, all implemented and measured, not
  roadmap bullets: **regret decomposition** (attributes every dollar of the
  gap to the oracle into capital-constrained / policy-conservatism /
  forecast-error buckets that sum exactly to the total), **cost-of-the-last-nine
  pricing** (the dashboard's Risk Slider tab prices each additional unit of
  confidence in dollars/year), and **reverse stress search** (binary-searches
  the smallest shock multiplier that exhausts the funding waterfall).
- We built a Tier2 bond sleeve exactly as the brief's barbell design asks for,
  tested it rigorously across five allocation levels, and found it
  **net-destroys value** at this dataset's realized rate volatility — so it's
  off by default, with the mechanism fully functional and documented rather
  than silently deleted. See `docs/DECISIONS.md` D9.
- Two real calibration bugs in our own hard governance limits
  (`max_single_sweep_frac`) were caught and fixed using the system's own
  correctness checks (an oracle-vs-itself regret test that should be exactly
  zero caught the second one). See `docs/DECISIONS.md` D6 and D10.
- Full decision log, including dead ends: `docs/DECISIONS.md`.

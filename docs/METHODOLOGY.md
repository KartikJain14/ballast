# Methodology

## 1. The decision loop

Hourly, event-driven: `Ingest → (sweep decision at 17:00) → Settle realized
flows → Fund any shortfall via the waterfall → Reconcile → Record`. The
policy never sees the realized row for the hour it's deciding — only
`HistoryView.as_of(t-1)` — enforced structurally by `engine/data.py`, not by
caller discipline (see tests/test_simulator.py's mutation test: perturbing
future rows leaves all prior decisions bit-for-bit identical).

## 2. The risk variable: overnight path-max `M`

For a night starting at 17:00, the risk that matters is not the hourly flow
or the terminal net flow — it's the **running maximum of cumulative outflow**
over the 15-hour lock window:

```
M = max_{k=1..15} Σ_{h=1..k} (withdrawal_h - deposit_h)
```

computed over the half-open window `[17:00, next 08:00)` — the 17:00 hour's
own flow is already locked away when it happens (in-window); the 08:00
hour's own flow happens after maturity frees the cash (out-of-window). Get
this boundary wrong and you under- or over-state `M` by 15-20%
(`docs/DECISIONS.md` D6b). Canonical, vectorized implementation:
`engine/risk/pathmax.py::compute_nightly_path_max`.

**Day-of-week conditioning is not optional.** `M` is bimodal: Friday/Saturday
nights average ~$28-32M, every other night averages ~$5-8M
(`docs/DATA_AUDIT.md` Section 11). All risk sizing in this codebase computes
`M` separately for the two regimes.

## 3. Newsvendor sizing

```
B* = F_M^-1(q*),   q* = c_u / (c_u + c_o)
```

- `c_u` (underage cost per dollar of shortfall) = the cheapest
  **policy-sanctioned** fallback, which is the Tier1 early-break haircut
  (0.5% in this dataset). Overdraft is *excluded* from this calculation even
  though it is ex-post cheaper (see Section 5) because it is barred ahead of
  Tier1 in the waterfall by policy, not by price.
- `c_o` (overage cost per dollar of buffer held idle) = foregone repo
  interest for the 15h window: `r_repo * (15/8760)`.

At this dataset's mean rates (0.5% break fee, 4.055% repo), `q* ≈ 0.9863`,
matching the brief's stated 97.7%-98.6% range. The policy overlay
`B = F_M^-1(max(q*, q_policy))` lets the treasurer's risk-appetite slider
(`q_policy`, default 0.995) dominate when it's stricter — a deliberate
governance floor over the pure economic optimum, not a bug.
`engine/risk/newsvendor.py`.

**v1 forecaster**: empirical quantiles of the realized, no-look-ahead
history of completed nights in the same regime (expanding/bounded-lookback
window, `engine/policy/ballast_policy.py`). This is deliberately simple —
see Section 7 for why we didn't invest in a heavier forecasting stack.

## 4. Accrual and cost conventions (every ambiguity named)

| Parameter | Default | Why |
|---|---|---|
| Repo accrual convention | Pro-rata 15h | Conservative (assigns less income than Act/360); both implemented, both reportable (`AccrualConvention`) |
| Haircut base | Amount withdrawn | Matches the brief's literal wording; `HaircutBase.AMOUNT_WITHDRAWN` means fee = haircut% × cash received, face value consumed = cash + fee |
| Bond index convention | Price-only | **Empirically confirmed**, not assumed: duration-approximation regression gives corr=0.9998, mean residual ≈ 0 (`docs/DATA_AUDIT.md` Section 12) |
| Opening capital / deposit base | $2,000,000,000 | The dataset's cumulative net flow declines to -$2.53B over the year; $20M (one hour of gross flow) produced a degenerate always-overdrawn simulation on first try (`docs/DECISIONS.md` D6d) |
| `max_single_sweep_frac` | 0.999 | A sanity backstop against a broken forecaster, NOT the thing that sizes the buffer — found binding (and silently substituting for the real sizing logic) at both 0.60 and 0.98 before landing here (`docs/DECISIONS.md` D6, D10) |
| `bond_dv01_cap_frac` | 0.0 | Tested at 5 allocation levels; net profit was highest at zero (Section 6) |

## 5. The overdraft trap, quantified

```
break_vs_overdraft_effective_apr(fee, hours) = fee * (8760 / hours)
```

A 0.5% flat break fee over a 15-hour overnight gap is equivalent to a
**292% annualized rate**; over 1 hour, **4,380%**. Conversely, at this
dataset's 12% overdraft APR, the break-even duration where a flat fee stops
being worse than pro-rata overdraft is **exactly 365.0 hours** for Tier1
(1,460h for Tier2, 10,950h for Tier3) — so overdraft is cheaper than
breaking ANY tier for every realistic gap in this problem, by ~24x in the
Tier1 case. A pure ex-post cost-minimizer would therefore never break a tier
and would always draw overdraft. **Ballast's funding waterfall tries
Tier1→Tier2→Tier3 before overdraft as a hard policy rule**, justified by
real, un-priced non-price consequences of overdraft reliance (regulatory
attention, correspondent-bank relationship risk, credit-line cancellation
risk) that the dataset's flat 12% APR doesn't charge for.
`engine/simulator/tiers.py`, `tests/test_tiers.py`.

## 6. The bond sleeve: built, tested, found not worth it

Per the brief's barbell design, core/stable balance beyond the overnight
threshold should earn Tier2 carry. The breakeven logic: carry accrues every
night a dollar sits untouched; the 2% liquidation haircut is a one-off cost
paid only when tapped — so only the rarely-tapped top slice of the buffer
(above roughly its 80th percentile) should sleeve into bonds, not the whole
thing. We implemented this (`decide_bond_target_usd`), wired it into the
simulator (buy-up-to-target only; never voluntarily sell, to avoid paying
the haircut for nothing), and swept the DV01 cap across {0%, 5%, 10%, 25%,
50%} of eligible cash on the full year:

| DV01 cap | Net profit (net interest − fees) |
|---|---|
| 0% | $14.70M |
| 5% | $14.61M |
| 10% | $14.53M |
| 25% | $14.17M |
| 50% | $13.93M |

Monotonically worse as allocation grows. This year's realized
`tier2_bond_yield_pct` ranges 3.76%-5.92% — real volatility that generates
mark-to-market losses (via the confirmed duration relationship) exceeding
the ~0.45%/yr carry pickup over repo. **Default is off**; the mechanism is
fully functional and configurable (`RiskPolicyConfig.bond_dv01_cap_frac`),
because the brief asked for the capability and a treasurer should be able to
opt in and see the live tradeoff, not have it silently withheld.

## 7. Why the forecasting stack stayed light

An advisor review (and our own numbers) showed the entire economic upside
available to a better *overnight-buffer forecaster* is bounded by
`oracle - ballast ≈ $600K/year`, because the oracle already has perfect
knowledge of `M` and barely beats a well-tuned static buffer. A heavier
stack (LightGBM quantile regression, conformal calibration, EVT tails,
Hawkes processes, neural challengers) would chase a fraction of that $600K
at a large build-time cost. We therefore kept the v1 forecaster to empirical,
regime-conditioned quantiles and invested the saved time in: (a) the
regret decomposition that explains *why* the ceiling is low, (b) the Stress
Lab, where the same day-of-week logic demonstrably matters more, and (c) the
bond-sleeve investigation (Section 6), which turned out to be a bigger real
lever to test (even though it came back negative). This tradeoff is
documented, not hidden — see `docs/DECISIONS.md`.

## 8. Regret decomposition

For every night, compare the policy's realized buffer cost against the
oracle's (`engine/risk/regret.py`). Nights are first split by whether cash
available at 17:00 was already below that night's realized `M` — on those,
the oracle can't do better either, so the regret contribution is exactly
zero by construction (both are capped by available cash, not forecast
skill); we report the *absolute* cost on those nights separately, since it's
a capital-adequacy fact, not a policy-quality one. On the remaining nights,
regret is split into **policy conservatism** (the q_policy floor being
stricter than the pure economic q*) and **forecast error** (everything
else), using the gap between q_policy and q* as the split fraction. The three
buckets sum exactly to total regret — verified in `tests/test_regret.py`.

## 9. Stress Lab mechanics

Scenario generators (`engine/stress/scenarios.py`) perturb a copy of the raw
dataframe — payday ×2, holiday ×3, an SVB-calibrated run (25% of deposit
base requested in 24h, doubled queue day 2, per the Fed OIG review), a
+200bp parallel bond shock applied via the confirmed duration relationship,
a -35% equity gap, haircut doubling, and a combined crisis. Because the main
scoreboard runs with an *unlimited* overdraft line (so "100% payout success"
stays a structural property of the waterfall, not a result), the Stress Lab
introduces a **finite** `overdraft_line_cap_usd` specifically to compute a
survival horizon. The cap (**$1,000,000,000**) was chosen because the
*unshocked* year survives at exactly that level — confirmed by testing a
grid of caps and finding $500M already fails organically due to the
dataset's structural decline (Section 4 of this doc) — so any default under
a scenario is cleanly attributable to the injected shock, not pre-existing
conditions. `reverse_stress_multiplier` binary-searches the smallest shock
multiplier that causes a default within the scenario window.

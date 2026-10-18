"""
Optional AI executive summary. Narrates numbers Ballast's own engine already
computed -- it is never asked to invent a figure, only to explain ones we
hand it. Fully optional: if GEMINI_API_KEY isn't set (via a local, gitignored
.env or the real environment), /api/summarize falls back to a template-based
summary built from the same numbers, so the product never depends on a paid
API to function (Operating Rule #3: free tooling only, works offline).

The key is read from the environment ONLY -- never hardcoded in source, never
logged, never echoed back in any response.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-flash-latest"


def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _template_summary(data: dict) -> str:
    sb = data["scoreboard"]
    ballast = sb["ballast"]
    best_naive = min((n for n in sb if n not in ("oracle", "ballast")), key=lambda n: sb[n]["total_cost_of_liquidity"])
    improvement = sb[best_naive]["total_cost_of_liquidity"] - ballast["total_cost_of_liquidity"]
    n_defaults = sum(1 for r in data["stress"].values() if not r["survived"])
    return (
        f"Over the analyzed period, Ballast earned ${ballast['net_interest_earned']:,.0f} in interest while "
        f"keeping every customer payout funded. Compared to the best alternative strategy tested, it "
        f"{'saved' if improvement >= 0 else 'cost'} ${abs(improvement):,.0f} in total liquidity costs. "
        f"The data shows {data['risk']['n_nights']} overnight windows with highly uneven risk — some nights "
        f"carry far more cash-drain risk than others, which is why a single fixed cash cushion is the wrong "
        f"approach. Under stress testing, {n_defaults} of {len(data['stress'])} simulated shock scenarios "
        f"would have exhausted the funding waterfall, underscoring the importance of active liquidity management."
    )


def generate_summary(data: dict) -> dict:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"text": _template_summary(data), "source": "template"}

    sb = data["scoreboard"]
    ballast = sb["ballast"]
    best_naive = min((n for n in sb if n not in ("oracle", "ballast")), key=lambda n: sb[n]["total_cost_of_liquidity"])

    improvement = sb[best_naive]["total_cost_of_liquidity"] - ballast["total_cost_of_liquidity"]
    facts = {
        "ballast_net_interest_earned_usd": round(ballast["net_interest_earned"]),
        "ballast_total_cost_of_liquidity_usd": round(ballast["total_cost_of_liquidity"]),
        "name_of_the_best_competing_non_ballast_strategy": best_naive,
        "that_competing_strategys_total_cost_of_liquidity_usd": round(sb[best_naive]["total_cost_of_liquidity"]),
        "ballast_dollars_saved_vs_that_competing_strategy": round(improvement),
        "theoretical_perfect_foresight_best_possible_cost_usd": round(sb.get("oracle", {}).get("total_cost_of_liquidity", 0)),
        "ballast_payout_success_rate_pct": round(ballast["payout_success_rate"] * 100, 3),
        "n_nights_analyzed": data["risk"]["n_nights"],
        "riskiest_night_of_week": max(data["risk"]["by_dow"], key=lambda k: data["risk"]["by_dow"][k]["mean"]),
        "calmest_night_of_week": min(data["risk"]["by_dow"], key=lambda k: data["risk"]["by_dow"][k]["mean"]),
        "nights_where_no_strategy_could_have_avoided_a_shortfall": data["regret"]["n_capital_constrained_nights"],
        "stress_scenarios_survived_count": sum(1 for r in data["stress"].values() if r["survived"]),
        "stress_scenarios_total_count": len(data["stress"]),
    }
    prompt = (
        "You are writing a 3-4 sentence executive summary for a bank treasurer, in plain English a "
        "non-technical reader can follow. The subject is 'Ballast', our liquidity strategy, being compared "
        "against competing strategies and a theoretical perfect-foresight upper bound. Use ONLY the facts "
        "given below -- do not invent any numbers, dates, or claims beyond them, and do not confuse Ballast "
        "with the competing strategy it is being compared against. Be direct and confident, not hedging. "
        "Quote every dollar figure EXACTLY as given below -- never round, approximate, or restate a figure as a "
        "different number (e.g. if a figure is 4,712, say 4,712, never '$0' or '$5,000'). "
        "Facts (JSON):\n" + json.dumps(facts, indent=2)
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
            text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"text": text, "source": "gemini"}
    except Exception:
        return {"text": _template_summary(data), "source": "template"}

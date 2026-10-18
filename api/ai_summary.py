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
import time
import urllib.request
from pathlib import Path

import requests

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


def _facts_and_prompt(data: dict) -> tuple[dict, str]:
    sb = data["scoreboard"]
    ballast = sb["ballast"]
    best_naive = min((n for n in sb if n not in ("oracle", "ballast")), key=lambda n: sb[n]["total_cost_of_liquidity"])
    improvement = sb[best_naive]["total_cost_of_liquidity"] - ballast["total_cost_of_liquidity"]
    # Pre-resolve the sign in Python (not left for the model to phrase from a
    # signed number against an awkward key name -- that produced sentences
    # like "...resulting in Ballast dollars saved of -17009637").
    comparison_word = "saved" if improvement >= 0 else "cost extra"
    usd = lambda v: f"${round(v):,}"
    # Dollar figures go to the model pre-formatted ("$21,352,965"): given bare
    # integers it quoted them as "$21352965", which reads badly on screen.
    facts = {
        "ballast_net_interest_earned": usd(ballast["net_interest_earned"]),
        "ballast_total_cost_of_liquidity": usd(ballast["total_cost_of_liquidity"]),
        "name_of_the_best_competing_non_ballast_strategy": best_naive,
        "that_competing_strategys_total_cost_of_liquidity": usd(sb[best_naive]["total_cost_of_liquidity"]),
        "ballast_vs_competing_strategy_comparison_word": comparison_word,
        "ballast_vs_competing_strategy_dollar_amount": usd(abs(improvement)),
        "theoretical_perfect_foresight_best_possible_cost": usd(sb.get("oracle", {}).get("total_cost_of_liquidity", 0)),
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
        "different number (e.g. if a figure is \"$4,712\", write $4,712 -- keep the dollar sign and the commas exactly). "
        "For the comparison to the competing strategy, write it as a natural sentence like "
        "'Ballast {comparison_word} {amount} compared to {strategy}' using the comparison word field exactly "
        "as given (it is already correctly worded for whether this was a saving or an extra cost) -- never say "
        "a negative dollar amount, and never quote a raw field name as if it were a sentence. "
        "Facts (JSON):\n" + json.dumps(facts, indent=2)
    )
    return facts, prompt


def _payload(prompt: str) -> dict:
    # thinkingBudget: 0 disables this model's extended internal reasoning phase
    # for this call. Measured impact for this prompt: WITHOUT it, the model
    # spends >30s "thinking" before emitting a single visible token (we
    # measured a hard read-timeout with zero bytes received); WITH it, the
    # first token arrives in ~10s and the rest streams in small chunks every
    # 100-300ms. This narration task needs no multi-step reasoning, so the
    # thinking phase was pure latency with no quality benefit here.
    return {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}


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

    _, prompt = _facts_and_prompt(data)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    payload = json.dumps(_payload(prompt)).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
            text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"text": text, "source": "gemini"}
    except Exception:
        return {"text": _template_summary(data), "source": "template"}


def generate_summary_stream(data: dict):
    """Yields small text chunks as they become available, for the frontend's
    chunk-by-chunk reveal effect. Real token streaming from Gemini's SSE
    endpoint when a key is configured; a paced word-by-word replay of the
    deterministic template otherwise (so the chunked UI effect is consistent
    either way -- we never fabricate a 'live' feel for data that isn't live,
    we just don't want the two code paths to look jarringly different)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        words = _template_summary(data).split(" ")
        for i, w in enumerate(words):
            yield w + (" " if i < len(words) - 1 else "")
            time.sleep(0.035)
        yield "\n__SOURCE__:template"
        return

    _, prompt = _facts_and_prompt(data)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse&key={key}"
    payload = _payload(prompt)
    got_any = False
    try:
        with requests.post(url, json=payload, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk_json = line[len("data:"):].strip()
                if not chunk_json:
                    continue
                try:
                    obj = json.loads(chunk_json)
                    text = obj["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if text:
                    got_any = True
                    yield text
        yield "\n__SOURCE__:gemini"
        return
    except Exception:
        if got_any:
            # Already streamed real, partial Gemini text -- do NOT also append
            # the template underneath it (that would read as garbled duplicate
            # text). Just end the stream; a partial-but-real answer beats a
            # partial-real-answer-plus-unrelated-template-glued-on.
            yield "\n__SOURCE__:gemini"
            return
    # Streaming never produced anything usable -- fall back to the template,
    # still paced as chunks so the UI doesn't jump straight to a wall of text.
    words = _template_summary(data).split(" ")
    for i, w in enumerate(words):
        yield w + (" " if i < len(words) - 1 else "")
        time.sleep(0.035)
    yield "\n__SOURCE__:template"

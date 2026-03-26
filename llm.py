import json
from openai import OpenAI
from config import (
    OPENAI_API_KEY,
    NANO_MODEL,
    DEEP_MODEL,
    NANO_BATCH_MAX_HEADLINES,
    LLM_NANO_MAX_COMPLETION_TOKENS,
    LLM_NANO_BATCH_MAX_COMPLETION_TOKENS,
    LLM_DEEP_TIERS_MAX_COMPLETION_TOKENS,
    LLM_DEEP_VALIDATE_MAX_COMPLETION_TOKENS,
)
from signals import normalize_trade_direction
from logger import log

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _call(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    *,
    max_completion_tokens: int,
) -> str:
    resp = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Failed to parse LLM JSON: {raw[:200]}")
        return {}


def _clip_text(value: object, max_chars: int) -> str:
    """Trim long LLM strings for storage, logs, and follow-up prompts."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return "…"
    return text[: max_chars - 1].rstrip() + "…"


# Hard caps after parse (prompt also requests brevity)
_DEEP_CAUSAL_CHAIN_MAX_CHARS = 480
_DEEP_INVALIDATION_MAX_CHARS = 200
_DEEP_VALIDATE_EXPLANATION_MAX_CHARS = 360
_VALIDATE_REASON_CONTEXT_CHARS = 900
_VALIDATE_CAUSAL_CONTEXT_CHARS = 600


def _normalize_tier_output(parsed: dict) -> dict:
    for tier in parsed.get("tiers") or []:
        if not isinstance(tier, dict):
            continue
        tier["causal_chain"] = _clip_text(
            tier.get("causal_chain"), _DEEP_CAUSAL_CHAIN_MAX_CHARS
        )
        tier["invalidation"] = _clip_text(
            tier.get("invalidation"), _DEEP_INVALIDATION_MAX_CHARS
        )
    return parsed


def _enforce_tier_directions_post_parse(parsed: dict) -> dict:
    """Immediately after LLM parse: canonicalize direction per tier or drop tickers (fail closed)."""
    for tier in parsed.get("tiers") or []:
        if not isinstance(tier, dict):
            continue
        raw = tier.get("direction")
        tickers = [x for x in (tier.get("tickers") or []) if isinstance(x, str) and x.strip()]
        if not tickers:
            continue
        resolved = [normalize_trade_direction(raw, sym.strip()) for sym in tickers]
        if any(d is None for d in resolved):
            log.warning(
                f"Tier {tier.get('tier')}: direction not parseable for all tickers; dropping tickers"
            )
            tier["tickers"] = []
            continue
        unique = set(resolved)
        if len(unique) != 1:
            log.warning(
                f"Tier {tier.get('tier')}: conflicting directions {resolved!r}; dropping tickers"
            )
            tier["tickers"] = []
            continue
        tier["direction"] = resolved[0]
    return parsed


# ---------- NANO: headline classification ----------

NANO_SYSTEM = """You are a strict financial news classifier for an **S&P 500–only** trading book. Be skeptical: almost all headlines are noise. Your job is triage only — flag almost nothing.

**Universe:** The only stocks this desk trades are **S&P 500 constituents**. A headline is in-scope only if the clearest price channel is to **one or more S&P 500 tickers** (named in text or an obvious large-cap SPX name), or a **macro/sector** story with a direct read on major S&P 500 sectors (e.g. rates → banks/XLF-style large caps). If the story is **only** about companies **not** in the S&P 500 (small-caps, foreign-only names, uranium juniors, etc.) with no credible SPX angle → set market_impact="none", actionable=false.

1. event_type: short label (e.g. EARNINGS_BEAT, RATE_CUT, MERGER, FDA_APPROVAL, LAYOFFS, MACRO_CPI, ANALYST_NOTE, OPINION_PIECE, ROUTINE_FILING).
2. tickers_mentioned: tickers explicitly named in the headline (may be empty); prefer noting S&P 500 symbols when present.
3. sentiment: positive, negative, or neutral (for the primary named companies or market tone).
4. magnitude: "low", "medium", or "high".
   - "high" ONLY for a discrete catalyst likely to reprice affected US equities over hours to a few sessions (not vague sentiment).
   - Routine "Company X reports earnings" without a clear surprise → "low" or "medium", not "high".
5. market_impact: "none" | "moderate" | "major"
   - "none": filler, opinion, recap, no new information, duplicate story, or no clear price channel.
   - "moderate": real news but incremental, already expected, or unlikely to move liquid names much.
   - "major": rare. Only for clear repricing events, e.g. named M&A with terms, regulatory/legal outcome (FDA, antitrust, major fine), earnings or guidance SURPRISE with numbers, rated macro print with immediate policy/market channel, CEO departure + strategy pivot, bankruptcy/restructuring headline, war/sanctions with direct commodity or sector shock. NOT: analyst PT changes without new facts, generic "markets rise", soft unverified rumor.
6. impact_one_liner: If market_impact is "major", one sentence: who moves and why. Otherwise use "" (empty string).
7. actionable: true ONLY if market_impact is "major" AND magnitude is "high" AND ALL hold:
   - Concrete catalyst (not columnist opinion or vague macro take).
   - At least one specific ticker OR a very narrow sector is clearly in scope.
   - A professional could state in one sentence WHY price should move and within roughly what horizon.
   If any doubt, actionable = false, market_impact != "major", magnitude != "high".

Consistency: actionable=true requires market_impact="major" and magnitude="high". Default bias: market_impact="none", actionable=false.

Respond with valid JSON only, including every key:
{
  "event_type": "...",
  "tickers_mentioned": [],
  "sentiment": "positive|negative|neutral",
  "magnitude": "low|medium|high",
  "market_impact": "none|moderate|major",
  "impact_one_liner": "",
  "actionable": false
}"""

NANO_USER = """Classify this headline:

Title: {title}
Summary: {summary}
"""

# ---------- NANO: batch triage — top actionable headlines ----------

NANO_BATCH_SYSTEM = """You are a financial news triage model. You receive a JSON array of headlines (each with id, title, summary).

**Tradable universe — S&P 500 only:** Include a headline only if the **best trading read** is on **S&P 500 index constituents** (ticker named or clearly implied large-cap SPX name), or a **macro / policy / commodity** item with a **direct, concrete** channel to specific S&P 500 names or liquid SPX-heavy sectors. **Do not** rank stories whose **only** actionable names are **outside** the S&P 500 (e.g. small-cap miners, non-SPX foreign listings, OTC) unless the headline **explicitly** ties to a clear SPX large-cap mechanism.

**Output size:** Return **up to** M items in `top_actionable`, where M = min(5, number of **qualifying** headlines). If **no** headline qualifies, return `{"top_actionable": []}`. Ranks must be **1..M** contiguous (1 = best). Do **not** pad with non-SPX stories to reach 5.

**Ranking (among qualifying only):** Order by (1) clarity of price channel to S&P 500 names, (2) novelty vs recap, (3) plausible repricing.

**Per item** (each row):
1. headline_id: must match an input id exactly; **no duplicate ids**.
2. rank: integer 1..M.
3. event_type, tickers_mentioned, sentiment, magnitude, market_impact, impact_one_liner — honest assessment.
4. actionable: true only if market_impact="major" AND magnitude="high" AND concrete catalyst **and** SPX-relevant; false allowed for weaker but still SPX-relevant picks you rank lower.
5. profit_angle_one_liner: one short sentence (S&P 500–focused), or "".

Respond with valid JSON only:
{
  "top_actionable": [ ... ]
}"""


def classify_headline(headline: dict) -> dict:
    """Use Nano to classify a single headline. Returns structured dict."""
    user_msg = NANO_USER.format(
        title=headline["title"],
        summary=headline.get("summary", ""),
    )
    raw = _call(
        NANO_MODEL, NANO_SYSTEM, user_msg, max_completion_tokens=LLM_NANO_MAX_COMPLETION_TOKENS
    )
    result = _parse_json(raw)
    result["headline_id"] = headline["id"]
    result["headline_title"] = headline["title"]
    result.setdefault("profit_angle_one_liner", "")
    return result


def rank_headlines_nano(headlines: list[dict]) -> list[tuple[dict, dict]] | None:
    """One Nano call: return up to 5 (headline, classification) pairs, S&P 500–focused, ranked best-first.

    Returns None if the response is unusable (caller may fall back to per-headline classification).
    """
    if not headlines:
        return []
    feed = headlines[: int(NANO_BATCH_MAX_HEADLINES)]
    payload = []
    for h in feed:
        payload.append(
            {
                "id": h["id"],
                "title": _clip_text(h.get("title") or "", 280),
                "summary": _clip_text(h.get("summary") or "", 450),
            }
        )
    user_msg = (
        f"There are {len(payload)} headlines below. Return top_actionable with **up to 5** items — "
        "only stories whose **primary** trading angle is **S&P 500** constituents (see system rules). "
        "Ranks 1..M contiguous; empty array if none qualify. Use each object's \"id\" as headline_id.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    raw = _call(
        NANO_MODEL,
        NANO_BATCH_SYSTEM,
        user_msg,
        max_completion_tokens=LLM_NANO_BATCH_MAX_COMPLETION_TOKENS,
    )
    parsed = _parse_json(raw)
    items = parsed.get("top_actionable")
    if not isinstance(items, list):
        log.warning("Nano batch: missing or invalid top_actionable; try per-headline triage")
        return None

    by_id = {h["id"]: h for h in feed}
    seen: set[str] = set()
    rows: list[tuple[int, dict, dict]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        hid = item.get("headline_id") or item.get("id")
        if not hid or hid not in by_id or hid in seen:
            continue
        try:
            rk = int(item.get("rank", 99))
        except (TypeError, ValueError):
            rk = 99
        classification = {
            "event_type": item.get("event_type", "unknown"),
            "tickers_mentioned": item.get("tickers_mentioned") or [],
            "sentiment": item.get("sentiment", "neutral"),
            "magnitude": item.get("magnitude", "low"),
            "market_impact": item.get("market_impact", "none"),
            "impact_one_liner": item.get("impact_one_liner") or "",
            "actionable": bool(item.get("actionable")),
            "profit_angle_one_liner": item.get("profit_angle_one_liner") or "",
            "headline_id": hid,
            "headline_title": by_id[hid]["title"],
            "nano_batch_rank": rk,
        }
        seen.add(hid)
        rows.append((rk, by_id[hid], classification))

    rows.sort(key=lambda x: x[0])
    out = [(h, c) for _, h, c in rows[:5]]
    if not out:
        log.info("Nano batch: no S&P 500–focused headlines returned (top_actionable empty or invalid)")
    return out


# ---------- GPT-5.1: tiered causal chain analysis (idea generation only) ----------

TIER_SYSTEM = """You are a skeptical buy-side analyst. "Edge" means a testable thesis with a clear channel from the headline to P&L — not creativity points. Obscure tickers with invented supply-chain stories are WORSE than no pick.

Tiers (S&P 500 only):

Tier 1 (Direct): Names explicitly in the headline or the single obvious first-order beneficiary (e.g. target in M&A, issuer of guidance). If the headline is ABOUT one company, Tier 1 MUST include that ticker. Liquid large-caps are ALLOWED here when they are the direct subject — edge is often speed + magnitude on the obvious name, not a hidden micro-cap guess.

Tier 2 (Second-order): Only if there is a DISCLOSED or widely documented link (customer/supplier/competitor named in filings or the article, regulatory peer set, same sub-industry with shared input costs). State the link in plain English. If you would need to invent a supply chain to connect the story to a stock, OMIT that stock.

Tier 3 (Third-order): Use sparingly. At most 1 stock. Only if magnitude is high and the logic chain is ONE additional clear step (e.g. rate shock → clear group of rate-sensitive names with similar balance sheets). No "creative" six-degree connections.

ANTI-HALLUCINATION:
- Do not claim specific customer/supplier relationships unless they are stated in the headline/summary or are standard industry facts any analyst would agree on. Otherwise cap confidence at 0.45 or omit the ticker.
- confidence must reflect evidence: 0.75+ only for Tier 1 direct or Tier 2 with a stated link. Tier 3 should rarely exceed 0.55.
- expected_move_pct: give a single number that is plausible for a single-session to few-day reaction; do not use fantasy sizes.

BREVITY (strict — long answers are wrong):
- causal_chain: at most 2 short sentences, under 55 words total. Mechanism only. No bullet lists, headers, "Furthermore", or essay-style analysis.
- invalidation: exactly one sentence, under 25 words.

Every tier entry MUST include:
- tickers: non-empty list of S&P 500 tickers only
- causal_chain: cause → effect per BREVITY limits above
- direction: enum ONLY — the JSON string "long" OR "short" (verbatim). Never ticker lists, sentences, or "A: long; B: long". Same side applies to every ticker in this tier's tickers array.
- expected_delay: e.g. "same day", "1-3 sessions"
- expected_move_pct: number
- confidence: 0.0-1.0 (calibrated as above)
- invalidation: one sentence — what price action or news in the next 1-3 sessions would prove this thesis wrong?
- evidence: "headline_explicit" | "standard_industry_link" | "inferred" — if "inferred", confidence must be <= 0.5

QUALITY OVER QUANTITY:
- Tier 2: at most 2 stocks unless the headline clearly implicates more with equal clarity.
- Tier 3: at most 1 stock.
- If nothing passes the bar, return empty tickers for that tier with a brief causal_chain explaining why (still valid JSON).

Respond with valid JSON matching this structure (all tier objects in a "tiers" array):
{
  "tiers": [
    {
      "tier": 1,
      "label": "direct",
      "tickers": ["XOM"],
      "causal_chain": "...",
      "direction": "long",
      "expected_delay": "same day",
      "expected_move_pct": 3.0,
      "confidence": 0.72,
      "invalidation": "...",
      "evidence": "headline_explicit"
    }
  ]
}"""

TIER_USER = """Analyze this news event for tradable S&P 500 ideas. Prefer fewer, higher-conviction names over a long list.
Keep every causal_chain and invalidation SHORT per system instructions (no long paragraphs).

Headline: {title}
Summary: {summary}
Event type: {event_type}
Sentiment: {sentiment}
Magnitude: {magnitude}
Market impact (Nano triage): {market_impact}
Impact summary: {impact_one_liner}
Profit angle (Nano): {profit_angle_one_liner}

Return Tier 1 (direct), Tier 2 (only with clear links), Tier 3 (optional, max one name; only if magnitude is high / impact is major).
If the story is weak or already fully priced with no discrete catalyst, say so and use empty tickers where appropriate."""


def analyze_tiers(headline: dict, classification: dict) -> dict:
    """Use GPT-5.1 to produce tiered causal chain analysis (idea generation)."""
    impact_line = (classification.get("impact_one_liner") or "").strip() or "(none)"
    profit_line = (classification.get("profit_angle_one_liner") or "").strip() or "(none)"
    user_msg = TIER_USER.format(
        title=headline["title"],
        summary=headline.get("summary", ""),
        event_type=classification.get("event_type", "unknown"),
        sentiment=classification.get("sentiment", "unknown"),
        magnitude=classification.get("magnitude", "unknown"),
        market_impact=classification.get("market_impact", "unknown"),
        impact_one_liner=impact_line,
        profit_angle_one_liner=profit_line,
    )
    raw = _call(
        DEEP_MODEL,
        TIER_SYSTEM,
        user_msg,
        temperature=0.4,
        max_completion_tokens=LLM_DEEP_TIERS_MAX_COMPLETION_TOKENS,
    )
    parsed = _parse_json(raw)
    parsed = _normalize_tier_output(parsed)
    return _enforce_tier_directions_post_parse(parsed)


# ---------- GPT-5.1: reason validation for open positions ----------

VALIDATE_SYSTEM = """You are monitoring an open trading position. Judge whether the ORIGINAL thesis is still supported by (1) recent headlines and (2) how price has behaved vs a simple benchmark (e.g. SPY/sector from the data string).

Rules:
- If the thesis was second-order or "inferred" and price has NOT moved in the trade's favor while the market or peers have, lean toward invalidation.
- If new headlines contradict the thesis or the catalyst is clearly stale, still_valid should be false.
- Do not invent new reasons to hold; only assess the original reason.
- explanation must be at most 2 short sentences (under 50 words). No bullet lists or essays.

Respond with valid JSON:
{
  "still_valid": true/false,
  "explanation": "brief, concrete, short",
  "urgency": "hold" | "watch_closely" | "exit_now"
}"""

VALIDATE_USER = """=== OPEN POSITION ===
Ticker: {ticker}
Direction: {direction}
Entry price: ${entry_price}
Current price: ${current_price}
Return: {return_pct:+.2f}%
Original reason: {reason}
Tier: {tier}
Causal chain: {causal_chain}
Held since: {entry_time}

=== RECENT NEWS HEADLINES ===
{recent_headlines}

=== CURRENT MARKET DATA ===
{market_data}

Is the original reason for this trade still valid?"""


def validate_reason(position: dict, ticker: str, current_price: float,
                    recent_headlines: str, market_data: str) -> dict:
    """Ask GPT-5.1 whether a position's original reason is still valid."""
    entry_price = position["entry_price"]
    direction = normalize_trade_direction(position.get("direction"), ticker)
    if direction not in ("long", "short"):
        log.warning(
            f"validate_reason {ticker}: invalid direction {position.get('direction')!r}, skipping LLM"
        )
        return {
            "still_valid": True,
            "explanation": "invalid_direction_skip_validation",
            "urgency": "hold",
        }
    if direction == "long":
        return_pct = (current_price - entry_price) / entry_price * 100
    else:
        return_pct = (entry_price - current_price) / entry_price * 100

    user_msg = VALIDATE_USER.format(
        ticker=ticker,
        direction=direction,
        entry_price=entry_price,
        current_price=current_price,
        return_pct=return_pct,
        reason=_clip_text(position.get("reason"), _VALIDATE_REASON_CONTEXT_CHARS),
        tier=position.get("tier", "?"),
        causal_chain=_clip_text(
            position.get("causal_chain"), _VALIDATE_CAUSAL_CONTEXT_CHARS
        ),
        entry_time=position.get("entry_time", "?"),
        recent_headlines=recent_headlines,
        market_data=market_data,
    )
    raw = _call(
        DEEP_MODEL,
        VALIDATE_SYSTEM,
        user_msg,
        temperature=0.2,
        max_completion_tokens=LLM_DEEP_VALIDATE_MAX_COMPLETION_TOKENS,
    )
    result = _parse_json(raw)
    result["explanation"] = _clip_text(
        result.get("explanation"), _DEEP_VALIDATE_EXPLANATION_MAX_CHARS
    )
    return result

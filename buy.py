"""Entry path: news-first — headlines → LLM classify → tiered causal chain → quant filter → execute."""
import math
from config import (
    get_sp500_tickers, MAX_POSITIONS, get_ticker_sector,
    get_position_size,
    ENTRY_SCORE_THRESHOLD,
    SHORT_ENTRY_SCORE_THRESHOLD,
    NEWS_LED_MIN_CONFIDENCE,
    ENTRY_SCORE_FLOOR_NEWS_LED,
    NEWS_LED_SHORT_MIN_CONFIDENCE,
    SHORT_ENTRY_SCORE_FLOOR_NEWS_LED,
    MAX_POSITIONS_PER_SECTOR, MAX_SECTOR_EXPOSURE_PCT,
    SECTOR_ETF_MAP,
    resolve_event_profile,
    NANO_REQUIRE_MAJOR_MARKET_IMPACT,
)
from state import (
    load_trade_history, compute_trade_stats, add_position,
    is_duplicate_event, record_event,
)
from news import get_new_headlines
from llm import analyze_tiers, classify_headline, rank_headlines_nano
from quant import (
    get_stock_snapshot, get_spy_snapshot, get_sector_snapshot,
    get_market_context, compute_opportunity_score, format_market_data,
)
from signals import (
    compute_entry_score,
    compute_rule_based_tp_sl,
    normalize_trade_direction,
)
from alpaca_client import (
    buy_stock,
    short_stock,
    get_current_price,
    get_account,
    assert_canonical_direction,
)
from logger import log, log_entry_trade, log_no_trade, log_error, log_news_cohort_snapshot


def _count_sector_positions(state: dict, sector: str) -> int:
    """Count how many open positions are in a given sector."""
    return sum(
        1 for p in state.get("open_positions", {}).values()
        if p.get("sector") == sector
    )


def _clamp_threshold(x: int) -> int:
    return max(0, min(98, x))


def _quant_allows_entry(
    direction: str, entry_score: int, llm_confidence: float, profile: dict
) -> tuple[bool, str]:
    """News-led gate with event-profile deltas on strict / floor thresholds."""
    st_long = _clamp_threshold(
        ENTRY_SCORE_THRESHOLD + int(profile.get("strict_threshold_delta", 0))
    )
    floor_long = _clamp_threshold(
        ENTRY_SCORE_FLOOR_NEWS_LED + int(profile.get("news_led_floor_delta", 0))
    )
    st_short = _clamp_threshold(
        SHORT_ENTRY_SCORE_THRESHOLD + int(profile.get("short_strict_threshold_delta", 0))
    )
    floor_short = _clamp_threshold(
        SHORT_ENTRY_SCORE_FLOOR_NEWS_LED + int(profile.get("short_news_led_floor_delta", 0))
    )
    if direction == "short":
        if llm_confidence >= NEWS_LED_SHORT_MIN_CONFIDENCE:
            if entry_score >= floor_short:
                return True, f"news_led_floor>={floor_short}"
            return False, f"news_led_short_need_score>={floor_short}"
        if entry_score >= st_short:
            return True, f"strict>={st_short}"
        return False, f"strict_short_need>={st_short}"
    if llm_confidence >= NEWS_LED_MIN_CONFIDENCE:
        if entry_score >= floor_long:
            return True, f"news_led_floor>={floor_long}"
        return False, f"news_led_long_need_score>={floor_long}"
    if entry_score >= st_long:
        return True, f"strict>={st_long}"
    return False, f"strict_long_need>={st_long}"


def _nano_passes_for_deep(classification: dict) -> tuple[bool, str]:
    """Gate before expensive tier analysis. When NANO_REQUIRE_MAJOR_MARKET_IMPACT, require major impact."""
    if not classification.get("actionable"):
        return False, "not_actionable"
    if not NANO_REQUIRE_MAJOR_MARKET_IMPACT:
        return True, "ok"
    if classification.get("market_impact") != "major":
        return False, "market_impact_not_major"
    if classification.get("magnitude") != "high":
        return False, "magnitude_not_high"
    return True, "ok"


def _sector_exposure_pct(state: dict, sector: str) -> float:
    """Percentage of total positions in a sector."""
    total = len(state.get("open_positions", {}))
    if total == 0:
        return 0.0
    sector_count = _count_sector_positions(state, sector)
    return sector_count / total * 100


def run_entry_path(state: dict) -> list[dict]:
    """Full buy/short entry pipeline. Returns list of entries made."""
    open_count = len(state.get("open_positions", {}))
    if open_count >= MAX_POSITIONS:
        log_no_trade(f"Already at max positions ({MAX_POSITIONS})")
        return []

    new_headlines = get_new_headlines(state)
    if not new_headlines:
        log_no_trade("No new headlines this cycle")
        return []

    sp500 = get_sp500_tickers()
    entries = []

    # Nano triage: one batch call returns up to 5 S&P 500–focused headlines; fallback = per-headline
    actionable_headlines = []
    ranked = rank_headlines_nano(new_headlines)
    if ranked is None:
        log.info("Nano batch triage unavailable; classifying headlines one-by-one")
        candidate_pairs: list[tuple[dict, dict]] = []
        for headline in new_headlines:
            try:
                classification = classify_headline(headline)
                candidate_pairs.append((headline, classification))
            except Exception as e:
                log_error("classify headline", str(e))
        ranked = candidate_pairs

    for headline, classification in ranked:
        try:
            from_batch = classification.get("nano_batch_rank") is not None
            if from_batch:
                # Batch always returns min(5,n) ranked picks; send all to Deep (strict Nano gate skipped).
                ok, gate_reason = True, "batch_top_ranked"
            else:
                ok, gate_reason = _nano_passes_for_deep(classification)
            mi = classification.get("market_impact", "?")
            mag = classification.get("magnitude", "?")
            rk = classification.get("nano_batch_rank")
            rank_note = f"rank={rk} " if rk is not None else ""
            triage_src = "nano_batch" if rk is not None else "per_headline"
            if ok:
                actionable_headlines.append((headline, classification))
                pa = (classification.get("profit_angle_one_liner") or "")[:100]
                extra = f" profit_angle={pa!r}" if pa else ""
                soft_note = (
                    " [soft_pick]"
                    if from_batch and not classification.get("actionable")
                    else ""
                )
                log.info(
                    f"ACTIONABLE [{triage_src}]{soft_note} {rank_note}{headline['title'][:80]} "
                    f"[{classification.get('event_type')}] "
                    f"impact={mi} mag={mag}{extra}"
                )
            elif not classification.get("actionable"):
                log.info(f"SKIPPED: {headline['title'][:80]} (not actionable)")
            else:
                log.info(
                    f"SKIPPED: {headline['title'][:80]} (nano_gate:{gate_reason} "
                    f"impact={mi} mag={mag})"
                )
        except Exception as e:
            log_error("Nano triage headline", str(e))

    if not actionable_headlines:
        log_no_trade("No headlines passed Nano triage for Deep analysis")
        return []

    # Pre-fetch SPY snapshot once per cycle
    spy_snapshot = get_spy_snapshot()

    for headline, classification in actionable_headlines:
        if open_count + len(entries) >= MAX_POSITIONS:
            break

        event_type = classification.get("event_type", "unknown")
        event_profile = resolve_event_profile(event_type)

        # Event deduplication
        if is_duplicate_event(state, event_type):
            log.info(f"DEDUP: Already traded on event '{event_type}', skipping")
            continue

        try:
            tiers_result = analyze_tiers(headline, classification)
            tiers = tiers_result.get("tiers", [])
            if not tiers:
                log_no_trade(f"No tiers returned for: {headline['title'][:60]}")
                continue

            log_news_cohort_snapshot({
                "headline_id": headline.get("id"),
                "title": (headline.get("title") or "")[:300],
                "event_type": event_type,
                "event_profile_id": event_profile["id"],
                "market_impact": classification.get("market_impact"),
                "impact_one_liner": (classification.get("impact_one_liner") or "")[:300],
                "tiers": [
                    {
                        "tier": t.get("tier"),
                        "tickers": t.get("tickers", []),
                        "direction": t.get("direction"),
                    }
                    for t in tiers
                ],
            })

            # Filter tickers, fetch snapshots, compute entry scores
            scored_candidates = []
            for tier in tiers:
                valid_tickers = [t for t in tier.get("tickers", []) if t in sp500]
                tier["tickers"] = valid_tickers
                raw_dir = tier.get("direction", "long")
                llm_confidence = tier.get("confidence", 0.5)
                expected_move = tier.get("expected_move_pct", 2.0)

                for ticker in valid_tickers:
                    direction = normalize_trade_direction(raw_dir, ticker)
                    if direction not in ("long", "short"):
                        log.info(
                            f"REJECTED {ticker}: direction not canonical "
                            f"({raw_dir!r})"
                        )
                        continue
                    if ticker in state.get("open_positions", {}):
                        continue

                    snapshot = get_stock_snapshot(ticker)
                    if snapshot is None:
                        continue

                    sector_snapshot = get_sector_snapshot(ticker)

                    entry_score = compute_entry_score(
                        snapshot=snapshot,
                        spy_snapshot=spy_snapshot,
                        sector_snapshot=sector_snapshot,
                        direction=direction,
                        llm_confidence=llm_confidence,
                        event_profile=event_profile,
                    )

                    opp_score = compute_opportunity_score(
                        snapshot, expected_move, direction
                    )

                    scored_candidates.append({
                        "ticker": ticker,
                        "snapshot": snapshot,
                        "entry_score": entry_score,
                        "opportunity_score": opp_score,
                        "tier": tier.get("tier", 1),
                        "direction": direction,
                        "expected_move_pct": expected_move,
                        "causal_chain": tier.get("causal_chain", ""),
                        "llm_confidence": llm_confidence,
                        "event_profile_id": event_profile["id"],
                    })

            if not scored_candidates:
                log_no_trade(f"No valid S&P 500 candidates for: {headline['title'][:60]}")
                continue

            # Rank: tier → LLM conviction → thesis room (opp) → tape score
            scored_candidates.sort(
                key=lambda x: (
                    -x["tier"],
                    -x["llm_confidence"],
                    -x["opportunity_score"],
                    -x["entry_score"],
                )
            )

            for cand in scored_candidates:
                if open_count + len(entries) >= MAX_POSITIONS:
                    break

                ticker = cand["ticker"]
                entry_score = cand["entry_score"]
                direction = cand["direction"]

                min_opp = float(event_profile.get("min_opportunity_score") or 0)
                if cand["opportunity_score"] < min_opp:
                    log.info(
                        f"REJECTED {ticker} ({direction}): opportunity_score="
                        f"{cand['opportunity_score']} < min {min_opp} [{event_profile['id']}]"
                    )
                    continue

                ok, gate = _quant_allows_entry(
                    direction, entry_score, cand["llm_confidence"], event_profile
                )
                if not ok:
                    log.info(
                        f"REJECTED {ticker} ({direction}): entry_score={entry_score} "
                        f"({gate}, conf={cand['llm_confidence']:.2f}, profile={event_profile['id']})"
                    )
                    continue
                quant_path = "news_led" if gate.startswith("news_led") else "strict"

                # Sector exposure check
                sector = get_ticker_sector(ticker)
                if _count_sector_positions(state, sector) >= MAX_POSITIONS_PER_SECTOR:
                    log.info(f"REJECTED {ticker}: sector {sector} at max positions")
                    continue
                if _sector_exposure_pct(state, sector) >= MAX_SECTOR_EXPOSURE_PCT:
                    log.info(f"REJECTED {ticker}: sector {sector} exposure too high")
                    continue

                # Confidence-based position sizing × event profile
                position_dollars = int(
                    get_position_size(entry_score)
                    * float(event_profile.get("position_size_multiplier", 1.0))
                )
                if position_dollars == 0:
                    continue

                current_price = get_current_price(ticker)
                if current_price is None:
                    current_price = cand["snapshot"]["current_price"]

                shares = max(1, math.floor(position_dollars / current_price))

                assert_canonical_direction(
                    direction, context=f"entry {ticker} pre_order"
                )

                # Execute trade
                if direction == "long":
                    result = buy_stock(
                        ticker,
                        shares,
                        position_direction="long",
                        context=f"entry {ticker}",
                    )
                else:
                    result = short_stock(
                        ticker,
                        shares,
                        position_direction="short",
                        context=f"entry {ticker}",
                    )

                if result:
                    assert_canonical_direction(
                        direction, context=f"add_position {ticker}"
                    )
                    tp, sl = compute_rule_based_tp_sl(
                        current_price,
                        direction,
                        take_profit_pct=event_profile["take_profit_pct"],
                        stop_loss_pct=event_profile["stop_loss_pct"],
                    )

                    add_position(state, ticker, {
                        "entry_price": current_price,
                        "direction": direction,
                        "shares": shares,
                        "reason": cand["causal_chain"],
                        "tier": cand["tier"],
                        "causal_chain": cand["causal_chain"],
                        "take_profit": tp,
                        "stop_loss": sl,
                        "confidence": cand["llm_confidence"],
                        "expected_timeframe": cand.get("expected_timeframe", "unknown"),
                        "event_type": event_type,
                        "event_profile_id": event_profile["id"],
                        "sector": sector,
                        "entry_score": entry_score,
                        "entry_volume_spike": cand["snapshot"].get("volume_spike", 1.0),
                        "max_hold_days": event_profile["max_hold_days"],
                        "min_hold_bars": event_profile["min_hold_bars"],
                        "trailing_stop_pct": event_profile["trailing_stop_pct"],
                        "momentum_fade_ret30m_min": event_profile["momentum_fade_ret30m_min"],
                        "rumor_fast_exit": event_profile["rumor_fast_exit"],
                        "opportunity_score": cand["opportunity_score"],
                        "quant_gate": gate,
                        "quant_path": quant_path,
                    })

                    record_event(state, event_type)

                    log_entry_trade(
                        ticker,
                        direction,
                        current_price,
                        shares,
                        cand["causal_chain"][:100],
                        cand["tier"],
                        cand["llm_confidence"],
                        event_profile_id=event_profile["id"],
                    )
                    log.info(
                        f"  -> entry_score={entry_score}, opp={cand['opportunity_score']}, "
                        f"profile={event_profile['id']}, size=${position_dollars}, sector={sector}"
                    )
                    entries.append({"ticker": ticker, "direction": direction})

        except Exception as e:
            log_error("entry pipeline for headline", str(e))

    if not entries:
        log_no_trade("No trades executed this cycle")

    return entries

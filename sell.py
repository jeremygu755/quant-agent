"""Exit path: quant signal waterfall, trailing stop, time exit, LLM thesis check as last resort."""
from state import remove_position, log_completed_trade
from quant import get_stock_snapshot, get_spy_snapshot, format_snapshot_for_llm
from signals import check_exit_signals, update_high_water_mark, normalize_trade_direction
from llm import validate_reason
from alpaca_client import (
    sell_stock,
    cover_short,
    get_current_price,
    get_position_side_for_ticker,
    assert_canonical_direction,
)
from news import fetch_headlines
from logger import log, log_exit_trade, log_error


def run_exit_path(state: dict) -> list[dict]:
    """Check all open positions for exit conditions. Returns list of exits made."""
    positions = state.get("open_positions", {})
    if not positions:
        log.info("No open positions to check")
        return []

    exits = []
    spy_snapshot = get_spy_snapshot()

    # Grab recent headlines for LLM reason validation (loaded only if needed)
    recent_headlines_text = None

    tickers_to_exit = []

    for ticker, pos in list(positions.items()):
        try:
            current_price = get_current_price(ticker)
            if current_price is None:
                snapshot = get_stock_snapshot(ticker)
                if snapshot:
                    current_price = snapshot["current_price"]
                else:
                    log.warning(f"Cannot get price for {ticker}, skipping exit check")
                    continue
            else:
                snapshot = get_stock_snapshot(ticker)

            pos["bars_held"] = int(pos.get("bars_held", 0)) + 1

            canon_dir = normalize_trade_direction(pos.get("direction"), ticker)
            if canon_dir is None:
                canon_dir = get_position_side_for_ticker(ticker)
                if canon_dir is not None:
                    log.warning(
                        f"{ticker}: direction from Alpaca (state was "
                        f"{pos.get('direction')!r}) -> {canon_dir!r}"
                    )
            if canon_dir is None:
                log.error(
                    f"{ticker}: cannot canonicalize direction "
                    f"{pos.get('direction')!r}; skipping exit checks this cycle"
                )
                continue

            if pos.get("direction") != canon_dir:
                was_noncanonical = pos.get("direction") not in ("long", "short")
                log.warning(
                    f"{ticker}: normalized direction {pos.get('direction')!r} -> {canon_dir!r}"
                )
                pos["direction"] = canon_dir
                if was_noncanonical:
                    pos["high_water_mark"] = pos["entry_price"]

            direction = pos["direction"]
            assert_canonical_direction(direction, context=f"exit loop {ticker}")
            entry_price = pos["entry_price"]

            # Trailing stop high-water mark (must run after direction is canonical)
            update_high_water_mark(pos, current_price)

            if direction == "long":
                return_pct = (current_price - entry_price) / entry_price * 100
            else:
                return_pct = (entry_price - current_price) / entry_price * 100

            # Run the quant signal waterfall
            exit_reason = check_exit_signals(
                pos=pos,
                current_price=current_price,
                snapshot=snapshot,
                spy_snapshot=spy_snapshot,
            )

            if exit_reason:
                log.info(f"QUANT EXIT {ticker}: {exit_reason}")
                tickers_to_exit.append((ticker, pos, current_price, return_pct, exit_reason))
                continue

            # Last resort: LLM thesis validation (only for positions held > 1 hour)
            entry_time_str = pos.get("entry_time", "")
            if not entry_time_str:
                continue

            from datetime import datetime, timezone
            try:
                entry_time = datetime.fromisoformat(entry_time_str)
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_held = 0

            if hours_held < 1:
                continue

            if recent_headlines_text is None:
                recent_headlines_raw = fetch_headlines()
                recent_headlines_text = "\n".join(
                    f"- {h['title']}" for h in recent_headlines_raw[:15]
                )

            market_data = format_snapshot_for_llm(snapshot) if snapshot else "(no data)"

            validation = validate_reason(
                position=pos,
                ticker=ticker,
                current_price=current_price,
                recent_headlines=recent_headlines_text,
                market_data=market_data,
            )

            if not validation.get("still_valid", True):
                exit_reason = f"thesis_invalidated: {validation.get('explanation', 'unknown')}"
                tickers_to_exit.append((ticker, pos, current_price, return_pct, exit_reason))
            elif validation.get("urgency") == "exit_now":
                exit_reason = f"urgent_exit: {validation.get('explanation', 'unknown')}"
                tickers_to_exit.append((ticker, pos, current_price, return_pct, exit_reason))

        except Exception as e:
            log_error(f"exit check {ticker}", str(e))

    # Execute all exits
    for ticker, pos, current_price, return_pct, exit_reason in tickers_to_exit:
        direction = pos["direction"]
        shares = pos["shares"]

        assert_canonical_direction(direction, context=f"exit {ticker} pre_order")

        if direction == "long":
            result = sell_stock(
                ticker,
                shares,
                position_direction="long",
                context=f"exit {ticker}",
            )
        else:
            result = cover_short(
                ticker,
                shares,
                position_direction="short",
                context=f"exit {ticker}",
            )

        if result:
            log_exit_trade(ticker, direction, exit_reason, pos["entry_price"],
                           current_price, round(return_pct, 2))

            log_completed_trade({
                "ticker": ticker,
                "direction": direction,
                "entry_price": pos["entry_price"],
                "exit_price": current_price,
                "return_pct": round(return_pct, 2),
                "shares": shares,
                "reason": pos.get("reason", ""),
                "exit_reason": exit_reason,
                "event_type": pos.get("event_type", "unknown"),
                "event_profile_id": pos.get("event_profile_id", "default"),
                "tier": pos.get("tier", "?"),
                "entry_time": pos.get("entry_time"),
                "entry_score": pos.get("entry_score", 0),
                "sector": pos.get("sector", "unknown"),
                "llm_confidence": pos.get("confidence"),
                "opportunity_score": pos.get("opportunity_score"),
                "quant_path": pos.get("quant_path", "unknown"),
                "quant_gate": pos.get("quant_gate", ""),
            })

            remove_position(state, ticker)
            exits.append({"ticker": ticker, "reason": exit_reason, "return_pct": return_pct})

    return exits

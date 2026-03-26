import json
import os
from datetime import datetime, timedelta, timezone

from config import STATE_FILE, HISTORY_FILE, DATA_DIR, MAX_HISTORY_DAYS, EVENT_DEDUP_HOURS
from logger import log
from signals import normalize_trade_direction


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _default_state() -> dict:
    return {
        "open_positions": {},
        "seen_headlines": [],
        "recent_events": [],
    }


def _backfill_open_position_directions(state: dict) -> None:
    """Normalize direction on load so sell logic always sees long|short when parseable."""
    for ticker, pos in list(state.get("open_positions", {}).items()):
        if not isinstance(pos, dict):
            continue
        if pos.get("direction") in ("long", "short"):
            continue
        canon = normalize_trade_direction(pos.get("direction"), ticker)
        if canon is not None:
            log.warning(
                f"load_state backfill {ticker}: {pos.get('direction')!r} -> {canon!r}"
            )
            pos["direction"] = canon
        else:
            log.warning(
                f"load_state: {ticker} direction {pos.get('direction')!r} "
                "not parseable; exit path will try Alpaca"
            )


def load_state() -> dict:
    _ensure_data_dir()
    if not os.path.exists(STATE_FILE):
        return _default_state()
    with open(STATE_FILE, "r") as f:
        state = json.load(f)
    if "recent_events" not in state:
        state["recent_events"] = []
    _backfill_open_position_directions(state)
    return state


def save_state(state: dict):
    _ensure_data_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def add_position(state: dict, ticker: str, position: dict):
    """Add an open position with all entry metadata."""
    now = datetime.now(timezone.utc).isoformat()
    entry_price = position["entry_price"]
    direction = position["direction"]

    state["open_positions"][ticker] = {
        "entry_price": entry_price,
        "entry_time": now,
        "direction": direction,
        "shares": position["shares"],
        "reason": position["reason"],
        "tier": position["tier"],
        "causal_chain": position["causal_chain"],
        "take_profit": position["take_profit"],
        "stop_loss": position["stop_loss"],
        "confidence": position["confidence"],
        "expected_timeframe": position.get("expected_timeframe", "unknown"),
        "event_type": position.get("event_type", "unknown"),
        "event_profile_id": position.get("event_profile_id", "default"),
        "sector": position.get("sector", "unknown"),
        "high_water_mark": entry_price,
        "entry_score": position.get("entry_score", 0),
        "entry_volume_spike": position.get("entry_volume_spike", 1.0),
        "max_hold_days": position.get("max_hold_days"),
        "min_hold_bars": position.get("min_hold_bars"),
        "trailing_stop_pct": position.get("trailing_stop_pct"),
        "momentum_fade_ret30m_min": position.get("momentum_fade_ret30m_min"),
        "rumor_fast_exit": position.get("rumor_fast_exit", False),
        "bars_held": 0,
        "opportunity_score": position.get("opportunity_score"),
        "quant_gate": position.get("quant_gate"),
        "quant_path": position.get("quant_path"),
    }


def remove_position(state: dict, ticker: str) -> dict | None:
    """Remove and return a closed position."""
    return state["open_positions"].pop(ticker, None)


def mark_headline_seen(state: dict, headline_id: str):
    if headline_id not in state["seen_headlines"]:
        state["seen_headlines"].append(headline_id)
    if len(state["seen_headlines"]) > 500:
        state["seen_headlines"] = state["seen_headlines"][-500:]


def is_headline_seen(state: dict, headline_id: str) -> bool:
    return headline_id in state["seen_headlines"]


# --- Event deduplication ---

def record_event(state: dict, event_type: str):
    """Record that we traded on this event type."""
    state["recent_events"].append({
        "event_type": event_type,
        "time": datetime.now(timezone.utc).isoformat(),
    })
    _prune_events(state)


def is_duplicate_event(state: dict, event_type: str) -> bool:
    """Check if we've already traded on this event type within EVENT_DEDUP_HOURS."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=EVENT_DEDUP_HOURS)
    for evt in state.get("recent_events", []):
        if evt["event_type"] == event_type:
            try:
                evt_time = datetime.fromisoformat(evt["time"])
                if evt_time.tzinfo is None:
                    evt_time = evt_time.replace(tzinfo=timezone.utc)
                if evt_time >= cutoff:
                    return True
            except (ValueError, TypeError):
                continue
    return False


def _prune_events(state: dict):
    """Remove events older than EVENT_DEDUP_HOURS."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=EVENT_DEDUP_HOURS)
    state["recent_events"] = [
        evt for evt in state.get("recent_events", [])
        if _event_after_cutoff(evt, cutoff)
    ]


def _event_after_cutoff(evt: dict, cutoff: datetime) -> bool:
    try:
        t = datetime.fromisoformat(evt["time"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t >= cutoff
    except (ValueError, TypeError):
        return False


# --- Trade History (history.jsonl) ---

def log_completed_trade(trade: dict):
    """Append a completed trade to history.jsonl."""
    _ensure_data_dir()
    trade["closed_at"] = datetime.now(timezone.utc).isoformat()
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def load_trade_history(days: int = MAX_HISTORY_DAYS) -> list[dict]:
    """Load trade history from the last N days."""
    _ensure_data_dir()
    if not os.path.exists(HISTORY_FILE):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    trades = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trade = json.loads(line)
            ticker = trade.get("ticker", "")
            d = normalize_trade_direction(trade.get("direction"), ticker)
            if d is not None and trade.get("direction") != d:
                trade["direction"] = d
                try:
                    ep = float(trade["entry_price"])
                    xp = float(trade["exit_price"])
                    if ep > 0:
                        if d == "long":
                            trade["return_pct"] = round((xp - ep) / ep * 100, 2)
                        else:
                            trade["return_pct"] = round((ep - xp) / ep * 100, 2)
                except (KeyError, TypeError, ValueError):
                    pass
            closed_at = datetime.fromisoformat(trade.get("closed_at", "2000-01-01"))
            if closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=timezone.utc)
            if closed_at >= cutoff:
                trades.append(trade)
    return trades


def compute_trade_stats(trades: list[dict]) -> dict:
    """Compute summary stats from trade history for LLM context."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "by_event_type": {},
            "by_ticker": {},
            "by_tier": {},
            "long_stats": {"trades": 0, "win_rate": 0.0},
            "short_stats": {"trades": 0, "win_rate": 0.0},
        }

    wins = [t for t in trades if t.get("return_pct", 0) > 0]
    total = len(trades)

    by_event: dict[str, list] = {}
    by_ticker: dict[str, list] = {}
    by_tier: dict[str, list] = {}
    longs = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]

    for t in trades:
        evt = t.get("event_type", "unknown")
        by_event.setdefault(evt, []).append(t)
        tkr = t.get("ticker", "?")
        by_ticker.setdefault(tkr, []).append(t)
        tier = str(t.get("tier", "?"))
        by_tier.setdefault(tier, []).append(t)

    def _summarize(group: list[dict]) -> dict:
        if not group:
            return {"trades": 0, "win_rate": 0.0, "avg_return_pct": 0.0}
        w = sum(1 for t in group if t.get("return_pct", 0) > 0)
        avg_ret = sum(t.get("return_pct", 0) for t in group) / len(group)
        return {
            "trades": len(group),
            "win_rate": round(w / len(group) * 100, 1),
            "avg_return_pct": round(avg_ret, 2),
        }

    return {
        "total_trades": total,
        "win_rate": round(len(wins) / total * 100, 1),
        "avg_return_pct": round(
            sum(t.get("return_pct", 0) for t in trades) / total, 2
        ),
        "by_event_type": {k: _summarize(v) for k, v in by_event.items()},
        "by_ticker": {k: _summarize(v) for k, v in by_ticker.items()},
        "by_tier": {k: _summarize(v) for k, v in by_tier.items()},
        "long_stats": _summarize(longs),
        "short_stats": _summarize(shorts),
    }


def backfill_history_jsonl() -> int:
    """Rewrite ``history.jsonl`` with canonical ``direction`` and recomputed ``return_pct`` when fixed.

    Run once manually: ``python -c \"from state import backfill_history_jsonl; backfill_history_jsonl()\"``
    """
    _ensure_data_dir()
    if not os.path.exists(HISTORY_FILE):
        return 0
    lines_out: list[str] = []
    n_dir_fixes = 0
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trade = json.loads(line)
            ticker = trade.get("ticker", "")
            d = normalize_trade_direction(trade.get("direction"), ticker)
            if d is not None:
                old = trade.get("direction")
                trade["direction"] = d
                if old != d:
                    n_dir_fixes += 1
                try:
                    ep = float(trade["entry_price"])
                    xp = float(trade["exit_price"])
                    if ep > 0:
                        if d == "long":
                            trade["return_pct"] = round((xp - ep) / ep * 100, 2)
                        else:
                            trade["return_pct"] = round((ep - xp) / ep * 100, 2)
                except (KeyError, TypeError, ValueError):
                    pass
            lines_out.append(json.dumps(trade, default=str))
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w") as out:
        out.write("\n".join(lines_out) + ("\n" if lines_out else ""))
    os.replace(tmp, HISTORY_FILE)
    log.info(
        f"backfill_history_jsonl: wrote {len(lines_out)} records ({n_dir_fixes} direction fixes)"
    )
    return n_dir_fixes

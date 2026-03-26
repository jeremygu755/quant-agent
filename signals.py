"""Pure quant signal scoring for entry and exit decisions. No LLM calls."""
import copy
import re
from datetime import datetime, timezone
from config import (
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_STOP_PCT, MAX_HOLD_DAYS,
    MIN_HOLD_BARS, SHORT_SPY_MAX_1H_PCT,
    EVENT_PROFILES,
    LONG_ENTRY_MIN_VOLUME_SPIKE,
    LONG_ENTRY_BAN_DECELERATING_MOMENTUM,
    LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H,
)


def _effective_event_profile(event_profile: dict | None) -> dict:
    if event_profile is not None:
        return event_profile
    return copy.deepcopy(EVENT_PROFILES["default"])


def normalize_trade_direction(raw: object, ticker: str) -> str | None:
    """Return exactly 'long' or 'short', or None if the value cannot be trusted (fail closed).

    Accepts only: exact \"long\"/\"short\", or prose containing ``TICKER: long`` / ``TICKER: short``
    for this symbol (case-insensitive ticker). Anything else returns None.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return None
    if s == "long":
        return "long"
    if s == "short":
        return "short"
    sym = re.escape(ticker.strip().upper().lower())
    matches = re.findall(rf"\b{sym}\s*:\s*(long|short)\b", s)
    if not matches:
        return None
    if len(set(matches)) != 1:
        return None
    return matches[0]


def compute_entry_score(
    snapshot: dict,
    spy_snapshot: dict | None,
    sector_snapshot: dict | None,
    direction: str,
    llm_confidence: float = 0.5,
    event_profile: dict | None = None,
) -> int:
    """Score a candidate stock for entry. Delegates to direction-specific scorer.

    ``event_profile`` comes from ``resolve_event_profile(event_type)`` in live trading;
    omit for generic scoring (e.g. backtest).
    """
    p = _effective_event_profile(event_profile)
    if direction == "short":
        return _score_short_entry(
            snapshot, spy_snapshot, sector_snapshot, llm_confidence, p
        )
    return _score_long_entry(
        snapshot, spy_snapshot, sector_snapshot, llm_confidence, p
    )


def _score_long_entry(
    snapshot: dict,
    spy_snapshot: dict | None,
    sector_snapshot: dict | None,
    llm_confidence: float,
    p: dict,
) -> int:
    """Score a long entry. Max ~100; pass ``ENTRY_SCORE_THRESHOLD`` to enter."""
    score = 0

    ret_5m = snapshot.get("pct_change_5m", 0.0)
    ret_30m = snapshot.get("pct_change_30m", 0.0)
    ret_1h = snapshot.get("pct_change_1h", 0.0)

    vol_spike = snapshot.get("volume_spike", 1.0)
    stock_ret_1h = snapshot.get("pct_change_1h", 0.0)
    spy_ret_1h = spy_snapshot.get("pct_change_1h", 0.0) if spy_snapshot else 0.0
    sector_ret_1h = sector_snapshot.get("pct_change_1h", 0.0) if sector_snapshot else 0.0
    abnormal_vs_sector = stock_ret_1h - sector_ret_1h

    if p.get("require_fast_5m_confirmation") and not (ret_5m > 0.2 and ret_30m > 0):
        return 0

    min_vol_profile = p.get("min_volume_spike")
    req_vol = float(LONG_ENTRY_MIN_VOLUME_SPIKE or 0)
    if min_vol_profile is not None:
        req_vol = max(req_vol, float(min_vol_profile))
    if req_vol > 0 and vol_spike < req_vol:
        return 0

    min_ab = p.get("min_sector_abnormal_long")
    if min_ab is not None and sector_snapshot is not None:
        if abnormal_vs_sector < min_ab:
            return 0

    if p.get("macro_alignment"):
        if spy_ret_1h < 0:
            return 0
        if sector_snapshot is not None and sector_ret_1h < 0:
            return 0

    if LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H is not None:
        if spy_snapshot is None:
            return 0
        if (stock_ret_1h - spy_snapshot.get("pct_change_1h", 0.0)) <= float(
            LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H
        ):
            return 0

    momentum = snapshot.get("momentum", "flat")
    if momentum == "reversing":
        return 0
    if LONG_ENTRY_BAN_DECELERATING_MOMENTUM and momentum == "decelerating":
        return 0

    if ret_30m > 0:
        score += 10
    if ret_1h > 0:
        score += 10
    if ret_5m > 0.3:
        score += 10

    if vol_spike >= 2.0:
        score += 20
    elif vol_spike >= 1.5:
        score += 10
    elif vol_spike >= 1.2:
        score += 5

    abnormal_vs_spy = stock_ret_1h - spy_ret_1h

    if abnormal_vs_spy > 1.0:
        score += 15
    elif abnormal_vs_spy > 0.5:
        score += 8

    if abnormal_vs_sector > 0.5:
        score += 10
    elif abnormal_vs_sector > 0.2:
        score += 5

    if abs(ret_5m) > abs(ret_30m) and ret_5m > 0:
        score += 10

    if sector_snapshot:
        sector_ret = sector_snapshot.get("pct_change_1h", 0.0)
        if sector_ret > 0.2:
            score += 10
        elif sector_ret > 0:
            score += 5

    if spy_snapshot:
        spy_ret = spy_snapshot.get("pct_change_1h", 0.0)
        if spy_ret < -0.5:
            score -= 10

    vol_ratio = snapshot.get("volatility_ratio", 1.0)
    if vol_ratio > 1.5:
        score += 5

    score += int(llm_confidence * 5)
    score += int(p.get("entry_score_adjustment", 0))

    return max(score, 0)


def _score_short_entry(
    snapshot: dict,
    spy_snapshot: dict | None,
    sector_snapshot: dict | None,
    llm_confidence: float,
    p: dict,
) -> int:
    """Score a short entry. Stricter than longs — requires bearish regime + strong weakness."""
    score = 0

    # HARD GATE: only short in a bearish or flat market
    if spy_snapshot:
        spy_ret_1h_gate = spy_snapshot.get("pct_change_1h", 0.0)
        if spy_ret_1h_gate > SHORT_SPY_MAX_1H_PCT:
            return 0  # market is bullish — no shorts allowed

    ret_5m = snapshot.get("pct_change_5m", 0.0)
    ret_30m = snapshot.get("pct_change_30m", 0.0)
    ret_1h = snapshot.get("pct_change_1h", 0.0)

    vol_spike = snapshot.get("volume_spike", 1.0)
    stock_ret_1h = snapshot.get("pct_change_1h", 0.0)
    spy_ret_1h = spy_snapshot.get("pct_change_1h", 0.0) if spy_snapshot else 0.0
    sector_ret_1h = sector_snapshot.get("pct_change_1h", 0.0) if sector_snapshot else 0.0
    weakness_vs_sector = sector_ret_1h - stock_ret_1h

    if p.get("require_fast_5m_confirmation") and not (ret_5m < -0.2 and ret_30m < 0):
        return 0

    min_vol = p.get("min_volume_spike")
    if min_vol is not None and vol_spike < min_vol:
        return 0

    min_ab_s = p.get("min_sector_abnormal_short")
    if min_ab_s is not None and sector_snapshot is not None:
        if weakness_vs_sector < min_ab_s:
            return 0

    if p.get("macro_alignment"):
        if spy_ret_1h > 0:
            return 0
        if sector_snapshot is not None and sector_ret_1h > 0:
            return 0

    # Momentum: stock must be falling (stricter thresholds than longs)
    if ret_30m < -0.1:
        score += 10
    if ret_1h < -0.2:
        score += 10
    if ret_5m < -0.3:
        score += 10

    # Volume spike
    if vol_spike >= 2.0:
        score += 20
    elif vol_spike >= 1.5:
        score += 10
    elif vol_spike >= 1.2:
        score += 5

    weakness_vs_spy = spy_ret_1h - stock_ret_1h  # positive = stock is weaker

    # Must show meaningful underperformance vs SPY
    if weakness_vs_spy < 0.3:
        return 0  # not weak enough relative to market

    if weakness_vs_spy > 1.5:
        score += 15
    elif weakness_vs_spy > 0.8:
        score += 10
    elif weakness_vs_spy > 0.3:
        score += 5

    if weakness_vs_sector > 0.8:
        score += 10
    elif weakness_vs_sector > 0.3:
        score += 5

    # Momentum acceleration (getting worse faster)
    if abs(ret_5m) > abs(ret_30m) and ret_5m < 0:
        score += 10

    momentum = snapshot.get("momentum", "flat")
    if momentum == "reversing":
        return 0

    # Sector confirmation: sector must also be weak (REQUIRED for shorts)
    if sector_snapshot:
        sector_ret = sector_snapshot.get("pct_change_1h", 0.0)
        if sector_ret < -0.3:
            score += 15  # strong sector weakness — more weight than longs
        elif sector_ret < -0.1:
            score += 8
        else:
            score -= 15  # sector is flat/up — penalize short heavily
    else:
        score -= 10  # no sector data — penalize

    # SPY confirmation bonus (market selling off)
    if spy_snapshot:
        spy_ret = spy_snapshot.get("pct_change_1h", 0.0)
        if spy_ret < -0.5:
            score += 10  # broad market selling — supports short thesis
        elif spy_ret < -0.2:
            score += 5

    vol_ratio = snapshot.get("volatility_ratio", 1.0)
    if vol_ratio > 1.5:
        score += 5

    score += int(llm_confidence * 5)
    score += int(p.get("entry_score_adjustment", 0))

    return max(score, 0)


def compute_rule_based_tp_sl(
    entry_price: float,
    direction: str,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> tuple[float, float]:
    """Compute take profit and stop loss prices; uses config defaults when pct omitted."""
    tp_pct = TAKE_PROFIT_PCT if take_profit_pct is None else take_profit_pct
    sl_pct = STOP_LOSS_PCT if stop_loss_pct is None else stop_loss_pct
    if direction == "long":
        tp = entry_price * (1 + tp_pct / 100)
        sl = entry_price * (1 - sl_pct / 100)
    else:
        tp = entry_price * (1 - tp_pct / 100)
        sl = entry_price * (1 + sl_pct / 100)
    return round(tp, 2), round(sl, 2)


# --- Exit Signals ---

def check_exit_signals(
    pos: dict,
    current_price: float,
    snapshot: dict | None,
    spy_snapshot: dict | None,
    as_of: datetime | None = None,
) -> str | None:
    """Check all quant exit signals. Returns exit reason string or None.

    Checks are ordered by priority — first trigger wins.
    If ``as_of`` is set (e.g. backtest bar time), time-based exit uses it; else wall-clock now.
    """
    direction = pos.get("direction")
    if direction not in ("long", "short"):
        return None

    entry_price = pos["entry_price"]

    if direction == "long":
        return_pct = (current_price - entry_price) / entry_price * 100
    else:
        return_pct = (entry_price - current_price) / entry_price * 100

    tp = pos.get("take_profit", 0)
    sl = pos.get("stop_loss", 0)

    # 1. Stop loss
    if direction == "long" and current_price <= sl:
        return f"stop_loss hit ({return_pct:+.2f}%, limit=${sl})"
    if direction == "short" and current_price >= sl:
        return f"stop_loss hit ({return_pct:+.2f}%, limit=${sl})"

    # 2. Take profit
    if direction == "long" and current_price >= tp:
        return f"take_profit hit ({return_pct:+.2f}%, target=${tp})"
    if direction == "short" and current_price <= tp:
        return f"take_profit hit ({return_pct:+.2f}%, target=${tp})"

    # 3. Trailing stop (price dropped from peak)
    trail_pct = pos.get("trailing_stop_pct") or TRAILING_STOP_PCT
    high_water = pos.get("high_water_mark", entry_price)
    if direction == "long":
        drop_from_peak = (high_water - current_price) / high_water * 100
        if drop_from_peak >= trail_pct and current_price > entry_price:
            return f"trailing_stop ({drop_from_peak:.2f}% drop from peak ${high_water:.2f})"
    else:
        rise_from_low = (current_price - high_water) / high_water * 100 if high_water > 0 else 0
        if rise_from_low >= trail_pct and current_price < entry_price:
            return f"trailing_stop ({rise_from_low:.2f}% rise from low ${high_water:.2f})"

    # Soft exits (4-7) only fire after the trade has had time to develop
    min_bars = pos.get("min_hold_bars") if pos.get("min_hold_bars") is not None else MIN_HOLD_BARS
    bars_held = int(pos.get("bars_held", 0))
    soft_exits_active = bars_held >= min_bars

    fade_min = pos.get("momentum_fade_ret30m_min")
    if fade_min is None:
        fade_min = 0.3

    if snapshot and soft_exits_active:
        # 4. Momentum fading — only if 5m return is < 15% of 30m return
        ret_5m = snapshot.get("pct_change_5m", 0.0)
        ret_30m = snapshot.get("pct_change_30m", 0.0)
        sign = 1.0 if direction == "long" else -1.0
        if ret_30m * sign > fade_min and ret_5m * sign < ret_30m * sign * 0.15:
            return f"momentum_fading (5m={ret_5m:+.2f}% vs 30m={ret_30m:+.2f}%)"

        # 5. Reversal detected
        momentum = snapshot.get("momentum", "flat")
        if momentum == "reversing" and return_pct > 0.5:
            return f"reversal_detected (was profitable, now reversing)"

        # 6. Volume drop — requires real entry spike (>2x) and severe drop (to 30%)
        entry_volume = pos.get("entry_volume_spike", 1.0)
        current_volume = snapshot.get("volume_spike", 1.0)
        if entry_volume > 2.0 and current_volume < entry_volume * 0.3:
            return f"volume_drop (entry={entry_volume:.1f}x → now={current_volume:.1f}x)"

        # 7. Abnormal return disappeared — tighter condition
        if spy_snapshot:
            stock_ret = snapshot.get("pct_change_1h", 0.0)
            spy_ret = spy_snapshot.get("pct_change_1h", 0.0)
            abnormal = abs(stock_ret - spy_ret)
            if abnormal < 0.15 and abs(return_pct) < 0.3:
                return f"abnormal_return_gone (stock={stock_ret:+.2f}% vs SPY={spy_ret:+.2f}%)"

    # 8. Time-based exit
    entry_time_str = pos.get("entry_time")
    if entry_time_str:
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            now = as_of if as_of is not None else datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            else:
                now = now.astimezone(timezone.utc)
            days_held = (now - entry_time).days
            max_days = pos.get("max_hold_days") or MAX_HOLD_DAYS
            if days_held >= max_days:
                return f"time_exit (held {days_held} days, max={max_days})"
        except (ValueError, TypeError):
            pass

    return None


def update_high_water_mark(pos: dict, current_price: float):
    """Update the high-water mark for trailing stop tracking."""
    direction = pos["direction"]
    hwm = pos.get("high_water_mark", pos["entry_price"])

    if direction == "long":
        pos["high_water_mark"] = max(hwm, current_price)
    else:
        pos["high_water_mark"] = min(hwm, current_price)

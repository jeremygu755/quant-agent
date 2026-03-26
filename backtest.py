"""Backtest engine — quant-only replay (no headlines). Live bot is news-led; see buy.run_entry_path."""
import json
import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from market_data import fetch_5m_ohlcv
from signals import compute_entry_score, check_exit_signals, update_high_water_mark, compute_rule_based_tp_sl
from config import ENTRY_SCORE_THRESHOLD, get_position_size


def _build_snapshots_from_row(row, prev_rows, ticker: str) -> dict:
    """Build a snapshot dict from a historical row for signal scoring."""
    current = row["Close"]
    idx = len(prev_rows)

    def _price_ago(n_bars):
        i = idx - n_bars
        return prev_rows.iloc[i]["Close"] if i >= 0 else current

    def _pct(a, b):
        return round((a - b) / b * 100, 3) if b != 0 else 0.0

    price_1bar = _price_ago(1)
    price_6bar = _price_ago(6)    # ~30 min with 5m bars
    price_12bar = _price_ago(12)  # ~1 hour
    price_3bar = _price_ago(3)    # ~15 min

    recent_move = _pct(current, _price_ago(3))
    prior_move = _pct(_price_ago(3), _price_ago(6))

    if abs(prior_move) < 0.001:
        momentum = "flat"
    elif abs(recent_move) > abs(prior_move):
        momentum = "accelerating"
    elif recent_move * prior_move < 0:
        momentum = "reversing"
    else:
        momentum = "decelerating"

    vol = row.get("Volume", 0)
    recent_vols = prev_rows["Volume"].tail(12)
    avg_vol = recent_vols.mean() if len(recent_vols) > 0 else 1

    return {
        "ticker": ticker,
        "current_price": round(current, 2),
        "pct_change_5m": _pct(current, price_1bar),
        "pct_change_30m": _pct(current, price_6bar),
        "pct_change_1h": _pct(current, price_12bar),
        "pct_change_1d": 0.0,
        "volume_spike": round(vol / avg_vol, 2) if avg_vol > 0 else 1.0,
        "volatility_ratio": 1.0,
        "momentum": momentum,
    }


def _spy_snapshot_asof(stock_bar_ts, spy_hist: pd.DataFrame | None) -> dict | None:
    """Build SPY snapshot using bars up to the same wall time as the stock bar (UTC)."""
    if spy_hist is None or spy_hist.empty:
        return None
    ts = stock_bar_ts
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    mask = spy_hist.index <= ts
    if not mask.any():
        return None
    j = int(mask.sum()) - 1
    if j < 13:
        return None
    spy_row = spy_hist.iloc[j]
    spy_prev = spy_hist.iloc[:j]
    return _build_snapshots_from_row(spy_row, spy_prev, "SPY")


def _dollar_path_from_trades(
    trades: list[dict],
    starting_equity: float,
    notional_per_trade: float,
    fee_bps_round_trip: float,
) -> dict:
    """Apply each trade P&L as fixed notional; subtract round-trip fee per trade (bps of notional)."""
    fee_per_trade = notional_per_trade * (fee_bps_round_trip / 10000.0)
    gross = sum(notional_per_trade * (t["return_pct"] / 100.0) for t in trades)
    total_fees = fee_per_trade * len(trades)
    net_pnl = gross - total_fees
    ending = starting_equity + net_pnl
    return {
        "starting_equity": round(starting_equity, 2),
        "ending_equity": round(ending, 2),
        "notional_per_trade": round(notional_per_trade, 2),
        "fee_bps_round_trip": fee_bps_round_trip,
        "total_gross_pnl_dollars": round(gross, 2),
        "total_fees_dollars": round(total_fees, 2),
        "total_net_pnl_dollars": round(net_pnl, 2),
        "portfolio_return_pct": round((ending - starting_equity) / starting_equity * 100, 4),
    }


def run_backtest(
    ticker: str,
    direction: str = "long",
    days: int = 5,
    llm_confidence: float = 0.6,
    starting_equity: float = 100_000.0,
    notional_per_trade: float | None = None,
    fee_bps_round_trip: float = 0.0,
) -> dict:
    """Run a basic backtest for a single ticker over the last N days.

    Uses 5-minute bars. Simulates entry scoring and exit signals.
    Returns a summary of all simulated trades.

    Dollar path: each closed trade contributes ``notional * return_pct/100`` gross P&L;
    ``fee_bps_round_trip`` is charged on notional once per round trip.
    ``portfolio_return_pct`` is net P&L vs ``starting_equity`` (sequential, non-overlapping trades).
    """
    hist = fetch_5m_ohlcv(ticker, days)

    if hist is None or hist.empty or len(hist) < 20:
        return {"error": f"Not enough data for {ticker}"}

    spy_hist = fetch_5m_ohlcv("SPY", days)

    trades = []
    open_trade = None
    cooldown = 0

    for i in range(13, len(hist)):
        row = hist.iloc[i]
        prev_rows = hist.iloc[:i]
        snapshot = _build_snapshots_from_row(row, prev_rows, ticker)

        bar_ts = hist.index[i]
        spy_snapshot = _spy_snapshot_asof(bar_ts, spy_hist)

        current_price = snapshot["current_price"]
        as_of = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
        if getattr(as_of, "tzinfo", None) is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)

        if open_trade is not None:
            update_high_water_mark(open_trade, current_price)
            exit_reason = check_exit_signals(
                pos=open_trade,
                current_price=current_price,
                snapshot=snapshot,
                spy_snapshot=spy_snapshot,
                as_of=as_of,
            )
            if exit_reason:
                entry_p = open_trade["entry_price"]
                if open_trade["direction"] == "long":
                    ret = (current_price - entry_p) / entry_p * 100
                else:
                    ret = (entry_p - current_price) / entry_p * 100

                trades.append({
                    "ticker": open_trade.get("ticker", ticker),
                    "direction": open_trade.get("direction", direction),
                    "entry_price": entry_p,
                    "exit_price": round(current_price, 2),
                    "return_pct": round(ret, 2),
                    "exit_reason": exit_reason,
                    "bars_held": open_trade.get("bars_held", 0),
                    "entry_score": open_trade.get("entry_score"),
                    "entry_momentum": open_trade.get("entry_momentum"),
                    "entry_vol_spike": open_trade.get("entry_vol_spike"),
                    "entry_ret_5m": open_trade.get("entry_ret_5m"),
                    "entry_ret_30m": open_trade.get("entry_ret_30m"),
                    "entry_ret_1h": open_trade.get("entry_ret_1h"),
                    "entry_spy_ret_1h": open_trade.get("entry_spy_ret_1h"),
                })
                open_trade = None
                cooldown = 6  # wait 30 min before re-entering
            else:
                open_trade["bars_held"] = open_trade.get("bars_held", 0) + 1
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        # Try entry
        score = compute_entry_score(
            snapshot=snapshot,
            spy_snapshot=spy_snapshot,
            sector_snapshot=None,
            direction=direction,
            llm_confidence=llm_confidence,
        )

        if score >= ENTRY_SCORE_THRESHOLD:
            tp, sl = compute_rule_based_tp_sl(current_price, direction)
            open_trade = {
                "entry_price": current_price,
                "direction": direction,
                "ticker": ticker,
                "take_profit": tp,
                "stop_loss": sl,
                "high_water_mark": current_price,
                "entry_volume_spike": snapshot.get("volume_spike", 1.0),
                "entry_time": hist.index[i].isoformat(),
                "bars_held": 0,
                "entry_score": score,
                "entry_momentum": snapshot.get("momentum"),
                "entry_vol_spike": round(snapshot.get("volume_spike", 1.0), 2),
                "entry_ret_5m": snapshot.get("pct_change_5m"),
                "entry_ret_30m": snapshot.get("pct_change_30m"),
                "entry_ret_1h": snapshot.get("pct_change_1h"),
                "entry_spy_ret_1h": spy_snapshot.get("pct_change_1h") if spy_snapshot else None,
            }

    # Close any open trade at end
    if open_trade is not None:
        final_price = hist["Close"].iloc[-1]
        entry_p = open_trade["entry_price"]
        if open_trade["direction"] == "long":
            ret = (final_price - entry_p) / entry_p * 100
        else:
            ret = (entry_p - final_price) / entry_p * 100
        trades.append({
            "ticker": open_trade.get("ticker", ticker),
            "direction": open_trade.get("direction", direction),
            "entry_price": entry_p,
            "exit_price": round(final_price, 2),
            "return_pct": round(ret, 2),
            "exit_reason": "end_of_data",
            "bars_held": open_trade.get("bars_held", 0),
            "entry_score": open_trade.get("entry_score"),
            "entry_momentum": open_trade.get("entry_momentum"),
            "entry_vol_spike": open_trade.get("entry_vol_spike"),
            "entry_ret_5m": open_trade.get("entry_ret_5m"),
            "entry_ret_30m": open_trade.get("entry_ret_30m"),
            "entry_ret_1h": open_trade.get("entry_ret_1h"),
            "entry_spy_ret_1h": open_trade.get("entry_spy_ret_1h"),
        })

    # Summary
    if not trades:
        return {
            "ticker": ticker,
            "direction": direction,
            "days": days,
            "total_trades": 0,
            "message": "No trades triggered",
        }

    wins = [t for t in trades if t["return_pct"] > 0]
    total_return = sum(t["return_pct"] for t in trades)
    per_trade_notional = (
        float(notional_per_trade) if notional_per_trade is not None else float(get_position_size(50))
    )
    dollar_summary = _dollar_path_from_trades(trades, starting_equity, per_trade_notional, fee_bps_round_trip)

    return {
        "ticker": ticker,
        "direction": direction,
        "days": days,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(trades) - len(wins),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_return_pct": round(total_return, 2),
        "avg_return_pct": round(total_return / len(trades), 2),
        "avg_bars_held": round(sum(t["bars_held"] for t in trades) / len(trades), 1),
        **dollar_summary,
        "trades": trades,
    }


def normalize_exit_kind(exit_reason: str) -> str:
    """Bucket exit strings from check_exit_signals into a short label."""
    if exit_reason.startswith("stop_loss"):
        return "stop_loss"
    if exit_reason.startswith("take_profit"):
        return "take_profit"
    if exit_reason.startswith("trailing_stop"):
        return "trailing_stop"
    if exit_reason.startswith("momentum_fading"):
        return "momentum_fading"
    if exit_reason.startswith("reversal_detected"):
        return "reversal_detected"
    if exit_reason.startswith("volume_drop"):
        return "volume_drop"
    if exit_reason.startswith("abnormal_return_gone"):
        return "abnormal_return_gone"
    if exit_reason.startswith("time_exit"):
        return "time_exit"
    if exit_reason == "end_of_data":
        return "end_of_data"
    return "other"


def summarize_trade_signals(trades: list[dict]) -> dict:
    """Aggregate win rate and avg return by exit kind and entry context."""
    by_exit: dict[str, list[float]] = {}
    by_momentum: dict[str, list[float]] = {}
    by_score_band: dict[str, list[float]] = {}
    by_vol: dict[str, list[float]] = {}

    def band(score):
        if score is None:
            return "unknown"
        if score < 60:
            return "score_50_59"
        if score < 70:
            return "score_60_69"
        if score < 80:
            return "score_70_79"
        return "score_80_plus"

    def vol_bucket(v):
        if v is None:
            return "vol_unknown"
        if v < 1.2:
            return "vol_lt_1.2"
        if v < 1.5:
            return "vol_1.2_1.5"
        if v < 2.0:
            return "vol_1.5_2.0"
        return "vol_ge_2.0"

    for t in trades:
        r = t.get("return_pct", 0.0)
        ek = normalize_exit_kind(t.get("exit_reason", ""))
        by_exit.setdefault(ek, []).append(r)
        by_momentum.setdefault(t.get("entry_momentum") or "unknown", []).append(r)
        by_score_band.setdefault(band(t.get("entry_score")), []).append(r)
        by_vol.setdefault(vol_bucket(t.get("entry_vol_spike")), []).append(r)

    def stats(rows: list[float]) -> dict:
        if not rows:
            return {"n": 0}
        wins = sum(1 for x in rows if x > 0)
        return {
            "n": len(rows),
            "win_rate_pct": round(wins / len(rows) * 100, 1),
            "avg_return_pct": round(sum(rows) / len(rows), 3),
            "sum_return_pct": round(sum(rows), 2),
        }

    return {
        "by_exit_kind": {k: stats(v) for k, v in sorted(by_exit.items(), key=lambda x: -len(x[1]))},
        "by_entry_momentum": {k: stats(v) for k, v in sorted(by_momentum.items(), key=lambda x: -len(x[1]))},
        "by_entry_score_band": {k: stats(v) for k, v in sorted(by_score_band.items())},
        "by_entry_volume_spike": {k: stats(v) for k, v in sorted(by_vol.items())},
    }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    direction = sys.argv[2] if len(sys.argv) > 2 else "long"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"Backtesting {ticker} ({direction}) over {days} days...")
    result = run_backtest(ticker, direction, days)
    print(json.dumps(result, indent=2))

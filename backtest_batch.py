"""
Multi-symbol quant backtest + per-signal attribution (entry flags + exit kinds).

Collects up to N closed trades across many tickers (5m Yahoo data), then:
- Full-pipeline summary (same rules as ``run_backtest``)
- Exit signal buckets (stop, TP, momentum_fading, …)
- Entry feature buckets (momentum, score band, volume) — from ``summarize_trade_signals``
- Per-entry-signal splits: performance when each long-entry condition was true vs false
  at the entry bar (mirrors ``signals._score_long_entry`` structure; sector omitted — backtest has no sector).

Usage:
  python backtest_batch.py
  python backtest_batch.py --target 200 --days 60 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any

from config import (
    get_position_size,
    get_sp500_tickers,
    LONG_ENTRY_BAN_DECELERATING_MOMENTUM,
    LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H,
    LONG_ENTRY_MIN_VOLUME_SPIKE,
)
from backtest import (
    _dollar_path_from_trades,
    run_backtest,
    summarize_trade_signals,
)


def _stats(returns: list[float]) -> dict:
    if not returns:
        return {"n": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "sum_return_pct": 0.0}
    wins = sum(1 for x in returns if x > 0)
    n = len(returns)
    return {
        "n": n,
        "win_rate_pct": round(wins / n * 100, 1),
        "avg_return_pct": round(sum(returns) / n, 3),
        "sum_return_pct": round(sum(returns), 2),
    }


def long_entry_signal_flags(t: dict) -> dict[str, bool]:
    """Boolean conditions aligned with long ``_score_long_entry`` (no sector data in backtest)."""
    r5 = float(t.get("entry_ret_5m") or 0.0)
    r30 = float(t.get("entry_ret_30m") or 0.0)
    r1h = float(t.get("entry_ret_1h") or 0.0)
    spy = t.get("entry_spy_ret_1h")
    vs_spy = (r1h - float(spy)) if spy is not None else None
    vol = float(t.get("entry_vol_spike") or 1.0)
    mom = str(t.get("entry_momentum") or "")

    min_v = float(LONG_ENTRY_MIN_VOLUME_SPIKE or 0)
    min_ab = LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H
    vol_gate = min_v <= 0 or vol >= min_v
    spy_gate = min_ab is None or (vs_spy is not None and vs_spy > float(min_ab))
    mom_gate = mom != "reversing" and (
        not LONG_ENTRY_BAN_DECELERATING_MOMENTUM or mom != "decelerating"
    )

    return {
        "ret_30m_positive": r30 > 0,
        "ret_1h_positive": r1h > 0,
        "ret_5m_gt_0_3": r5 > 0.3,
        "volume_spike_ge_1_2": vol >= 1.2,
        "volume_spike_ge_1_5": vol >= 1.5,
        "volume_spike_ge_2": vol >= 2.0,
        "vs_spy_1h_gt_0_5": vs_spy is not None and vs_spy > 0.5,
        "vs_spy_1h_gt_1_0": vs_spy is not None and vs_spy > 1.0,
        "five_m_stronger_than_30m": abs(r5) > abs(r30) and r5 > 0,
        "momentum_accelerating": mom == "accelerating",
        "momentum_decelerating": mom == "decelerating",
        "momentum_flat": mom == "flat",
        "momentum_not_reversing": mom != "reversing",
        "spy_1h_very_weak_underlying": spy is not None and float(spy) < -0.5,
        "long_config_hard_gates": vol_gate and spy_gate and mom_gate,
    }


def short_entry_signal_flags(t: dict) -> dict[str, bool]:
    """Boolean conditions aligned with short ``_score_short_entry`` (minimal; no sector in backtest)."""
    r5 = float(t.get("entry_ret_5m") or 0.0)
    r30 = float(t.get("entry_ret_30m") or 0.0)
    r1h = float(t.get("entry_ret_1h") or 0.0)
    spy = t.get("entry_spy_ret_1h")
    weakness_vs_spy = (float(spy) - r1h) if spy is not None else None
    vol = float(t.get("entry_vol_spike") or 1.0)
    mom = str(t.get("entry_momentum") or "")

    return {
        "short_ret_30m_lt_neg_0_1": r30 < -0.1,
        "short_ret_1h_lt_neg_0_2": r1h < -0.2,
        "short_ret_5m_lt_neg_0_3": r5 < -0.3,
        "short_volume_ge_1_2": vol >= 1.2,
        "short_volume_ge_1_5": vol >= 1.5,
        "short_volume_ge_2": vol >= 2.0,
        "short_weak_vs_spy_ge_0_3": weakness_vs_spy is not None and weakness_vs_spy >= 0.3,
        "short_five_m_faster_down": abs(r5) > abs(r30) and r5 < 0,
        "short_momentum_not_reversing": mom != "reversing",
    }


def summarize_entry_signal_splits(trades: list[dict]) -> dict[str, Any]:
    """For each atomic entry flag, stats when condition true vs false at entry."""
    longs = [t for t in trades if t.get("direction", "long") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]

    def _split(group: list[dict], flag_fn) -> dict:
        if not group:
            return {}
        keys = sorted(flag_fn(group[0]).keys())
        out: dict[str, dict] = {}
        for key in keys:
            true_r: list[float] = []
            false_r: list[float] = []
            for t in group:
                r = float(t.get("return_pct") or 0.0)
                if flag_fn(t)[key]:
                    true_r.append(r)
                else:
                    false_r.append(r)
            out[key] = {
                "when_true": _stats(true_r),
                "when_false": _stats(false_r),
            }
        return out

    return {
        "long": _split(longs, long_entry_signal_flags),
        "short": _split(shorts, short_entry_signal_flags),
    }


def collect_batch_trades(
    target_trades: int = 200,
    days: int = 60,
    direction: str = "long",
    llm_confidence: float = 0.6,
    seed: int = 42,
    tickers: list[str] | None = None,
    max_trades_per_ticker: int | None = None,
) -> tuple[list[dict], dict]:
    """Run ``run_backtest`` over shuffled symbols until ``target_trades`` or list exhausted.

    If ``max_trades_per_ticker`` is set, take at most that many closes from each symbol
    (spreads sample across more names; e.g. ``1`` ≈ one trade per symbol up to target).
    """
    if tickers is None:
        tickers = sorted(get_sp500_tickers())
        random.seed(seed)
        random.shuffle(tickers)
    if not tickers:
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "JPM", "V", "UNH",
        ] * 30

    all_trades: list[dict] = []
    meta: list[dict] = []
    errors = 0
    per_sym: dict[str, int] = {}
    symbols_attempted = 0

    for sym in tickers:
        symbols_attempted += 1
        if len(all_trades) >= target_trades:
            break
        r = run_backtest(sym, direction, days, llm_confidence=llm_confidence)
        if r.get("error"):
            errors += 1
            continue
        tr = r.get("trades") or []
        if not tr:
            continue
        taken = 0
        for row in tr:
            if len(all_trades) >= target_trades:
                break
            if max_trades_per_ticker is not None:
                if per_sym.get(sym, 0) >= max_trades_per_ticker:
                    break
            row = dict(row)
            row.setdefault("ticker", sym)
            row.setdefault("direction", direction)
            all_trades.append(row)
            per_sym[sym] = per_sym.get(sym, 0) + 1
            taken += 1
        if taken:
            meta.append({"ticker": sym, "trades_taken": taken, "trades_available": len(tr)})

    clipped = all_trades[:target_trades]
    return clipped, {
        "symbols_attempted": symbols_attempted,
        "tickers_with_trades": len(meta),
        "fetch_errors": errors,
        "target": target_trades,
        "actual": len(clipped),
        "unique_symbols_in_batch": len({t.get("ticker") for t in clipped}),
        "max_trades_per_ticker": max_trades_per_ticker,
    }


def build_full_report(
    trades: list[dict],
    starting_equity: float = 100_000.0,
    notional_per_trade: float | None = None,
    fee_bps: float = 0.0,
) -> dict:
    ntz = float(notional_per_trade) if notional_per_trade is not None else float(get_position_size(50))
    dollar = _dollar_path_from_trades(trades, starting_equity, ntz, fee_bps)
    wins = [t for t in trades if float(t.get("return_pct") or 0) > 0]
    total_ret = sum(float(t.get("return_pct") or 0) for t in trades)

    return {
        "batch_meta": {
            "trade_count": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
            "sum_return_pct": round(total_ret, 2),
            "avg_return_pct": round(total_ret / len(trades), 3) if trades else 0.0,
        },
        "dollar_path": dollar,
        "exit_and_entry_context": summarize_trade_signals(trades),
        "entry_signal_splits": summarize_entry_signal_splits(trades),
        "caveat": (
            "Quant-only backtest (no news/event profiles). Entry splits are correlational: "
            "all trades already passed ENTRY_SCORE_THRESHOLD, so 'when_false' is not independent."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-symbol backtest + signal breakdown")
    p.add_argument("--target", type=int, default=200, help="Target closed trades")
    p.add_argument("--days", type=int, default=60, help="Lookback days per symbol (5m)")
    p.add_argument("--direction", choices=("long", "short"), default="long")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-confidence", type=float, default=0.6, dest="llm_confidence")
    p.add_argument(
        "--max-per-ticker",
        type=int,
        default=None,
        metavar="N",
        help="Cap trades taken per symbol (e.g. 1 = diversify across up to target symbols)",
    )
    args = p.parse_args()

    trades, scan_info = collect_batch_trades(
        target_trades=args.target,
        days=args.days,
        direction=args.direction,
        llm_confidence=args.llm_confidence,
        seed=args.seed,
        max_trades_per_ticker=args.max_per_ticker,
    )
    report = {
        "scan": scan_info,
        **build_full_report(trades),
    }
    if not trades:
        report["error"] = f"No trades collected (try more tickers or longer --days). scan={scan_info}"
    print(json.dumps(report, indent=2))
    return 0 if trades else 1


if __name__ == "__main__":
    raise SystemExit(main())

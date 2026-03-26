"""
Post-hoc trade analytics from history.jsonl.

Measures:
- Trades / win rate / P&L proxies by event_type, event_profile_id
- Buckets for LLM confidence and opportunity_score at entry
- strict vs news_led quant path (see note below)

``quant_path``:
- ``strict`` — passed the higher entry-score threshold (tape had to confirm hard).
- ``news_led`` — passed only the news-led floor with higher LLM confidence (softer quant bar).

That compares *two real execution paths*, not a full counterfactual "LLM-only, zero quant"
(which would require logging every tier candidate and simulating fills without gates).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from config import MAX_HISTORY_DAYS
from state import load_trade_history


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def trade_llm_confidence(t: dict) -> float | None:
    return _f(t.get("llm_confidence", t.get("confidence")))


def trade_opportunity(t: dict) -> float | None:
    return _f(t.get("opportunity_score"))


def bucket_llm_confidence(c: float | None) -> str:
    if c is None:
        return "unknown"
    if c < 0.5:
        return "conf_0.00_0.50"
    if c < 0.65:
        return "conf_0.50_0.65"
    if c < 0.8:
        return "conf_0.65_0.80"
    return "conf_0.80_1.00"


def bucket_opportunity(o: float | None) -> str:
    if o is None:
        return "unknown"
    if o < 0:
        return "opp_negative"
    if o < 1.0:
        return "opp_0_1"
    if o < 3.0:
        return "opp_1_3"
    if o < 5.0:
        return "opp_3_5"
    return "opp_5_plus"


def approx_dollar_pnl(t: dict) -> float:
    """Position P&L in dollars from return_pct and entry notional."""
    rp = _f(t.get("return_pct")) or 0.0
    sh = int(t.get("shares") or 0)
    ep = _f(t.get("entry_price")) or 0.0
    return sh * ep * (rp / 100.0)


def summarize(trades: list[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "sum_return_pct": 0.0,
            "avg_return_pct": 0.0,
            "sum_dollar_pnl_approx": 0.0,
        }
    wins = [t for t in trades if (_f(t.get("return_pct")) or 0) > 0]
    rets = [_f(t.get("return_pct")) or 0.0 for t in trades]
    dollars = [approx_dollar_pnl(t) for t in trades]
    n = len(trades)
    return {
        "trades": n,
        "wins": len(wins),
        "win_rate_pct": round(len(wins) / n * 100, 1),
        "sum_return_pct": round(sum(rets), 2),
        "avg_return_pct": round(sum(rets) / n, 3),
        "sum_dollar_pnl_approx": round(sum(dollars), 2),
    }


def group_by(trades: list[dict], key_fn) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        out[key_fn(t)].append(t)
    return dict(out)


def compute_analytics(trades: list[dict]) -> dict:
    by_event_type = group_by(trades, lambda t: str(t.get("event_type", "unknown")))
    by_profile = group_by(trades, lambda t: str(t.get("event_profile_id", "unknown")))
    by_conf = group_by(trades, lambda t: bucket_llm_confidence(trade_llm_confidence(t)))
    by_opp = group_by(trades, lambda t: bucket_opportunity(trade_opportunity(t)))
    by_quant_path = group_by(trades, lambda t: str(t.get("quant_path", "unknown")))

    return {
        "overall": summarize(trades),
        "by_event_type": {k: summarize(v) for k, v in sorted(by_event_type.items())},
        "by_event_profile_id": {k: summarize(v) for k, v in sorted(by_profile.items())},
        "by_llm_confidence_bucket": {k: summarize(v) for k, v in sorted(by_conf.items())},
        "by_opportunity_score_bucket": {k: summarize(v) for k, v in sorted(by_opp.items())},
        "by_quant_path": {k: summarize(v) for k, v in sorted(by_quant_path.items())},
        "_note": (
            "quant_path strict vs news_led compares execution paths. "
            "True LLM-only (no quant) is not observed without shadow logging of rejected candidates."
        ),
    }


def print_report(data: dict) -> None:
    print(json.dumps(data, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Analytics from data/history.jsonl")
    p.add_argument("--days", type=int, default=MAX_HISTORY_DAYS, help="Lookback days")
    args = p.parse_args()
    trades = load_trade_history(args.days)
    if not trades:
        print(json.dumps({"error": "no trades in window", "days": args.days}, indent=2))
        return 1
    print_report(compute_analytics(trades))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

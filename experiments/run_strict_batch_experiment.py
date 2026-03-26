#!/usr/bin/env python3
"""
One-off batch experiments. Patches backtest module only in-process; does not change config/signals.

Usage:
  python experiments/run_strict_batch_experiment.py baseline-stops   # 200 trades, analyze stop exits
  python experiments/run_strict_batch_experiment.py strict-200       # score>=75, cap |5m|, target 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest as bt
from backtest_batch import build_full_report, collect_batch_trades

# Defaults aligned with user request
SEED = 42
DAYS = 60
TARGET = 200
# Long-only: reject entries where |5m return| is already very stretched (chase risk)
MAX_ABS_RET_5M_PCT = 0.85


def _is_stop_exit(reason: str) -> bool:
    r = (reason or "").lower()
    return "stop_loss" in r or "trailing_stop" in r


def analyze_stop_trades(all_trades: list[dict]) -> dict:
    stops = [t for t in all_trades if _is_stop_exit(str(t.get("exit_reason", "")))]
    others = [t for t in all_trades if not _is_stop_exit(str(t.get("exit_reason", "")))]

    def _nums(rows, key):
        return [float(t[key]) for t in rows if t.get(key) is not None]

    def _summ(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
        }

    r5_s = _nums(stops, "entry_ret_5m")
    r5_o = _nums(others, "entry_ret_5m")
    vol_s = _nums(stops, "entry_vol_spike")
    vol_o = _nums(others, "entry_vol_spike")
    r1h_s = _nums(stops, "entry_ret_1h")
    r1h_o = _nums(others, "entry_ret_1h")
    spy_s = _nums(stops, "entry_spy_ret_1h")
    spy_o = _nums(others, "entry_spy_ret_1h")
    vs_spy_s = [a - b for a, b in zip(r1h_s, spy_s) if True]  # need paired - simplify below

    paired_vs: list[float] = []
    for t in stops:
        r1 = t.get("entry_ret_1h")
        sp = t.get("entry_spy_ret_1h")
        if r1 is not None and sp is not None:
            paired_vs.append(float(r1) - float(sp))
    paired_vs_o: list[float] = []
    for t in others:
        r1 = t.get("entry_ret_1h")
        sp = t.get("entry_spy_ret_1h")
        if r1 is not None and sp is not None:
            paired_vs_o.append(float(r1) - float(sp))

    mom_s = Counter(str(t.get("entry_momentum") or "?") for t in stops)
    stretched_s = sum(1 for t in stops if abs(float(t.get("entry_ret_5m") or 0)) >= 0.5)
    stretched_o = sum(1 for t in others if abs(float(t.get("entry_ret_5m") or 0)) >= 0.5)

    by_ticker = Counter(str(t.get("ticker", "?")) for t in stops)
    ticker_all = Counter(str(t.get("ticker", "?")) for t in all_trades)

    return {
        "stop_trade_count": len(stops),
        "other_trade_count": len(others),
        "stop_share_pct": round(100 * len(stops) / len(all_trades), 1) if all_trades else 0,
        "entry_ret_5m": {"stops": _summ(r5_s), "non_stops": _summ(r5_o)},
        "entry_vol_spike": {"stops": _summ(vol_s), "non_stops": _summ(vol_o)},
        "entry_ret_1h": {"stops": _summ(r1h_s), "non_stops": _summ(r1h_o)},
        "entry_spy_ret_1h": {"stops": _summ(spy_s), "non_stops": _summ(spy_o)},
        "abnormal_vs_spy_1h": {"stops": _summ(paired_vs), "non_stops": _summ(paired_vs_o)},
        "momentum_distribution_stops": dict(mom_s),
        "pct_entry_abs_ret5m_ge_0_5": {
            "stops": round(100 * stretched_s / len(stops), 1) if stops else 0,
            "non_stops": round(100 * stretched_o / len(others), 1) if others else 0,
        },
        "stop_counts_by_ticker_top15": dict(by_ticker.most_common(15)),
        "stop_rate_by_ticker_min5_stops": {
            sym: {"stops": c, "total": ticker_all[sym], "stop_rate_pct": round(100 * c / ticker_all[sym], 1)}
            for sym, c in by_ticker.items()
            if c >= 5
        },
    }


def symbol_pnl_drag(all_trades: list[dict]) -> dict:
    by_sym: dict[str, list[float]] = {}
    for t in all_trades:
        sym = str(t.get("ticker", "?"))
        by_sym.setdefault(sym, []).append(float(t.get("return_pct") or 0))
    rows = []
    for sym, rets in by_sym.items():
        rows.append(
            {
                "ticker": sym,
                "n": len(rets),
                "sum_return_pct": round(sum(rets), 2),
                "avg_return_pct": round(sum(rets) / len(rets), 4),
            }
        )
    rows.sort(key=lambda x: x["sum_return_pct"])
    return {"worst_10_by_sum_pnl_pct": rows[:10], "best_5_by_sum_pnl_pct": rows[-5:][::-1]}


def run_baseline_stops() -> None:
    trades, scan = collect_batch_trades(
        target_trades=TARGET, days=DAYS, direction="long", seed=SEED
    )
    report = {"scan": scan, **build_full_report(trades)}
    report["stop_loss_deep_dive"] = analyze_stop_trades(trades)
    report["symbol_pnl"] = symbol_pnl_drag(trades)
    print(json.dumps(report, indent=2))


def run_strict_200() -> None:
    orig_score_fn = bt.compute_entry_score
    orig_thr = bt.ENTRY_SCORE_THRESHOLD

    def wrapped(snapshot, spy_snapshot=None, sector_snapshot=None, direction="long", llm_confidence=0.6, event_profile=None):
        if direction == "long":
            r5 = abs(float(snapshot.get("pct_change_5m") or 0))
            if r5 > MAX_ABS_RET_5M_PCT:
                return 0
        return orig_score_fn(
            snapshot, spy_snapshot, sector_snapshot, direction, llm_confidence, event_profile
        )

    bt.compute_entry_score = wrapped
    bt.ENTRY_SCORE_THRESHOLD = 75

    try:
        trades, scan = collect_batch_trades(
            target_trades=TARGET, days=DAYS, direction="long", seed=SEED
        )
        report = {
            "experiment": {
                "ENTRY_SCORE_THRESHOLD": 75,
                "max_abs_entry_ret_5m_pct": MAX_ABS_RET_5M_PCT,
                "note": "vol>=2, vs_spy>1, no decelerating still from config.signals",
            },
            "scan": scan,
            **build_full_report(trades),
        }
        report["stop_loss_deep_dive"] = analyze_stop_trades(trades)
        report["symbol_pnl"] = symbol_pnl_drag(trades)
        print(json.dumps(report, indent=2))
    finally:
        bt.compute_entry_score = orig_score_fn
        bt.ENTRY_SCORE_THRESHOLD = orig_thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baseline-stops", "strict-200"))
    args = ap.parse_args()
    if args.mode == "baseline-stops":
        run_baseline_stops()
    else:
        run_strict_200()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

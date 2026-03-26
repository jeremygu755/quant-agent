#!/usr/bin/env python3
"""One-shot full pipeline test: news → Nano classify → Deep tiers → Yahoo snapshots →
event profile + quant gates → (optional) Alpaca quote + order.

Default: in-memory state + fixture headline + **dry-run orders** (no broker orders).
Set env ``FULL_FLOW_REAL_ORDERS=1`` or ``--live-orders`` to submit real **paper** orders.

Usage:
  cd "quant agent" && source venv/bin/activate && python full_flow_test.py
  FULL_FLOW_REAL_ORDERS=1 python full_flow_test.py --live-orders
  python full_flow_test.py --rss   # use real RSS + fresh state (still dry orders by default)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))


def _dry_order(ticker: str, qty: int, side: str):
    print(f"  [DRY ORDER] {side} {ticker} x{qty}")
    return {"order_id": "dry-run", "status": "accepted"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Full quant-agent flow smoke test")
    parser.add_argument(
        "--rss",
        action="store_true",
        help="Use real Yahoo RSS via get_new_headlines (marks headlines seen in state)",
    )
    parser.add_argument(
        "--live-orders",
        action="store_true",
        help="Submit real Alpaca paper orders (no dry-run patches)",
    )
    parser.add_argument(
        "--persist-state",
        action="store_true",
        help="load/save data/state.json (default: in-memory only)",
    )
    parser.add_argument(
        "--run-exit",
        action="store_true",
        help="Also run sell.run_exit_path after entry (dry exit patches unless --live-orders)",
    )
    args = parser.parse_args()

    live = args.live_orders or os.environ.get("FULL_FLOW_REAL_ORDERS", "").lower() in (
        "1",
        "true",
        "yes",
    )

    from state import _default_state, load_state, save_state
    import buy as buy_mod
    from buy import run_entry_path
    from sell import run_exit_path
    from alpaca_client import get_account, get_current_price

    if args.persist_state:
        state = load_state()
        print("State: loaded from disk")
    else:
        state = _default_state()
        print("State: in-memory (not saving to disk)")

    fixture = {
        "id": f"full-flow-{time.time()}",
        "title": "Apple AAPL tops Q1 earnings estimates as iPhone revenue grows",
        "summary": "Earnings beat; guidance discussed. Large-cap name for pipeline test.",
        "link": "https://example.com/full-flow-test",
        "published": "",
        "source": "fixture",
    }

    def fake_headlines(s):
        return [fixture]

    print("\n=== Alpaca (read) ===")
    try:
        acct = get_account()
        print("  account:", acct)
        print("  SPY quote:", get_current_price("SPY"))
    except Exception as e:
        print("  Alpaca read failed:", e)

    print("\n=== Entry path (LLM + Yahoo + quant + gates) ===")
    with ExitStack() as stack:
        if not args.rss:
            stack.enter_context(patch.object(buy_mod, "get_new_headlines", side_effect=fake_headlines))
        if not live:
            stack.enter_context(
                patch.object(buy_mod, "buy_stock", side_effect=lambda t, q: _dry_order(t, q, "BUY"))
            )
            stack.enter_context(
                patch.object(
                    buy_mod, "short_stock", side_effect=lambda t, q: _dry_order(t, q, "SHORT")
                )
            )
        entries = run_entry_path(state)

    print("  entries:", entries)
    print("  open_positions keys:", list(state.get("open_positions", {}).keys()))

    if args.run_exit:
        print("\n=== Exit path ===")
        import sell as sell_mod

        with ExitStack() as stack:
            if not live:
                stack.enter_context(
                    patch.object(
                        sell_mod, "sell_stock", side_effect=lambda t, q: _dry_order(t, q, "SELL")
                    )
                )
                stack.enter_context(
                    patch.object(
                        sell_mod, "cover_short", side_effect=lambda t, q: _dry_order(t, q, "COVER")
                    )
                )
            exits = run_exit_path(state)
        print("  exits:", exits)
        print("  open_positions after:", list(state.get("open_positions", {}).keys()))

    if args.persist_state:
        save_state(state)
        print("\nState saved to disk.")

    if not entries and not state.get("open_positions"):
        print(
            "\nNote: No trade opened (LLM tickers may not overlap S&P, or gates rejected). "
            "That still validates classify + tiers + scoring path when logs show candidates."
        )

    print("\nDone. Orders were", "LIVE PAPER" if live else "DRY-RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

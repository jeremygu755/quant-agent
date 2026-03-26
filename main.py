"""
Quant Agent — news-led trading bot (headlines + causal tiers drive ideas; quant filters risk).
Loops every 30 minutes: exit path first, then entry path.
Only runs during market hours (Mon-Fri 9:30 AM - 4:00 PM ET).

Env QUANT_REREAD_HEADLINES_ONCE=1 — on first cycle only, re-offer RSS items to Nano even if already in seen_headlines
(still enforces NEWS_MAX_AGE_MINUTES). Normal dedup afterward.
"""
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from config import CYCLE_INTERVAL_SECONDS, get_sp500_tickers
from state import load_state, save_state
from sell import run_exit_path
from buy import run_entry_path
from alpaca_client import get_account, get_positions
from logger import log, log_cycle_start, log_cycle_end, log_error

ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    """Check if US stock market is currently open (Mon-Fri 9:30-16:00 ET)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def sync_from_alpaca(state: dict):
    """Pull cash and positions from Alpaca as source of truth."""
    try:
        acct = get_account()
        if acct.get("cash", 0) > 0:
            log.info(f"Alpaca sync: cash=${acct['cash']:.2f}, equity=${acct['equity']:.2f}")

        alpaca_positions = get_positions()
        alpaca_tickers = {p["ticker"] for p in alpaca_positions}
        state_tickers = set(state.get("open_positions", {}).keys())

        # Remove state positions that Alpaca no longer has (filled exits, manual closes)
        for ticker in state_tickers - alpaca_tickers:
            log.warning(f"Alpaca sync: removing {ticker} from state (not in Alpaca)")
            state["open_positions"].pop(ticker, None)

        # Warn about Alpaca positions not in state (manual trades)
        for ticker in alpaca_tickers - state_tickers:
            log.warning(f"Alpaca sync: {ticker} in Alpaca but not in state (manual trade?)")

    except Exception as e:
        log.warning(f"Alpaca sync failed: {e}")


def run_cycle(cycle_num: int):
    log_cycle_start(cycle_num)
    state = load_state()

    # Sync from Alpaca at start of each cycle
    sync_from_alpaca(state)
    save_state(state)

    # --- EXIT PATH (runs first) ---
    log.info("--- EXIT PATH ---")
    try:
        exits = run_exit_path(state)
        if exits:
            log.info(f"Exited {len(exits)} position(s): {[e['ticker'] for e in exits]}")
        save_state(state)
    except Exception as e:
        log_error("exit_path", f"{e}\n{traceback.format_exc()}")

    # --- ENTRY PATH (runs second) ---
    log.info("--- ENTRY PATH ---")
    try:
        entries = run_entry_path(state)
        if entries:
            log.info(f"Entered {len(entries)} position(s): {[e['ticker'] for e in entries]}")
        save_state(state)
    except Exception as e:
        log_error("entry_path", f"{e}\n{traceback.format_exc()}")

    log_cycle_end(cycle_num)


def main():
    log.info("Quant Agent starting up...")

    sp500 = get_sp500_tickers()
    log.info(f"Loaded {len(sp500)} S&P 500 tickers")

    cycle_num = 0
    while True:
        if not is_market_open():
            now_et = datetime.now(ET)
            log.info(
                f"Market closed ({now_et.strftime('%A %H:%M ET')}). "
                f"Sleeping {CYCLE_INTERVAL_SECONDS // 60} minutes..."
            )
            time.sleep(CYCLE_INTERVAL_SECONDS)
            continue

        cycle_num += 1
        try:
            run_cycle(cycle_num)
        except Exception as e:
            log_error("main_loop", f"{e}\n{traceback.format_exc()}")

        log.info(f"Sleeping {CYCLE_INTERVAL_SECONDS // 60} minutes until next cycle...")
        time.sleep(CYCLE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

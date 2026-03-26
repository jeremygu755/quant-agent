import json
import logging
import sys
from datetime import datetime, timezone
from config import DECISIONS_FILE, DATA_DIR
import os

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("quant-agent")


def log_decision(decision: dict):
    """Append a structured decision record to decisions.jsonl."""
    decision["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(decision, default=str) + "\n")


def log_cycle_start(cycle_num: int):
    log.info(f"=== CYCLE {cycle_num} START ===")
    log_decision({"event": "cycle_start", "cycle": cycle_num})


def log_cycle_end(cycle_num: int):
    log.info(f"=== CYCLE {cycle_num} END ===")
    log_decision({"event": "cycle_end", "cycle": cycle_num})


def log_exit_trade(ticker: str, direction: str, reason: str, entry_price: float,
                   exit_price: float, return_pct: float):
    log.info(
        f"EXIT {direction.upper()} {ticker}: "
        f"entry=${entry_price:.2f} exit=${exit_price:.2f} return={return_pct:+.2f}% "
        f"reason={reason}"
    )
    log_decision({
        "event": "exit",
        "ticker": ticker,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": return_pct,
        "reason": reason,
    })


def log_entry_trade(
    ticker: str,
    direction: str,
    price: float,
    shares: int,
    reason: str,
    tier: int,
    confidence: float,
    event_profile_id: str | None = None,
):
    log.info(
        f"ENTRY {direction.upper()} {ticker}: "
        f"${price:.2f} x{shares} tier={tier} conf={confidence:.0%} "
        f"reason={reason}"
    )
    row = {
        "event": "entry",
        "ticker": ticker,
        "direction": direction,
        "price": price,
        "shares": shares,
        "reason": reason,
        "tier": tier,
        "confidence": confidence,
    }
    if event_profile_id:
        row["event_profile_id"] = event_profile_id
    log_decision(row)


def log_news_cohort_snapshot(payload: dict):
    """Record headline + tier tickers for news-cohort replay / evaluation."""
    log_decision({"event": "news_cohort", **payload})


def log_no_trade(reason: str):
    log.info(f"NO TRADE: {reason}")
    log_decision({"event": "no_trade", "reason": reason})


def log_error(context: str, error: str):
    log.error(f"{context}: {error}")
    log_decision({"event": "error", "context": context, "error": error})

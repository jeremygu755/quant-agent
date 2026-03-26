"""Utilities for news-cohort evaluation using decisions.jsonl records.

Each live cycle may append ``event: news_cohort`` rows (see logger.log_news_cohort_snapshot).
Filter those to rebuild which tickers were in play for which headline/event_profile,
without running the generic random-ticker quant backtest.
"""
import json
from config import DECISIONS_FILE


def iter_news_cohort_records(decisions_path: str | None = None):
    """Yield decoded ``news_cohort`` decision dicts newest-first if file missing is ok."""
    path = decisions_path or DECISIONS_FILE
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "news_cohort":
            yield row


def load_recent_news_cohorts(limit: int = 50) -> list[dict]:
    """Return up to ``limit`` recent news cohort snapshots."""
    out = []
    for row in iter_news_cohort_records():
        out.append(row)
        if len(out) >= limit:
            break
    return out

import os
import feedparser
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from config import YAHOO_RSS_URLS, NEWS_MAX_AGE_MINUTES
from state import is_headline_seen, mark_headline_seen
from logger import log

# Set env QUANT_REREAD_HEADLINES_ONCE=1 when starting the agent: first ``get_new_headlines``
# ignores ``seen_headlines`` so RSS items are re-eligible for Nano (one fetch per process only).
_REREAD_HEADLINES_ONCE_CONSUMED = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _parse_published_time(published_str: str) -> datetime | None:
    """Parse RSS published time string into a timezone-aware datetime."""
    if not published_str:
        return None
    try:
        dt = parsedate_to_datetime(published_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _is_fresh(published_str: str) -> bool:
    """Return True if the headline was published within NEWS_MAX_AGE_MINUTES."""
    pub_time = _parse_published_time(published_str)
    if pub_time is None:
        return True  # if we can't parse the time, let it through
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=NEWS_MAX_AGE_MINUTES)
    return pub_time >= cutoff


def fetch_headlines() -> list[dict]:
    """Fetch headlines from all configured RSS feeds."""
    all_entries = []
    for url in YAHOO_RSS_URLS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                all_entries.append({
                    "id": entry.get("link", entry.get("id", entry.get("title", ""))),
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": url,
                })
        except Exception as e:
            log.warning(f"Failed to fetch RSS from {url}: {e}")
    return all_entries


def get_new_headlines(state: dict) -> list[dict]:
    """Return headlines not yet seen (or all candidates if one-shot reread), fresh, and mark seen."""
    global _REREAD_HEADLINES_ONCE_CONSUMED
    raw = fetch_headlines()
    bypass_dedup = _env_truthy("QUANT_REREAD_HEADLINES_ONCE") and not _REREAD_HEADLINES_ONCE_CONSUMED
    if bypass_dedup:
        _REREAD_HEADLINES_ONCE_CONSUMED = True
        log.info(
            "QUANT_REREAD_HEADLINES_ONCE: skipping seen_headlines for this fetch only "
            f"({len(raw)} raw); freshness filter still applies"
        )

    new = []
    stale_count = 0
    for entry in raw:
        hid = entry["id"]
        if not bypass_dedup and is_headline_seen(state, hid):
            continue
        mark_headline_seen(state, hid)
        if not _is_fresh(entry.get("published", "")):
            stale_count += 1
            continue
        new.append(entry)

    log.info(f"Fetched {len(raw)} headlines, {len(new)} new+fresh, {stale_count} stale skipped")
    return new

import copy
import io
import os
from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Single OHLCV vendor for 5m + daily (quant + backtest): "alpaca" (default, same family as broker) or "yahoo".
_mdp = (
    os.getenv("MARKET_DATA_PROVIDER") or os.getenv("MARKET_DATA_BARS_PROVIDER") or "alpaca"
).strip().lower()
MARKET_DATA_PROVIDER = _mdp if _mdp in ("yahoo", "alpaca") else "alpaca"
MARKET_DATA_BARS_PROVIDER = MARKET_DATA_PROVIDER  # backward compat for env MARKET_DATA_BARS_PROVIDER only

NANO_MODEL = "gpt-4.1-nano"
DEEP_MODEL = "gpt-5.1"

# Hard cap on model output size (completion tokens). Increase if you see JSON parse failures.
LLM_NANO_MAX_COMPLETION_TOKENS = 512
# Batch triage: rank up to N headlines in one Nano call (see ``rank_headlines_nano``).
NANO_BATCH_MAX_HEADLINES = 45
LLM_NANO_BATCH_MAX_COMPLETION_TOKENS = 1024
LLM_DEEP_TIERS_MAX_COMPLETION_TOKENS = 2048
LLM_DEEP_VALIDATE_MAX_COMPLETION_TOKENS = 512

CYCLE_INTERVAL_SECONDS = 1800  # 30 minutes
MAX_POSITIONS = 6
MAX_HISTORY_DAYS = 30

# --- Quant signal thresholds (live entries are news-led; quant confirms / vetoes) ---
# Strict path: weak or missing LLM conviction — require full tape alignment.
ENTRY_SCORE_THRESHOLD = 60  # skip 50–59 band (weak edge in batch backtests)
SHORT_ENTRY_SCORE_THRESHOLD = 65  # shorts require higher conviction
# Long-only hard gates in ``signals._score_long_entry`` (backtest + live quant score).
LONG_ENTRY_MIN_VOLUME_SPIKE = 2.0
LONG_ENTRY_BAN_DECELERATING_MOMENTUM = True
# Stock 1h% minus SPY 1h% must exceed this; requires SPY snapshot. None disables.
LONG_ENTRY_MIN_ABNORMAL_VS_SPY_1H = 1.0
# News-led path: strong tier confidence — thesis comes from headlines; quant still needs this floor.
NEWS_LED_MIN_CONFIDENCE = 0.65
ENTRY_SCORE_FLOOR_NEWS_LED = 60  # aligned with ENTRY_SCORE_THRESHOLD (no sub-60 longs on tape)
NEWS_LED_SHORT_MIN_CONFIDENCE = 0.72
SHORT_ENTRY_SCORE_FLOOR_NEWS_LED = 55
SHORT_SPY_MAX_1H_PCT = 0.0  # SPY 1h return must be <= this to allow shorts (bearish regime)
TAKE_PROFIT_PCT = 2.0
STOP_LOSS_PCT = 1.0
TRAILING_STOP_PCT = 1.0
MAX_HOLD_DAYS = 5
MAX_SECTOR_EXPOSURE_PCT = 30.0
MAX_POSITIONS_PER_SECTOR = 2
NEWS_MAX_AGE_MINUTES = 60
# One-shot RSS reread: ``QUANT_REREAD_HEADLINES_ONCE=1`` → first ``get_new_headlines`` in that process
# ignores ``seen_headlines`` (still drops items older than NEWS_MAX_AGE_MINUTES). See ``news.py``.
# When True, run Deep (tier analysis) only if Nano sets market_impact to "major" (and actionable).
# Set False to use legacy gate: actionable only.
NANO_REQUIRE_MAJOR_MARKET_IMPACT = True
EVENT_DEDUP_HOURS = 4
MIN_HOLD_BARS = 3  # soft exits disabled until trade has been held this many bars (~15 min)


def _base_event_profile(profile_id: str) -> dict:
    """Defaults match global TP/SL/hold; overrides per news regime."""
    return {
        "id": profile_id,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "trailing_stop_pct": TRAILING_STOP_PCT,
        "max_hold_days": MAX_HOLD_DAYS,
        "min_hold_bars": MIN_HOLD_BARS,
        "position_size_multiplier": 1.0,
        "strict_threshold_delta": 0,
        "short_strict_threshold_delta": 0,
        "news_led_floor_delta": 0,
        "short_news_led_floor_delta": 0,
        "min_opportunity_score": 0.0,
        "entry_score_adjustment": 0,
        "require_fast_5m_confirmation": False,
        "min_sector_abnormal_long": None,
        "min_sector_abnormal_short": None,
        "min_volume_spike": None,
        "macro_alignment": False,
        "rumor_fast_exit": False,
        "momentum_fade_ret30m_min": 0.3,
    }


def _ep(profile_id: str, **kwargs) -> dict:
    b = _base_event_profile(profile_id)
    b.update(kwargs)
    return b


# Profiles keyed by id; resolved from Nano event_type via EVENT_TYPE_PATTERN_ORDER.
EVENT_PROFILES: dict[str, dict] = {
    "default": _base_event_profile("default"),
    "earnings_guidance": _ep(
        "earnings_guidance",
        take_profit_pct=3.5,
        stop_loss_pct=1.5,
        trailing_stop_pct=1.5,
        max_hold_days=8,
        min_hold_bars=2,
        position_size_multiplier=1.1,
        entry_score_adjustment=6,
        strict_threshold_delta=0,
        news_led_floor_delta=0,
        short_strict_threshold_delta=-2,
        short_news_led_floor_delta=-3,
    ),
    "rumor_analyst": _ep(
        "rumor_analyst",
        take_profit_pct=1.2,
        stop_loss_pct=0.8,
        trailing_stop_pct=0.6,
        max_hold_days=2,
        min_hold_bars=1,
        position_size_multiplier=0.55,
        strict_threshold_delta=5,
        short_strict_threshold_delta=4,
        news_led_floor_delta=8,
        short_news_led_floor_delta=6,
        min_opportunity_score=0.5,
        require_fast_5m_confirmation=True,
        rumor_fast_exit=True,
        momentum_fade_ret30m_min=0.15,
    ),
    "sector_sympathy": _ep(
        "sector_sympathy",
        take_profit_pct=2.2,
        stop_loss_pct=1.1,
        trailing_stop_pct=1.0,
        max_hold_days=5,
        min_hold_bars=3,
        position_size_multiplier=0.95,
        min_sector_abnormal_long=0.35,
        min_sector_abnormal_short=0.35,
        min_volume_spike=1.5,
        strict_threshold_delta=2,
        news_led_floor_delta=4,
    ),
    "macro": _ep(
        "macro",
        take_profit_pct=2.5,
        stop_loss_pct=1.2,
        trailing_stop_pct=1.2,
        max_hold_days=6,
        min_hold_bars=3,
        position_size_multiplier=0.9,
        macro_alignment=True,
        strict_threshold_delta=2,
        short_strict_threshold_delta=2,
        news_led_floor_delta=2,
    ),
}

# First matching substring wins (event_type uppercased).
EVENT_TYPE_PATTERN_ORDER: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "EARNINGS",
            "GUIDANCE",
            "EPS",
            "REVENUE",
            "EBITDA",
            "QUARTERLY",
            "BEAT",
            "MISS",
        ),
        "earnings_guidance",
    ),
    (
        (
            "RUMOR",
            "SPECULATION",
            "ANALYST",
            "UPGRADE",
            "DOWNGRADE",
            "PRICE_TARGET",
            "REITERAT",
        ),
        "rumor_analyst",
    ),
    (
        ("SYMPATHY", "PEER", "SUPPLY_CHAIN", "CORRELATED", "CONTAGION"),
        "sector_sympathy",
    ),
    (
        (
            "FED",
            "RATE_CUT",
            "RATE_HIKE",
            "CPI",
            "PCE",
            "NFP",
            "JOBS_REPORT",
            "GDP",
            "INFLATION",
            "MACRO",
            "TREASURY",
            "YIELD",
            "RECESSION",
        ),
        "macro",
    ),
]


def resolve_event_profile(event_type: str | None) -> dict:
    """Map classifier event_type string to a mutable profile copy."""
    et = (event_type or "").upper()
    for patterns, profile_id in EVENT_TYPE_PATTERN_ORDER:
        if any(p in et for p in patterns):
            return copy.deepcopy(EVENT_PROFILES[profile_id])
    return copy.deepcopy(EVENT_PROFILES["default"])


# Position sizing by entry score
POSITION_SIZE_MAP = {
    80: 5000,   # score 80-100 → $5k
    65: 3500,   # score 65-79  → $3.5k
    50: 2000,   # score 50-64  → $2k
}

SECTOR_ETF_MAP = {
    "energy": "XLE",
    "technology": "XLK",
    "financials": "XLF",
    "healthcare": "XLV",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication_services": "XLC",
}

# Reverse map: ETF ticker → sector name
ETF_TO_SECTOR = {v: k for k, v in SECTOR_ETF_MAP.items()}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
HISTORY_FILE = os.path.join(DATA_DIR, "history.jsonl")
DECISIONS_FILE = os.path.join(DATA_DIR, "decisions.jsonl")

YAHOO_RSS_URLS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://finance.yahoo.com/rss/topstories",
]

_sp500_cache: set[str] | None = None
_ticker_sector_cache: dict[str, str] | None = None


def get_sp500_tickers() -> set[str]:
    """Pull current S&P 500 tickers from Wikipedia. Cached after first call."""
    global _sp500_cache
    if _sp500_cache is not None:
        return _sp500_cache
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (quant-agent)"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        _sp500_cache = set(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        _sp500_cache = set()
    return _sp500_cache


def get_ticker_sector(ticker: str) -> str:
    """Get GICS sector for an S&P 500 ticker. Returns 'unknown' if not found."""
    global _ticker_sector_cache
    if _ticker_sector_cache is None:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": "Mozilla/5.0 (quant-agent)"},
                timeout=15,
            )
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            df = tables[0]
            df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
            _ticker_sector_cache = dict(zip(df["Symbol"], df["GICS Sector"]))
        except Exception:
            _ticker_sector_cache = {}
    sector = _ticker_sector_cache.get(ticker, "unknown")
    return sector.lower().replace(" ", "_") if sector != "unknown" else "unknown"


def get_position_size(entry_score: int) -> int:
    """Return dollar amount to invest based on entry score."""
    for threshold, size in sorted(POSITION_SIZE_MAP.items(), reverse=True):
        if entry_score >= threshold:
            return size
    return 0

"""Unified OHLCV: 5m and daily from one vendor (Alpaca or Yahoo).

Default is **Alpaca** so intraday + daily match your execution stack; falls back to Yahoo
if keys are missing or the API errors. Configure with ``MARKET_DATA_PROVIDER`` (or legacy
``MARKET_DATA_BARS_PROVIDER``). ``ALPACA_BAR_FEED`` selects iex / sip / delayed_sip.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    MARKET_DATA_PROVIDER,
)
from logger import log

_ET = "America/New_York"


def _alpaca_feed():
    from alpaca.data.enums import DataFeed

    raw = os.getenv("ALPACA_BAR_FEED", "iex").strip().lower()
    return {
        "iex": DataFeed.IEX,
        "sip": DataFeed.SIP,
        "delayed_sip": DataFeed.DELAYED_SIP,
    }.get(raw, DataFeed.IEX)


def _yahoo_to_standard_utc(hist: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo index to UTC; keep Open High Low Close Volume columns."""
    if hist.empty:
        return hist
    out = hist.copy()
    idx = out.index
    if idx.tz is None:
        out.index = idx.tz_localize(_ET, ambiguous="infer", nonexistent="shift_forward")
    else:
        out.index = idx.tz_convert(timezone.utc)
    return out.sort_index()


def _alpaca_slice_to_standard(df_sym: pd.DataFrame) -> pd.DataFrame:
    """Single-symbol Alpaca BarSet slice → same columns as Yahoo-shaped frames."""
    if df_sym.empty:
        return df_sym
    idx = pd.DatetimeIndex(df_sym.index)
    if idx.tz is None:
        idx = idx.tz_localize(timezone.utc)
    else:
        idx = idx.tz_convert(timezone.utc)
    return pd.DataFrame(
        {
            "Open": df_sym["open"].astype(float),
            "High": df_sym["high"].astype(float),
            "Low": df_sym["low"].astype(float),
            "Close": df_sym["close"].astype(float),
            "Volume": df_sym["volume"].astype(float),
        },
        index=idx,
    ).sort_index()


def _alpaca_bars(
    sym: str,
    timeframe,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    """Fetch Alpaca bars for one symbol; return standard OHLCV or None."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=_alpaca_feed(),
        )
        barset = client.get_stock_bars(req)
        df = barset.df
        if df is None or df.empty:
            return None
        try:
            sub = df.xs(sym, level=0)
        except KeyError:
            log.warning(f"Alpaca bars: no rows for {sym}")
            return None
        return _alpaca_slice_to_standard(sub)
    except Exception as e:
        log.warning(f"Alpaca bars failed for {sym}: {e}")
        return None


def fetch_5m_ohlcv(ticker: str, days: int) -> pd.DataFrame | None:
    """5-minute OHLCV, UTC index. Bar timestamp = bar open (Alpaca convention)."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    sym = ticker.strip().upper()
    days = max(1, int(days))

    if MARKET_DATA_PROVIDER == "alpaca":
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 1)
        out = _alpaca_bars(sym, TimeFrame(5, TimeFrameUnit.Minute), start, end)
        if out is not None and len(out) >= 5:
            return out
        log.warning(f"Alpaca 5m insufficient for {sym}; falling back to Yahoo")

    try:
        stock = yf.Ticker(sym)
        period = f"{max(days, 5)}d"
        hist = stock.history(period=period, interval="5m")
        if hist.empty or len(hist) < 5:
            return None
        return _yahoo_to_standard_utc(hist)
    except Exception as e:
        log.warning(f"Yahoo 5m fetch failed for {sym}: {e}")
        return None


def fetch_daily_ohlcv(ticker: str, days: int) -> pd.DataFrame | None:
    """Daily OHLCV (UTC index) for volume/volatility vs 20d features."""
    from alpaca.data.timeframe import TimeFrame

    sym = ticker.strip().upper()
    days = max(5, int(days))
    # Extra calendar slack so we get enough trading days (weekends/holidays).
    cal_days = max(days + 35, 45)

    if MARKET_DATA_PROVIDER == "alpaca":
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=cal_days)
        out = _alpaca_bars(sym, TimeFrame.Day, start, end)
        if out is not None and len(out) >= 2:
            return out.tail(days + 10)
        log.warning(f"Alpaca daily insufficient for {sym}; falling back to Yahoo")

    try:
        stock = yf.Ticker(sym)
        hist = stock.history(period=f"{cal_days}d", interval="1d")
        if hist.empty:
            return None
        return _yahoo_to_standard_utc(hist)
    except Exception as e:
        log.warning(f"Yahoo daily fetch failed for {sym}: {e}")
        return None


def last_bar_timestamp_utc(hist_5m: pd.DataFrame) -> datetime | None:
    """UTC timestamp of the latest 5m bar (for logging / reconciliation)."""
    if hist_5m is None or hist_5m.empty:
        return None
    ts = hist_5m.index[-1]
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

"""Market snapshots from unified OHLCV (see ``MARKET_DATA_PROVIDER`` in config).

Default vendor is **Alpaca** for both 5m and daily so tape matches broker; Yahoo is used
only if Alpaca fails or API keys are missing.
"""
from datetime import datetime, timedelta, timezone

from config import MARKET_DATA_PROVIDER, SECTOR_ETF_MAP, get_ticker_sector
from logger import log
from market_data import fetch_5m_ohlcv, fetch_daily_ohlcv, last_bar_timestamp_utc


def _safe_pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 3)


def get_stock_snapshot(ticker: str) -> dict | None:
    """Pull current price, volume, short-term returns, and momentum for a ticker."""
    try:
        hist_5m = fetch_5m_ohlcv(ticker, days=5)
        hist_daily = fetch_daily_ohlcv(ticker, days=30)

        if hist_5m is None or hist_daily is None or hist_5m.empty or hist_daily.empty:
            return None

        current_price = hist_5m["Close"].iloc[-1]
        current_volume = hist_5m["Volume"].iloc[-1]

        now_idx = hist_5m.index[-1]
        price_5m_ago = _find_price_at(hist_5m, now_idx - timedelta(minutes=5))
        price_30m_ago = _find_price_at(hist_5m, now_idx - timedelta(minutes=30))
        price_1h_ago = _find_price_at(hist_5m, now_idx - timedelta(hours=1))
        price_1d_ago = hist_daily["Close"].iloc[-2] if len(hist_daily) >= 2 else current_price

        avg_volume_20d = hist_daily["Volume"].tail(20).mean() if len(hist_daily) >= 20 else hist_daily["Volume"].mean()
        intraday_vol_avg = hist_5m["Volume"].tail(12).mean()  # last hour of 5m bars
        volume_spike = round(current_volume / avg_volume_20d, 2) if avg_volume_20d > 0 else 1.0

        returns_daily = hist_daily["Close"].pct_change().dropna()
        vol_5d = returns_daily.tail(5).std() if len(returns_daily) >= 5 else 0
        vol_30d = returns_daily.std() if len(returns_daily) > 1 else 1
        volatility_ratio = round(vol_5d / vol_30d, 2) if vol_30d > 0 else 1.0

        pct_5m = _safe_pct_change(current_price, price_5m_ago)
        pct_30m = _safe_pct_change(current_price, price_30m_ago)
        pct_1h = _safe_pct_change(current_price, price_1h_ago)
        pct_1d = _safe_pct_change(current_price, price_1d_ago)

        price_15m_ago = _find_price_at(hist_5m, now_idx - timedelta(minutes=15))
        recent_move = _safe_pct_change(current_price, price_15m_ago)
        prior_move = _safe_pct_change(price_15m_ago, price_30m_ago)
        if abs(prior_move) < 0.001:
            momentum = "flat"
        elif abs(recent_move) > abs(prior_move):
            momentum = "accelerating"
        elif recent_move * prior_move < 0:
            momentum = "reversing"
        else:
            momentum = "decelerating"

        last_ts = last_bar_timestamp_utc(hist_5m)

        return {
            "ticker": ticker,
            "market_data_provider": MARKET_DATA_PROVIDER,
            "last_bar_ts_utc": last_ts.isoformat() if last_ts else None,
            "current_price": round(current_price, 2),
            "price_5m_ago": round(price_5m_ago, 2),
            "price_30m_ago": round(price_30m_ago, 2),
            "price_1h_ago": round(price_1h_ago, 2),
            "price_1d_ago": round(price_1d_ago, 2),
            "pct_change_5m": pct_5m,
            "pct_change_30m": pct_30m,
            "pct_change_1h": pct_1h,
            "pct_change_1d": pct_1d,
            "current_volume": int(current_volume),
            "avg_volume_20d": int(avg_volume_20d),
            "volume_spike": volume_spike,
            "volatility_ratio": volatility_ratio,
            "momentum": momentum,
        }
    except Exception as e:
        log.warning(f"Failed to get snapshot for {ticker}: {e}")
        return None


def _find_price_at(hist, target_time) -> float:
    """Find the closest price at or before target_time."""
    mask = hist.index <= target_time
    if mask.any():
        return hist.loc[mask, "Close"].iloc[-1]
    return hist["Close"].iloc[0]


def get_spy_snapshot() -> dict | None:
    """Convenience: get SPY snapshot."""
    return get_stock_snapshot("SPY")


def get_sector_snapshot(ticker: str) -> dict | None:
    """Get the sector ETF snapshot for a given stock ticker."""
    sector = get_ticker_sector(ticker)
    etf = SECTOR_ETF_MAP.get(sector)
    if not etf:
        return None
    return get_stock_snapshot(etf)


def get_market_context() -> dict:
    """Get SPY + sector ETF snapshots for broad market context."""
    context = {}
    spy = get_stock_snapshot("SPY")
    if spy:
        context["SPY"] = spy

    for sector, etf in SECTOR_ETF_MAP.items():
        snap = get_stock_snapshot(etf)
        if snap:
            context[f"{sector}_{etf}"] = snap

    return context


def compute_opportunity_score(ticker_snapshot: dict, expected_move_pct: float,
                              direction: str) -> float:
    """How much room is left in the trade: expected move minus actual move so far."""
    actual_move = ticker_snapshot.get("pct_change_1h", 0.0)
    if direction == "short":
        actual_move = -actual_move
    remaining = expected_move_pct - actual_move
    return round(max(remaining, 0), 2)


def format_snapshot_for_llm(snapshot: dict) -> str:
    """Format a single stock snapshot as readable text for the LLM."""
    if not snapshot:
        return "  (no data available)"
    return (
        f"  {snapshot['ticker']}: ${snapshot['current_price']}\n"
        f"    5m: {snapshot.get('pct_change_5m', 0):+.2f}% | "
        f"30min: {snapshot['pct_change_30m']:+.2f}% | "
        f"1h: {snapshot['pct_change_1h']:+.2f}% | "
        f"1d: {snapshot['pct_change_1d']:+.2f}%\n"
        f"    Volume: {snapshot['volume_spike']}x 20d avg | "
        f"Volatility: {snapshot['volatility_ratio']}x normal\n"
        f"    Momentum: {snapshot['momentum']}"
    )


def format_market_data(snapshots: dict[str, dict], market_ctx: dict) -> str:
    """Build the full market data string for the LLM prompt."""
    lines = ["--- Candidate Stocks ---"]
    for ticker, snap in snapshots.items():
        lines.append(format_snapshot_for_llm(snap))

    lines.append("\n--- Broad Market Context ---")
    if "SPY" in market_ctx:
        lines.append(format_snapshot_for_llm(market_ctx["SPY"]))
    for key, snap in market_ctx.items():
        if key != "SPY":
            lines.append(format_snapshot_for_llm(snap))

    return "\n".join(lines)

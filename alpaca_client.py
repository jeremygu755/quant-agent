from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from logger import log

_trading_client = None
_data_client = None


def _get_trading_client() -> TradingClient:
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    return _trading_client


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


def get_account() -> dict:
    try:
        acct = _get_trading_client().get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
        }
    except Exception as e:
        log.error(f"Failed to get account: {e}")
        return {"cash": 0, "equity": 0, "buying_power": 0}


def get_positions() -> list[dict]:
    try:
        positions = _get_trading_client().get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": int(p.qty),
                "side": p.side.value,
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]
    except Exception as e:
        log.error(f"Failed to get positions: {e}")
        return []


def get_position_side_for_ticker(ticker: str) -> str | None:
    """Alpaca position side for symbol: 'long' or 'short', or None if not held."""
    sym = ticker.strip().upper()
    for p in get_positions():
        if p["ticker"] == sym:
            s = str(p.get("side", "")).lower()
            if s in ("long", "short"):
                return s
    return None


def assert_canonical_direction(direction: str, *, context: str) -> None:
    """Require direction in {long, short}; log and raise before broker calls."""
    if direction not in ("long", "short"):
        log.error(f"{context}: invalid direction {direction!r} (expected 'long' or 'short')")
        raise ValueError(
            f"{context}: direction must be 'long' or 'short', got {direction!r}"
        )


def _submit_order(ticker: str, qty: int, side: OrderSide, label: str) -> dict | None:
    try:
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = _get_trading_client().submit_order(req)
        log.info(f"{label} order submitted: {ticker} x{qty}, order_id={order.id}")
        return {"order_id": str(order.id), "status": str(order.status)}
    except Exception as e:
        log.error(f"{label} order failed for {ticker}: {e}")
        return None


def buy_stock(
    ticker: str, qty: int, *, position_direction: str = "long", context: str = "buy_stock"
) -> dict | None:
    assert_canonical_direction(position_direction, context=f"{context} position_direction")
    if position_direction != "long":
        raise ValueError(f"{context}: buy_stock opens a long; got position_direction={position_direction!r}")
    return _submit_order(ticker, qty, OrderSide.BUY, "BUY")


def sell_stock(
    ticker: str, qty: int, *, position_direction: str = "long", context: str = "sell_stock"
) -> dict | None:
    assert_canonical_direction(position_direction, context=f"{context} position_direction")
    if position_direction != "long":
        raise ValueError(f"{context}: sell_stock closes a long; got position_direction={position_direction!r}")
    return _submit_order(ticker, qty, OrderSide.SELL, "SELL")


def short_stock(
    ticker: str, qty: int, *, position_direction: str = "short", context: str = "short_stock"
) -> dict | None:
    assert_canonical_direction(position_direction, context=f"{context} position_direction")
    if position_direction != "short":
        raise ValueError(f"{context}: short_stock opens a short; got position_direction={position_direction!r}")
    return _submit_order(ticker, qty, OrderSide.SELL, "SHORT")


def cover_short(
    ticker: str, qty: int, *, position_direction: str = "short", context: str = "cover_short"
) -> dict | None:
    assert_canonical_direction(position_direction, context=f"{context} position_direction")
    if position_direction != "short":
        raise ValueError(f"{context}: cover_short closes a short; got position_direction={position_direction!r}")
    return _submit_order(ticker, qty, OrderSide.BUY, "COVER")


def get_current_price(ticker: str) -> float | None:
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        trades = _get_data_client().get_stock_latest_trade(req)
        if ticker in trades:
            return float(trades[ticker].price)
        return None
    except Exception as e:
        log.warning(f"Failed to get price for {ticker}: {e}")
        return None

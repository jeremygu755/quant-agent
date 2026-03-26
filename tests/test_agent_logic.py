"""Unit tests for direction normalization, TP/SL, exits, tier enforcement, and order guards."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpaca_client import (  # noqa: E402
    assert_canonical_direction,
    buy_stock,
    cover_short,
    sell_stock,
    short_stock,
)
from llm import _enforce_tier_directions_post_parse  # noqa: E402
from signals import (  # noqa: E402
    check_exit_signals,
    compute_rule_based_tp_sl,
    normalize_trade_direction,
    update_high_water_mark,
)
from buy import _nano_passes_for_deep  # noqa: E402


class TestNormalizeTradeDirection(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(normalize_trade_direction("long", "X"), "long")
        self.assertEqual(normalize_trade_direction("short", "X"), "short")
        self.assertEqual(normalize_trade_direction("  LONG  ", "X"), "long")

    def test_prose_multi(self):
        s = "ANET: long; DDOG: long; SNOW: long"
        self.assertEqual(normalize_trade_direction(s, "DDOG"), "long")
        self.assertEqual(normalize_trade_direction(s, "ANET"), "long")

    def test_fail_closed(self):
        self.assertIsNone(normalize_trade_direction("", "X"))
        self.assertIsNone(normalize_trade_direction("random text", "X"))
        prose = "ANET: long; DDOG: long"
        self.assertIsNone(normalize_trade_direction(prose, "ZZZ"))

    def test_conflicting(self):
        bad = "ddog: long; ddog: short"
        self.assertIsNone(normalize_trade_direction(bad, "DDOG"))


class TestComputeRuleBasedTpSl(unittest.TestCase):
    def test_long(self):
        tp, sl = compute_rule_based_tp_sl(100.0, "long", 2.0, 1.0)
        self.assertEqual(tp, 102.0)
        self.assertEqual(sl, 99.0)

    def test_short(self):
        tp, sl = compute_rule_based_tp_sl(100.0, "short", 2.0, 1.0)
        self.assertEqual(tp, 98.0)
        self.assertEqual(sl, 101.0)


class TestCheckExitSignals(unittest.TestCase):
    def base_pos_long(self):
        return {
            "direction": "long",
            "entry_price": 100.0,
            "take_profit": 102.0,
            "stop_loss": 99.0,
            "high_water_mark": 101.0,
            "bars_held": 10,
            "min_hold_bars": 3,
            "trailing_stop_pct": 1.0,
            "entry_volume_spike": 1.0,
        }

    def test_invalid_direction_returns_none(self):
        pos = self.base_pos_long()
        pos["direction"] = "garbage"
        self.assertIsNone(
            check_exit_signals(pos, 100.5, {"pct_change_5m": 0}, {"pct_change_1h": 0})
        )

    def test_stop_loss_long(self):
        pos = self.base_pos_long()
        r = check_exit_signals(pos, 98.0, None, None)
        self.assertIsNotNone(r)
        self.assertIn("stop_loss", r)

    def test_take_profit_short(self):
        pos = {
            "direction": "short",
            "entry_price": 100.0,
            "take_profit": 98.0,
            "stop_loss": 101.0,
            "high_water_mark": 99.0,
            "bars_held": 10,
            "min_hold_bars": 3,
            "trailing_stop_pct": 1.0,
            "entry_volume_spike": 1.0,
        }
        r = check_exit_signals(pos, 97.5, None, None)
        self.assertIsNotNone(r)
        self.assertIn("take_profit", r)


class TestUpdateHighWaterMark(unittest.TestCase):
    def test_long_tracks_peak(self):
        pos = {"direction": "long", "entry_price": 100.0, "high_water_mark": 100.0}
        update_high_water_mark(pos, 105.0)
        self.assertEqual(pos["high_water_mark"], 105.0)

    def test_short_tracks_trough(self):
        pos = {"direction": "short", "entry_price": 100.0, "high_water_mark": 100.0}
        update_high_water_mark(pos, 95.0)
        self.assertEqual(pos["high_water_mark"], 95.0)


class TestEnforceTierDirections(unittest.TestCase):
    def test_canonical_passthrough(self):
        parsed = {
            "tiers": [
                {
                    "tier": 1,
                    "tickers": ["AAPL", "MSFT"],
                    "direction": "long",
                }
            ]
        }
        out = _enforce_tier_directions_post_parse(parsed)
        self.assertEqual(out["tiers"][0]["tickers"], ["AAPL", "MSFT"])
        self.assertEqual(out["tiers"][0]["direction"], "long")

    def test_prose_unified(self):
        parsed = {
            "tiers": [
                {
                    "tier": 2,
                    "tickers": ["DDOG", "ANET"],
                    "direction": "DDOG: long; ANET: long",
                }
            ]
        }
        out = _enforce_tier_directions_post_parse(parsed)
        self.assertEqual(out["tiers"][0]["tickers"], ["DDOG", "ANET"])
        self.assertEqual(out["tiers"][0]["direction"], "long")

    def test_mixed_side_clears_tickers(self):
        parsed = {
            "tiers": [
                {
                    "tier": 2,
                    "tickers": ["X", "Y"],
                    "direction": "X: long; Y: short",
                }
            ]
        }
        out = _enforce_tier_directions_post_parse(parsed)
        self.assertEqual(out["tiers"][0]["tickers"], [])


class TestAssertCanonicalDirection(unittest.TestCase):
    def test_ok(self):
        assert_canonical_direction("long", context="test")
        assert_canonical_direction("short", context="test")

    def test_bad_raises(self):
        with self.assertRaises(ValueError):
            assert_canonical_direction("maybe", context="test")


class TestAlpacaOrderDirectionGuards(unittest.TestCase):
    @patch("alpaca_client._submit_order")
    def test_buy_stock_rejects_non_long_opening(self, _m):
        with self.assertRaises(ValueError):
            buy_stock("SPY", 1, position_direction="short", context="test")

    @patch("alpaca_client._submit_order")
    def test_short_stock_rejects_non_short_opening(self, _m):
        with self.assertRaises(ValueError):
            short_stock("SPY", 1, position_direction="long", context="test")

    @patch("alpaca_client._submit_order")
    def test_sell_stock_rejects_non_long_close(self, _m):
        with self.assertRaises(ValueError):
            sell_stock("SPY", 1, position_direction="short", context="test")

    @patch("alpaca_client._submit_order")
    def test_cover_rejects_non_short_close(self, _m):
        with self.assertRaises(ValueError):
            cover_short("SPY", 1, position_direction="long", context="test")


class TestNanoPassesForDeep(unittest.TestCase):
    def test_major_passes(self):
        c = {
            "actionable": True,
            "market_impact": "major",
            "magnitude": "high",
        }
        self.assertTrue(_nano_passes_for_deep(c)[0])

    def test_not_actionable(self):
        self.assertFalse(_nano_passes_for_deep({"actionable": False})[0])


if __name__ == "__main__":
    unittest.main()

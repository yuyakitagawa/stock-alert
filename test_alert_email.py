"""Unit tests for alert_email.py helper functions"""
import unittest
import os
import sys
sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from alert_email import (
    get_judgment,
    build_sparkline_svg,
    build_priority_actions,
    build_diff_section,
    build_sector_warning,
)


def _make_result(code="1234", name="テスト株", net=0.0, signal="hold",
                 drop_prob=30.0, vol=25.0, rel20=None, prices_close=None):
    return {
        "code": code, "name": name, "prob": 40.0 + net, "drop_prob": drop_prob,
        "net": net, "close": 1000.0, "signal": signal,
        "vol": vol, "vol_label": "🟡中", "recommend": "⏳ 様子見",
        "rel5": None, "rel20": rel20,
        "prices_close": prices_close or list(range(100, 160)),
    }


class TestGetJudgment(unittest.TestCase):
    def test_strong_buy(self):
        label, color = get_judgment(15)
        self.assertIn("強気", label)
        self.assertEqual(color, "#1a7a1a")

    def test_sell(self):
        label, _ = get_judgment(-20)
        self.assertIn("売り", label)

    def test_neutral(self):
        label, _ = get_judgment(0)
        self.assertIn("中立", label)

    def test_boundary_15(self):
        label, _ = get_judgment(15)
        self.assertIn("強気買い", label)

    def test_boundary_minus5(self):
        label, _ = get_judgment(-5)
        self.assertIn("中立", label)

    def test_boundary_minus5_1(self):
        label, _ = get_judgment(-5.1)
        self.assertIn("弱気", label)


class TestBuildSparklineSvg(unittest.TestCase):
    def test_returns_svg(self):
        prices = list(range(100, 160))
        svg = build_sparkline_svg(prices)
        self.assertIn("<svg", svg)
        self.assertIn("polyline", svg)

    def test_green_when_rising(self):
        prices = list(range(100, 160))
        svg = build_sparkline_svg(prices)
        self.assertIn("#0a7a0a", svg)

    def test_red_when_falling(self):
        prices = list(range(159, 99, -1))
        svg = build_sparkline_svg(prices)
        self.assertIn("#c0392b", svg)

    def test_empty_for_flat(self):
        svg = build_sparkline_svg([100] * 60)
        self.assertEqual(svg, "")

    def test_empty_for_short_data(self):
        svg = build_sparkline_svg([100])
        self.assertEqual(svg, "")

    def test_uses_last_60_bars(self):
        prices = [50] * 100 + [100] * 60  # last 60 are all 100 (flat)
        svg = build_sparkline_svg(prices)
        self.assertEqual(svg, "")  # flat → empty


class TestBuildPriorityActions(unittest.TestCase):
    def test_sell_comes_first(self):
        results = [
            _make_result("1111", "売り株", net=-8.0, signal="sell", drop_prob=45.0),
            _make_result("2222", "強気株", net=18.0, signal="hold"),
        ]
        actions = build_priority_actions(results)
        self.assertEqual(actions[0]["emoji"], "🔴")

    def test_max_3_actions(self):
        results = [_make_result(str(i), f"株{i}", net=-8.0, signal="sell") for i in range(10)]
        actions = build_priority_actions(results)
        self.assertLessEqual(len(actions), 3)

    def test_strong_buy_included(self):
        results = [_make_result("3333", "強気株", net=20.0, signal="hold")]
        actions = build_priority_actions(results)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["emoji"], "🟢")

    def test_empty_when_no_signals(self):
        results = [_make_result("4444", "普通株", net=2.0, signal="hold")]
        actions = build_priority_actions(results)
        self.assertEqual(actions, [])


class TestBuildDiffSection(unittest.TestCase):
    def test_shows_significant_change(self):
        results = [_make_result("1234", "テスト株", net=10.0)]
        prev    = {"1234": {"net": 5.0, "code": "1234", "name": "テスト株", "signal": "hold"}}
        html = build_diff_section(results, prev)
        self.assertIn("▲", html)
        self.assertIn("テスト株", html)

    def test_empty_when_no_prev(self):
        results = [_make_result("1234", net=10.0)]
        html = build_diff_section(results, {})
        self.assertEqual(html, "")

    def test_empty_when_small_change(self):
        results = [_make_result("1234", net=5.1)]
        prev    = {"1234": {"net": 4.0, "code": "1234", "name": "株", "signal": "hold"}}
        html = build_diff_section(results, prev)
        self.assertEqual(html, "")  # 1.1% < 3% threshold

    def test_decline_shown_with_down_arrow(self):
        results = [_make_result("1234", net=-5.0)]
        prev    = {"1234": {"net": 5.0, "code": "1234", "name": "株", "signal": "hold"}}
        html = build_diff_section(results, prev)
        self.assertIn("▼", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)

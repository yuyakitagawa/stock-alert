"""Unit tests for alert_email.py helper functions"""
import unittest
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from alert_email import (
    get_judgment,
    build_sparkline_svg,
    build_priority_actions,
    build_diff_section,
    build_sector_warning,
    _row_code_str,
    _row_net_percent,
    _safe_float,
    _tiered_sell_signal,
)


def _make_result(code="1234", name="テスト株", net=0.0, signal="hold",
                 drop_prob=30.0, vol=25.0, rel20=None, prices_close=None):
    return {
        "code": code, "name": name, "prob": 40.0 + net, "drop_prob": drop_prob,
        "net": net, "close": 1000.0, "signal": signal,
        "vol": vol, "vol_label": "🟡中", "recommend": "⏳ 方向感なし",
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
    """ランキングCSVは環境依存のため ranking_df で注入して決定的にする。"""

    def test_sell_comes_first(self):
        results = [
            _make_result("1111", "売り株", net=-8.0, signal="sell", drop_prob=45.0),
            _make_result("2222", "強気株", net=18.0, signal="hold"),
        ]
        actions = build_priority_actions(results, ranking_df=pd.DataFrame())
        self.assertEqual(actions[0]["emoji"], "🔴")

    def test_max_3_actions(self):
        results = [_make_result(str(i), f"株{i}", net=-8.0, signal="sell") for i in range(10)]
        actions = build_priority_actions(results, ranking_df=pd.DataFrame())
        self.assertLessEqual(len(actions), 3)

    def test_ranking_buy_when_no_sells(self):
        results = [_make_result("3333", "保有のみ", net=20.0, signal="hold")]
        ranking = pd.DataFrame([
            {"銘柄コード": 9999, "銘柄名": "スクリーン候補", "ネット(%)": 10.0, "ボラ(%)": 22.0},
        ])
        actions = build_priority_actions(results, ranking_df=ranking)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["emoji"], "✅")

    def test_empty_when_no_signals(self):
        results = [_make_result("4444", "普通株", net=2.0, signal="hold")]
        actions = build_priority_actions(results, ranking_df=pd.DataFrame())
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


class TestRowCodeStr(unittest.TestCase):
    def test_normal_int(self):
        self.assertEqual(_row_code_str({"銘柄コード": 1234}), "1234")

    def test_normal_float(self):
        self.assertEqual(_row_code_str({"銘柄コード": 1234.0}), "1234")

    def test_string_numeric(self):
        self.assertEqual(_row_code_str({"銘柄コード": "1234"}), "1234")

    def test_nan_returns_none(self):
        self.assertIsNone(_row_code_str({"銘柄コード": float("nan")}))

    def test_none_returns_none(self):
        self.assertIsNone(_row_code_str({"銘柄コード": None}))

    def test_non_numeric_string_returns_none(self):
        self.assertIsNone(_row_code_str({"銘柄コード": "ABC"}))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_row_code_str({"銘柄コード": ""}))


class TestAbnormalDataRobustness(unittest.TestCase):
    """異常値を含むranking DataFrameでクラッシュしないことを確認"""

    def _make_ranking(self, rows):
        return pd.DataFrame(rows)

    def test_priority_actions_with_nan_code(self):
        results = [_make_result("1111", "保有株", net=5.0, signal="hold")]
        ranking = self._make_ranking([
            {"銘柄コード": float("nan"), "銘柄名": "欠損コード株", "ネット(%)": 10.0, "ボラ(%)": 22.0, "下落確率(%)": 5.0},
            {"銘柄コード": 9999, "銘柄名": "正常株", "ネット(%)": 10.0, "ボラ(%)": 22.0, "下落確率(%)": 5.0},
        ])
        actions = build_priority_actions(results, ranking_df=ranking)
        # NaNコードでクラッシュせず、正常な行は処理される
        self.assertIsInstance(actions, list)

    def test_priority_actions_with_non_numeric_code(self):
        results = [_make_result("1111", "保有株", net=5.0, signal="hold")]
        ranking = self._make_ranking([
            {"銘柄コード": "INVALID", "銘柄名": "不正コード株", "ネット(%)": 10.0, "ボラ(%)": 22.0, "下落確率(%)": 5.0},
        ])
        actions = build_priority_actions(results, ranking_df=ranking)
        self.assertIsInstance(actions, list)

    def test_priority_actions_with_missing_net(self):
        results = [_make_result("1111", "保有株", net=5.0, signal="hold")]
        ranking = self._make_ranking([
            {"銘柄コード": 9999, "銘柄名": "ネット欠損株", "ネット(%)": None, "ボラ(%)": 22.0, "下落確率(%)": 5.0},
        ])
        actions = build_priority_actions(results, ranking_df=ranking)
        self.assertEqual(actions, [])

    def test_priority_actions_with_string_drop_prob(self):
        results = [_make_result("1111", "保有株", net=5.0, signal="hold")]
        ranking = self._make_ranking([
            {"銘柄コード": 9999, "銘柄名": "下落確率文字列", "ネット(%)": 10.0, "ボラ(%)": 22.0, "下落確率(%)": "-"},
        ])
        # _safe_float("-") → None として扱われ、クラッシュしない
        actions = build_priority_actions(results, ranking_df=ranking)
        self.assertIsInstance(actions, list)

    def test_safe_float_with_invalid(self):
        self.assertIsNone(_safe_float("-"))
        self.assertIsNone(_safe_float(None))
        self.assertIsNone(_safe_float("N/A"))
        self.assertEqual(_safe_float(3.14), 3.14)
        self.assertEqual(_safe_float(5), 5.0)


class TestTieredSellSignal(unittest.TestCase):
    """保有日数別段階的売りシグナルのテスト"""

    def test_early_phase_normal_threshold(self):
        """0-3日: net<-5%のみ売り"""
        self.assertEqual(_tiered_sell_signal(-5.1, 2), "sell")
        self.assertEqual(_tiered_sell_signal(-4.9, 2), "hold")
        self.assertEqual(_tiered_sell_signal(0.0,  2), "hold")

    def test_mid_phase_tighter_threshold(self):
        """4-63日: net<6%で売り（A買い基準を下回ったら乗り換え）"""
        self.assertEqual(_tiered_sell_signal(5.9,  45), "sell")
        self.assertEqual(_tiered_sell_signal(6.0,  45), "hold")
        # 通常なら hold のままだが中期では sell になるケース
        self.assertEqual(_tiered_sell_signal(1.0,  45), "sell")
        self.assertEqual(_tiered_sell_signal(1.0,   3), "hold")

    def test_late_phase_tightest_threshold(self):
        """63日超: net<6%で売り（モデルホライズン外）"""
        self.assertEqual(_tiered_sell_signal(5.9,  70), "sell")
        self.assertEqual(_tiered_sell_signal(6.0,  70), "hold")
        self.assertEqual(_tiered_sell_signal(3.0,  70), "sell")
        # 中期も後期も同じ閾値(net<6%)
        self.assertEqual(_tiered_sell_signal(5.9,  45), "sell")
        self.assertEqual(_tiered_sell_signal(5.9,  70), "sell")

    def test_none_holding_days_uses_normal(self):
        """holding_days=None（新規）は通常閾値を使う"""
        self.assertEqual(_tiered_sell_signal(-5.1, None), "sell")
        self.assertEqual(_tiered_sell_signal(-4.9, None), "hold")
        self.assertEqual(_tiered_sell_signal(3.0,  None), "hold")

    def test_boundary_days(self):
        """境界日数の確認（3日以下は通常、4日から中期、63日以下は中期、64日から後期）"""
        self.assertEqual(_tiered_sell_signal(1.0,   3), "hold")   # 3日は通常
        self.assertEqual(_tiered_sell_signal(1.0,   4), "sell")   # 4日から中期(net<6%で売り)
        self.assertEqual(_tiered_sell_signal(6.0,  63), "hold")   # 63日は中期(net=6.0はhold)
        self.assertEqual(_tiered_sell_signal(5.9,  63), "sell")   # 63日は中期(net<6.0はsell)
        self.assertEqual(_tiered_sell_signal(5.9,  64), "sell")   # 64日から後期


class TestCandidateFilters(unittest.TestCase):
    """net_min/drop_max/conflict除外 フィルタの動作確認"""

    def _ranking(self, rows):
        return pd.DataFrame(rows)

    def _titles(self, actions):
        return [a["title"] for a in actions]

    def test_conflict_excluded(self):
        """net≥10 かつ drop≥5 のコンフリクト銘柄は除外される"""
        ranking = self._ranking([
            {"銘柄コード": 9001, "銘柄名": "コンフリクト株", "ネット(%)": 12.0, "ボラ(%)": 25.0, "下落確率(%)": 6.0},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertFalse(any("コンフリクト株" in t for t in self._titles(actions)))

    def test_high_net_low_drop_included(self):
        """net≥10 かつ drop<5 はコンフリクトではなく候補に残る"""
        ranking = self._ranking([
            {"銘柄コード": 9002, "銘柄名": "高net低drop株", "ネット(%)": 12.0, "ボラ(%)": 25.0, "下落確率(%)": 4.0},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertTrue(any("高net低drop株" in t for t in self._titles(actions)))

    def test_net_at_new_min_included(self):
        """net=6.0（新下限ちょうど）は候補に含まれる"""
        ranking = self._ranking([
            {"銘柄コード": 9003, "銘柄名": "ネット6株", "ネット(%)": 6.0, "ボラ(%)": 25.0, "下落確率(%)": 3.0},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertTrue(any("ネット6株" in t for t in self._titles(actions)))

    def test_net_below_min_excluded(self):
        """net=5.9（下限未満）は除外される"""
        ranking = self._ranking([
            {"銘柄コード": 9004, "銘柄名": "ネット低株", "ネット(%)": 5.9, "ボラ(%)": 25.0, "下落確率(%)": 3.0},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertFalse(any("ネット低株" in t for t in self._titles(actions)))

    def test_drop_prob_below_new_max_included(self):
        """drop_prob=11.9%（新上限未満）は除外されない"""
        ranking = self._ranking([
            {"銘柄コード": 9005, "銘柄名": "drop11株", "ネット(%)": 7.0, "ボラ(%)": 25.0, "下落確率(%)": 11.9},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertTrue(any("drop11株" in t for t in self._titles(actions)))

    def test_drop_prob_above_new_max_excluded(self):
        """drop_prob=12.1%（新上限超過）は除外される"""
        ranking = self._ranking([
            {"銘柄コード": 9006, "銘柄名": "drop13株", "ネット(%)": 7.0, "ボラ(%)": 25.0, "下落確率(%)": 12.1},
        ])
        actions = build_priority_actions([], ranking_df=ranking)
        self.assertFalse(any("drop13株" in t for t in self._titles(actions)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

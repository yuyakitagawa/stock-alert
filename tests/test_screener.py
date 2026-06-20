"""
test_screener.py
スクリーナー v1 条件のユニットテスト

実行: python3 tests/test_screener.py
"""

import os
import sys
import unittest
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "core"))

from unittest.mock import patch
from screener import apply_screener_v1, apply_sector_concentration_filter


def _make_universe(rows):
    cols = ["code", "name", "momentum", "momentum_20d", "vol", "score",
            "close", "slope_up", "vr2060", "rel_strength_3m"]
    defaults = {
        "code": "0000", "name": "テスト", "momentum": 10.0,
        "momentum_20d": 0.0, "vol": 30.0, "score": 8.0, "close": 500.0,
        "slope_up": True, "vr2060": 1.2, "rel_strength_3m": 0.06,
        "rel_strength_20d": 0.01,
        "rsi": 55.0, "turnover_m": 100.0,
    }
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestApplyScreenerV1(unittest.TestCase):

    def test_all_conditions_met_passes(self):
        """株価・流動性を満たす銘柄が通過する"""
        df = _make_universe([{
            "code": "A", "momentum": 10.0, "vol": 30.0, "close": 500.0,
        }])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_price_below_300_excluded(self):
        """株価 < 300円 は除外される"""
        df = _make_universe([{"code": "A", "close": 299.0}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_momentum_above_30pct_passes(self):
        """3ヶ月モメンタム > +30% も通過（モデルに選別を委任）"""
        df = _make_universe([{"code": "A", "momentum": 50.0}])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_low_liquidity_excluded(self):
        """20日平均売買代金 < 50百万円は除外される"""
        df = _make_universe([{"code": "A", "turnover_m": 49.9}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_liquidity_at_threshold_passes(self):
        """20日平均売買代金 = 50百万円ちょうどは通過する"""
        df = _make_universe([{"code": "A", "turnover_m": 50.0}])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_multiple_stocks_filtered_correctly(self):
        """株価・流動性のみでフィルタリング。モメンタム/ボラはモデルに委任"""
        df = _make_universe([
            {"code": "OK",  "momentum": 10.0, "vol": 30.0, "close": 500.0},
            {"code": "P1",  "momentum": 4.0},   # モメンタム低い → 通過（モデルが判断）
            {"code": "P2",  "vol": 10.0},       # ボラ低い → 通過（モデルが判断）
            {"code": "NG",  "close": 200.0},    # 株価下限未満 → 除外
        ])
        result = apply_screener_v1(df)
        self.assertEqual(len(result), 3)
        self.assertNotIn("NG", result["code"].values)


class TestSectorConcentrationFilter(unittest.TestCase):

    def test_under_threshold_keeps_all(self):
        """同セクター2銘柄は除外しない（閾値以下）"""
        df = pd.DataFrame([{"code": "A"}, {"code": "B"}, {"code": "C"}])
        with patch("screener.get_sector_cached",
                   side_effect=lambda c: {"A": "機械", "B": "機械", "C": "電気機器"}[c]):
            kept, excluded = apply_sector_concentration_filter(df)
        self.assertEqual(len(kept), 3)
        self.assertEqual(excluded, [])

    def test_three_in_same_sector_excludes_all(self):
        """同セクター3銘柄が出たら、そのセクター全銘柄を除外"""
        df = pd.DataFrame([{"code": "A"}, {"code": "B"}, {"code": "C"}, {"code": "D"}])
        with patch("screener.get_sector_cached",
                   side_effect=lambda c: {"A": "不動産業", "B": "不動産業", "C": "不動産業", "D": "機械"}[c]):
            kept, excluded = apply_sector_concentration_filter(df)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept.iloc[0]["code"], "D")
        self.assertEqual(excluded, [("不動産業", 3)])

    def test_empty_input(self):
        """空のDataFrameは何もせず返す"""
        df = pd.DataFrame(columns=["code"])
        kept, excluded = apply_sector_concentration_filter(df)
        self.assertTrue(kept.empty)
        self.assertEqual(excluded, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

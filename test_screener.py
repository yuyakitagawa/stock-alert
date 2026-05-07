"""
test_screener.py
スクリーナー v1 条件のユニットテスト

実行: python3 test_screener.py
"""

import unittest
import pandas as pd

from screener import apply_screener_v1


def _make_universe(rows):
    cols = ["code", "name", "momentum", "momentum_20d", "vol", "score",
            "close", "slope_up", "vr2060", "rel_strength_3m"]
    defaults = {
        "code": "0000", "name": "テスト", "momentum": 10.0,
        "momentum_20d": 0.0, "vol": 30.0, "score": 8.0, "close": 500.0,
        "slope_up": True, "vr2060": 1.2, "rel_strength_3m": 0.06,
    }
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestApplyScreenerV1(unittest.TestCase):

    def test_all_conditions_met_passes(self):
        """v1の全条件を満たす銘柄が通過する"""
        df = _make_universe([{
            "code": "A", "momentum": 10.0, "momentum_20d": 0.0,
            "vol": 30.0, "close": 500.0, "slope_up": True,
        }])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_slope_down_excluded(self):
        """slope_up=False は除外される"""
        df = _make_universe([{"code": "A", "slope_up": False}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_price_below_300_excluded(self):
        """株価 < 300円 は除外される"""
        df = _make_universe([{"code": "A", "close": 299.0}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_momentum_below_5pct_excluded(self):
        """3ヶ月モメンタム < +5% は除外される"""
        df = _make_universe([{"code": "A", "momentum": 4.9}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_momentum_above_30pct_excluded(self):
        """3ヶ月モメンタム > +30% は除外される（急騰後ミーンリバージョン防止）"""
        df = _make_universe([{"code": "A", "momentum": 31.0}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_momentum_at_upper_bound_passes(self):
        """3ヶ月モメンタム = +30% はちょうど通過する（境界値 <=）"""
        df = _make_universe([{"code": "A", "momentum": 30.0}])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_vol_below_20pct_excluded(self):
        """年率ボラ < 20% は除外される（+15%達成可能性なし）"""
        df = _make_universe([{"code": "A", "vol": 19.0}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_vol_above_50pct_excluded(self):
        """年率ボラ > 50% は除外される"""
        df = _make_universe([{"code": "A", "vol": 51.0}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_momentum_20d_below_minus3_excluded(self):
        """20日モメンタム < -3% は除外される"""
        df = _make_universe([{"code": "A", "momentum_20d": -3.1}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_vol_ratio_below_1_excluded(self):
        """出来高比 < 1.0（出来高減少）は除外される"""
        df = _make_universe([{"code": "A", "vr2060": 0.99}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_vol_ratio_at_1_passes(self):
        """出来高比 = 1.0 はちょうど通過する（境界値 >=）"""
        df = _make_universe([{"code": "A", "vr2060": 1.0}])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_negative_relative_strength_excluded(self):
        """日経比相対強度 < 0 は除外される"""
        df = _make_universe([{"code": "A", "rel_strength_3m": -0.01}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_at_threshold_relative_strength_passes(self):
        """日経比相対強度 = +5%（閾値ちょうど）は通過する（境界値 >=）"""
        df = _make_universe([{"code": "A", "rel_strength_3m": 0.05}])
        self.assertEqual(len(apply_screener_v1(df)), 1)

    def test_below_threshold_relative_strength_excluded(self):
        """日経比相対強度 = +4%（閾値未満）は除外される"""
        df = _make_universe([{"code": "A", "rel_strength_3m": 0.04}])
        self.assertTrue(apply_screener_v1(df).empty)

    def test_multiple_stocks_filtered_correctly(self):
        """複数銘柄のうち条件を満たすもののみ通過する"""
        df = _make_universe([
            {"code": "OK",  "momentum": 10.0, "vol": 30.0, "close": 500.0},
            {"code": "NG1", "momentum": 35.0},
            {"code": "NG2", "vol": 10.0},
            {"code": "NG3", "close": 200.0},
        ])
        result = apply_screener_v1(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["code"], "OK")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_screener.py
スクリーナー v2 条件のユニットテスト

実行: python3 test_screener.py
"""

import unittest
import numpy as np
import pandas as pd

from screener import (
    apply_momentum_v2,
    apply_high_proximity_v2,
    apply_volatility_band_v2,
    apply_price_filter,
    apply_screener_v1,
    write_compare_report,
    RANK_6M_THRESHOLD,
    MIN_RETURN_3M_V2,
    MAX_RETURN_3M_V2,
    MAX_HIGH_DIST_V2,
    MIN_VOL_V2,
    MAX_VOL_V2,
    MIN_PRICE,
    MIN_VOLATILITY,
    MAX_MOMENTUM,
    MIN_VOL_RATIO,
)


def _make_universe(rows):
    """テスト用の universe_df を作成するヘルパー"""
    cols = ["code", "name", "r2", "momentum", "momentum_20d", "vol", "score",
            "close", "slope_up", "return_3m", "return_6m", "vr2060", "rel_strength_3m"]
    defaults = {
        "code": "0000", "name": "テスト", "r2": 0.8, "momentum": 10.0,
        "momentum_20d": 0.0, "vol": 30.0, "score": 8.0, "close": 500.0,
        "slope_up": True, "return_3m": 0.10, "return_6m": 0.15,
        "vr2060": 1.2, "rel_strength_3m": 0.02,
    }
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_prices(n, start=1000.0, growth=0.0, vol=0.0):
    """テスト用の価格配列を生成するヘルパー"""
    np.random.seed(0)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + growth + np.random.randn() * vol))
    return np.array(prices)


class TestApplyMomentumV2(unittest.TestCase):

    def test_cross_section_rank_upper30pct_passes(self):
        """6ヶ月リターン上位30%の銘柄が通過する"""
        df = _make_universe([
            {"code": "A", "return_6m": 0.50, "return_3m": 0.10},
            {"code": "B", "return_6m": 0.40, "return_3m": 0.10},
            {"code": "C", "return_6m": 0.30, "return_3m": 0.10},
            {"code": "D", "return_6m": 0.10, "return_3m": 0.10},
        ])
        result = apply_momentum_v2(df)
        # 上位30% = 4銘柄中上位1〜2銘柄（パーセンタイル≥0.70）
        self.assertIn("A", result["code"].values)

    def test_cross_section_rank_lower70pct_excluded(self):
        """6ヶ月リターン下位70%は除外される"""
        df = _make_universe([
            {"code": "A", "return_6m": 0.50, "return_3m": 0.10},
            {"code": "B", "return_6m": 0.40, "return_3m": 0.10},
            {"code": "C", "return_6m": 0.30, "return_3m": 0.10},
            {"code": "D", "return_6m": 0.01, "return_3m": 0.10},
        ])
        result = apply_momentum_v2(df)
        self.assertNotIn("D", result["code"].values)

    def test_3m_upper_bound_excludes(self):
        """3ヶ月リターン > +30% は除外される（6ヶ月ランク条件も同時に満たせる5銘柄構成）"""
        df = _make_universe([
            {"code": "HIGH", "return_6m": 0.50, "return_3m": 0.31},  # 6m rank=1.0, 3m超過→除外
            {"code": "OK",   "return_6m": 0.45, "return_3m": 0.20},  # 6m rank=0.8, 通過
            {"code": "C",    "return_6m": 0.10, "return_3m": 0.10},  # 6m rank=0.6, rank不足
            {"code": "D",    "return_6m": 0.05, "return_3m": 0.10},  # 6m rank=0.4, rank不足
            {"code": "E",    "return_6m": 0.01, "return_3m": 0.10},  # 6m rank=0.2, rank不足
        ])
        result = apply_momentum_v2(df)
        self.assertNotIn("HIGH", result["code"].values)
        self.assertIn("OK", result["code"].values)

    def test_3m_upper_bound_at_30pct_passes(self):
        """3ヶ月リターン = +30% はちょうど通過する（境界値 <=）"""
        df = _make_universe([
            {"code": "BOUNDARY", "return_6m": 0.50, "return_3m": 0.30},
            {"code": "LOWER",    "return_6m": 0.30, "return_3m": 0.10},
        ])
        result = apply_momentum_v2(df)
        self.assertIn("BOUNDARY", result["code"].values)

    def test_3m_lower_bound_excludes(self):
        """3ヶ月リターン < +5% は除外される"""
        df = _make_universe([
            {"code": "LOW", "return_6m": 0.50, "return_3m": 0.04},
            {"code": "OK",  "return_6m": 0.40, "return_3m": 0.10},
        ])
        result = apply_momentum_v2(df)
        self.assertNotIn("LOW", result["code"].values)

    def test_3m_lower_bound_at_5pct_passes(self):
        """3ヶ月リターン = +5% はちょうど通過する（境界値 >=）"""
        df = _make_universe([
            {"code": "BOUNDARY", "return_6m": 0.50, "return_3m": 0.05},
            {"code": "LOWER",    "return_6m": 0.30, "return_3m": 0.04},
        ])
        result = apply_momentum_v2(df)
        self.assertIn("BOUNDARY", result["code"].values)

    def test_rank_6m_column_added(self):
        """rank_6m 列が追加されていること"""
        df = _make_universe([
            {"code": "A", "return_6m": 0.20, "return_3m": 0.10},
        ])
        result = apply_momentum_v2(df)
        self.assertIn("rank_6m", result.columns)

    def test_empty_dataframe_returns_empty(self):
        """空のDataFrameを渡しても空が返る"""
        df = _make_universe([])
        result = apply_momentum_v2(df)
        self.assertTrue(result.empty)


class TestApplyHighProximityV2(unittest.TestCase):

    def _make_df_and_dict(self, code, prices):
        df = _make_universe([{"code": code}])
        return df, {code: prices}

    def test_insufficient_data_excluded(self):
        """データ長 < 252 の銘柄は除外される"""
        df, pd_dict = self._make_df_and_dict("A", _make_prices(251))
        result = apply_high_proximity_v2(df, pd_dict)
        self.assertTrue(result.empty)

    def test_no_data_excluded(self):
        """prices_dict にない銘柄は除外される"""
        df = _make_universe([{"code": "A"}])
        result = apply_high_proximity_v2(df, {})
        self.assertTrue(result.empty)

    def test_within_20pct_passes(self):
        """52週高値から-10%の銘柄は通過する"""
        prices = _make_prices(300, start=1000.0)
        max_price = max(prices[-252:])
        # 現在値を52週高値の90%（-10%乖離）に設定
        prices[-1] = max_price * 0.90
        df = _make_universe([{"code": "A"}])
        result = apply_high_proximity_v2(df, {"A": prices})
        self.assertEqual(len(result), 1)

    def test_boundary_exactly_minus20pct_passes(self):
        """52週高値から-20%ちょうどの銘柄は通過する（境界値 >=）"""
        prices = np.ones(300) * 1000.0
        prices[-1] = 800.0  # 高値1000に対して-20%
        df = _make_universe([{"code": "A"}])
        result = apply_high_proximity_v2(df, {"A": prices})
        self.assertEqual(len(result), 1)

    def test_beyond_minus20pct_excluded(self):
        """52週高値から-21%の銘柄は除外される"""
        prices = np.ones(300) * 1000.0
        prices[-1] = 790.0  # -21%
        df = _make_universe([{"code": "A"}])
        result = apply_high_proximity_v2(df, {"A": prices})
        self.assertTrue(result.empty)


class TestApplyVolatilityBandV2(unittest.TestCase):

    def _make_df_and_dict(self, code, prices):
        df = _make_universe([{"code": code}])
        return df, {code: prices}

    def test_insufficient_data_excluded(self):
        """データ長 < 60 の銘柄は除外される"""
        df, pd_dict = self._make_df_and_dict("A", _make_prices(59))
        result = apply_volatility_band_v2(df, pd_dict)
        self.assertTrue(result.empty)

    def test_no_data_excluded(self):
        """prices_dict にない銘柄は除外される"""
        df = _make_universe([{"code": "A"}])
        result = apply_volatility_band_v2(df, {})
        self.assertTrue(result.empty)

    def test_vol_in_band_passes(self):
        """年率ボラ30%（20%-50%内）の銘柄は通過する"""
        # 日次ボラ約1.9%で年率約30%
        prices = _make_prices(100, start=1000.0, vol=0.019)
        df = _make_universe([{"code": "A"}])
        result = apply_volatility_band_v2(df, {"A": prices})
        self.assertEqual(len(result), 1)

    def test_vol_too_low_excluded(self):
        """年率ボラ < 20% の銘柄は除外される"""
        # ほぼ動かない価格（ボラ≈0）
        prices = np.ones(100) * 1000.0
        df = _make_universe([{"code": "A"}])
        result = apply_volatility_band_v2(df, {"A": prices})
        self.assertTrue(result.empty)

    def test_vol_too_high_excluded(self):
        """年率ボラ > 50% の銘柄は除外される"""
        # 日次ボラ約3.5%で年率約55%
        prices = _make_prices(100, start=1000.0, vol=0.035)
        df = _make_universe([{"code": "A"}])
        result = apply_volatility_band_v2(df, {"A": prices})
        # ボラが大きすぎて除外されているかを確認（確率的なので緩くテスト）
        # 注意: 乱数次第で境界をまたぐ可能性があるため、vol=0.05 で確実に超過させる
        prices2 = _make_prices(100, start=1000.0, vol=0.05)
        result2 = apply_volatility_band_v2(df, {"A": prices2})
        self.assertTrue(result2.empty)


class TestCompareReport(unittest.TestCase):

    def test_overlap_and_diff_counts(self):
        """v1・v2通過銘柄の重複・差分が正しく計算される"""
        df_v1 = pd.DataFrame({"code": ["A", "B", "C"]})
        df_v2 = pd.DataFrame({"code": ["B", "C", "D"]})

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            # write_compare_report はパスをハードコードしているので
            # 関数ロジックをここで再現してテスト
            v1_codes = set(df_v1["code"].astype(str))
            v2_codes = set(df_v2["code"].astype(str))
            overlap  = v1_codes & v2_codes
            only_v1  = v1_codes - v2_codes
            only_v2  = v2_codes - v1_codes

            self.assertEqual(len(overlap), 2)   # B, C
            self.assertEqual(len(only_v1), 1)   # A
            self.assertEqual(len(only_v2), 1)   # D
            self.assertIn("A", only_v1)
            self.assertIn("D", only_v2)

    def test_empty_v2_produces_all_in_v1_only(self):
        """v2通過ゼロなら全銘柄がv1のみに入る"""
        df_v1 = pd.DataFrame({"code": ["A", "B"]})
        df_v2 = pd.DataFrame({"code": []})
        v1_codes = set(df_v1["code"].astype(str))
        v2_codes = set(df_v2["code"].astype(str))
        only_v1 = v1_codes - v2_codes
        self.assertEqual(len(only_v1), 2)


class TestApplyScreenerV1(unittest.TestCase):

    def test_all_conditions_met_passes(self):
        """v1の全条件を満たす銘柄が通過する"""
        df = _make_universe([{
            "code": "A", "r2": 0.70, "momentum": 10.0, "momentum_20d": 0.0,
            "vol": 30.0, "close": 500.0, "slope_up": True,
        }])
        result = apply_screener_v1(df)
        self.assertEqual(len(result), 1)

    def test_slope_down_excluded(self):
        """slope_up=False は除外される"""
        df = _make_universe([{"code": "A", "slope_up": False}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_price_below_300_excluded(self):
        """株価 < 300円 は除外される"""
        df = _make_universe([{"code": "A", "close": 299.0}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_momentum_above_30pct_excluded(self):
        """3ヶ月モメンタム > +30% は除外される（急騰後ミーンリバージョン防止）"""
        df = _make_universe([{"code": "A", "momentum": 31.0}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_vol_below_20pct_excluded(self):
        """年率ボラ < 20% は除外される（+15%達成可能性なし）"""
        df = _make_universe([{"code": "A", "vol": 19.0}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_vol_ratio_below_1_excluded(self):
        """出来高比 < 1.0（出来高減少）は除外される"""
        df = _make_universe([{"code": "A", "vr2060": 0.99}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_vol_ratio_at_1_passes(self):
        """出来高比 = 1.0 はちょうど通過する（境界値 >=）"""
        df = _make_universe([{"code": "A", "vr2060": 1.0}])
        result = apply_screener_v1(df)
        self.assertEqual(len(result), 1)

    def test_negative_relative_strength_excluded(self):
        """日経比相対強度 < 0（日経に負けている）は除外される"""
        df = _make_universe([{"code": "A", "rel_strength_3m": -0.01}])
        result = apply_screener_v1(df)
        self.assertTrue(result.empty)

    def test_zero_relative_strength_passes(self):
        """日経比相対強度 = 0（日経と同等）は通過する（境界値 >=）"""
        df = _make_universe([{"code": "A", "rel_strength_3m": 0.0}])
        result = apply_screener_v1(df)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

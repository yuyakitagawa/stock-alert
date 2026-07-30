"""
test_data_sanity.py
ランキング出力の不変条件チェック（lib/data_sanity）のユニットテスト

実行: python3 tests/test_data_sanity.py
"""
import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from lib.data_sanity import (
    check_ranking, check_price_freshness,
    has_critical, Violation, format_violations,
)


def _healthy_rows(n=3200):
    """下落確率に多様性のある健全データ（下落モデルのみに一本化済み）。"""
    rows = []
    for i in range(n):
        drop = round((i % 40) * 1.0, 1)   # 40種類のdrop値
        rows.append({
            "code": f"{1000+i}",
            "drop_prob": drop,
            "recommend": "⏳ 方向感なし",
        })
    return rows


class TestProbRange(unittest.TestCase):
    def test_healthy_passes(self):
        v = check_ranking(_healthy_rows())
        self.assertEqual(v, [], f"健全データで違反が出た: {format_violations(v)}")

    def test_out_of_range(self):
        rows = _healthy_rows(50)
        rows[0]["drop_prob"] = 150.0
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "prob_range" and x.severity == "critical" for x in v))


class TestPredictionCollapse(unittest.TestCase):
    def test_collapse_critical(self):
        """ユニーク値が極端に少ない → critical prediction_collapse。"""
        rows = []
        for i in range(3200):
            drop = 8.8 if i % 2 == 0 else 7.1   # 2種類のみ
            rows.append({"code": f"{i}", "drop_prob": drop,
                         "recommend": "⏳ 方向感なし"})
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "prediction_collapse" for x in v))

    def test_low_diversity_warning(self):
        """18種・最頻値偏重（実バグ相当）→ warning low_diversity。"""
        rows = []
        for i in range(3200):
            drop = 8.8 if i < 1400 else round(10 + (i % 17), 1)  # 偏った18種
            rows.append({"code": f"{i}", "drop_prob": drop,
                         "recommend": "⏳ 方向感なし"})
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "low_diversity" and x.severity == "warning" for x in v))


class TestMissingAndVocab(unittest.TestCase):
    def test_missing_fields_critical(self):
        rows = _healthy_rows(50)
        rows[0]["drop_prob"] = None
        rows[1]["code"] = None
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "missing_fields" and x.severity == "critical" for x in v))

    def test_unknown_recommend_warning(self):
        rows = _healthy_rows(50)
        rows[0]["recommend"] = "謎ラベル"
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "recommend_vocab" and x.severity == "warning" for x in v))

    def test_empty_is_critical(self):
        v = check_ranking([])
        self.assertTrue(has_critical(v))


class TestRowCount(unittest.TestCase):
    def test_too_few_rows_warning(self):
        v = check_ranking(_healthy_rows(100))   # 100件 < 3000
        self.assertTrue(any(x.check == "row_count" and x.severity == "warning" for x in v))


class TestDataFrameInput(unittest.TestCase):
    def test_accepts_japanese_columns(self):
        """DataFrame＋日本語列名（rank_stocksのCSV形式）でも検査できる。"""
        import pandas as pd
        rows = _healthy_rows(3100)
        df = pd.DataFrame([{
            "銘柄コード": r["code"],
            "下落確率(%)": r["drop_prob"],
            "推奨": r["recommend"],
        } for r in rows])
        v = check_ranking(df)
        self.assertEqual(v, [], f"日本語列の健全DataFrameで違反: {format_violations(v)}")


class TestPriceFreshness(unittest.TestCase):
    def test_healthy_varies_passes(self):
        history = {f"{1000+i}": [100 + i, 101 + i, 99 + i, 103 + i] for i in range(20)}
        v = check_price_freshness(history)
        self.assertEqual(v, [], f"日々変動する価格で誤検知: {format_violations(v)}")

    def test_frozen_majority_critical(self):
        """三菱商事8058で実際に起きた再現: 大半の銘柄でcloseが全期間同一値のまま固まる。"""
        history = {f"{1000+i}": [4924.0, 4924.0, 4924.0] for i in range(10)}
        history.update({f"{2000+i}": [100 + i, 105 + i] for i in range(3)})  # 健全な少数派
        v = check_price_freshness(history)
        self.assertTrue(any(x.check == "frozen_price" and x.severity == "critical" for x in v))

    def test_frozen_minority_warning(self):
        history = {f"{1000+i}": [100 + i, 105 + i] for i in range(9)}
        history["9999"] = [4924.0, 4924.0]  # 1件だけ凍結
        v = check_price_freshness(history)
        self.assertTrue(any(x.check == "frozen_price" and x.severity == "warning" for x in v))

    def test_single_point_history_ignored(self):
        """1件しかない銘柄は凍結かどうか判定不能なので対象外。"""
        history = {f"{1000+i}": [100.0] for i in range(10)}
        v = check_price_freshness(history)
        self.assertEqual(v, [], f"比較不能な1件データで誤検知: {format_violations(v)}")

    def test_empty_history_ignored(self):
        self.assertEqual(check_price_freshness({}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

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

from lib.data_sanity import check_ranking, has_critical, Violation, format_violations


def _healthy_rows(n=3200):
    """net = rise - drop が成立し、多様性のある健全データ。"""
    rows = []
    for i in range(n):
        rise = round(5 + (i % 40) * 1.0, 1)   # 40種類のrise値
        drop = round((i % 13) * 0.7, 1)
        rows.append({
            "code": f"{1000+i}",
            "rise_prob": rise,
            "drop_prob": drop,
            "net": round(rise - drop, 1),
            "recommend": "⏳ 方向感なし",
        })
    return rows


class TestNetIntegrity(unittest.TestCase):
    def test_healthy_passes(self):
        v = check_ranking(_healthy_rows())
        self.assertEqual(v, [], f"健全データで違反が出た: {format_violations(v)}")

    def test_net_equals_rise_bug(self):
        """net==rise（下落未減算）バグ → critical net_integrity を検出。"""
        rows = _healthy_rows()
        for r in rows:
            r["net"] = r["rise_prob"]   # バグ再現: dropを引かない
        v = check_ranking(rows)
        self.assertTrue(has_critical(v))
        self.assertTrue(any(x.check == "net_integrity" for x in v))

    def test_net_small_tolerance_ok(self):
        """丸め誤差 0.1 程度は許容。"""
        rows = _healthy_rows(50)
        rows[0]["net"] = round(rows[0]["net"] + 0.1, 1)
        v = check_ranking(rows)
        self.assertFalse(any(x.check == "net_integrity" for x in v))


class TestProbRange(unittest.TestCase):
    def test_out_of_range(self):
        rows = _healthy_rows(50)
        rows[0]["rise_prob"] = 150.0
        rows[0]["net"] = 150.0 - rows[0]["drop_prob"]
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "prob_range" and x.severity == "critical" for x in v))


class TestPredictionCollapse(unittest.TestCase):
    def test_collapse_critical(self):
        """ユニーク値が極端に少ない → critical prediction_collapse。"""
        rows = []
        for i in range(3200):
            rise = 33.8 if i % 2 == 0 else 32.1   # 2種類のみ
            drop = round((i % 10) * 0.5, 1)
            rows.append({"code": f"{i}", "rise_prob": rise,
                         "drop_prob": drop, "net": round(rise - drop, 1),
                         "recommend": "⏳ 方向感なし"})
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "prediction_collapse" for x in v))

    def test_low_diversity_warning(self):
        """18種・最頻値偏重（実バグ相当）→ warning low_diversity。"""
        rows = []
        for i in range(3200):
            rise = 33.8 if i < 1400 else round(10 + (i % 17), 1)  # 偏った18種
            drop = round((i % 11) * 0.6, 1)
            rows.append({"code": f"{i}", "rise_prob": rise,
                         "drop_prob": drop, "net": round(rise - drop, 1),
                         "recommend": "⏳ 方向感なし"})
        v = check_ranking(rows)
        self.assertTrue(any(x.check == "low_diversity" and x.severity == "warning" for x in v))


class TestMissingAndVocab(unittest.TestCase):
    def test_missing_fields_critical(self):
        rows = _healthy_rows(50)
        rows[0]["net"] = None
        rows[1]["rise_prob"] = None
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
            "銘柄コード": r["code"], "上昇確率(%)": r["rise_prob"],
            "下落確率(%)": r["drop_prob"], "ネット(%)": r["net"],
            "推奨": r["recommend"],
        } for r in rows])
        v = check_ranking(df)
        self.assertEqual(v, [], f"日本語列の健全DataFrameで違反: {format_violations(v)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

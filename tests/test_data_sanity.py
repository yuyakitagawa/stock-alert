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
    check_ranking, check_site, check_pages, has_critical, Violation, format_violations,
)


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


class TestCheckSite(unittest.TestCase):
    def _ctx(self):
        rows = _healthy_rows(3100)
        for r in rows:
            r["date"] = "2026-06-11"
        return {
            "date": "2026-06-11",
            "rankings": rows,
            "stock_meta": [{"code": r["code"], "sector": "電気機器"} for r in rows],
            "gen_ai_analyses": [{"code": r["code"], "summary": "解析あり",
                             "verdict": "様子見", "date": "2026-06-11"} for r in rows[:10]],
            "earnings": [{"code": rows[0]["code"]}],
            "expected_ai": 10,
        }

    def test_healthy_site_passes(self):
        v = check_site(self._ctx())
        self.assertEqual(v, [], f"健全サイトで違反: {format_violations(v)}")

    def test_stale_rankings_critical(self):
        ctx = self._ctx()
        for r in ctx["rankings"]:
            r["date"] = "2026-06-10"   # 本日ではない
        v = check_site(ctx)
        self.assertTrue(any(x.check == "stale_rankings" and x.severity == "critical" for x in v))

    def test_meta_coverage_gap(self):
        ctx = self._ctx()
        ctx["stock_meta"] = ctx["stock_meta"][:100]   # 大半が欠損
        v = check_site(ctx)
        self.assertTrue(any(x.check == "meta_coverage" and x.severity == "critical" for x in v))

    def test_ai_empty_warning(self):
        ctx = self._ctx()
        ctx["gen_ai_analyses"][0]["summary"] = ""
        v = check_site(ctx)
        self.assertTrue(any(x.check == "ai_empty" for x in v))

    def test_empty_rankings_critical(self):
        v = check_site({"date": "2026-06-11", "rankings": []})
        self.assertTrue(any(x.check == "rankings_empty" and x.severity == "critical" for x in v))

    def test_partial_context_skips_missing(self):
        """未提供のキーは検査しない（earnings/gen_ai_analyses無し）。"""
        rows = _healthy_rows(3100)
        for r in rows:
            r["date"] = "2026-06-11"
        v = check_site({"date": "2026-06-11", "rankings": rows})
        self.assertEqual(v, [], f"部分contextで誤検知: {format_violations(v)}")

    def test_description_full_coverage_ok(self):
        """対象銘柄すべてに会社説明があれば違反なし。"""
        ctx = self._ctx()
        ctx["desc_targets"] = ["2811", "7309", "4452"]
        ctx["descriptions"] = [{"code": c, "summary": "説明あり"} for c in ctx["desc_targets"]]
        v = check_site(ctx)
        self.assertFalse(any(x.check == "description_coverage" for x in v))

    def test_description_missing_warning(self):
        """対象銘柄の一部に説明が無ければ warning で指摘。"""
        ctx = self._ctx()
        ctx["desc_targets"] = ["2811", "7309", "4452", "1801"]
        ctx["descriptions"] = [{"code": "2811", "summary": "説明あり"},
                               {"code": "7309", "summary": "説明あり"},
                               {"code": "4452", "summary": "説明あり"}]  # 1801(建設)欠損
        v = check_site(ctx)
        self.assertTrue(any(x.check == "description_coverage" and x.severity == "warning" for x in v))

    def test_description_empty_summary_counts_as_missing(self):
        """summaryが空文字でも欠損扱い（『概要情報を取得できませんでした』状態）。"""
        ctx = self._ctx()
        ctx["desc_targets"] = ["2811", "1801"]
        ctx["descriptions"] = [{"code": "2811", "summary": "説明あり"},
                               {"code": "1801", "summary": "   "}]
        v = check_site(ctx)
        self.assertTrue(any(x.check == "description_coverage" for x in v))

    def test_description_partial_missing_is_warning(self):
        """一部だけ欠損（説明が1件以上ある）なら warning に留める。"""
        ctx = self._ctx()
        ctx["desc_targets"] = ["2811", "7309", "4452", "1801"]
        ctx["descriptions"] = [{"code": "2811", "summary": "説明あり"}]  # 3/4欠損だが1件あり
        v = check_site(ctx)
        self.assertTrue(any(x.check == "description_coverage" and x.severity == "warning" for x in v))

    def test_description_all_missing_critical(self):
        """対象に説明が1件も無い＝パイプライン故障 → critical。"""
        ctx = self._ctx()
        ctx["desc_targets"] = ["2811", "7309", "4452", "1801"]
        ctx["descriptions"] = []  # 全欠損
        v = check_site(ctx)
        self.assertTrue(any(x.check == "description_coverage" and x.severity == "critical" for x in v))


class TestCheckPages(unittest.TestCase):
    def _ok(self, route="/rankings", expect=None):
        return {"route": route, "status": 200,
                "body": "x" * 5000 + (expect or ""), "expect": expect}

    def test_healthy_pages_pass(self):
        results = [self._ok("/"), self._ok("/rankings"),
                   self._ok("/watchlist", "お得度"), self._ok("/stocks/7203", "AIスコア")]
        v = check_pages(results)
        self.assertEqual(v, [], f"健全ページで違反: {format_violations(v)}")

    def test_http_error_critical(self):
        v = check_pages([{"route": "/rankings", "status": 500, "body": ""}])
        self.assertTrue(any(x.check == "page_status" and x.severity == "critical" for x in v))

    def test_fetch_error_critical(self):
        v = check_pages([{"route": "/", "error": "timeout"}])
        self.assertTrue(any(x.check == "page_fetch" and x.severity == "critical" for x in v))

    def test_error_boundary_critical(self):
        body = "x" * 2000 + "エラーが発生しました" + "x" * 2000
        v = check_pages([{"route": "/stocks/7203", "status": 200, "body": body}])
        self.assertTrue(any(x.check == "page_error" and x.severity == "critical" for x in v))

    def test_empty_body_critical(self):
        v = check_pages([{"route": "/activity", "status": 200, "body": "短い"}])
        self.assertTrue(any(x.check == "page_empty" and x.severity == "critical" for x in v))

    def test_missing_expected_content_critical(self):
        """期待文言（例: 銘柄ページの『AIスコア』）が無ければ404/欠落として critical。"""
        body = "x" * 5000  # AIスコアを含まない
        v = check_pages([{"route": "/stocks/9999999", "status": 200, "body": body, "expect": "AIスコア"}])
        self.assertTrue(any(x.check == "page_content" and x.severity == "critical" for x in v))

    def test_notfound_string_does_not_false_positive(self):
        """健全ページにboundary由来の『ページが見つかりません』が混じっても誤検知しない。"""
        body = "x" * 5000 + "ページが見つかりません"  # 全ページに埋め込まれる文字列
        v = check_pages([{"route": "/", "status": 200, "body": body}])
        self.assertEqual(v, [], f"boundary文字列で誤検知: {format_violations(v)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""tools/seo_loop.py のユニットテスト（判定の純関数のみ。APIは叩かない）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import seo_loop as s  # noqa: E402

SITE = "https://kujira-watch.com"


def _r(keys, clicks, impressions, position):
    return {"keys": list(keys), "clicks": clicks, "impressions": impressions,
            "ctr": clicks / impressions if impressions else 0.0, "position": position}


def test_brand_queries_are_excluded_from_almost_there():
    rows = [_r(("クジラウォッチ 使い方", f"{SITE}/faq"), 0, 50, 5.0),
            _r(("大量保有 速報", f"{SITE}/"), 1, 50, 5.0)]
    got = [r["keys"][0] for r in s.almost_there(rows)]
    assert got == ["大量保有 速報"]


def test_almost_there_bounds_are_3_to_20_and_first_page_first():
    rows = [_r(("a", f"{SITE}/x"), 5, 100, 2.0),       # 既に上位
            _r(("b", f"{SITE}/x"), 0, 100, 21.0),      # 届かない
            _r(("c", f"{SITE}/x"), 0, 500, 15.0),
            _r(("d", f"{SITE}/x"), 0, 20, 4.0)]
    assert [r["keys"][0] for r in s.almost_there(rows)] == ["d", "c"]


def test_best_page_per_query_picks_most_impressions():
    rows = [_r(("q", f"{SITE}/a"), 0, 10, 8.0), _r(("q", f"{SITE}/b"), 0, 30, 12.0)]
    assert s.best_page_per_query(rows)["q"]["keys"][1] == f"{SITE}/b"


def test_no_clicks_needs_over_500_impressions_and_ctr_below_half_percent():
    rows = [_r((f"{SITE}/a",), 1, 1000, 8.0),   # 0.1% → 対象
            _r((f"{SITE}/b",), 10, 1000, 8.0),  # 1.0% → 対象外
            _r((f"{SITE}/c",), 0, 500, 8.0)]    # 500ちょうど → 対象外
    assert [r["keys"][0] for r in s.no_clicks(rows)] == [f"{SITE}/a"]


def test_decaying_ignores_small_pages_and_catches_vanished_ones():
    prev = [_r((f"{SITE}/a",), 10, 100, 5.0), _r((f"{SITE}/b",), 4, 100, 5.0),
            _r((f"{SITE}/c",), 10, 100, 5.0), _r((f"{SITE}/gone",), 8, 100, 5.0)]
    now = [_r((f"{SITE}/a",), 6, 100, 7.0), _r((f"{SITE}/b",), 0, 100, 5.0),
           _r((f"{SITE}/c",), 8, 100, 5.0)]
    got = {c["keys"][0]: round(d, 2) for c, _, d in s.decaying(now, prev)}
    assert got == {f"{SITE}/a": 0.4, f"{SITE}/gone": 1.0}


def test_ai_mode_counts_words_for_english_and_chars_for_japanese():
    assert s.is_ai_mode("how do i find activist investors buying japanese stocks")
    assert not s.is_ai_mode("activist investors japan stocks list today")  # 6語・英語は字数で拾わない
    assert s.is_ai_mode("アクティビストが大量保有した銘柄はその後上がるのか")
    assert not s.is_ai_mode("大量保有 報告書 見方")


def test_wrong_intent_flags_comparison_query_on_stock_page():
    assert s.intent_mismatch("トヨタ ホンダ 比較", f"{SITE}/stocks/7203") == "comparison"
    assert s.intent_mismatch("大量保有 ランキング", f"{SITE}/ranking/buys") is None
    assert s.intent_mismatch("大量保有報告書とは", f"{SITE}/stocks/7203") == "definition"
    assert s.intent_mismatch("7203 株価", f"{SITE}/stocks/7203") is None


def test_untargeted_requires_title_and_skips_unknown_titles():
    rows = [_r(("村上ファンド 銘柄", f"{SITE}/articles/1"), 1, 40, 6.0),
            _r(("エフィッシモ", f"{SITE}/articles/2"), 1, 40, 6.0),
            _r(("シティ 保有", f"{SITE}/articles/3"), 1, 40, 6.0)]
    titles = {f"{SITE}/articles/1": "旧村上ファンド系の買い増し一覧",
              f"{SITE}/articles/2": "ストラテジックキャピタルの動向"}
    got = [r["keys"][0] for r in s.untargeted(rows, titles)]
    assert got == ["エフィッシモ"]  # 1はtitleに語あり、3はtitle不明で見送り


def test_join_conversions_matches_by_path_and_flags_leaks():
    pages = [_r((f"{SITE}/articles/a",), 40, 400, 5.0), _r((f"{SITE}/guides/b/",), 12, 100, 5.0),
             _r((f"{SITE}/stocks/1",), 3, 100, 5.0)]
    landing = {"/articles/a": [45.0, 0.0], "/guides/b": [14.0, 6.0], "/guides/b?utm=x": [1.0, 1.0]}
    joined = s.join_conversions(pages, landing)
    by = {p: (g, ss, c) for p, g, ss, c in joined}
    assert by["/guides/b"] == (12, 15.0, 7.0)
    assert [j[0] for j in s.leaks(joined)] == ["/articles/a"]
    assert [j[0] for j in s.converters(joined)] == ["/guides/b"]


def test_render_marks_empty_sections():
    text = s.render([("x", "do", []), ("y", "do", ["- a"])])
    assert "## x（0件）" in text and "- 該当なし" in text and "- a" in text

"""記事本文に差し込む解説図（web/article_figures.py）のユニットテスト。
描画はPillowのみ・ネットワーク無しなので、差し込み位置のロジックと図の生成可否を検証する。

実行: python3 tests/test_article_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.article_figures as m

# 本番(GitHub Actions)はNoto Sans CJK、ローカル(mac)はヒラギノ。どちらも無い環境では
# 図が作れない（＝Noneを返す）のが正しい挙動なので、描画そのもののテストは飛ばす。
HAS_FONT = m._font(10) is not None

FACT_SHEET = {
    "stock_name": "テスト製薬",
    "stock_code": "4568",
    "filer_name": "テストアセットマネジメント",
    "holding_ratio": 8.32,
    "context_facts": {
        "holding_history": {
            "count": 3,
            "first_date": "2024-03-15",
            "first_ratio": 5.02,
            "points": [
                {"date": "2024-03-15", "ratio": 5.02},
                {"date": "2025-06-20", "ratio": 7.05},
                {"date": "2026-08-21", "ratio": 8.32},
            ],
        },
        "stock_other_filers": [
            {"name": "他社アセット", "ratio": 6.21},
            {"name": "他社証券", "ratio": 5.44},
        ],
        "filer_other_holdings": [
            {"name": "他銘柄A", "code": "1111", "ratio": 9.1, "sector": "小売業"},
            {"name": "他銘柄B", "code": "2222", "ratio": 6.7, "sector": "サービス業"},
        ],
    },
}


def _body(paragraph_texts: list) -> str:
    return "".join(f"<p>{t}</p>" for t in paragraph_texts)


def test_insert_figures_after_matching_paragraph():
    body = _body(["直答文", "他社アセットなど他の大株主と比べると", "指標の話", "※推測: 締め"])
    out = m.insert_figures_into_body(body, [{"html": "<figure>F</figure>", "anchors": ["他社アセット"]}])
    assert out == "<p>直答文</p><p>他社アセットなど他の大株主と比べると</p><figure>F</figure><p>指標の話</p><p>※推測: 締め</p>"


def test_insert_figures_never_lands_after_last_paragraph():
    """最終段落は「※推測」の締め。その後ろは株価チャートの位置なので図は入れない。"""
    body = _body(["直答文", "本文", "本文", "※推測: ポートフォリオの話"])
    out = m.insert_figures_into_body(body, [{"html": "<figure>F</figure>", "anchors": ["ポートフォリオ"]}])
    assert out.endswith("<figure>F</figure><p>※推測: ポートフォリオの話</p>")


def test_insert_figures_never_lands_before_second_paragraph():
    """1段落目は検索クエリへの直答文なので、その直後より前には差し込まない。"""
    body = _body(["直答文", "本文", "本文", "本文", "※推測"])
    out = m.insert_figures_into_body(body, [{"html": "<figure>F</figure>", "anchors": ["直答文"]}])
    assert out.startswith("<p>直答文</p><p>本文</p>")


def test_insert_figures_without_anchors_spreads_and_keeps_order():
    body = _body(["直答文", "本文2", "本文3", "本文4", "本文5", "※推測"])
    figures = [{"html": f"<figure>F{i}</figure>", "anchors": []} for i in range(3)]
    out = m.insert_figures_into_body(body, figures)
    assert out.index("F0") < out.index("F1") < out.index("F2")
    assert out.count("<figure>") == 3
    assert not out.endswith("</figure>")  # 最終段落の後ろには来ない


def test_insert_figures_appends_when_body_has_too_few_paragraphs():
    out = m.insert_figures_into_body("<p>一段落だけ</p>", [{"html": "<figure>F</figure>", "anchors": []}])
    assert out == "<p>一段落だけ</p><figure>F</figure>"


def test_insert_figures_returns_body_unchanged_without_figures():
    body = _body(["直答文", "本文", "※推測"])
    assert m.insert_figures_into_body(body, []) == body


def test_figure_html_contains_alt_and_caption():
    html = m.figure_html("https://example.com/a.png", "代替テキスト", "キャプション")
    assert '<img src="https://example.com/a.png" alt="代替テキスト">' in html
    assert "<figcaption>キャプション</figcaption>" in html


def test_holding_history_figure_needs_two_disclosures():
    assert m.holding_history_figure("テスト製薬", "テストAM", [{"date": "2026-08-21", "ratio": 8.32}]) is None


def test_horizontal_bar_figure_needs_two_rows():
    assert m.shareholders_figure("テスト製薬", "テストAM", 8.32, []) is None


def test_build_article_figures_empty_without_context_facts():
    assert m.build_article_figures({"stock_name": "テスト製薬", "filer_name": "テストAM"}) == []


def test_build_article_figures_makes_three_png_figures():
    if not HAS_FONT:
        return
    figures = m.build_article_figures(FACT_SHEET)
    assert [f["filename"] for f in figures] == [
        "holding-history.png", "shareholders.png", "filer-portfolio.png",
    ]
    for f in figures:
        assert f["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
        assert f["alt"] and f["caption"] and f["alt_en"] and f["caption_en"] and f["anchors"]


def test_buyback_article_figures_empty_without_prior():
    assert m.buyback_article_figures({"stock_name": "テスト製薬", "amount_oku": 300.0, "prior": []}) == []


def test_buyback_article_figures_uses_prior_and_current():
    if not HAS_FONT:
        return
    figures = m.buyback_article_figures({
        "stock_name": "テスト製薬",
        "disc_date": "2026-08-21",
        "amount_oku": 300.0,
        "prior": [{"disclosed_at": "2025-05-08T06:00:00+00:00", "max_amount_yen": 10_000_000_000}],
    })
    assert len(figures) == 1 and figures[0]["bytes"][:8] == b"\x89PNG\r\n\x1a\n"
    assert "2025年" in figures[0]["anchors"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed" + ("" if HAS_FONT else "（フォント未検出のため描画テストは省略）"))

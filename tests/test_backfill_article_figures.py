"""既存記事への解説図バックフィル（tools/backfill_article_figures.py）のユニットテスト。
microCMS/Supabaseはモックし、対象選定と本文へのマージのみ検証する。

実行: python3 tests/test_backfill_article_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.backfill_article_figures as m

CHART = '<figure><img src="c.png" alt="テスト製薬（4568）株価推移（直近3ヶ月）"></figure>'
BODY = "<p>直答文</p><p>本文2</p><p>本文3</p><p>※推測</p>" + CHART


def test_has_explainer_figure_ignores_price_chart():
    assert m.has_explainer_figure(BODY) is False
    assert m.has_explainer_figure(BODY + "<figure>解説図</figure>") is True


def test_merge_into_body_keeps_chart_last():
    out = m.merge_into_body(BODY, [{"html": "<figure>F</figure>", "anchors": ["本文3"]}])
    assert out == "<p>直答文</p><p>本文2</p><p>本文3</p><figure>F</figure><p>※推測</p>" + CHART


def test_merge_into_body_without_chart():
    out = m.merge_into_body("<p>a</p><p>b</p><p>c</p>", [{"html": "<figure>F</figure>", "anchors": []}])
    assert out.endswith("<p>c</p>") and "<figure>F</figure>" in out


def test_select_candidates_skips_done_and_undated():
    articles = [
        {"id": "1", "stockCode": "4568", "dealDate": "2026-08-20T00:00:00.000Z", "body": BODY},
        {"id": "2", "stockCode": "4568", "dealDate": "2026-08-19T00:00:00.000Z",
         "body": BODY + "<figure>解説図</figure>"},          # 処理済み
        {"id": "3", "stockCode": "", "dealDate": "2026-08-18T00:00:00.000Z", "body": BODY},  # コード無し
        {"id": "4", "stockCode": "7203", "dealDate": "2026-06-01T00:00:00.000Z", "body": BODY},
    ]
    assert [a["id"] for a in m.select_candidates(articles)] == ["1", "4"]
    assert [a["id"] for a in m.select_candidates(articles, days=30, today="2026-08-25")] == ["1"]
    assert [a["id"] for a in m.select_candidates(articles, limit=1)] == ["1"]


def test_is_buyback_from_deal_type_or_tags():
    assert m.is_buyback({"dealType": ["自社株買い"]}) is True
    assert m.is_buyback({"dealType": [], "tags": "自社株買い,消却"}) is True
    assert m.is_buyback({"dealType": ["アクティビスト"], "tags": "EDINET"}) is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

"""lib/gsc_search_guard.py のユニットテスト。
ネットワーク（GSC）は呼ばず、URL→記事idの抽出と振り分け・fail-closedのみ検証する。

実行: python3 tests/test_gsc_search_guard.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import gsc_search_guard as g


def test_article_id_from_url_covers_ja_en_and_subdomain():
    assert g.article_id_from_url("https://kujira-watch.com/articles/4k_noajrha") == "4k_noajrha"
    assert g.article_id_from_url("https://kujira-watch.com/en/articles/zt5zant_bf") == "zt5zant_bf"
    assert g.article_id_from_url("https://en.kujira-watch.com/articles/9wty--sei4") == "9wty--sei4"
    assert g.article_id_from_url("https://kujira-watch.com/stocks/4425") is None
    assert g.article_id_from_url("https://kujira-watch.com/articles") is None


def test_ids_with_impressions_skips_zero_and_non_articles():
    rows = [
        {"keys": ["https://kujira-watch.com/articles/a1"], "impressions": 3},
        {"keys": ["https://kujira-watch.com/articles/a2"], "impressions": 0},
        {"keys": ["https://kujira-watch.com/en/articles/a3"], "impressions": 1},
        {"keys": ["https://kujira-watch.com/investors/2808"], "impressions": 30},
    ]
    assert g.ids_with_impressions(rows) == {"a1", "a3"}


def test_split_protected_keeps_searched_articles():
    targets = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}]
    deletable, kept = g.split_protected(targets, {"a2"})
    assert [a["id"] for a in deletable] == ["a1", "a3"]
    assert [a["id"] for a in kept] == ["a2"]


def test_filter_deletable_aborts_when_gsc_unavailable():
    original = g.searched_article_ids
    g.searched_article_ids = lambda days=g.LOOKBACK_DAYS: None
    try:
        assert g.filter_deletable([{"id": "a1"}]) is None
    finally:
        g.searched_article_ids = original


def test_filter_deletable_removes_protected():
    original = g.searched_article_ids
    g.searched_article_ids = lambda days=g.LOOKBACK_DAYS: {"a1"}
    try:
        assert [a["id"] for a in g.filter_deletable([{"id": "a1"}, {"id": "a2"}])] == ["a2"]
    finally:
        g.searched_article_ids = original


if __name__ == "__main__":
    test_article_id_from_url_covers_ja_en_and_subdomain()
    test_ids_with_impressions_skips_zero_and_non_articles()
    test_split_protected_keeps_searched_articles()
    test_filter_deletable_aborts_when_gsc_unavailable()
    test_filter_deletable_removes_protected()
    print("test_gsc_search_guard: 5件 OK")

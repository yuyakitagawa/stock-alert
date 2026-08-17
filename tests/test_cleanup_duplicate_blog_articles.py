"""重複記事クリーンアップ（tools/cleanup_duplicate_blog_articles）のユニットテスト。
ネットワーク（microCMS）はモックし、グループ化・削除対象選定のロジックのみ検証する。

実行: python3 tests/test_cleanup_duplicate_blog_articles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.cleanup_duplicate_blog_articles import find_duplicates


def test_find_duplicates_keeps_earliest_and_deletes_later():
    """同一銘柄・同一開示日・同一提出者・同一比率変化幅の記事は先発1件を残し
    後発を削除対象にする（先発はX投稿等で既にリンクされている可能性が高いため）。"""
    articles = [
        {"id": "late", "stockCode": "4812", "dealDate": "2026-08-17T00:00:00.000Z",
         "filerName": "Oasis", "ratioChangePct": 1.2, "createdAt": "2026-08-17T09:30:00.000Z"},
        {"id": "early", "stockCode": "4812", "dealDate": "2026-08-17T00:00:00.000Z",
         "filerName": "Oasis", "ratioChangePct": 1.2, "createdAt": "2026-08-17T08:10:00.000Z"},
    ]
    dups = find_duplicates(articles)
    assert [a["id"] for a in dups] == ["late"]


def test_find_duplicates_ignores_different_filer_or_date():
    """提出者・開示日・銘柄のいずれかが違えば別イベントとして削除しない。"""
    articles = [
        {"id": "a", "stockCode": "402A", "dealDate": "2025-08-20T00:00:00.000Z",
         "filerName": "グローバル・ブレイン", "createdAt": "2025-08-20T08:00:00.000Z"},
        {"id": "b", "stockCode": "402A", "dealDate": "2025-08-20T00:00:00.000Z",
         "filerName": "31VENTURES", "createdAt": "2025-08-20T09:00:00.000Z"},
        {"id": "c", "stockCode": "402A", "dealDate": "2025-08-21T00:00:00.000Z",
         "filerName": "グローバル・ブレイン", "createdAt": "2025-08-21T08:00:00.000Z"},
    ]
    assert find_duplicates(articles) == []


def test_find_duplicates_skips_articles_without_filer_name():
    """filerName送信開始(2026-08-15)前の旧記事は突き合わせできないため対象外。"""
    articles = [
        {"id": "a", "stockCode": "7203", "dealDate": "2026-08-14T00:00:00.000Z",
         "filerName": "", "createdAt": "2026-08-14T08:00:00.000Z"},
        {"id": "b", "stockCode": "7203", "dealDate": "2026-08-14T00:00:00.000Z",
         "createdAt": "2026-08-14T09:00:00.000Z"},
    ]
    assert find_duplicates(articles) == []


def test_find_duplicates_keeps_same_filer_with_different_ratio_change():
    """同一提出者が同日に複数の報告書を出す実例（2936 2025-08-13 橋本舜2件）は
    別イベントなので削除しない。already_published()の判定キーと揃える。"""
    articles = [
        {"id": "a", "stockCode": "2936", "dealDate": "2025-08-13T00:00:00.000Z",
         "filerName": "橋本舜", "ratioChangePct": 1.05, "createdAt": "2025-08-13T09:50:00.000Z"},
        {"id": "b", "stockCode": "2936", "dealDate": "2025-08-13T00:00:00.000Z",
         "filerName": "橋本舜", "ratioChangePct": 2.30, "createdAt": "2025-08-13T10:07:00.000Z"},
    ]
    assert find_duplicates(articles) == []


if __name__ == "__main__":
    test_find_duplicates_keeps_earliest_and_deletes_later()
    test_find_duplicates_ignores_different_filer_or_date()
    test_find_duplicates_skips_articles_without_filer_name()
    test_find_duplicates_keeps_same_filer_with_different_ratio_change()
    print("全テスト成功 (4件)")

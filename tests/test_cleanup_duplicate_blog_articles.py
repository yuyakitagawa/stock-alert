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


def test_find_duplicates_matches_legacy_articles_by_title():
    """filerName送信開始(2026-08-15)前の旧記事は、同一銘柄・同一開示日でタイトルまで
    一致すれば同一開示の重複と見なす。旧記事は概算金額でしか突き合わせておらず、
    株価キャッシュの更新のたびに同じ開示が再投稿された（実例: 9706に同一記事が11件）。"""
    articles = [
        {"id": "a", "stockCode": "9706", "dealDate": "2025-10-07T00:00:00.000Z",
         "title": "みずほ銀行が日本空港ビルデングを3.54%保有", "dealAmount": 120.0,
         "createdAt": "2025-10-07T08:00:00.000Z"},
        {"id": "b", "stockCode": "9706", "dealDate": "2025-10-07T00:00:00.000Z",
         "title": "みずほ銀行が日本空港ビルデングを3.54%保有", "dealAmount": 121.4,
         "createdAt": "2025-10-07T09:00:00.000Z"},
        {"id": "c", "stockCode": "9706", "dealDate": "2025-10-07T00:00:00.000Z",
         "title": "みずほ銀行が日本空港ビルデングを3.54%保有", "dealAmount": 119.2,
         "createdAt": "2025-10-07T10:00:00.000Z"},
    ]
    assert [a["id"] for a in find_duplicates(articles)] == ["b", "c"]


def test_find_duplicates_keeps_legacy_articles_with_different_titles():
    """旧記事でもタイトルが違えば別の開示なので削除しない。"""
    articles = [
        {"id": "a", "stockCode": "9706", "dealDate": "2025-10-07T00:00:00.000Z",
         "title": "みずほ銀行が日本空港ビルデングを3.54%保有",
         "createdAt": "2025-10-07T08:00:00.000Z"},
        {"id": "b", "stockCode": "9706", "dealDate": "2025-10-07T00:00:00.000Z",
         "title": "ブラックロックが日本空港ビルデングを1.17%に減らす",
         "createdAt": "2025-10-07T09:00:00.000Z"},
    ]
    assert find_duplicates(articles) == []


def test_find_duplicates_skips_articles_without_matchable_key():
    """タイトルも提出者も無い記事、銘柄コードや開示日が欠けた記事は突き合わせ不能。
    誤って別開示を消さないよう対象外にする。"""
    articles = [
        {"id": "a", "stockCode": "7203", "dealDate": "2026-08-14T00:00:00.000Z",
         "title": "", "createdAt": "2026-08-14T08:00:00.000Z"},
        {"id": "b", "stockCode": "7203", "dealDate": "2026-08-14T00:00:00.000Z",
         "createdAt": "2026-08-14T09:00:00.000Z"},
        {"id": "c", "stockCode": "", "dealDate": "2026-08-14T00:00:00.000Z",
         "title": "同じタイトル", "createdAt": "2026-08-14T08:00:00.000Z"},
        {"id": "d", "stockCode": "", "dealDate": "2026-08-14T00:00:00.000Z",
         "title": "同じタイトル", "createdAt": "2026-08-14T09:00:00.000Z"},
        {"id": "e", "stockCode": "7203", "title": "開示日なし",
         "createdAt": "2026-08-14T08:00:00.000Z"},
        {"id": "f", "stockCode": "7203", "title": "開示日なし",
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


def test_find_duplicates_catches_buyback_articles_without_filer_name():
    """自社株買い記事は発行体自身の開示でfilerNameを持たない。旧実装ではfilerNameが
    空という理由だけで丸ごと対象外になり、already_published()のすり抜けを回収できず
    同一開示から13本の重複記事が公開された（2026-08-25、コンヴァノ6574）。
    tagsで自社株買いと判別し、銘柄×開示日で重複と見なす。"""
    articles = [
        {"id": "late", "stockCode": "6574", "dealDate": "2026-08-24T00:00:00.000Z",
         "tags": "自社株買い", "createdAt": "2026-08-25T02:15:41.000Z"},
        {"id": "early", "stockCode": "6574", "dealDate": "2026-08-24T00:00:00.000Z",
         "tags": "自社株買い,消却", "createdAt": "2026-08-24T14:38:21.000Z"},
        {"id": "later", "stockCode": "6574", "dealDate": "2026-08-24T00:00:00.000Z",
         "tags": "自社株買い", "createdAt": "2026-08-25T03:40:51.000Z"},
    ]
    dups = find_duplicates(articles)
    assert sorted(a["id"] for a in dups) == ["late", "later"]


def test_find_duplicates_keeps_buyback_articles_of_different_stock_or_date():
    """銘柄か開示日が違えば別の取得枠決議なので削除しない。"""
    articles = [
        {"id": "a", "stockCode": "6574", "dealDate": "2026-08-24T00:00:00.000Z",
         "tags": "自社株買い", "createdAt": "2026-08-24T14:38:21.000Z"},
        {"id": "b", "stockCode": "9706", "dealDate": "2026-08-24T00:00:00.000Z",
         "tags": "自社株買い", "createdAt": "2026-08-24T15:10:00.000Z"},
        {"id": "c", "stockCode": "6574", "dealDate": "2026-05-12T00:00:00.000Z",
         "tags": "自社株買い", "createdAt": "2026-05-12T09:00:00.000Z"},
    ]
    assert find_duplicates(articles) == []


if __name__ == "__main__":
    test_find_duplicates_keeps_earliest_and_deletes_later()
    test_find_duplicates_ignores_different_filer_or_date()
    test_find_duplicates_matches_legacy_articles_by_title()
    test_find_duplicates_keeps_legacy_articles_with_different_titles()
    test_find_duplicates_skips_articles_without_matchable_key()
    test_find_duplicates_keeps_same_filer_with_different_ratio_change()
    test_find_duplicates_catches_buyback_articles_without_filer_name()
    test_find_duplicates_keeps_buyback_articles_of_different_stock_or_date()
    print("全テスト成功 (8件)")

"""tools/delete_low_value_blog_articles.py のユニットテスト。
ネットワーク（microCMS）は呼ばず、抽出ロジックのみ検証する。

実行: python3 tests/test_delete_low_value_blog_articles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.delete_low_value_blog_articles import find_low_value


def test_find_low_value_picks_only_below_threshold():
    articles = [
        {"id": "keep-amount", "dealAmount": 3.0, "ratioChangePct": 0.02},
        {"id": "keep-ratio", "dealAmount": 0.5, "ratioChangePct": -1.2},
        {"id": "drop", "dealAmount": 0.0, "ratioChangePct": 0.01},
    ]
    assert [a["id"] for a in find_low_value(articles)] == ["drop"]


def test_find_low_value_treats_missing_fields_as_zero():
    # ratioChangePctが無い旧記事は変化幅0扱いになり、金額だけで判定される
    assert [a["id"] for a in find_low_value([{"id": "old", "dealAmount": None}])] == ["old"]
    assert find_low_value([{"id": "old-big", "dealAmount": 12.0}]) == []


if __name__ == "__main__":
    test_find_low_value_picks_only_below_threshold()
    test_find_low_value_treats_missing_fields_as_zero()
    print("全テスト成功 (2件)")

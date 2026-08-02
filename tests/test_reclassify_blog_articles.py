"""tools/reclassify_blog_articles.py のロジックのユニットテスト。

実行: python3 tests/test_reclassify_blog_articles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.reclassify_blog_articles import normalize_deal_type


def test_normalize_deal_type_passes_through_plain_string():
    assert normalize_deal_type("アクティビスト") == "アクティビスト"


def test_normalize_deal_type_unwraps_single_element_list():
    assert normalize_deal_type(["日系証券銀行"]) == "日系証券銀行"


def test_normalize_deal_type_empty_list_returns_none():
    """空配列(値未設定)でIndexErrorにならず、Noneを返す
    （本番実行で実際に発生したバグの再発防止）。"""
    assert normalize_deal_type([]) is None


def test_normalize_deal_type_none_passes_through():
    assert normalize_deal_type(None) is None


if __name__ == "__main__":
    test_normalize_deal_type_passes_through_plain_string()
    test_normalize_deal_type_unwraps_single_element_list()
    test_normalize_deal_type_empty_list_returns_none()
    test_normalize_deal_type_none_passes_through()
    print("OK: test_reclassify_blog_articles (4 tests)")

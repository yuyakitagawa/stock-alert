"""tools/reclassify_blog_articles.py のロジックのユニットテスト。

実行: python3 tests/test_reclassify_blog_articles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.reclassify_blog_articles import normalize_deal_type, build_put_payload


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


def test_build_put_payload_strips_metadata_and_overwrites_deal_type():
    """PUTは完全上書きのため、id等のメタ情報は送らずdealTypeだけ書き換えて他は保持する。"""
    article = {
        "id": "abc123", "createdAt": "2026-07-20T00:00:00.000Z",
        "updatedAt": "2026-07-20T00:00:00.000Z", "publishedAt": "2026-07-20T00:00:00.000Z",
        "revisedAt": "2026-07-20T00:00:00.000Z",
        "title": "タイトル", "body": "<p>本文</p>", "stockName": "テスト自動車",
        "stockCode": "7203", "dealType": "インサイダー買い", "dealDate": "2026-07-20T00:00:00.000Z",
        "dealAmount": 12.3,
    }
    payload = build_put_payload(article, "個人")
    assert "id" not in payload
    assert "createdAt" not in payload
    assert "updatedAt" not in payload
    assert "publishedAt" not in payload
    assert "revisedAt" not in payload
    assert payload["dealType"] == "個人"
    assert payload["title"] == "タイトル"
    assert payload["stockCode"] == "7203"
    assert payload["dealAmount"] == 12.3


if __name__ == "__main__":
    test_normalize_deal_type_passes_through_plain_string()
    test_normalize_deal_type_unwraps_single_element_list()
    test_normalize_deal_type_empty_list_returns_none()
    test_normalize_deal_type_none_passes_through()
    test_build_put_payload_strips_metadata_and_overwrites_deal_type()
    print("OK: test_reclassify_blog_articles (5 tests)")

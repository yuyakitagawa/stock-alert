"""ブログ記事自動投稿（web/publish_blog_articles）のロジックのユニットテスト。
ネットワーク（microCMS/yfinance/Supabase/Claude）は全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_publish_blog_articles.py
"""
import os
import sys
import json
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.publish_blog_articles as m


def test_estimate_deal_amount_oku_calculation():
    with mock.patch.object(m, "shares_outstanding", return_value=1_000_000_000), \
         mock.patch.object(m, "get_price_at_date", return_value=2000.0):
        # 10億株 × 2000円 × 5% = 1000億円
        assert m.estimate_deal_amount_oku("7203", 5.0, "2026-07-20") == 1000.0


def test_estimate_deal_amount_oku_none_when_no_change():
    assert m.estimate_deal_amount_oku("7203", 0, "2026-07-20") is None


def test_estimate_deal_amount_oku_none_when_shares_missing():
    with mock.patch.object(m, "shares_outstanding", return_value=None):
        assert m.estimate_deal_amount_oku("7203", 5.0, "2026-07-20") is None


def test_generate_article_body_parses_plain_json():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>", "dealType": "日系ファンド買い"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"title": "タイトル", "body": "<p>本文</p>", "dealType": "日系ファンド買い"}


def test_generate_article_body_strips_code_fence():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>", "dealType": "インサイダー買い"})
    fenced = f"```json\n{raw}\n```"
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(fenced)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"title": "タイトル", "body": "<p>本文</p>", "dealType": "インサイダー買い"}


def test_generate_article_body_none_on_empty_title():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "", "body": "<p>本文</p>", "dealType": "その他"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.generate_article_body(fact_sheet) is None


def test_generate_article_body_falls_back_to_sonota_on_invalid_deal_type():
    """Claudeが決められた選択肢以外を返したら「その他」に丸める。"""
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>", "dealType": "謎の分類"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.generate_article_body(fact_sheet)
    assert result["dealType"] == "その他"


def test_build_and_publish_excludes_sell_and_maps_fields():
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
        {"issuer_code": "9999", "name": "テスト商事", "filer_name": "アセットマネジメント株式会社",
         "holding_ratio": 6.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "変更報告書"},
        {"issuer_code": "1234", "name": "売却テスト", "filer_name": "ファンド株式会社",
         "holding_ratio": 4.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "株式の譲渡・売却による変更報告書"},
        {"issuer_code": "6502", "name": "東芝型テスト", "filer_name": "キオクシアファンド",
         "holding_ratio": 15.10, "holding_ratio_prior": 16.10, "disc_date": "2026-07-20",
         "doc_type_code": "360", "doc_description": "変更報告書"},  # 概要はキーワード無しだが比率減少=売り
    ]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "generate_article_body",
                            side_effect=[
                                {"title": "テストタイトル1", "body": "<p>本文</p>", "dealType": "インサイダー買い"},
                                {"title": "テストタイトル2", "body": "<p>本文</p>", "dealType": "日系ファンド買い"},
                            ]), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert len(results) == 2  # 売却は除外される（概要文言に頼らず保有比率の増減でも判定）
    assert [r["stockCode"] for r in results] == ["7203", "9999"]  # |比率|降順
    assert results[0]["dealType"] == "インサイダー買い"
    assert results[1]["dealType"] == "日系ファンド買い"
    assert results[0]["dealDate"] == "2026-07-20T00:00:00.000Z"
    assert results[0]["dealAmount"] == 12.3


def test_build_and_publish_skips_when_already_published():
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=True):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []


def test_build_and_publish_skips_when_amount_unestimable():
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=None):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []


class _FakeResponse:
    def __init__(self, status_code, text, json_data=None):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


def test_publish_article_retries_as_array_on_type_mismatch():
    """セレクトフィールドが複数選択(配列)設定の場合、'has unexpected data type' を
    検知してその項目だけ配列に包んで一度だけ再送信する。"""
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(201, "", {"id": "retried-id"}),
    ]
    payload = {"title": "t", "dealType": "機関投資家買い"}
    with mock.patch.object(m, "_post_once", side_effect=responses) as post_mock:
        content_id = m.publish_article(payload)
    assert content_id == "retried-id"
    assert post_mock.call_count == 2
    retried_payload = post_mock.call_args_list[1].args[0]
    assert retried_payload["dealType"] == ["機関投資家買い"]


def test_publish_article_gives_up_when_same_field_fails_twice():
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
    ]
    payload = {"title": "t", "dealType": "機関投資家買い"}
    with mock.patch.object(m, "_post_once", side_effect=responses):
        content_id = m.publish_article(payload)
    assert content_id is None


def test_build_and_publish_stops_early_on_permission_error():
    """1件目でAPIキーの権限エラーが出たら、2件目以降はClaude呼び出しごと打ち切る
    （無駄なトークン消費を防ぐ）。"""
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
        {"issuer_code": "9999", "name": "テスト商事", "filer_name": "アセットマネジメント株式会社",
         "holding_ratio": 6.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "変更報告書"},
    ]
    generate_calls = []

    def _track_generate(fact_sheet):
        generate_calls.append(fact_sheet)
        return {"title": "テストタイトル", "body": "<p>本文</p>", "dealType": "その他"}

    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "generate_article_body", side_effect=_track_generate), \
         mock.patch.object(m, "publish_article",
                            side_effect=m.MicroCMSPermissionError("HTTP 400: forbidden")):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert results == []
    assert len(generate_calls) == 1  # 2件目はClaudeを呼ばずに打ち切られる


def _fact_sheet():
    return {"stock_name": "テスト", "stock_code": "7203", "filer_name": "X",
            "doc_type_label": "大量保有報告書", "holding_ratio": 8.5,
            "disc_date": "2026-07-20", "deal_amount_oku": 12.3}


def _fake_client(text):
    class _Block:
        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self, text):
            self._text = text

        def create(self, **kwargs):
            return _Resp(self._text)

    class _Client:
        def __init__(self, text):
            self.messages = _Messages(text)

    return _Client(text)


if __name__ == "__main__":
    test_estimate_deal_amount_oku_calculation()
    test_estimate_deal_amount_oku_none_when_no_change()
    test_estimate_deal_amount_oku_none_when_shares_missing()
    test_generate_article_body_parses_plain_json()
    test_generate_article_body_strips_code_fence()
    test_generate_article_body_none_on_empty_title()
    test_generate_article_body_falls_back_to_sonota_on_invalid_deal_type()
    test_build_and_publish_excludes_sell_and_maps_fields()
    test_build_and_publish_skips_when_already_published()
    test_build_and_publish_skips_when_amount_unestimable()
    test_build_and_publish_stops_early_on_permission_error()
    test_publish_article_retries_as_array_on_type_mismatch()
    test_publish_article_gives_up_when_same_field_fails_twice()
    print("全テスト成功 (13件)")

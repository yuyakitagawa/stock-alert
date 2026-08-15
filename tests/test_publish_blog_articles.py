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


def test_shares_outstanding_retries_then_succeeds():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(
        side_effect=[Exception("rate limited"), Exception("rate limited"), {"sharesOutstanding": 5_000_000}]
    )
    with mock.patch("yfinance.Ticker", return_value=ticker), mock.patch("time.sleep"):
        assert m.shares_outstanding("7203") == 5_000_000.0


def test_shares_outstanding_falls_back_to_implied_shares_outstanding():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(return_value={"impliedSharesOutstanding": 3_000_000})
    with mock.patch("yfinance.Ticker", return_value=ticker):
        assert m.shares_outstanding("3269") == 3_000_000.0


def test_shares_outstanding_returns_none_after_exhausting_retries():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(side_effect=Exception("rate limited"))
    with mock.patch("yfinance.Ticker", return_value=ticker), mock.patch("time.sleep"):
        assert m.shares_outstanding("7203") is None


def test_generate_article_body_parses_plain_json():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"title": "タイトル", "body": "<p>本文</p>"}


def test_generate_article_body_strips_code_fence():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    fenced = f"```json\n{raw}\n```"
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(fenced)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"title": "タイトル", "body": "<p>本文</p>"}


def test_generate_article_body_none_on_empty_title():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "", "body": "<p>本文</p>"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.generate_article_body(fact_sheet) is None


def test_classify_filer_returns_cached_master_row_without_calling_claude():
    """edinet_filer_classificationに登録済みの提出者はClaudeを呼ばずマスターの値を返す。"""
    cached = {"category": "外資系伝統運用会社", "is_foreign": True, "description": "米大手運用会社"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.classify_filer("Ｆｉｄｅｌｉｔｙ")
    assert result == cached
    assert select_mock.called
    assert not anthropic_mock.called


def test_classify_filer_asks_claude_and_persists_when_not_cached():
    """マスター未登録の提出者はClaudeに判定させ、結果をedinet_filer_classificationへ保存する。"""
    raw = json.dumps({"category": "アクティビスト", "is_foreign": True, "description": "海外の物言う株主"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.classify_filer("新規ファンド")
    assert result == {"category": "アクティビスト", "is_foreign": True, "description": "海外の物言う株主"}
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["filer_name"] == "新規ファンド"
    assert saved_rows[0]["confidence"] == "low"


def test_classify_filer_falls_back_to_sonota_on_invalid_category():
    """Claudeが決められた選択肢以外を返したら「その他」に丸める。"""
    raw = json.dumps({"category": "謎の分類", "is_foreign": False, "description": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.classify_filer("謎の提出者")
    assert result["category"] == "その他"


def test_build_and_publish_includes_sell_and_tags_them():
    """売り方向（概要のキーワード or 保有比率の減少で判定）も除外せず記事化し、
    tagsに"売り"を付与して買いと区別する（買い側はtagsを変えない後方互換）。"""
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
         mock.patch.object(m, "classify_filer",
                            side_effect=[
                                {"category": "外資系伝統運用会社", "is_foreign": True, "description": ""},
                                {"category": "個人", "is_foreign": False, "description": ""},
                                {"category": "国内アセットマネジメント", "is_foreign": False, "description": ""},
                                {"category": "PE・メザニンファンド", "is_foreign": False, "description": ""},
                            ]), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=4, dry_run=False)

    assert len(results) == 4  # 売りも除外されない
    # |比率|降順: 6502(15.10) > 7203(8.5) > 9999(6.0) > 1234(4.0)
    assert [r["stockCode"] for r in results] == ["6502", "7203", "9999", "1234"]
    by_code = {r["stockCode"]: r for r in results}
    assert by_code["6502"]["tags"] == "EDINET,自動生成,売り"  # 比率減少による売り判定
    assert by_code["1234"]["tags"] == "EDINET,自動生成,売り"  # キーワードによる売り判定
    assert by_code["7203"]["tags"] == "EDINET,自動生成"  # 買いはtags不変
    assert by_code["9999"]["tags"] == "EDINET,自動生成"
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


def test_get_featured_article_ids_preserves_pool_order_without_reordering():
    """kujira-watch側getFeaturedArticles()と同じロジック: プールはmicroCMS側で
    -dealDate,-dealAmount順（日付優先→同日内は金額降順）に返ってくるので、
    Python側では並べ替えず先頭count件のidをそのまま採用する。プール全体を
    金額だけで並べ替えると、投稿数が少ない日に数日前の大型取引が「注目」を
    占有し続けてしまう（実際に発生したバグ）ため、この並べ替えはしない。"""
    pool = [
        {"id": "today-big"},
        {"id": "today-small"},
        {"id": "older-huge"},
        {"id": "older-medium"},
    ]
    resp = _FakeResponse(200, "", {"contents": pool})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", return_value=resp):
        ids = m.get_featured_article_ids(pool_size=20, count=2)
    assert ids == {"today-big", "today-small"}


def test_get_featured_article_ids_returns_empty_set_on_http_error():
    resp = _FakeResponse(500, "server error")
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", return_value=resp):
        assert m.get_featured_article_ids() == set()


def test_get_featured_article_ids_returns_empty_set_on_exception():
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=Exception("timeout")):
        assert m.get_featured_article_ids() == set()


class _FakeResponse:
    def __init__(self, status_code, text, json_data=None, content=b""):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_publish_article_retries_as_array_on_type_mismatch():
    """セレクトフィールドが複数選択(配列)設定の場合、'has unexpected data type' を
    検知してその項目だけ配列に包んで一度だけ再送信する。"""
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(201, "", {"id": "retried-id"}),
    ]
    payload = {"title": "t", "dealType": "個人"}
    with mock.patch.object(m, "_post_once", side_effect=responses) as post_mock:
        content_id = m.publish_article(payload)
    assert content_id == "retried-id"
    assert post_mock.call_count == 2
    retried_payload = post_mock.call_args_list[1].args[0]
    assert retried_payload["dealType"] == ["個人"]


def test_publish_article_drops_non_string_field_on_type_mismatch():
    """eyecatch等のオブジェクト値フィールドは配列化では直せないため、
    そのフィールドを除外して再送信し、記事自体は投稿される。"""
    responses = [
        _FakeResponse(400, '{"message":"\'eyecatch\' has unexpected data type."}'),
        _FakeResponse(201, "", {"id": "no-eyecatch-id"}),
    ]
    payload = {"title": "t", "eyecatch": {"url": "https://example.test/x.png"}}
    with mock.patch.object(m, "_post_once", side_effect=responses) as post_mock:
        content_id = m.publish_article(payload)
    assert content_id == "no-eyecatch-id"
    assert post_mock.call_count == 2
    retried_payload = post_mock.call_args_list[1].args[0]
    assert "eyecatch" not in retried_payload


def test_publish_article_gives_up_when_same_field_fails_twice():
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
    ]
    payload = {"title": "t", "dealType": "個人"}
    with mock.patch.object(m, "_post_once", side_effect=responses):
        content_id = m.publish_article(payload)
    assert content_id is None


def test_update_article_retries_as_array_on_type_mismatch():
    """update_article()（PATCH）もpublish_article()（POST）と同じ型不一致リトライを行う
    （tools/reclassify_blog_articles.py の一括再分類・tools/rewrite_thin_blog_articles.py の
    本文リライトで使う）。"""
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(200, "", {"id": "content-1"}),
    ]
    payload = {"dealType": "アクティビスト"}
    with mock.patch.object(m, "_patch_once", side_effect=responses) as patch_mock:
        ok = m.update_article("content-1", payload)
    assert ok is True
    assert patch_mock.call_count == 2
    retried_payload = patch_mock.call_args_list[1].args[1]
    assert retried_payload["dealType"] == ["アクティビスト"]


def test_update_article_returns_false_on_failure():
    responses = [_FakeResponse(400, '{"message":"invalid"}')]
    with mock.patch.object(m, "_patch_once", side_effect=responses):
        ok = m.update_article("content-1", {"dealType": "その他"})
    assert ok is False


class _FakeDraw:
    """textbboxの幅を文字数×10pxで返すダミーdraw（実フォント無しで折り返しロジックだけ検証する）。"""

    def textbbox(self, xy, text, font=None):
        return (0, 0, len(text) * 10, 20)


def test_wrap_text_lines_breaks_on_width():
    lines = m._wrap_text_lines(_FakeDraw(), "あいうえおかきくけこ", font=None, max_width=50)
    assert lines == ["あいうえお", "かきくけこ"]


def test_wrap_text_lines_respects_max_lines():
    lines = m._wrap_text_lines(_FakeDraw(), "あ" * 30, font=None, max_width=50, max_lines=2)
    assert len(lines) == 2


def test_search_pexels_photo_returns_none_without_api_key():
    with mock.patch.object(m, "PEXELS_API_KEY", ""):
        assert m.search_pexels_photo("finance") is None


def test_search_pexels_photo_returns_bytes_and_photographer_on_success():
    search_resp = _FakeResponse(200, "", {"photos": [{
        "src": {"large": "https://example.test/a.jpg"}, "photographer": "Jane Doe",
    }]})
    photo_resp = _FakeResponse(200, "", content=b"fake-image-bytes")
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=[search_resp, photo_resp]):
        result = m.search_pexels_photo("finance")
    assert result == {"bytes": b"fake-image-bytes", "photographer": "Jane Doe"}


def test_search_pexels_photo_defaults_photographer_when_missing():
    search_resp = _FakeResponse(200, "", {"photos": [{"src": {"large": "https://example.test/a.jpg"}}]})
    photo_resp = _FakeResponse(200, "", content=b"fake-image-bytes")
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=[search_resp, photo_resp]):
        result = m.search_pexels_photo("finance")
    assert result["photographer"] == "Pexels"


def test_search_pexels_photo_returns_none_when_no_results():
    search_resp = _FakeResponse(200, "", {"photos": []})
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", return_value=search_resp):
        assert m.search_pexels_photo("finance") is None


def test_search_pexels_photo_returns_none_on_exception():
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=Exception("timeout")):
        assert m.search_pexels_photo("finance") is None


def test_upload_eyecatch_returns_url_on_success():
    resp = _FakeResponse(201, "", {"url": "https://images.microcms-assets.io/assets/x/y.png"})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        url = m.upload_eyecatch(b"png-bytes")
    assert url == "https://images.microcms-assets.io/assets/x/y.png"


def test_upload_eyecatch_returns_none_on_failure():
    resp = _FakeResponse(403, "forbidden")
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        assert m.upload_eyecatch(b"png-bytes") is None


_SAMPLE_EYECATCH_CARD = {
    "filer_name": "Oasis Management",
    "stock_name": "アインHD",
    "holding_ratio": 20.93,
    "badge_label": "📈 買い増し",
    "disc_date": "2026-07-20",
}


def test_build_eyecatch_for_article_none_without_pexels_key():
    with mock.patch.object(m, "PEXELS_API_KEY", ""):
        assert m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD) is None


def test_build_eyecatch_for_article_none_when_generation_fails():
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch.object(m, "generate_eyecatch_image", return_value=None):
        assert m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD) is None


def test_build_eyecatch_for_article_returns_url_dict_on_success():
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch.object(m, "generate_eyecatch_image", return_value=b"png-bytes"), \
         mock.patch.object(m, "upload_eyecatch", return_value="https://images.microcms-assets.io/x.png"):
        result = m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD)
    assert result == {"url": "https://images.microcms-assets.io/x.png"}


def test_build_and_publish_includes_eyecatch_when_available():
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
    ]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article",
                            return_value={"url": "https://images.microcms-assets.io/x.png"}) as eyecatch_mock, \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert results[0]["eyecatch"] == {"url": "https://images.microcms-assets.io/x.png"}
    eyecatch_mock.assert_called_once_with("個人", {
        "filer_name": "個人 太郎",
        "stock_name": "テスト自動車",
        "holding_ratio": 8.5,
        "badge_label": "📈 新規取得",
        "disc_date": "2026-07-20",
    })


def test_build_and_publish_skips_eyecatch_on_dry_run():
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
    ]
    with mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article") as eyecatch_mock:
        m.build_and_publish(days=3, max_articles=3, dry_run=True)

    eyecatch_mock.assert_not_called()


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
        return {"title": "テストタイトル", "body": "<p>本文</p>"}

    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "その他", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body", side_effect=_track_generate), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article",
                            side_effect=m.MicroCMSPermissionError("HTTP 400: forbidden")):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert results == []
    assert len(generate_calls) == 1  # 2件目はClaudeを呼ばずに打ち切られる


def _fact_sheet():
    return {"stock_name": "テスト", "stock_code": "7203", "filer_name": "X",
            "doc_type_label": "大量保有報告書", "holding_ratio": 8.5,
            "disc_date": "2026-07-20", "deal_amount_oku": 12.3}


def test_dp_level_label_thresholds():
    assert m.dp_level_label(35) == "高"
    assert m.dp_level_label(25) == "やや高"
    assert m.dp_level_label(18) == "中"
    assert m.dp_level_label(10) == "やや低"
    assert m.dp_level_label(3) == "低"


def test_get_pit_ranking_snapshot_queries_as_of_disc_date():
    """記事公開時点(post-hoc)ではなく、開示日以前で直近のスナップショットを取る
    （先読みバイアス防止、CLAUDE.md PIT規律）。"""
    with mock.patch.object(m.sb, "select_one", return_value={"close": 3000, "drop_prob": 12.0}) as select_mock:
        result = m.get_pit_ranking_snapshot("7203", "2026-07-20")
    assert result == {"close": 3000, "drop_prob": 12.0}
    query = select_mock.call_args.args[1]
    assert "code=eq.7203" in query
    assert "date=lte.2026-07-20" in query


def _capturing_client(text):
    calls = []

    class _Block:
        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Resp(text)

    class _Client:
        def __init__(self):
            self.messages = _Messages()

    return _Client(), calls


def test_generate_article_body_includes_context_when_available():
    fact_sheet = _fact_sheet()
    fact_sheet["context_close"] = 3000.0
    fact_sheet["context_dp_level"] = "やや低"
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "やや低" in prompt
    assert "3,000円" in prompt


def test_generate_article_body_omits_context_when_unavailable():
    fact_sheet = _fact_sheet()  # context_close/context_dp_level 無し
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "下落リスク水準" not in prompt


def test_generate_article_body_includes_ratio_increase_when_available():
    """既存開示からの増加分が分かる場合、変化幅(ポイント)をプロンプトに織り込む。"""
    fact_sheet = _fact_sheet()
    fact_sheet["ratio_change_pct"] = 2.48
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "これまでの開示から2.48ポイント増加" in prompt


def test_generate_article_body_describes_new_position_when_change_equals_ratio():
    """過去開示が無く変化幅=保有比率そのものの場合は「新規保有」の文脈で伝える（実際には
    5%未満だった保証は無いため、データで確認できる範囲の表現に留める）。"""
    fact_sheet = _fact_sheet()
    fact_sheet["ratio_change_pct"] = fact_sheet["holding_ratio"]
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "新規保有" in prompt
    assert "ポイント増加" not in prompt


def test_generate_article_body_uses_buy_wording_by_default():
    """directionを指定しない場合は従来通り「取得」「推定取得金額」として扱う（後方互換）。"""
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "推定取得金額" in prompt
    assert "推定売却金額" not in prompt


def test_generate_article_body_uses_sell_wording_when_direction_is_sell():
    """direction="sell"なら「売却」「推定売却金額」の文言でプロンプトを構成する。"""
    fact_sheet = _fact_sheet()
    fact_sheet["direction"] = "sell"
    fact_sheet["deal_amount_label"] = "推定売却金額"
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "推定売却金額" in prompt
    assert "推定取得金額" not in prompt
    assert "この売却が今後" in prompt


def test_generate_article_body_includes_company_description_when_available():
    """事業内容の事実があればプロンプトに織り込み、冒頭で触れるよう指示する。"""
    fact_sheet = _fact_sheet()
    fact_sheet["company_description"] = "美容院ブランドのライセンス展開を行う企業"
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "美容院ブランドのライセンス展開を行う企業" in prompt


def test_generate_article_body_always_requests_labelled_speculation():
    """事業内容・下落リスク文脈の有無に関わらず、「※推測:」ラベル付きの1文を必ず要求する。"""
    fact_sheet = _fact_sheet()  # company_description/context 無し
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "※推測:" in prompt
    assert "創作しないでください" in prompt


def test_generate_article_body_prompt_requests_english_translation():
    """kujira-watch(/en)向けにtitleEn/bodyEnもJSONに含めるよう1回の呼び出しでプロンプトに要求する
    （JA/ENを別々に生成すると事実がズレたりAPI呼び出しが倍になるため）。"""
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>", "titleEn": "Title", "bodyEn": "<p>Body</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        result = m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "titleEn" in prompt
    assert "bodyEn" in prompt
    assert result["titleEn"] == "Title"
    assert result["bodyEn"] == "<p>Body</p>"


def test_build_and_publish_includes_english_fields_when_generated():
    """generate_article_body()がtitleEn/bodyEnを返した場合、publish_article()へのpayloadに含める。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>",
                                          "titleEn": "Test Title", "bodyEn": "<p>Body</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)
    assert results[0]["titleEn"] == "Test Title"
    assert results[0]["bodyEn"] == "<p>Body</p>"


def test_build_and_publish_omits_english_fields_when_not_generated():
    """titleEn/bodyEnが無い（部分失敗・後方互換ケース）場合はpayloadにキー自体を含めない。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)
    assert "titleEn" not in results[0]
    assert "bodyEn" not in results[0]


def test_get_company_description_returns_cached_without_calling_claude():
    cached = {"description": "美容院チェーンを展開する企業"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを展開する企業"
    assert select_mock.called
    assert not anthropic_mock.called


def test_get_company_description_asks_claude_and_persists_when_not_cached():
    raw = json.dumps({"description": "美容院チェーンを展開する企業"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを展開する企業"
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["code"] == "9439"
    assert saved_rows[0]["description"] == "美容院チェーンを展開する企業"


def test_get_company_description_returns_empty_without_api_key_when_not_cached():
    with mock.patch.object(m, "ANTHROPIC_API_KEY", ""), \
         mock.patch.object(m.sb, "select_one", return_value=None):
        assert m.get_company_description("9439", "エム・エイチ・グループ") == ""


def test_get_filer_profile_returns_cached_without_calling_claude():
    cached = {"profile": "1990年代設立の国内独立系運用会社。"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.get_filer_profile("テストファンド", "独立系ブティックAM")
    assert result == "1990年代設立の国内独立系運用会社。"
    assert select_mock.called
    assert not anthropic_mock.called


def test_get_filer_profile_asks_claude_and_persists_when_not_cached():
    raw = json.dumps({"profile": "1990年代設立の国内独立系運用会社。"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_filer_profile("テストファンド", "独立系ブティックAM")
    assert result == "1990年代設立の国内独立系運用会社。"
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["filer_name"] == "テストファンド"
    # categoryも含めること: PostgreSQLはON CONFLICT時のUPDATE分岐でも候補行構築時点で
    # NOT NULL制約(category)を評価するため、欠くと既存行の更新のつもりでも失敗する。
    assert saved_rows[0]["category"] == "独立系ブティックAM"
    assert saved_rows[0]["profile"] == "1990年代設立の国内独立系運用会社。"


def test_get_filer_profile_returns_empty_without_api_key_when_not_cached():
    with mock.patch.object(m, "ANTHROPIC_API_KEY", ""), \
         mock.patch.object(m.sb, "select_one", return_value=None):
        assert m.get_filer_profile("テストファンド", "独立系ブティックAM") == ""


def test_get_filer_profile_returns_empty_when_claude_returns_blank():
    """情報が乏しい個人名義等の提出者は空文字のまま(創作させない)。"""
    raw = json.dumps({"profile": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_filer_profile("個人 太郎", "個人")
    assert result == ""
    upsert_mock.assert_not_called()


def test_upload_price_chart_returns_url_on_success():
    resp = _FakeResponse(201, "", {"url": "https://images.microcms-assets.io/assets/x/chart.png"})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        url = m.upload_price_chart(b"png-bytes")
    assert url == "https://images.microcms-assets.io/assets/x/chart.png"


def test_build_price_chart_for_article_none_when_generation_fails():
    with mock.patch.object(m, "generate_price_chart_image", return_value=None):
        assert m.build_price_chart_for_article("7203", "テスト自動車") is None


def test_build_price_chart_for_article_returns_url_on_success():
    with mock.patch.object(m, "generate_price_chart_image", return_value=b"png-bytes"), \
         mock.patch.object(m, "upload_price_chart",
                            return_value="https://images.microcms-assets.io/x/chart.png"):
        assert m.build_price_chart_for_article("7203", "テスト自動車") == \
            "https://images.microcms-assets.io/x/chart.png"


def test_build_and_publish_embeds_chart_image_in_body():
    """チャートが生成できた場合、本文HTMLの末尾に<img>タグとして埋め込む。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article",
                            return_value="https://images.microcms-assets.io/x/chart.png"), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert "https://images.microcms-assets.io/x/chart.png" in results[0]["body"]
    assert "<p>本文</p>" in results[0]["body"]


def test_build_and_publish_includes_pit_context_in_fact_sheet():
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    captured = {}

    def _fake_generate(fact_sheet):
        captured.update(fact_sheet)
        return {"title": "t", "body": "<p>本文</p>"}

    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "get_pit_ranking_snapshot", return_value={"close": 3000.0, "drop_prob": 25.0}), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body", side_effect=_fake_generate), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert captured["context_close"] == 3000.0
    assert captured["context_dp_level"] == "やや高"
    assert captured["ratio_change_pct"] == 8.5


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
    test_shares_outstanding_retries_then_succeeds()
    test_shares_outstanding_falls_back_to_implied_shares_outstanding()
    test_shares_outstanding_returns_none_after_exhausting_retries()
    test_generate_article_body_parses_plain_json()
    test_generate_article_body_strips_code_fence()
    test_generate_article_body_none_on_empty_title()
    test_classify_filer_returns_cached_master_row_without_calling_claude()
    test_classify_filer_asks_claude_and_persists_when_not_cached()
    test_classify_filer_falls_back_to_sonota_on_invalid_category()
    test_dp_level_label_thresholds()
    test_get_pit_ranking_snapshot_queries_as_of_disc_date()
    test_generate_article_body_includes_context_when_available()
    test_generate_article_body_omits_context_when_unavailable()
    test_generate_article_body_includes_ratio_increase_when_available()
    test_generate_article_body_describes_new_position_when_change_equals_ratio()
    test_generate_article_body_uses_buy_wording_by_default()
    test_generate_article_body_uses_sell_wording_when_direction_is_sell()
    test_generate_article_body_prompt_requests_english_translation()
    test_build_and_publish_includes_english_fields_when_generated()
    test_build_and_publish_omits_english_fields_when_not_generated()
    test_build_and_publish_includes_pit_context_in_fact_sheet()
    test_build_and_publish_includes_sell_and_tags_them()
    test_build_and_publish_skips_when_already_published()
    test_build_and_publish_skips_when_amount_unestimable()
    test_build_and_publish_stops_early_on_permission_error()
    test_publish_article_retries_as_array_on_type_mismatch()
    test_publish_article_drops_non_string_field_on_type_mismatch()
    test_publish_article_gives_up_when_same_field_fails_twice()
    test_update_article_retries_as_array_on_type_mismatch()
    test_update_article_returns_false_on_failure()
    test_wrap_text_lines_breaks_on_width()
    test_wrap_text_lines_respects_max_lines()
    test_search_pexels_photo_returns_none_without_api_key()
    test_search_pexels_photo_returns_bytes_and_photographer_on_success()
    test_search_pexels_photo_defaults_photographer_when_missing()
    test_search_pexels_photo_returns_none_when_no_results()
    test_search_pexels_photo_returns_none_on_exception()
    test_upload_eyecatch_returns_url_on_success()
    test_upload_eyecatch_returns_none_on_failure()
    test_build_eyecatch_for_article_none_without_pexels_key()
    test_build_eyecatch_for_article_none_when_generation_fails()
    test_build_eyecatch_for_article_returns_url_dict_on_success()
    test_build_and_publish_includes_eyecatch_when_available()
    test_build_and_publish_skips_eyecatch_on_dry_run()
    test_generate_article_body_includes_company_description_when_available()
    test_generate_article_body_always_requests_labelled_speculation()
    test_get_company_description_returns_cached_without_calling_claude()
    test_get_company_description_asks_claude_and_persists_when_not_cached()
    test_get_company_description_returns_empty_without_api_key_when_not_cached()
    test_upload_price_chart_returns_url_on_success()
    test_build_price_chart_for_article_none_when_generation_fails()
    test_build_price_chart_for_article_returns_url_on_success()
    test_build_and_publish_embeds_chart_image_in_body()
    test_get_featured_article_ids_preserves_pool_order_without_reordering()
    test_get_featured_article_ids_returns_empty_set_on_http_error()
    test_get_featured_article_ids_returns_empty_set_on_exception()
    test_get_filer_profile_returns_cached_without_calling_claude()
    test_get_filer_profile_asks_claude_and_persists_when_not_cached()
    test_get_filer_profile_returns_empty_without_api_key_when_not_cached()
    test_get_filer_profile_returns_empty_when_claude_returns_blank()
    print("全テスト成功 (63件)")

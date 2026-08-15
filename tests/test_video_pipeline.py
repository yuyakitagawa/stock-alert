"""自動動画投稿パイプライン（video/）のロジックのユニットテスト。
ネットワーク(microCMS / Claude / YouTube / TikTok)は全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_video_pipeline.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import video.build_script as bs
import video.tiktok_client as tk
import video.youtube_client as yt

YT_ENV = {
    "YOUTUBE_CLIENT_ID": "cid",
    "YOUTUBE_CLIENT_SECRET": "csec",
    "YOUTUBE_REFRESH_TOKEN": "rtok",
}
TK_ENV = {
    "TIKTOK_CLIENT_KEY": "ckey",
    "TIKTOK_CLIENT_SECRET": "csec",
    "TIKTOK_REFRESH_TOKEN": "rtok",
}

PROPS = {
    "stockName": "東陽テクニカ",
    "stockCode": "8151",
    "filerName": "シンプレクス・アセット・マネジメント",
    "dealTypeLabel": "国内アセットマネジメント",
    "direction": "sell",
    "dealAmountOku": 40.1,
    "holdingRatio": 8.77,
    "discDate": "2026-08-14",
    "hook": "東陽テクニカを40億円超で売却",
    "bullets": ["要点1", "要点2", "要点3"],
    "closing": "続きはクジラウォッチで",
    "articleId": "abc123",
}


# ---------------- 記事の選定 ----------------

def test_pick_article_takes_largest_among_featured():
    """注目枠に入っている記事の中で金額規模が最大のものを選ぶ。"""
    articles = [
        {"id": "a", "dealAmount": 10.0},
        {"id": "b", "dealAmount": 80.0},
        {"id": "c", "dealAmount": 50.0},
    ]
    picked = bs.pick_article(articles, featured_ids={"a", "c"})
    assert picked["id"] == "c"


def test_pick_article_returns_none_when_no_featured_overlap():
    """新着記事があっても注目枠に1件も入っていなければ動画を作らない。"""
    articles = [{"id": "a", "dealAmount": 999.0}]
    assert bs.pick_article(articles, featured_ids={"z"}) is None


# ---------------- props の組み立て ----------------

def test_deal_type_label_unwraps_select_array():
    """microCMSのセレクト型は配列で返るため先頭要素をラベルに使う。"""
    assert bs.deal_type_label({"dealType": ["国内アセットマネジメント"]}) == "国内アセットマネジメント"


def test_deal_type_label_falls_back_when_missing():
    assert bs.deal_type_label({}) == "大量保有報告書"


def test_build_props_marks_sell_from_tags():
    article = {"stockName": "A社", "stockCode": "1234", "tags": "EDINET,自動生成,売り",
               "dealType": ["個人"], "dealDate": "2026-08-14T00:00:00.000Z", "dealAmount": 12.3,
               "body": "<p>保有比率を6.10%に低下</p>"}
    props = bs.build_props(article, {"hook": "h", "bullets": ["1", "2", "3"], "closing": "c"})
    assert props["direction"] == "sell"
    assert props["discDate"] == "2026-08-14"
    assert props["dealAmountOku"] == 12.3


def test_build_props_defaults_missing_filer_name_to_empty():
    """古い記事はfilerNameが未設定（microCMSは空フィールドを返さない）。動画側で行ごと省く。"""
    article = {"stockName": "A社", "stockCode": "1234", "tags": "", "dealType": ["個人"],
               "dealDate": "2026-08-14T00:00:00.000Z", "dealAmount": 1.0, "body": "<p>5.00%</p>"}
    props = bs.build_props(article, {"hook": "h", "bullets": ["1", "2", "3"], "closing": "c"})
    assert props["filerName"] == ""


def test_extract_holding_ratio_takes_last_percentage_in_body():
    """本文は「前回◯%→今回◯%」の順で書かれるため、末尾側が今回の保有比率になる。"""
    article = {"body": "<p>従来の5.02%から積み増し、今回7.35%となりました。</p>"}
    assert bs.extract_holding_ratio(article) == 7.35


def test_extract_holding_ratio_returns_zero_when_absent():
    assert bs.extract_holding_ratio({"body": "<p>比率の記載なし</p>"}) == 0.0


# ---------------- 台本の生成 ----------------

def _claude_response(text: str):
    resp = mock.Mock()
    resp.content = [mock.Mock(text=text)]
    client = mock.Mock()
    client.messages.create.return_value = resp
    return client


def test_generate_script_parses_json():
    payload = '{"hook": "40億円の売却", "bullets": ["a", "b", "c"], "closing": "続きはこちら"}'
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=_claude_response(payload)):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert script["hook"] == "40億円の売却"
    assert script["bullets"] == ["a", "b", "c"]


def test_generate_script_retries_once_when_bullets_too_long():
    """字数超過は動画のレイアウトを壊すため、一度だけ作り直す。"""
    long_bullet = "あ" * (bs.BULLET_MAX_CHARS + 10)
    first = f'{{"hook": "短い", "bullets": ["{long_bullet}", "b", "c"], "closing": "c"}}'
    second = '{"hook": "短い", "bullets": ["a", "b", "c"], "closing": "c"}'
    client = mock.Mock()
    client.messages.create.side_effect = [
        mock.Mock(content=[mock.Mock(text=first)]),
        mock.Mock(content=[mock.Mock(text=second)]),
    ]
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert client.messages.create.call_count == 2
    assert script["bullets"] == ["a", "b", "c"]


def test_generate_script_trims_when_retry_still_too_long():
    """作り直しても長い場合は末尾を詰めてレイアウト崩れを防ぐ。"""
    long_bullet = "あ" * (bs.BULLET_MAX_CHARS + 10)
    payload = f'{{"hook": "短い", "bullets": ["{long_bullet}", "b", "c"], "closing": "c"}}'
    client = mock.Mock()
    client.messages.create.return_value = mock.Mock(content=[mock.Mock(text=payload)])
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert len(script["bullets"][0]) == bs.BULLET_MAX_CHARS
    assert script["bullets"][0].endswith("…")


def test_generate_script_returns_none_without_api_key():
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", ""):
        assert bs.generate_script({"title": "t", "body": "b", "tags": ""}) is None


# ---------------- YouTube ----------------

def test_youtube_title_includes_stock_and_amount():
    title = yt.build_title(PROPS)
    assert "東陽テクニカ" in title
    assert "40.1億円" in title
    assert "売却" in title
    assert "#Shorts" in title


def test_youtube_title_truncated_but_keeps_shorts_tag():
    props = {**PROPS, "stockName": "あ" * 60, "filerName": "い" * 60}
    title = yt.build_title(props)
    assert len(title) <= yt.TITLE_MAX_CHARS
    assert title.endswith("#Shorts")


def test_youtube_description_has_article_url_with_utm():
    desc = yt.build_description(PROPS)
    assert f"{yt.SITE_URL}/articles/abc123" in desc
    assert "utm_source=youtube" in desc
    assert "投資勧誘・投資助言ではありません" in desc


def test_youtube_description_falls_back_to_site_root_without_article_id():
    desc = yt.build_description({**PROPS, "articleId": None})
    assert "/articles/" not in desc


def test_youtube_upload_skipped_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert yt.upload("/tmp/none.mp4", PROPS) is None


# ---------------- TikTok ----------------

def test_tiktok_caption_includes_stock_and_hashtags():
    caption = tk.build_caption(PROPS)
    assert "東陽テクニカ" in caption
    assert "#EDINET" in caption


def test_tiktok_caption_truncates_long_head():
    caption = tk.build_caption({**PROPS, "stockName": "あ" * 200})
    head = caption.split("\n")[0]
    assert len(head) <= tk.CAPTION_MAX_CHARS
    assert head.endswith("…")


def test_tiktok_upload_skipped_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert tk.upload("/tmp/none.mp4", PROPS) is None


def test_tiktok_uses_inbox_endpoint_by_default(tmp_path):
    """アプリ審査前は直接公開できないため、既定では下書き(inbox)へ送る。"""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 100)
    posted = {}

    def fake_post(url, **kwargs):
        posted["url"] = url
        return mock.Mock(ok=True, json=lambda: {"data": {"upload_url": "https://up", "publish_id": "p1"}})

    with mock.patch.dict(os.environ, TK_ENV, clear=True), \
         mock.patch.object(tk, "_access_token", return_value="tok"), \
         mock.patch.object(tk, "_upload_bytes", return_value=True), \
         mock.patch("requests.post", side_effect=fake_post):
        publish_id = tk.upload(str(video), PROPS)

    assert publish_id == "p1"
    assert posted["url"] == tk.INBOX_INIT_URL


def test_tiktok_direct_post_falls_back_to_self_only_when_public_not_allowed(tmp_path):
    """審査未通過アカウントでは一般公開が選べないため SELF_ONLY へ落とす。"""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 100)
    sent = {}

    def fake_post(url, **kwargs):
        if url == tk.DIRECT_INIT_URL:
            sent["payload"] = kwargs.get("json")
            return mock.Mock(ok=True, json=lambda: {"data": {"upload_url": "https://up", "publish_id": "p2"}})
        return mock.Mock(ok=True, json=lambda: {"data": {"privacy_level_options": ["SELF_ONLY"]}})

    with mock.patch.dict(os.environ, {**TK_ENV, "TIKTOK_DIRECT_POST": "1"}, clear=True), \
         mock.patch.object(tk, "_access_token", return_value="tok"), \
         mock.patch.object(tk, "_upload_bytes", return_value=True), \
         mock.patch("requests.post", side_effect=fake_post):
        publish_id = tk.upload(str(video), PROPS)

    assert publish_id == "p2"
    assert sent["payload"]["post_info"]["privacy_level"] == "SELF_ONLY"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

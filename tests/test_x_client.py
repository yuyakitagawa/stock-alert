"""X(Twitter)自動投稿（web/x_client）のロジックのユニットテスト。
ネットワーク(X API)は全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_x_client.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_client as m

X_ENV = {
    "X_API_KEY": "key",
    "X_API_KEY_SECRET": "key_secret",
    "X_ACCESS_TOKEN": "token",
    "X_ACCESS_TOKEN_SECRET": "token_secret",
}


def test_build_tweet_text_includes_title_amount_and_url():
    text = m.build_tweet_text("ソフトバンクG、大量保有報告書を提出", 120.5, is_sell=False, article_id="abc123")
    assert "ソフトバンクG、大量保有報告書を提出" in text
    assert "推定取得金額: 120.5億円" in text
    assert f"{m.SITE_URL}/articles/abc123" in text
    assert "#EDINET" in text


def test_build_tweet_text_sell_direction_label():
    text = m.build_tweet_text("〇〇HD、株式を一部売却", 30.0, is_sell=True, article_id="xyz")
    assert "推定売却金額: 30.0億円" in text


def test_build_tweet_text_appends_sanitized_stock_hashtag():
    text = m.build_tweet_text("タイトル", 10.0, is_sell=False, article_id="id1",
                              stock_name="シンプレクス・アセット（8151）")
    # 「・」「（）」はハッシュタグを切ってしまうため除去される
    assert text.endswith("#EDINET #大量保有報告書 #シンプレクスアセット8151")


def test_build_tweet_text_omits_stock_hashtag_when_name_empty():
    text = m.build_tweet_text("タイトル", 10.0, is_sell=False, article_id="id1")
    assert text.endswith("#EDINET #大量保有報告書")


def test_build_tweet_text_truncates_long_title():
    long_title = "あ" * 200
    text = m.build_tweet_text(long_title, 10.0, is_sell=False, article_id="id1")
    body_line = text.split("\n")[0]
    assert len(body_line) <= m.TWEET_BODY_MAX_CHARS
    assert body_line.endswith("…")


def test_auth_returns_none_when_env_vars_missing():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m._auth() is None


def test_auth_returns_oauth1_when_all_env_vars_set():
    with mock.patch.dict(os.environ, X_ENV, clear=True):
        auth = m._auth()
        assert auth is not None


def test_post_tweet_skips_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(m.requests, "post") as mock_post:
        assert m.post_tweet("hello") is False
        mock_post.assert_not_called()


def test_post_tweet_returns_true_on_success():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "post", return_value=mock.Mock(status_code=201)):
        assert m.post_tweet("hello") is True


def test_post_tweet_returns_false_on_http_error():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(
             m.requests, "post",
             return_value=mock.Mock(status_code=403, text="Forbidden"),
         ):
        assert m.post_tweet("hello") is False


def test_upload_media_returns_media_id_on_success():
    resp = mock.Mock(ok=True, json=lambda: {"media_id_string": "999"})
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "post", return_value=resp) as mock_post:
        assert m.upload_media(b"png-bytes") == "999"
    assert mock_post.call_args.kwargs["files"]["media"][0] == "chart.png"


def test_upload_media_returns_none_on_http_error():
    resp = mock.Mock(ok=False, status_code=400, text="Bad Request")
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "post", return_value=resp):
        assert m.upload_media(b"png-bytes") is None


def test_upload_media_returns_none_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(m.requests, "post") as mock_post:
        assert m.upload_media(b"png-bytes") is None
        mock_post.assert_not_called()


def test_post_tweet_includes_media_ids_when_media_id_given():
    captured = {}
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(
             m.requests, "post",
             side_effect=lambda url, auth, json, timeout: captured.update(json) or mock.Mock(status_code=201),
         ):
        assert m.post_tweet("hello", media_id="999") is True
    assert captured["media"] == {"media_ids": ["999"]}


def test_post_top_articles_skips_without_auth():
    published = [{"id": "1", "title": "t", "dealAmount": 100.0, "tags": "",
                  "stockCode": "1234", "stockName": "テスト"}]
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m.post_top_articles(published, featured_ids={"1"}) == 0


def test_post_top_articles_picks_largest_deal_amount_first_among_featured():
    published = [
        {"id": "1", "title": "小さい取引", "dealAmount": 10.0, "tags": "",
         "stockCode": "1111", "stockName": "小さい会社"},
        {"id": "2", "title": "大きい取引", "dealAmount": 500.0, "tags": "",
         "stockCode": "2222", "stockName": "大きい会社"},
        {"id": None, "title": "dry-runなのでID無し", "dealAmount": 999.0, "tags": "",
         "stockCode": "3333", "stockName": "対象外"},
    ]
    posted_texts = []
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=None), \
         mock.patch.object(m, "post_tweet", side_effect=lambda text, media_id=None: posted_texts.append(text) or True):
        posted = m.post_top_articles(published, featured_ids={"1", "2"}, top_n=1)

    assert posted == 1
    assert "大きい取引" in posted_texts[0]


def test_post_top_articles_excludes_articles_not_in_featured_ids():
    """当日一番金額が大きくても、ホームページの「注目」に入っていなければ投稿しない
    （8/9運用で9.6億円の記事が投稿された一方、8/7の1406億円の記事が「注目」に
    居座り続けていた実例を踏まえた仕様）。"""
    published = [
        {"id": "1", "title": "今日の中では一番大きいが注目には入らない", "dealAmount": 9.6, "tags": "",
         "stockCode": "1111", "stockName": "小さい会社"},
    ]
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=None), \
         mock.patch.object(m, "post_tweet") as mock_tweet:
        posted = m.post_top_articles(published, featured_ids={"other-id"})

    assert posted == 0
    mock_tweet.assert_not_called()


def test_post_top_articles_attaches_chart_when_generated():
    """チャート画像が生成できた場合、アップロードしてmedia_idを付けて投稿する。"""
    published = [{"id": "1", "title": "大きい取引", "dealAmount": 500.0, "tags": "",
                  "stockCode": "2222", "stockName": "大きい会社"}]
    calls = []
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=b"png-bytes"), \
         mock.patch.object(m, "upload_media", return_value="777"), \
         mock.patch.object(m, "post_tweet",
                            side_effect=lambda text, media_id=None: calls.append(media_id) or True):
        posted = m.post_top_articles(published, featured_ids={"1"}, top_n=1)

    assert posted == 1
    assert calls == ["777"]


def test_post_top_articles_posts_without_chart_when_generation_fails():
    """チャート生成に失敗してもテキストのみで投稿を続行する。"""
    published = [{"id": "1", "title": "大きい取引", "dealAmount": 500.0, "tags": "",
                  "stockCode": "2222", "stockName": "大きい会社"}]
    calls = []
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=None), \
         mock.patch.object(m, "post_tweet",
                            side_effect=lambda text, media_id=None: calls.append(media_id) or True):
        posted = m.post_top_articles(published, featured_ids={"1"}, top_n=1)

    assert posted == 1
    assert calls == [None]


def test_build_daily_summary_text_includes_count_total_and_extremes():
    articles = [
        {"stockName": "大型買い", "dealAmount": 40.0, "tags": "EDINET,自動生成", "filerName": "買いファンド"},
        {"stockName": "大型売り", "dealAmount": 30.0, "tags": "EDINET,自動生成,売り", "filerName": "売りファンド"},
        {"stockName": "小型買い", "dealAmount": 5.5, "tags": "EDINET,自動生成", "filerName": ""},
    ]
    text = m.build_daily_summary_text(articles, "2026-08-15")
    assert "🐋 本日のクジラ｜8/15の大量保有報告書" in text
    assert "3件・合計75.5億円" in text
    assert "大型買い +40.0億円（買いファンド）" in text
    assert "大型売り -30.0億円（売りファンド）" in text
    assert f"{m.SITE_URL}/date/2026-08-15" in text
    assert "utm_campaign=daily_summary" in text
    # 取り上げた最大買い増し・最大売却の銘柄タグが付く
    assert text.endswith("#EDINET #大量保有報告書 #大型買い #大型売り")


def test_build_video_tweet_text_includes_shorts_url_and_stock_tag():
    props = {"stockName": "東陽テクニカ", "filerName": "テストファンド",
             "direction": "sell", "dealAmountOku": 40.1}
    text = m.build_video_tweet_text(props, "abc123XYZ00")
    assert "🎬 1分解説｜東陽テクニカをテストファンドが推定40.1億円売却" in text
    assert "https://youtube.com/shorts/abc123XYZ00" in text
    assert text.endswith("#Shorts #大量保有報告書 #東陽テクニカ")


def test_build_video_tweet_text_uses_generic_subject_without_filer():
    props = {"stockName": "テスト社", "filerName": "", "direction": "buy", "dealAmountOku": 5.0}
    text = m.build_video_tweet_text(props, "vid")
    assert "テスト社を大口投資家が推定5.0億円取得" in text


def test_post_video_tweet_skips_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.post_video_tweet({"stockName": "A"}, "vid") is False
        tweet_mock.assert_not_called()


def test_post_video_tweet_posts_when_authed():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "post_tweet", return_value=True) as tweet_mock:
        assert m.post_video_tweet(
            {"stockName": "A", "filerName": "F", "direction": "buy", "dealAmountOku": 1.0}, "vid"
        ) is True
    assert "youtube.com/shorts/vid" in tweet_mock.call_args.args[0]


def test_build_daily_summary_text_none_when_no_articles():
    assert m.build_daily_summary_text([], "2026-08-15") is None


def test_build_daily_summary_text_uses_total_count_when_truncated():
    articles = [{"stockName": "A", "dealAmount": 1.0, "tags": "", "filerName": ""}]
    text = m.build_daily_summary_text(articles, "2026-08-15", total_count=120)
    assert "120件" in text


def test_post_daily_summary_only_fires_at_final_run_hour():
    """19時JST(10時UTC)の最終便以外の時間帯では投稿しない（1日1回の重複ガード）。"""
    from datetime import datetime, timezone

    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date") as fetch_mock:
        assert m.post_daily_summary(now_utc=datetime(2026, 8, 15, 5, 0, tzinfo=timezone.utc)) is False
        fetch_mock.assert_not_called()


def test_post_daily_summary_posts_at_final_run_hour():
    from datetime import datetime, timezone

    articles = [{"stockName": "A", "dealAmount": 10.0, "tags": "", "filerName": ""}]
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date", return_value=(articles, 1)), \
         mock.patch.object(m, "post_tweet", return_value=True) as tweet_mock:
        # 10時UTC＝19時JSTなのでJST日付は当日
        ok = m.post_daily_summary(now_utc=datetime(2026, 8, 15, 10, 30, tzinfo=timezone.utc))
    assert ok is True
    posted_text = tweet_mock.call_args.args[0]
    assert "8/15" in posted_text


def test_post_daily_summary_skips_when_no_articles():
    from datetime import datetime, timezone

    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date", return_value=([], 0)), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        ok = m.post_daily_summary(now_utc=datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc))
    assert ok is False
    tweet_mock.assert_not_called()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

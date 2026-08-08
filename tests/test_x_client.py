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


def test_post_top_articles_skips_without_auth():
    published = [{"id": "1", "title": "t", "dealAmount": 100.0, "tags": ""}]
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m.post_top_articles(published) == 0


def test_post_top_articles_picks_largest_deal_amount_first():
    published = [
        {"id": "1", "title": "小さい取引", "dealAmount": 10.0, "tags": ""},
        {"id": "2", "title": "大きい取引", "dealAmount": 500.0, "tags": ""},
        {"id": None, "title": "dry-runなのでID無し", "dealAmount": 999.0, "tags": ""},
    ]
    posted_texts = []
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "post_tweet", side_effect=lambda text: posted_texts.append(text) or True):
        posted = m.post_top_articles(published, top_n=1)

    assert posted == 1
    assert "大きい取引" in posted_texts[0]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

"""X(Twitter)自動投稿（web/x_client, web/x_insight, web/x_followup）のロジックのユニットテスト。
ネットワーク(X API)とSupabaseは全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_x_client.py
"""
import os
import sys
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_client as m

X_ENV = {
    "X_API_KEY": "key",
    "X_API_KEY_SECRET": "key_secret",
    "X_ACCESS_TOKEN": "token",
    "X_ACCESS_TOKEN_SECRET": "token_secret",
}

BUY_ARTICLE = {
    "id": "a1", "title": "記事タイトル", "stockName": "東陽テクニカ", "stockCode": "8151",
    "filerName": "三井住友トラスト・アセットマネジメント株式会社", "dealAmount": 120.5,
    "tags": "EDINET,自動生成", "ratioChangePct": 1.81, "holdingRatio": 5.02,
    "dealDate": "2026-08-19T00:00:00.000Z",
}
CORRECTION_ARTICLE = {
    "id": "c1", "title": "訂正記事", "stockName": "太陽誘電", "stockCode": "6976",
    "filerName": "テストファンド", "dealAmount": 0.0, "tags": "EDINET,自動生成,訂正,売り",
    "ratioChangePct": -10.81, "holdingRatio": 4.41,
}


def _ok_response(tweet_id="111"):
    return mock.Mock(status_code=201, json=lambda: {"data": {"id": tweet_id}})


# ------------------------------------------------------------------ 本文の組み立て

def test_build_tweet_text_leads_with_who_did_what():
    """1行目は記事タイトルの流用ではなく「誰が・どの銘柄を・どうした」のフックにする。"""
    text = m.build_tweet_text(BUY_ARTICLE)
    first = text.splitlines()[0]
    assert first.startswith("🟢 ")
    assert "東陽テクニカ(8151)" in first
    assert "買い増し" in first
    assert "記事タイトル" not in text


def test_build_tweet_text_second_line_has_amount_and_ratio_change():
    text = m.build_tweet_text(BUY_ARTICLE)
    assert text.splitlines()[1] == "約120億円・保有比率 3.21%→5.02%"


def test_build_tweet_text_labels_new_position():
    """直前の保有割合が0なら「買い増し」ではなく「新規取得」。"""
    text = m.build_tweet_text({**BUY_ARTICLE, "ratioChangePct": 5.02, "holdingRatio": 5.02})
    assert "を新規取得" in text.splitlines()[0]


def test_build_tweet_text_sell_uses_red_mark():
    text = m.build_tweet_text({**BUY_ARTICLE, "tags": "EDINET,自動生成,売り",
                               "ratioChangePct": -2.0, "holdingRatio": 3.02})
    assert text.splitlines()[0].startswith("🔴 ")
    assert "を売却" in text.splitlines()[0]


def test_build_tweet_text_uses_two_searchable_hashtags_only():
    """#EDINETや`#社名`は検索母数が無いため付けず、銘柄はコード付きで本文に素で書く。"""
    text = m.build_tweet_text(BUY_ARTICLE)
    assert text.endswith("#日本株 #大量保有報告書")
    assert "#EDINET" not in text
    assert "#東陽テクニカ" not in text


def test_build_tweet_text_includes_insight_line_when_it_fits():
    text = m.build_tweet_text(BUY_ARTICLE, insight="この提出者の開示は過去1434件、東陽テクニカでは2回目")
    assert "この提出者の開示は過去1434件" in text


def test_build_tweet_text_drops_insight_when_too_long():
    text = m.build_tweet_text(BUY_ARTICLE, insight="あ" * 200)
    assert "あ" * 200 not in text
    assert m.weighted_len(text) <= 270


def test_build_tweet_text_puts_url_in_body_only_when_given():
    """既定ではURLは自己リプライに置くのでlink_urlは空。A/Bのlink_in_body側でのみ本文に入る。"""
    assert m.SITE_URL not in m.build_tweet_text(BUY_ARTICLE)
    text = m.build_tweet_text(BUY_ARTICLE, link_url=f"{m.SITE_URL}/articles/a1")
    assert f"{m.SITE_URL}/articles/a1" in text


def test_build_tweet_text_correction_shows_ratio_not_amount():
    text = m.build_tweet_text(CORRECTION_ARTICLE)
    assert text.splitlines()[0].startswith("⚠️ 訂正｜")
    assert "保有比率 15.22%→4.41%（-10.81pt）" in text
    assert "億円" not in text


def test_format_oku_rounds_for_readability():
    assert m._format_oku(120.5) == "約120億円"
    assert m._format_oku(8.44) == "約8.4億円"
    assert m._format_oku(12000) == "約1.2兆円"
    assert m._format_oku(0) == ""


def test_build_article_alt_carries_the_numbers_in_the_card():
    alt = m.build_article_alt(BUY_ARTICLE)
    assert "東陽テクニカ（8151）" in alt
    assert "3.21%→5.02%" in alt
    assert "EDINET" in alt


# ------------------------------------------------------------------ 認証・投稿API

def test_auth_returns_none_when_env_vars_missing():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m._auth() is None


def test_auth_returns_oauth1_when_all_env_vars_set():
    with mock.patch.dict(os.environ, X_ENV, clear=True):
        assert m._auth() is not None


def test_post_tweet_skips_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(m.requests, "post") as mock_post:
        assert m.post_tweet("hello") is None
        mock_post.assert_not_called()


def test_post_tweet_returns_tweet_id_and_logs_it():
    """boolではなくtweet_idを返し、x_postsに記録する（どの投稿が伸びたかを追うため）。"""
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "post", return_value=_ok_response("12345")), \
         mock.patch.object(m, "log_post") as log_mock:
        assert m.post_tweet("hello", kind="article") == "12345"
    assert log_mock.call_args.args[0] == "12345"
    assert log_mock.call_args.args[1] == "article"


def test_post_tweet_returns_none_on_http_error():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "verify_auth", return_value={"access_level": "read", "reason": ""}), \
         mock.patch.object(m.requests, "post",
                           return_value=mock.Mock(status_code=403, text="Forbidden")):
        assert m.post_tweet("hello") is None


def test_card_stock_line_keeps_stock_code():
    """社名が長くても証券コードは残す（銘柄検索の手掛かりを消さない）。フォント無しの環境では飛ばす。"""
    from PIL import Image, ImageDraw

    import web.x_card_image as card

    if card._font_path() is None:
        return
    draw = ImageDraw.Draw(Image.new("RGB", (card.CARD_W, card.CARD_H)))
    max_w = card.CARD_W - card.PAD * 2
    for name in ("ニチレイ", "セブン&アイ・ホールディングス", "あ" * 40):
        label, font = card._stock_line(draw, name, "3382", max_w)
        assert label.endswith("（3382）"), label
        assert draw.textlength(label, font=font) <= max_w, label


def test_build_article_media_attaches_card_only():
    """Xは複数画像を左右に並べて両方切り落とすため、添付は数字カード1枚だけにする。"""
    chart = mock.Mock(side_effect=AssertionError("カードが作れた時にチャートを作ってはいけない"))
    with mock.patch("web.x_card_image.build_deal_card", return_value=b"card"), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", chart), \
         mock.patch.object(m, "upload_media", return_value="m1"):
        assert m.build_article_media(BUY_ARTICLE) == ["m1"]


def test_build_article_media_falls_back_to_chart():
    """カードが作れない（フォント欠如等）ときだけチャートを代替に使う。これも1枚。"""
    with mock.patch("web.x_card_image.build_deal_card", return_value=None), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=b"chart"), \
         mock.patch.object(m, "upload_media", return_value="m2"):
        assert m.build_article_media(BUY_ARTICLE) == ["m2"]


def test_build_article_media_without_any_image():
    with mock.patch("web.x_card_image.build_deal_card", return_value=None), \
         mock.patch("web.publish_blog_articles.generate_price_chart_image", return_value=None):
        assert m.build_article_media(BUY_ARTICLE) == []


def test_post_tweet_includes_media_ids_and_reply_target():
    captured = {}
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "log_post"), \
         mock.patch.object(
             m.requests, "post",
             side_effect=lambda url, auth, json, timeout: captured.update(json) or _ok_response()):
        m.post_tweet("hello", media_ids=["999", "888"], reply_to="777")
    assert captured["media"] == {"media_ids": ["999", "888"]}
    assert captured["reply"] == {"in_reply_to_tweet_id": "777"}


def test_publish_puts_link_in_a_self_reply_when_ab_flag_on():
    """X_LINK_IN_REPLY=1 のときだけURLを2投稿目に分ける（既定は本文）。"""
    calls = []
    with mock.patch.dict(os.environ, {**X_ENV, "X_LINK_IN_REPLY": "1"}, clear=True), \
         mock.patch.object(m, "post_tweet",
                           side_effect=lambda text, **kw: calls.append((text, kw)) or "100"):
        assert m.publish("本文", link_line="https://example.com", kind="article") == "100"
    assert len(calls) == 2
    assert "https://example.com" not in calls[0][0]
    assert "https://example.com" in calls[1][0]
    assert calls[1][1]["reply_to"] == "100"


def test_publish_keeps_link_in_body_by_default():
    """親投稿にURLが無いと、スレッドを開かない読者にリンクが届かないため既定は本文。"""
    calls = []
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "post_tweet",
                           side_effect=lambda text, **kw: calls.append(text) or "100"):
        m.publish("本文", link_line="https://example.com", kind="article")
    assert len(calls) == 1
    assert calls[0].endswith("https://example.com")


def test_upload_media_sets_alt_text():
    """画像内の数字が読み上げ環境に伝わらないため、altを必ず設定する。"""
    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return mock.Mock(ok=True, json=lambda: {"media_id_string": "999"})

    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "post", side_effect=fake_post):
        assert m.upload_media(b"png-bytes", alt_text="保有比率 3.21%→5.02%") == "999"
    assert posts[0][0] == m.MEDIA_UPLOAD_URL
    assert posts[1][0] == m.MEDIA_METADATA_URL
    assert posts[1][1]["json"]["alt_text"]["text"] == "保有比率 3.21%→5.02%"


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


# ------------------------------------------------------------------ 投稿時間帯

def test_within_posting_hours_rejects_late_night_jst():
    # 20時UTC = 翌5時JST
    assert m.within_posting_hours(datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc)) is False
    # 3時UTC = 12時JST
    assert m.within_posting_hours(datetime(2026, 8, 19, 3, 0, tzinfo=timezone.utc)) is True


def test_post_top_articles_skips_outside_posting_hours():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "publish") as publish_mock:
        posted = m.post_top_articles([BUY_ARTICLE], featured_ids={"a1"},
                                     now_utc=datetime(2026, 8, 19, 20, 0, tzinfo=timezone.utc))
    assert posted == 0
    publish_mock.assert_not_called()


# ------------------------------------------------------------------ 記事投稿の選別

def _patch_article_deps(publish_side_effect):
    """post_top_articlesの外部依存（画像生成・Supabase・投稿）をまとめてモックする。"""
    return [
        mock.patch.object(m, "build_article_media", return_value=[]),
        mock.patch.object(m, "find_prior_tweet", return_value=None),
        mock.patch("web.x_insight.fetch_filer_context", return_value={}),
        mock.patch.object(m, "publish", side_effect=publish_side_effect),
    ]


def _run_post_top(published, featured_ids, top_n=m.ARTICLES_PER_RUN, publish_side_effect=None):
    calls = []
    side_effect = publish_side_effect or (lambda body, **kw: calls.append((body, kw)) or "1")
    patches = _patch_article_deps(side_effect)
    with mock.patch.dict(os.environ, X_ENV, clear=True):
        for p in patches:
            p.start()
        try:
            posted = m.post_top_articles(published, featured_ids, top_n=top_n,
                                         now_utc=datetime(2026, 8, 19, 3, 0, tzinfo=timezone.utc))
        finally:
            for p in patches:
                p.stop()
    return posted, calls


def test_post_top_articles_skips_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m.post_top_articles([BUY_ARTICLE], featured_ids={"a1"}) == 0


def test_post_top_articles_posts_one_article_per_run():
    """以前は1回の実行で3件連投していた。毎時バッチなので1件ずつで十分分散する。"""
    published = [
        {**BUY_ARTICLE, "id": "1", "stockName": "小さい会社", "dealAmount": 10.0},
        {**BUY_ARTICLE, "id": "2", "stockName": "大きい会社", "dealAmount": 500.0},
    ]
    posted, calls = _run_post_top(published, {"1", "2"})
    assert posted == 1
    assert "大きい会社" in calls[0][0]


def test_post_top_articles_excludes_articles_not_in_featured_ids():
    """当日一番金額が大きくても、ホームページの「注目」に入っていなければ投稿しない。"""
    posted, calls = _run_post_top([{**BUY_ARTICLE, "id": "1"}], {"other-id"})
    assert posted == 0
    assert calls == []


def test_post_top_articles_posts_every_correction_without_cap():
    """訂正報告書は既報の前提を覆すため件数制限を設けない（施策7）。"""
    published = [
        {**CORRECTION_ARTICLE, "id": "c1", "ratioChangePct": -10.81},
        {**CORRECTION_ARTICLE, "id": "c2", "ratioChangePct": -3.0, "stockCode": "1111"},
        {**BUY_ARTICLE, "id": "f1"},
    ]
    posted, calls = _run_post_top(published, {"f1"})
    assert posted == 3
    assert calls[0][1]["kind"] == "correction"
    assert calls[1][1]["kind"] == "correction"
    assert calls[2][1]["kind"] == "article"


def test_post_top_articles_replies_correction_to_the_prior_post_on_the_same_stock():
    calls = []
    patches = [
        mock.patch.object(m, "build_article_media", return_value=[]),
        mock.patch.object(m, "find_prior_tweet", return_value="900"),
        mock.patch("web.x_insight.fetch_filer_context", return_value={}),
        mock.patch.object(m, "publish", side_effect=lambda body, **kw: calls.append(kw) or "1"),
    ]
    with mock.patch.dict(os.environ, X_ENV, clear=True):
        for p in patches:
            p.start()
        try:
            m.post_top_articles([CORRECTION_ARTICLE], featured_ids=set(),
                                now_utc=datetime(2026, 8, 19, 3, 0, tzinfo=timezone.utc))
        finally:
            for p in patches:
                p.stop()
    assert calls[0]["reply_to"] == "900"


# ------------------------------------------------------------------ 日次サマリー

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
    assert text.endswith("#日本株 #大量保有報告書")
    assert m.SITE_URL not in text  # URLは自己リプライ側


def test_build_daily_summary_text_none_when_no_articles():
    assert m.build_daily_summary_text([], "2026-08-15") is None


def test_build_daily_summary_text_uses_total_count_when_truncated():
    articles = [{"stockName": "A", "dealAmount": 1.0, "tags": "", "filerName": ""}]
    text = m.build_daily_summary_text(articles, "2026-08-15", total_count=120)
    assert "120件" in text


def test_post_daily_summary_only_fires_at_the_21jst_run():
    """21時JST(12時UTC)の便以外では投稿しない（1日1回の重複ガード）。"""
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date") as fetch_mock:
        assert m.post_daily_summary(now_utc=datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)) is False
        fetch_mock.assert_not_called()


def test_post_daily_summary_posts_at_the_final_run():
    articles = [{"stockName": "A", "stockCode": "1234", "dealAmount": 10.0, "tags": "", "filerName": ""}]
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date", return_value=(articles, 1)), \
         mock.patch.object(m, "build_daily_summary_media", return_value=["m1"]), \
         mock.patch.object(m, "publish", return_value="1") as publish_mock:
        ok = m.post_daily_summary(now_utc=datetime(2026, 8, 15, 12, 30, tzinfo=timezone.utc))
    assert ok is True
    assert "8/15" in publish_mock.call_args.args[0]
    assert publish_mock.call_args.kwargs["media_ids"] == ["m1"]


def test_post_daily_summary_skips_when_no_articles():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "fetch_articles_by_deal_date", return_value=([], 0)), \
         mock.patch.object(m, "publish") as publish_mock:
        assert m.post_daily_summary(now_utc=datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)) is False
    publish_mock.assert_not_called()


# ------------------------------------------------------------------ 動画クロス投稿

def test_build_video_tweet_text_includes_shorts_url_and_tags():
    props = {"stockName": "東陽テクニカ", "filerName": "テストファンド",
             "direction": "sell", "dealAmountOku": 40.1}
    text = m.build_video_tweet_text(props, "abc123XYZ00")
    assert "🎬 1分解説｜テストファンドが東陽テクニカを約40億円売却" in text
    assert "https://youtube.com/shorts/abc123XYZ00" in text
    assert text.endswith("#Shorts #日本株 #大量保有報告書")


def test_build_video_tweet_text_uses_generic_subject_without_filer():
    props = {"stockName": "テスト社", "filerName": "", "direction": "buy", "dealAmountOku": 5.0}
    assert "大口投資家がテスト社を約5.0億円取得" in m.build_video_tweet_text(props, "vid")


def test_post_video_tweet_skips_without_auth():
    with mock.patch.dict(os.environ, {}, clear=True), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.post_video_tweet({"stockName": "A"}, "vid") is False
        tweet_mock.assert_not_called()


def test_post_video_tweet_posts_when_authed():
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m, "post_tweet", return_value="1") as tweet_mock:
        assert m.post_video_tweet(
            {"stockName": "A", "filerName": "F", "direction": "buy", "dealAmountOku": 1.0}, "vid"
        ) is True
    assert "youtube.com/shorts/vid" in tweet_mock.call_args.args[0]


# ------------------------------------------------------------------ トークン権限の確認

def _verify_response(status, headers, body=None):
    resp = mock.MagicMock()
    resp.status_code = status
    resp.headers = headers
    resp.json.return_value = body or {}
    resp.text = ""
    return resp


def test_verify_auth_reports_read_only_token():
    """App権限をRead and Writeにしても、変更前に発行したトークンは読み取り専用のまま
    （2026-08-18の日次便が403 You are not permitted to perform this action で失敗した原因）。"""
    resp = _verify_response(200, {"x-access-level": "read"}, {"screen_name": "kujira"})
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "get", return_value=resp):
        result = m.verify_auth()
    assert result["ok"] is False
    assert result["access_level"] == "read"
    assert "再発行" in result["reason"]


def test_verify_auth_ok_for_read_write_token():
    resp = _verify_response(200, {"x-access-level": "read-write"}, {"screen_name": "kujira"})
    with mock.patch.dict(os.environ, X_ENV, clear=True), \
         mock.patch.object(m.requests, "get", return_value=resp):
        result = m.verify_auth()
    assert result["ok"] is True
    assert result["screen_name"] == "kujira"


def test_verify_auth_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert m.verify_auth()["ok"] is False


# ------------------------------------------------------------------ 解釈行（web/x_insight）

def test_build_insight_line_marks_first_appearance():
    import web.x_insight as ins

    assert ins.build_insight_line("東陽テクニカ", {"filer_disclosures": 1, "stock_disclosures": 1}) \
        == "この提出者がEDINETに登場するのは初"


def test_build_insight_line_counts_repeat_filings_on_the_same_stock():
    import web.x_insight as ins

    line = ins.build_insight_line("東陽テクニカ", {"filer_disclosures": 1434, "stock_disclosures": 3})
    assert line == "この提出者の開示は過去1434件、東陽テクニカでは3回目"


def test_build_insight_line_empty_without_context():
    import web.x_insight as ins

    assert ins.build_insight_line("東陽テクニカ", {}) == ""


# ------------------------------------------------------------------ 答え合わせ（web/x_followup）

def test_build_followup_text_reports_average_and_both_extremes():
    """勝った銘柄だけを出さない（平均・上昇数・最も下げた銘柄まで出す）。"""
    import web.x_followup as fu

    holdings = [{"issuer_code": c, "issuer_name": n} for c, n in
                [("1111", "上げた会社"), ("2222", "下げた会社"), ("3333", "横ばい会社"),
                 ("4444", "D社"), ("5555", "E社")]]
    returns = {"1111": 30.0, "2222": -20.0, "3333": 1.0, "4444": 5.0, "5555": -2.0}
    stats = fu.build_stats(holdings, returns)
    text = fu.build_followup_text("2026-05-20", stats)
    assert "5/20に大量保有報告書が出た5銘柄のその後" in text
    assert "平均 +2.8%" in text
    assert "上がったのは 3/5銘柄" in text
    assert "上げた会社(1111) +30.0%" in text
    assert "下げた会社(2222) -20.0%" in text
    assert text.endswith("#日本株 #大量保有報告書")


def test_build_followup_text_none_when_too_few_samples():
    import web.x_followup as fu

    holdings = [{"issuer_code": "1111", "issuer_name": "A"}]
    stats = fu.build_stats(holdings, {"1111": 5.0})
    assert fu.build_followup_text("2026-05-20", stats) is None


def test_fetch_returns_excludes_split_adjusted_series():
    """株式分割で終値が不連続な銘柄は、実態とかけ離れたリターンになるため除外する。"""
    import web.x_followup as fu

    rows = [
        {"code": "1111", "date": "2026-05-20", "close": 1000.0},
        {"code": "1111", "date": "2026-05-21", "close": 250.0},   # 4分割（-75%）
        {"code": "2222", "date": "2026-05-20", "close": 1000.0},
        {"code": "2222", "date": "2026-05-21", "close": 1100.0},
    ]
    with mock.patch.object(fu.sb, "select", return_value=rows):
        out = fu.fetch_returns(["1111", "2222"], "2026-05-20")
    assert "1111" not in out
    assert round(out["2222"], 1) == 10.0


def test_fetch_returns_skips_series_starting_long_after_disclosure():
    import web.x_followup as fu

    rows = [
        {"code": "1111", "date": "2026-06-20", "close": 1000.0},
        {"code": "1111", "date": "2026-06-21", "close": 1100.0},
    ]
    with mock.patch.object(fu.sb, "select", return_value=rows):
        assert fu.fetch_returns(["1111"], "2026-05-20") == {}


# ------------------------------------------------------------------ メトリクス収集（web/x_metrics）

def test_fetch_target_tweet_ids_url_encodes_the_timestamp():
    """ISO文字列の"+00:00"を生でクエリに載せると空白と解釈されPostgRESTが400を返す。"""
    import web.x_metrics as xm

    captured = {}
    with mock.patch.object(xm.sb, "select",
                           side_effect=lambda table, query: captured.update(q=query) or []):
        xm.fetch_target_tweet_ids()
    assert "+00:00" not in captured["q"]
    assert "%2B00%3A00" in captured["q"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

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
import video.tts as tts_mod
import video.background as bg_mod
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
    "scenes": [
        {"kind": "hook", "caption": "40億円超の売却", "narration": "東陽テクニカの株式が40億円分売却されました。"},
        {"kind": "company", "caption": "電子計測の専業", "narration": "電子計測装置を手がける会社です。"},
        {"kind": "deal", "caption": "8.77%まで低下", "narration": "保有比率は8.77パーセントまで低下しました。"},
        {"kind": "filer", "caption": "国内独立系AM", "narration": "国内独立系の資産運用会社による売却です。"},
        {"kind": "change", "caption": "利益確定売りか", "narration": "値上がり後の利益確定の可能性があります。"},
        {"kind": "outlook", "caption": "需給に注意", "narration": "今後の需給の変化に注意が必要です。"},
        {"kind": "cta", "caption": "続きはブログで", "narration": "詳しくはクジラウォッチで。"},
    ],
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
    props = bs.build_props(article, {"scenes": PROPS["scenes"]})
    assert props["direction"] == "sell"
    assert props["discDate"] == "2026-08-14"
    assert props["dealAmountOku"] == 12.3


def test_build_props_defaults_missing_filer_name_to_empty():
    """古い記事はfilerNameが未設定（microCMSは空フィールドを返さない）。動画側で行ごと省く。"""
    article = {"stockName": "A社", "stockCode": "1234", "tags": "", "dealType": ["個人"],
               "dealDate": "2026-08-14T00:00:00.000Z", "dealAmount": 1.0, "body": "<p>5.00%</p>"}
    props = bs.build_props(article, {"scenes": PROPS["scenes"]})
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


def _script_json(caption_len: int = 10, narration_len: int = 60) -> str:
    """新形式（hook/sections/closing、各narration+caption）の正常なClaude出力を作る。"""
    import json as _json
    cap = "あ" * caption_len
    nar = "い" * (narration_len - 1) + "。"
    section = {"narration": nar, "caption": cap}
    return _json.dumps({
        "hook": dict(section),
        "sections": [
            {"kind": k, **section} for k, _ in bs.SECTION_SPEC
        ],
        "closing": dict(section),
    }, ensure_ascii=False)


def test_generate_script_parses_json_into_flat_scenes():
    """hook/sections/closing がフラットなシーン列（kind付き）に展開される。"""
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=_claude_response(_script_json())):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    kinds = [sc["kind"] for sc in script["scenes"]]
    assert kinds == ["hook"] + [k for k, _ in bs.SECTION_SPEC] + ["cta"]
    assert all(sc["narration"] and sc["caption"] for sc in script["scenes"])


def test_generate_script_overrides_kind_by_expected_order():
    """Claudeがkindを誤っても、期待順（SECTION_SPEC）で上書きされる。"""
    import json as _json
    data = _json.loads(_script_json())
    for sec in data["sections"]:
        sec["kind"] = "wrong"
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=_claude_response(_json.dumps(data, ensure_ascii=False))):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert [sc["kind"] for sc in script["scenes"][1:-1]] == [k for k, _ in bs.SECTION_SPEC]


def test_generate_script_retries_once_when_caption_too_long():
    """字数超過は動画のレイアウトを壊すため、一度だけ作り直す。"""
    client = mock.Mock()
    client.messages.create.side_effect = [
        mock.Mock(content=[mock.Mock(text=_script_json(caption_len=bs.CAPTION_MAX_CHARS + 10))]),
        mock.Mock(content=[mock.Mock(text=_script_json())]),
    ]
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert client.messages.create.call_count == 2
    assert all(len(sc["caption"]) <= bs.CAPTION_MAX_CHARS for sc in script["scenes"])


def test_generate_script_trims_when_retry_still_too_long():
    """作り直しても長い場合は末尾を詰めてレイアウト崩れを防ぐ。"""
    payload = _script_json(caption_len=bs.CAPTION_MAX_CHARS + 10)
    client = mock.Mock()
    client.messages.create.return_value = mock.Mock(content=[mock.Mock(text=payload)])
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert all(len(sc["caption"]) <= bs.CAPTION_MAX_CHARS for sc in script["scenes"])


def test_generate_script_returns_none_without_api_key():
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", ""):
        assert bs.generate_script({"title": "t", "body": "b", "tags": ""}) is None


def test_generate_script_returns_none_when_sections_missing():
    """sectionsが5件に満たない不正な出力は捨てる（不完全な動画を作らない）。"""
    payload = '{"hook": {"narration": "a", "caption": "b"}, "sections": [], "closing": {"narration": "c", "caption": "d"}}'
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=_claude_response(payload)):
        assert bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""}) is None


def test_trim_narration_cuts_at_sentence_boundary():
    """ナレーションの切り詰めは文の途中ではなく句点で切る（読み上げが不自然になるため）。"""
    text = "最初の文です。" * 20  # 140字
    trimmed = bs._trim_narration(text, bs.NARRATION_MAX_CHARS)
    assert len(trimmed) <= bs.NARRATION_MAX_CHARS
    assert trimmed.endswith("。")


def test_trim_narration_falls_back_when_no_period():
    text = "あ" * 120
    trimmed = bs._trim_narration(text, bs.NARRATION_MAX_CHARS)
    assert len(trimmed) == bs.NARRATION_MAX_CHARS
    assert trimmed.endswith("…")


# ---------------- ナレーション（VOICEVOX） ----------------

def test_narrate_sections_skips_when_engine_unavailable(tmp_path):
    """VOICEVOXエンジンに繋がらない場合は無音扱い（動画は止めない）。"""
    with mock.patch.object(tts_mod, "engine_available", return_value=False):
        assert tts_mod.narrate_sections([{"narration": "テスト。"}], str(tmp_path)) is False


def test_narrate_sections_writes_audio_and_duration(tmp_path):
    """成功時は各シーンに audio ファイル名と durationSec が書き込まれる。"""
    sections = [{"narration": "テストです。"}, {"narration": "二つ目です。"}]

    def fake_synthesize(text, out_path):
        with open(out_path, "wb") as f:
            f.write(b"RIFF")
        return True

    with mock.patch.object(tts_mod, "engine_available", return_value=True), \
         mock.patch.object(tts_mod, "synthesize", side_effect=fake_synthesize), \
         mock.patch.object(tts_mod, "duration_seconds", return_value=2.5):
        ok = tts_mod.narrate_sections(sections, str(tmp_path))

    assert ok is True
    assert sections[0]["audio"] == "narration_0.wav"
    assert sections[1]["audio"] == "narration_1.wav"
    assert all(s["durationSec"] == 2.5 for s in sections)


def test_narrate_sections_all_or_nothing(tmp_path):
    """1つでも合成に失敗したら全体を無音扱いにする（一部だけ音が出る動画を防ぐ）。"""
    sections = [{"narration": "一つ目。"}, {"narration": "二つ目。"}]
    with mock.patch.object(tts_mod, "engine_available", return_value=True), \
         mock.patch.object(tts_mod, "synthesize", side_effect=[True, False]):
        assert tts_mod.narrate_sections(sections, str(tmp_path)) is False


# ---------------- 株価チャートシーン ----------------

def _fake_prices(closes):
    import pandas as pd
    return pd.DataFrame({"Close": closes})


def test_build_price_scene_up_trend():
    closes = [100.0] * 43 + [120.0] * 20
    with mock.patch("lib.utils.get_prices", return_value=_fake_prices(closes)):
        scene = bs.build_price_scene("9627")
    assert scene["kind"] == "chart"
    assert "+20%" in scene["caption"]
    assert "上昇" in scene["narration"]
    assert scene["closes"][-1] == 120.0
    assert len(scene["closes"]) <= 63


def test_build_price_scene_down_trend_uses_minus_sign():
    closes = [200.0] * 43 + [150.0] * 20
    with mock.patch("lib.utils.get_prices", return_value=_fake_prices(closes)):
        scene = bs.build_price_scene("9627")
    assert "−25%" in scene["caption"]
    assert "下落" in scene["narration"]


def test_build_price_scene_returns_none_without_data():
    with mock.patch("lib.utils.get_prices", return_value=None):
        assert bs.build_price_scene("9627") is None


# ---------------- 背景動画（Pexels） ----------------

def test_pick_video_file_prefers_portrait_and_min_height():
    """縦向きかつMIN_HEIGHT以上のファイルだけが候補になり、過剰な解像度は選ばない。"""
    videos = [
        {"duration": 15, "video_files": [
            {"height": 720, "width": 1280, "link": "landscape"},          # 横向き→除外
            {"height": 1080, "width": 608, "link": "small-portrait"},     # 低解像度→除外
            {"height": 1920, "width": 1080, "link": "hd-portrait"},
            {"height": 3840, "width": 2160, "link": "uhd-portrait"},
        ]},
    ]
    picked = bg_mod.pick_video_file(videos)
    assert picked["file"]["link"] == "hd-portrait"
    assert picked["duration"] == 15


def test_pick_video_file_returns_none_when_no_portrait():
    videos = [{"duration": 10, "video_files": [{"height": 720, "width": 1280, "link": "x"}]}]
    assert bg_mod.pick_video_file(videos) is None


def test_background_fetch_pool_skips_without_api_key(tmp_path):
    with mock.patch.dict(os.environ, {}, clear=True):
        assert bg_mod.fetch_pool(str(tmp_path)) == []


def test_assign_backgrounds_avoids_consecutive_repeat():
    """プールが2本以上あれば、隣り合うシーンに同じ背景を割り当てない。"""
    scenes = [{"kind": k} for k in ("hook", "company", "deal", "filer", "change", "outlook", "chart", "cta")]
    pool = [
        {"filename": "bg_0.mp4", "durationSec": 10.0, "people": False},
        {"filename": "bg_1.mp4", "durationSec": 12.0, "people": False},
    ]
    bg_mod.assign_backgrounds(scenes, pool)
    names = [s["backgroundVideo"] for s in scenes]
    assert all(names[i] != names[i + 1] for i in range(len(names) - 1))
    assert all(s["backgroundVideoDurationSec"] > 0 for s in scenes)


def test_assign_backgrounds_first_scene_prefers_people():
    """先頭シーン（hook）にはプール内の人物素材を優先して割り当てる。"""
    scenes = [{"kind": k} for k in ("hook", "company", "deal")]
    pool = [
        {"filename": "bg_0.mp4", "durationSec": 10.0, "people": False},
        {"filename": "bg_1.mp4", "durationSec": 12.0, "people": False},
        {"filename": "bg_2.mp4", "durationSec": 9.0, "people": True},
    ]
    for _ in range(10):  # ランダム性があるため繰り返して常に人物になることを確認
        bg_mod.assign_backgrounds(scenes, pool)
        assert scenes[0]["backgroundVideo"] == "bg_2.mp4"


def test_assign_backgrounds_first_scene_falls_back_without_people():
    """プールに人物素材が無ければ先頭シーンも通常の割当になる。"""
    scenes = [{"kind": "hook"}]
    pool = [{"filename": "bg_0.mp4", "durationSec": 10.0, "people": False}]
    bg_mod.assign_backgrounds(scenes, pool)
    assert scenes[0]["backgroundVideo"] == "bg_0.mp4"


def test_assign_backgrounds_single_video_reused():
    """プールが1本しか無ければ全シーンで使い回す。"""
    scenes = [{"kind": "hook"}, {"kind": "cta"}]
    bg_mod.assign_backgrounds(scenes, [{"filename": "bg_0.mp4", "durationSec": 8.0, "people": False}])
    assert [s["backgroundVideo"] for s in scenes] == ["bg_0.mp4", "bg_0.mp4"]


def test_assign_backgrounds_noop_with_empty_pool():
    scenes = [{"kind": "hook"}]
    bg_mod.assign_backgrounds(scenes, [])
    assert "backgroundVideo" not in scenes[0]


# ---------------- YouTube ----------------

def test_youtube_title_includes_stock_and_amount():
    title = yt.build_title(PROPS)
    assert "東陽テクニカ" in title
    assert "40.1億円" in title
    assert "売却" in title
    assert "#Shorts" in title


def test_youtube_title_falls_back_when_filer_name_missing():
    """filerName未設定の記事で「【銘柄】が…取得」という主語のねじれを防ぐ。"""
    title = yt.build_title({**PROPS, "filerName": ""})
    assert "大口投資家が" in title
    assert "】が" not in title


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

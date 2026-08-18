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
import video.line_notify as ln
import video.youtube_client as yt
import video.render as render_mod

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


def test_parse_article_id_accepts_bare_id():
    """記事IDをそのまま渡した場合はそのまま返す。"""
    assert bs.parse_article_id("lfu9qf5t6u3l") == "lfu9qf5t6u3l"


def test_parse_article_id_extracts_from_url():
    """記事URLを丸ごと貼っても記事IDだけを取り出す（オーナーのコピペ運用のため）。"""
    assert bs.parse_article_id(
        "https://kujira-watch.com/articles/lfu9qf5t6u3l"
    ) == "lfu9qf5t6u3l"


def test_parse_article_id_strips_whitespace_and_trailing_slash():
    """前後の空白・末尾スラッシュを落として実IDだけにする。"""
    assert bs.parse_article_id("  https://kujira-watch.com/articles/abc123/  ") == "abc123"
    assert bs.parse_article_id(" abc123 ") == "abc123"


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

def test_duration_seconds_reads_wav_exactly(tmp_path):
    """WAVの長さはwaveモジュールで正確に読む。ffprobe不在のCIで文字数概算に落ちて
    全シーンの音声が尻切れになった実障害（2026-08-17）の再発防止。"""
    import struct, wave as wave_mod
    path = str(tmp_path / "t.wav")
    with wave_mod.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(struct.pack("<h", 0) * 60000)  # 2.5秒ぶんの無音
    assert abs(tts_mod.duration_seconds(path, text="あ" * 90) - 2.5) < 0.001


def test_duration_seconds_falls_back_to_estimate_for_broken_file(tmp_path):
    path = str(tmp_path / "broken.wav")
    open(path, "wb").write(b"not a wav")
    with mock.patch("shutil.which", return_value=None):
        est = tts_mod.duration_seconds(path, text="あ" * 75)
    assert est == 10.0  # 75 / 7.5


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
        {"filename": "bg_0.mp4", "durationSec": 10.0},
        {"filename": "bg_1.mp4", "durationSec": 12.0},
    ]
    bg_mod.assign_backgrounds(scenes, pool)
    names = [s["backgroundVideo"] for s in scenes]
    assert all(names[i] != names[i + 1] for i in range(len(names) - 1))
    assert all(s["backgroundVideoDurationSec"] > 0 for s in scenes)



def test_assign_backgrounds_single_video_reused():
    """プールが1本しか無ければ全シーンで使い回す。"""
    scenes = [{"kind": "hook"}, {"kind": "cta"}]
    bg_mod.assign_backgrounds(scenes, [{"filename": "bg_0.mp4", "durationSec": 8.0}])
    assert [s["backgroundVideo"] for s in scenes] == ["bg_0.mp4", "bg_0.mp4"]


def test_assign_backgrounds_noop_with_empty_pool():
    scenes = [{"kind": "hook"}]
    bg_mod.assign_backgrounds(scenes, [])
    assert "backgroundVideo" not in scenes[0]


# ---------------- LINE通知 ----------------

def test_line_message_includes_caption_and_youtube_url():
    msg = ln.build_message({"articleTitle": "テスト記事", "stockName": "A社"},
                           "キャプション本文 #tag", youtube_id="abc123", tiktok_publish_id="p1")
    assert "テスト記事" in msg
    assert "https://youtube.com/shorts/abc123" in msg
    assert "キャプション本文 #tag" in msg
    assert "コピー用" in msg


def test_line_message_omits_tiktok_block_when_not_uploaded():
    msg = ln.build_message({"articleTitle": "t"}, "cap", youtube_id="abc", tiktok_publish_id=None)
    assert "キャプション" not in msg
    assert "youtube.com" in msg


def test_line_notify_skips_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert ln.notify({"articleTitle": "t"}, "cap", youtube_id="abc") is False


def test_line_notify_skips_when_nothing_posted():
    with mock.patch.dict(os.environ, {"LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_USER_ID": "u"}, clear=True):
        assert ln.notify({"articleTitle": "t"}, "cap") is False


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

def test_tiktok_caption_includes_deal_details():
    """定型の誘導文ではなく取引詳細（比率・開示日・要点）で構成する（2026-08-17指示）。"""
    caption = tk.build_caption(PROPS)
    assert "東陽テクニカ" in caption
    assert "8151" in caption
    assert "保有比率8.77%" in caption
    assert "8/14開示" in caption
    assert "#東陽テクニカ" in caption
    # 廃止した定型文が混ざっていないこと
    assert "プロフィール" not in caption
    assert "VOICEVOX" not in caption
    # 台本の要点が箇条書きで入ること（company/change/outlook/chartから最大3つ）
    assert "・電子計測の専業" in caption


def test_tiktok_caption_falls_back_when_filer_name_missing():
    """filerName未設定の記事でも主語が欠けない（「大口投資家」に置換される）。"""
    caption = tk.build_caption({**PROPS, "filerName": ""})
    assert "大口投資家が" in caption


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


# ---------------- 音量正規化（配信基準 -14 LUFS） ----------------

LOUDNORM_JSON = """
[Parsed_loudnorm_0 @ 0x7f] 
{
	"input_i" : "-25.15",
	"input_tp" : "-6.10",
	"input_lra" : "3.70",
	"input_thresh" : "-35.60",
	"target_offset" : "0.20"
}
"""


def test_measure_loudness_parses_ffmpeg_json():
    """loudnormの測定値はstderr末尾のJSONに出るので、そこから読み取る。"""
    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stderr=LOUDNORM_JSON)):
        stats = render_mod._measure_loudness("dummy.mp4")
    assert stats["input_i"] == "-25.15"
    assert stats["input_thresh"] == "-35.60"


def test_measure_loudness_returns_none_when_unparsable():
    """測定できなければNoneを返し、呼び出し側は音量そのままで続行する。"""
    with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stderr="no json here")):
        assert render_mod._measure_loudness("dummy.mp4") is None


def test_normalize_loudness_skips_when_no_audio_stream():
    """無音の動画（TTS失敗時）は正規化せずFalseを返す。"""
    with mock.patch.object(render_mod, "_has_audio_stream", return_value=False):
        assert render_mod.normalize_loudness("dummy.mp4") is False


def test_normalize_loudness_passes_measured_values_to_second_pass(tmp_path):
    """2パス目には1パス目の測定値をそのまま渡す（これが無いと目標音量に届かない）。"""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        # 2パス目は出力ファイルを作る想定
        pathlib_path = cmd[-1]
        with open(pathlib_path, "wb") as f:
            f.write(b"y" * 10)
        return mock.Mock(returncode=0, stderr="")

    with mock.patch.object(render_mod, "_has_audio_stream", return_value=True), \
         mock.patch.object(render_mod, "_measure_loudness",
                           return_value={"input_i": "-25.15", "input_tp": "-6.10",
                                         "input_lra": "3.70", "input_thresh": "-35.60",
                                         "target_offset": "0.20"}), \
         mock.patch("subprocess.run", side_effect=fake_run):
        assert render_mod.normalize_loudness(str(video)) is True

    filt = calls["cmd"][calls["cmd"].index("-af") + 1]
    assert "measured_I=-25.15" in filt
    assert "measured_TP=-6.10" in filt
    assert f"I={render_mod.TARGET_LUFS}" in filt
    # 映像は再エンコードしない（画質を落とさないため）
    assert "copy" in calls["cmd"]
    assert video.read_bytes() == b"y" * 10


def test_normalize_loudness_keeps_original_when_ffmpeg_fails(tmp_path):
    """正規化に失敗しても元のmp4を残す（音量のために投稿を止めない）。"""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"orig")

    with mock.patch.object(render_mod, "_has_audio_stream", return_value=True), \
         mock.patch.object(render_mod, "_measure_loudness",
                           return_value={"input_i": "-25.0", "input_tp": "-6.0",
                                         "input_lra": "3.0", "input_thresh": "-35.0",
                                         "target_offset": "0.0"}), \
         mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr="boom")):
        assert render_mod.normalize_loudness(str(video)) is False

    assert video.read_bytes() == b"orig"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

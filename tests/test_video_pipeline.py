"""自動動画投稿パイプライン（video/）のロジックのユニットテスト。
ネットワーク(microCMS / Claude / YouTube)は全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_video_pipeline.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import video.build_script as bs
import video.tts as tts_mod
import video.background as bg_mod
import video.line_notify as ln
import video.youtube_client as yt
import video.render as render_mod
import video.audio_gen as audio_gen
import video.post_text as pt


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
    """古い記事はfilerNameが未設定（microCMSは空フィールドを返さない）。開示データからも
    特定できなければ空文字にして、動画側で行ごと省く。"""
    article = {"stockName": "A社", "stockCode": "1234", "tags": "", "dealType": ["個人"],
               "dealDate": "2026-08-14T00:00:00.000Z", "dealAmount": 1.0, "body": "<p>5.00%</p>"}
    with mock.patch("lib.db.sb.select", return_value=[]):
        props = bs.build_props(article, {"scenes": PROPS["scenes"]})
    assert props["filerName"] == ""


def _old_article(body: str) -> dict:
    return {"stockName": "サンクゼール", "stockCode": "2937", "tags": "売り",
            "dealDate": "2026-08-10T00:00:00.000Z", "dealAmount": 14.8, "body": body}


def test_resolve_filer_name_picks_the_candidate_named_in_the_body():
    """同一銘柄・同一開示日には複数の提出者がいるのが普通なので、開示データだけでは
    一意に決まらない。本文に名前が書かれているものだけを採る。"""
    article = _old_article("<p>個人投資家の久世良太氏が保有株式の売却を進めることが明らかに。</p>")
    rows = [{"filer_name": "久世　良太"}, {"filer_name": "公益財団法人サンクゼール財団"}]
    with mock.patch("lib.db.sb.select", return_value=rows):
        assert bs.resolve_filer_name(article) == "久世　良太"


def test_resolve_filer_name_returns_empty_when_ambiguous():
    """本文に複数の候補が出てくる回は特定できない。誤った提出者名を出すより総称に落とす。"""
    article = _old_article("<p>久世良太氏とサンクゼール財団が保有。</p>")
    rows = [{"filer_name": "久世　良太"}, {"filer_name": "公益財団法人サンクゼール財団"}]
    with mock.patch("lib.db.sb.select", return_value=rows):
        assert bs.resolve_filer_name(article) == ""


def test_resolve_filer_name_returns_empty_without_disclosure_rows():
    with mock.patch("lib.db.sb.select", return_value=[]):
        assert bs.resolve_filer_name(_old_article("<p>本文</p>")) == ""


def test_resolve_filer_name_keeps_existing_value_without_lookup():
    """filerNameがある記事では照会しない（毎回Supabaseを叩かない）。"""
    article = dict(_old_article("<p>本文</p>"), filerName="既にある名前")
    with mock.patch("lib.db.sb.select", side_effect=AssertionError("照会してはいけない")):
        assert bs.resolve_filer_name(article) == "既にある名前"


def test_normalize_name_absorbs_width_space_and_corporate_suffix():
    assert bs._normalize_name("久世　良太") == bs._normalize_name("久世良太")
    assert bs._normalize_name("アースエレメンツ・キャピタル株式会社") == \
        bs._normalize_name("アースエレメンツキャピタル")


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


def _script_json(caption_len: int = 10, narration_len: int = 40) -> str:
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
    """sectionsがSECTION_SPECの件数に満たない不正な出力は捨てる（不完全な動画を作らない）。"""
    payload = '{"hook": {"narration": "a", "caption": "b"}, "sections": [], "closing": {"narration": "c", "caption": "d"}}'
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=_claude_response(payload)):
        assert bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""}) is None


def test_generate_script_feeds_the_problem_back_on_retry():
    """作り直しでは「何字の文が長すぎたか」をプロンプトに足して伝える。同じ指示を
    そのまま投げ直すと同じ長さが返り、2026-08-19に投稿0件になった。"""
    import json as _json
    over = _json.loads(_script_json())
    over["sections"][0]["narration"] = "長い文です。" * 15  # 90字
    client = mock.Mock()
    client.messages.create.side_effect = [
        mock.Mock(content=[mock.Mock(text=_json.dumps(over, ensure_ascii=False))]),
        mock.Mock(content=[mock.Mock(text=_script_json())]),
    ]
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        assert bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""}) is not None
    retry_prompt = client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "前回の出力の問題" in retry_prompt
    assert "90字" in retry_prompt
    assert "narration" in retry_prompt


def test_retry_feedback_distinguishes_caption_from_narration():
    """captionの26字超過を「narrationが長い」と伝えると見当違いの直しをさせる
    （2026-08-20の博報堂DYの回が3回とも外して投稿0件になった）。"""
    import json as _json
    over = _json.loads(_script_json())
    over["sections"][0]["caption"] = "あ" * (bs.CAPTION_MAX_CHARS + 3)
    client = mock.Mock()
    client.messages.create.side_effect = [
        mock.Mock(content=[mock.Mock(text=_json.dumps(over, ensure_ascii=False))]),
        mock.Mock(content=[mock.Mock(text=_script_json())]),
    ]
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    retry_prompt = client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "caption" in retry_prompt and f"{bs.CAPTION_MAX_CHARS + 3}字" in retry_prompt


def test_prompt_forbids_takeover_wording():
    """0.38%の取得を「買収」と読み上げた回があった。開示内容を超える語は禁止する。"""
    client = mock.Mock()
    client.messages.create.return_value = mock.Mock(content=[mock.Mock(text=_script_json())])
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "買収" in prompt and "使わないでください" in prompt
    # 新規保有を「買い増し」と読み上げた回があった（日本製鉄×JPモルガン 0.38%）
    assert "新規保有なら「買い増し」とは書かず" in prompt


def test_generate_script_drops_scene_that_stays_broken_instead_of_the_video():
    """1シーンの読み上げ文が直らないだけで動画を丸ごと諦めると、その日の投稿が飛ぶ
    （2026-08-19・20と2日続けて0件になった）。会社説明はシーンごと落として動画は出す。"""
    import json as _json
    data = _json.loads(_script_json())
    data["sections"][0]["narration"] = "あ" * (bs.NARRATION_MAX_CHARS + 30)  # 句点なし→…で切れる
    payload = _json.dumps(data, ensure_ascii=False)
    client = mock.Mock()
    client.messages.create.return_value = mock.Mock(content=[mock.Mock(text=payload)])
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert script is not None
    assert "company" not in [sc["kind"] for sc in script["scenes"]]
    assert all(not bs.is_broken_narration(sc["narration"]) for sc in script["scenes"])


def test_salvage_scenes_rebuilds_hook_from_facts():
    """hookは動画の要なので落とせない。壊れていたら記事の事実だけで組み直す。"""
    article = {"stockName": "日本製鉄", "filerName": "JPモルガン", "dealAmount": 137.7,
               "body": "<p>0.38%</p>", "tags": "買い"}
    scenes = [
        {"kind": "hook", "caption": "x", "narration": "途中で切れた文…"},
        {"kind": "cta", "caption": "y", "narration": "完結した文です。"},
    ]
    salvaged = bs.salvage_scenes(scenes, article)
    hook = salvaged[0]
    assert hook["kind"] == "hook"
    assert not bs.is_broken_narration(hook["narration"])
    assert "日本製鉄" in hook["narration"] and "137.7" in hook["narration"]


def test_salvage_scenes_uses_deal_type_label_for_long_filer_names():
    """英語の正式名は読み上げが長くなるので、20字を超えたら分類ラベルで代える。"""
    article = {"stockName": "電通総研", "filerName": "Oasis Management Company Ltd.",
               "dealType": ["アクティビスト"], "dealAmount": 262.9,
               "body": "<p>5.01%</p>", "tags": "買い"}
    salvaged = bs.salvage_scenes(
        [{"kind": "hook", "caption": "x", "narration": "切れた…"},
         {"kind": "cta", "caption": "y", "narration": "完結した文です。"}], article)
    assert "アクティビスト" in salvaged[0]["narration"]


def test_salvage_scenes_drops_change_without_previous_ratio():
    """前回比率が無ければ「推移」を語れないので、定型文も作らず落とす。"""
    article = {"stockName": "A", "filerName": "B", "dealAmount": 10,
               "body": "<p>5.01%</p>", "tags": "買い"}
    salvaged = bs.salvage_scenes(
        [{"kind": "hook", "caption": "x", "narration": "完結した文です。"},
         {"kind": "change", "caption": "y", "narration": "切れた…"},
         {"kind": "cta", "caption": "z", "narration": "完結した文です。"}], article)
    assert [sc["kind"] for sc in salvaged] == ["hook", "cta"]


def test_salvage_scenes_returns_none_without_hook_or_cta():
    article = {"stockName": "A", "filerName": "B", "dealAmount": 10, "body": "", "tags": ""}
    assert bs.salvage_scenes([{"kind": "deal", "caption": "x", "narration": "完結した文です。"}],
                             article) is None


def test_generate_script_accepts_after_retry_fixes_broken_narration():
    """1回目が途中で切れていても、作り直しで直れば台本として採用する。"""
    import json as _json
    broken = _json.loads(_script_json())
    broken["sections"][0]["narration"] = "あ" * (bs.NARRATION_MAX_CHARS + 30)
    client = mock.Mock()
    client.messages.create.side_effect = [
        mock.Mock(content=[mock.Mock(text=_json.dumps(broken, ensure_ascii=False))]),
        mock.Mock(content=[mock.Mock(text=_script_json())]),
    ]
    with mock.patch.object(bs, "ANTHROPIC_API_KEY", "key"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        script = bs.generate_script({"title": "t", "body": "<p>b</p>", "tags": ""})
    assert script is not None
    assert all(not bs.is_broken_narration(sc["narration"]) for sc in script["scenes"])


def test_is_broken_narration_detects_truncation():
    assert bs.is_broken_narration("保有比率を引き上げました。こ…")
    assert bs.is_broken_narration("句点で終わらない文")
    assert not bs.is_broken_narration("句点で終わる文です。")


def test_script_has_no_outlook_section():
    """outlook（推測シーン）は廃止済み。復活させるとSceneViewに描き先が無い。"""
    assert "outlook" not in [k for k, _ in bs.SECTION_SPEC]


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

def _fake_prices(closes, start="2026-05-01"):
    """lib.utils.get_prices と同じ形（DatetimeIndex＋Close列）のダミー。
    build_price_scene が index から日付ラベルと開示日の位置を作るため、indexも本物に寄せる。"""
    import pandas as pd
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_extract_prev_holding_ratio_takes_second_last_percentage():
    """本文は「前回◯%→今回◯%」の順なので、末尾から2番目が前回の比率になる。"""
    article = {"body": "<p>前回3.40%から今回5.21%へ</p>", "tags": "買い"}
    assert bs.extract_prev_holding_ratio(article) == 3.40


def test_extract_prev_holding_ratio_returns_none_when_direction_contradicts():
    """買いなのに比率が減っている＝拾い方を間違えているので、前回不明として扱う
    （誤った数字を出すくらいなら change シーンごと落とす）。"""
    article = {"body": "<p>8.00%から5.21%へ</p>", "tags": "買い"}
    assert bs.extract_prev_holding_ratio(article) is None


def test_extract_prev_holding_ratio_accepts_decrease_for_sell():
    article = {"body": "<p>8.00%から5.21%へ</p>", "tags": "売り"}
    assert bs.extract_prev_holding_ratio(article) == 8.00


def test_extract_prev_holding_ratio_returns_none_with_single_percentage():
    assert bs.extract_prev_holding_ratio({"body": "<p>5.21%</p>", "tags": "買い"}) is None


def test_build_props_includes_prev_holding_ratio():
    article = {"stockName": "A", "stockCode": "1", "dealAmount": 10,
               "body": "<p>3.40%から5.21%へ</p>", "tags": "買い", "dealDate": "2026-08-14"}
    props = bs.build_props(article, {"scenes": []})
    assert props["prevHoldingRatio"] == 3.40
    assert props["holdingRatio"] == 5.21


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


def test_build_price_scene_marks_disclosure_position():
    """開示日がチャートの範囲内なら、その位置（discIndex）を返して縦線を引かせる。"""
    closes = [100.0] * 63
    with mock.patch("lib.utils.get_prices", return_value=_fake_prices(closes, start="2026-06-01")):
        scene = bs.build_price_scene("9627", "2026-08-01")
    assert len(scene["dates"]) == len(scene["closes"])
    assert scene["dates"][scene["discIndex"]] == "2026-08-01"


def test_build_price_scene_omits_disclosure_before_chart_range():
    """開示日がチャートより前なら縦線は引かない（範囲外に線を出さない）。"""
    closes = [100.0] * 63
    with mock.patch("lib.utils.get_prices", return_value=_fake_prices(closes, start="2026-06-01")):
        scene = bs.build_price_scene("9627", "2026-01-01")
    assert "discIndex" not in scene


def test_build_price_scene_without_disc_date_has_no_index():
    closes = [100.0] * 63
    with mock.patch("lib.utils.get_prices", return_value=_fake_prices(closes)):
        scene = bs.build_price_scene("9627")
    assert "discIndex" not in scene
    assert len(scene["dates"]) == len(scene["closes"])


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


def test_has_rejected_subject_matches_word_not_substring():
    """人物クリップは弾き、地名などの部分一致（germany の man）では弾かない。"""
    assert bg_mod.has_rejected_subject("https://www.pexels.com/video/woman-eating-sushi-12345/")
    assert bg_mod.has_rejected_subject("https://www.pexels.com/video/a-person-walking-99/")
    assert not bg_mod.has_rejected_subject("https://www.pexels.com/video/aerial-view-of-germany-77/")
    assert not bg_mod.has_rejected_subject("https://www.pexels.com/video/ocean-waves-at-sunset-42/")


def test_pick_video_file_skips_clips_with_people():
    """自然系のクエリでもPexelsは人物クリップを返す。金融解説の画としてノイズなので使わない。"""
    people = {"duration": 15, "url": "https://www.pexels.com/video/woman-in-a-forest-1/",
              "video_files": [{"height": 1920, "width": 1080, "link": "people"}]}
    nature = {"duration": 15, "url": "https://www.pexels.com/video/forest-canopy-2/",
              "video_files": [{"height": 1920, "width": 1080, "link": "nature"}]}
    assert bg_mod.pick_video_file([people]) is None
    assert bg_mod.pick_video_file([people, nature])["file"]["link"] == "nature"


def test_pick_video_file_returns_none_when_no_portrait():
    videos = [{"duration": 10, "video_files": [{"height": 720, "width": 1280, "link": "x"}]}]
    assert bg_mod.pick_video_file(videos) is None


def test_background_fetch_pool_skips_without_api_key(tmp_path):
    with mock.patch.dict(os.environ, {}, clear=True):
        assert bg_mod.fetch_pool(str(tmp_path)) == []


def test_assign_backgrounds_only_targets_text_only_scenes():
    """実写背景は company / filer にだけ敷く。数字を読ませるシーン（hook/deal/change/chart）と
    ctaは、実写の明部に数字が沈まないようブランドのグラデーション背景に固定する。"""
    scenes = [{"kind": k} for k in ("hook", "company", "deal", "filer", "change", "chart", "cta")]
    pool = [
        {"filename": "bg_0.mp4", "durationSec": 10.0},
        {"filename": "bg_1.mp4", "durationSec": 12.0},
    ]
    bg_mod.assign_backgrounds(scenes, pool)
    assigned = {s["kind"] for s in scenes if s.get("backgroundVideo")}
    assert assigned == {"company", "filer"}
    assert all(s["backgroundVideoDurationSec"] > 0 for s in scenes if s.get("backgroundVideo"))


def test_assign_backgrounds_avoids_consecutive_repeat():
    """プールが2本以上あれば、続けて同じ背景を割り当てない。"""
    scenes = [{"kind": k} for k in ("company", "filer")]
    pool = [
        {"filename": "bg_0.mp4", "durationSec": 10.0},
        {"filename": "bg_1.mp4", "durationSec": 12.0},
    ]
    bg_mod.assign_backgrounds(scenes, pool)
    assert scenes[0]["backgroundVideo"] != scenes[1]["backgroundVideo"]


def test_assign_backgrounds_single_video_reused():
    """プールが1本しか無ければ対象シーンで使い回す。"""
    scenes = [{"kind": "company"}, {"kind": "filer"}]
    bg_mod.assign_backgrounds(scenes, [{"filename": "bg_0.mp4", "durationSec": 8.0}])
    assert [s["backgroundVideo"] for s in scenes] == ["bg_0.mp4", "bg_0.mp4"]


def test_assign_backgrounds_noop_with_empty_pool():
    scenes = [{"kind": "company"}]
    bg_mod.assign_backgrounds(scenes, [])
    assert "backgroundVideo" not in scenes[0]


# ---------------- 効果音・BGM（自前生成） ----------------

def test_ensure_sound_effects_writes_playable_wavs(tmp_path):
    """効果音とBGMは外部素材を持たずnumpyで合成する（毎日の全自動運用でライセンス確認が要らない）。"""
    import wave as wave_mod
    assert audio_gen.ensure_sound_effects(str(tmp_path))
    for name in audio_gen.GENERATED_AUDIO:
        with wave_mod.open(str(tmp_path / name)) as w:
            assert w.getnchannels() == 1
            assert w.getframerate() == audio_gen.SAMPLE_RATE
            assert w.getnframes() > 0


def test_bgm_loops_without_a_click(tmp_path):
    """BGMは12秒でループする。末尾と先頭の段差が普通のサンプル間差分より大きいと
    ループのたびにプチッと鳴るため、段差が通常の範囲に収まっていることを確かめる。"""
    import numpy as np
    import wave as wave_mod

    audio_gen.ensure_sound_effects(str(tmp_path))
    with wave_mod.open(str(tmp_path / "bgm.wav")) as w:
        assert w.getnframes() == int(audio_gen.SAMPLE_RATE * audio_gen.BGM_LOOP_SEC)
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
    loop_step = abs(x[0] - x[-1])
    typical_step = float(np.percentile(np.abs(np.diff(x)), 99))
    assert loop_step <= typical_step * 3


def test_bgm_leaves_headroom():
    """クリップするとナレーションの上で歪む。ピークに余裕を持たせる。"""
    import numpy as np

    assert float(np.max(np.abs(audio_gen._bgm()))) <= 0.6


def test_ensure_sound_effects_is_deterministic(tmp_path):
    """同じ波形が毎回できること（差分レビューでSEの変更に気づけるように）。"""
    a, b = tmp_path / "a", tmp_path / "b"
    audio_gen.ensure_sound_effects(str(a))
    audio_gen.ensure_sound_effects(str(b))
    for name in audio_gen.GENERATED_AUDIO:
        assert (a / name).read_bytes() == (b / name).read_bytes()


# ---------------- LINE通知 ----------------

def test_line_message_includes_title_and_youtube_url():
    msg = ln.build_message({"articleTitle": "テスト記事", "stockName": "A社"}, youtube_id="abc123")
    assert "テスト記事" in msg
    assert "https://youtube.com/shorts/abc123" in msg


def test_line_notify_skips_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert ln.notify({"articleTitle": "t"}, youtube_id="abc") is False


def test_line_notify_skips_when_nothing_posted():
    with mock.patch.dict(os.environ, {"LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_USER_ID": "u"}, clear=True):
        assert ln.notify({"articleTitle": "t"}) is False


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


def test_youtube_title_includes_holding_ratio():
    """保有比率まで入れると一覧で「何%になったのか」が分かり、クリックの理由が増える。"""
    assert "保有比率8.77%へ" in yt.build_title(PROPS)


def test_youtube_title_omits_ratio_when_unknown():
    props = dict(PROPS, holdingRatio=0.0)
    title = yt.build_title(props)
    assert "保有比率" not in title
    assert title.endswith(" #Shorts")


def test_youtube_description_puts_article_url_above_the_fold():
    """Shortsの説明文は冒頭しか畳まずに見えない。記事URLが折りたたみの下（8行目）に
    あったため導線に到達できなかった（2026-08-19）。先頭3行以内に置くこと。"""
    lines = yt.build_description(PROPS).split("\n")
    url_line = next(i for i, ln in enumerate(lines) if ln.startswith("https://kujira-watch.com/articles/"))
    assert url_line <= 2


def test_youtube_description_names_the_site_correctly():
    """動画側で「クジラウォッチ」と勝手に名乗った事故（2026-08-19）を投稿文でも防ぐ。"""
    desc = yt.build_description(PROPS)
    assert pt.SITE_NAME in desc
    assert "クジラウォッチ" not in desc


def test_youtube_description_leads_with_searchable_hashtags():
    """説明文の先頭3つのハッシュタグはタイトル上部に出る枠。機能タグ(#Shorts)ではなく
    実際に検索される語を先に置く。"""
    tags = yt.build_description(PROPS).rstrip().split("\n")[-1].split()
    assert tags[:2] == ["#日本株", "#大量保有報告書"]
    assert "#Shorts" in tags and tags.index("#Shorts") >= 3


def test_post_text_hashtag_strips_unusable_characters():
    """銘柄名の空白や「．」をそのまま「#」に続けるとタグが途中で切れて本文が漏れる
    （例: Ｊ．フロント リテイリング）。"""
    assert pt.hashtag("Ｊ．フロント リテイリング") == "#Ｊフロントリテイリング"
    assert pt.hashtag("アインホールディングス") == "#アインホールディングス"
    assert pt.hashtag("") == ""
    assert pt.hashtag("・（）") == ""


def test_youtube_description_uses_sanitized_stock_hashtag():
    desc = yt.build_description(dict(PROPS, stockName="Ｊ．フロント リテイリング"))
    assert "#Ｊフロントリテイリング" in desc
    assert "#Ｊ．フロント" not in desc


def test_article_url_falls_back_to_site_root():
    assert pt.article_url("", "youtube").startswith(pt.SITE_URL + "?utm_source=youtube")
    assert pt.article_url("abc", "youtube").startswith(pt.SITE_URL + "/articles/abc?")


def test_youtube_upload_skipped_without_credentials():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert yt.upload("/tmp/none.mp4", PROPS) is None


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


def test_render_rejects_props_with_truncated_narration():
    """古いprops JSONを再レンダリングする経路でも、切れた字幕は書き出させない。"""
    assert render_mod.has_broken_narration({"scenes": [{"narration": "途中で切れた文…"}]})
    assert not render_mod.has_broken_narration({"scenes": [{"narration": "完結した文です。"}]})


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


def test_normalize_loudness_warns_when_ffprobe_missing(capsys):
    """ffmpeg/ffprobe が無い環境では音量正規化が丸ごと効かない。無言で飛ばすと
    -25 LUFS のまま投稿され続けるので必ずログに残す（CIのUbuntu 24.04で実際に起きた）。"""
    with mock.patch("subprocess.run", side_effect=FileNotFoundError("ffprobe")):
        assert render_mod.normalize_loudness("x.mp4") is False
    assert "ffprobe が見つからない" in capsys.readouterr().out


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


# ---------------- Canva製ブランド素材（エンドカード・サムネイル） ----------------

import video.thumbnail as thumb_mod


def test_stage_end_card_copies_canva_asset_and_sets_prop(tmp_path):
    with mock.patch.object(render_mod, "PUBLIC_DIR", str(tmp_path)):
        props = {}
        staged = render_mod._stage_end_card(props)
    assert props["endCard"] == "cta_endcard.png"
    assert staged == [str(tmp_path / "cta_endcard.png")]
    assert os.path.exists(staged[0])


def test_stage_end_card_falls_back_to_text_cta_without_asset(tmp_path):
    with mock.patch.object(render_mod, "ASSETS_DIR", str(tmp_path)):
        props = {"endCard": "stale"}
        assert render_mod._stage_end_card(props) == []
    assert "endCard" not in props


def test_thumbnail_compose_writes_1280x720_png(tmp_path):
    from PIL import Image

    out = thumb_mod.compose(dict(PROPS), str(tmp_path / "t.png"))
    assert out and os.path.exists(out)
    assert Image.open(out).size == (1280, 720)


def test_thumbnail_compose_skips_without_base(tmp_path):
    with mock.patch.object(thumb_mod, "BASE_IMAGE", str(tmp_path / "none.png")):
        assert thumb_mod.compose(dict(PROPS), str(tmp_path / "t.png")) is None


def test_thumbnail_format_amount_matches_remotion():
    assert thumb_mod.format_amount(1893.4) == "1,893"
    assert thumb_mod.format_amount(40.1) == "40.1"
    assert thumb_mod.format_amount(40.0) == "40"


def test_set_thumbnail_skips_without_credentials(tmp_path):
    img = tmp_path / "t.png"
    img.write_bytes(b"x")
    with mock.patch.object(yt, "_access_token", return_value=None), \
         mock.patch.object(yt.requests, "post") as post:
        assert yt.set_thumbnail("vid", str(img)) is False
    post.assert_not_called()


def test_set_thumbnail_posts_png_to_thumbnails_set(tmp_path):
    img = tmp_path / "t.png"
    img.write_bytes(b"png")
    res = mock.Mock(ok=True)
    with mock.patch.object(yt, "_access_token", return_value="tok"), \
         mock.patch.object(yt.requests, "post", return_value=res) as post:
        assert yt.set_thumbnail("vid123", str(img)) is True
    assert post.call_args.kwargs["params"]["videoId"] == "vid123"
    assert post.call_args.kwargs["headers"]["Content-Type"] == "image/png"


def test_publish_aborts_before_rendering_when_youtube_token_is_dead():
    """トークン失効の日に230秒かけて書き出さない（2026-08-25に74.9MBが丸損した）。"""
    import video.publish_video as pv
    with mock.patch.object(pv.build_script, "build", return_value={"scenes": []}), \
         mock.patch.object(pv.youtube_client, "is_configured", return_value=True), \
         mock.patch.object(pv.youtube_client, "check_auth", return_value=False), \
         mock.patch.object(pv.render, "render") as render_call:
        rc = pv.run()
    assert rc == 1, rc
    render_call.assert_not_called()


def test_publish_records_the_upload_for_the_heartbeat(tmp_path):
    """公開した事実をその場でyoutube_videosへ残す。統計収集は手動実行なので、
    それ待ちだと当日のハートビートに「動画0本」と誤判定される。"""
    import video.publish_video as pv
    video_file = tmp_path / "out"
    with mock.patch.object(pv, "OUT_DIR", str(video_file)), \
         mock.patch.dict(os.environ, {"X_API_KEY": ""}), \
         mock.patch.object(pv.build_script, "build", return_value={"scenes": [], "stockCode": "6501"}), \
         mock.patch.object(pv.youtube_client, "is_configured", return_value=True), \
         mock.patch.object(pv.youtube_client, "check_auth", return_value=True), \
         mock.patch.object(pv.youtube_client, "build_title", return_value="【日立建機】…"), \
         mock.patch.object(pv.youtube_client, "upload", return_value="vid1"), \
         mock.patch.object(pv.youtube_client, "set_thumbnail"), \
         mock.patch.object(pv.thumbnail, "compose", return_value=""), \
         mock.patch.object(pv.tts, "narrate_sections", return_value=False), \
         mock.patch.object(pv.background, "fetch_pool", return_value=[]), \
         mock.patch.object(pv.background, "assign_backgrounds"), \
         mock.patch.object(pv.render, "render", return_value=True), \
         mock.patch.object(pv.line_notify, "notify"), \
         mock.patch.object(pv.youtube_metrics, "record_upload") as rec:
        rc = pv.run(keep_video=True)
    assert rc == 0, rc
    assert rec.call_args.args == ("vid1", "【日立建機】…"), rec.call_args


def test_publish_renders_when_youtube_credentials_are_absent():
    """Secrets未登録の段階は動画の生成自体が目的なので止めない。"""
    import video.publish_video as pv
    with mock.patch.object(pv.build_script, "build", return_value={"scenes": []}), \
         mock.patch.object(pv.youtube_client, "is_configured", return_value=False), \
         mock.patch.object(pv.youtube_client, "check_auth") as check, \
         mock.patch.object(pv.tts, "narrate_sections", return_value=False), \
         mock.patch.object(pv.background, "fetch_pool", return_value=[]), \
         mock.patch.object(pv.background, "assign_backgrounds"), \
         mock.patch.object(pv.render, "render", return_value=False):
        rc = pv.run()
    check.assert_not_called()
    assert rc == 1, rc  # renderがFalse＝レンダリング失敗までは到達している

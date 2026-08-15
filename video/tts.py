"""
video/tts.py

ナレーション音声の合成（VOICEVOX）。

VOICEVOX は無料・登録不要・商用利用可（クレジット表記のみ必要）の日本語音声合成エンジン。
ずんだもん等の声は日本の TikTok / YouTube Shorts の解説動画で広く使われており、
視聴者の馴染みがある。クレジット表記（例: 「VOICEVOX:ずんだもん」）は
動画のCTAシーン・YouTube説明文・TikTokキャプションに入れる（利用規約上の必須事項）。

エンジンはHTTPサーバとして動く。CI（video_post.yml）では公式Dockerイメージ
`voicevox/voicevox_engine:cpu-ubuntu20.04-latest` をジョブ内で起動し、ローカルでは
公式配布のアプリ/エンジンを起動しておく（デフォルト http://127.0.0.1:50021）。
エンジンに繋がらない・合成に失敗した場合は None/False を返し、呼び出し側は
**無音の動画**を書き出して投稿を続行する（音声のためだけに動画投稿そのものを止めない）。

合成手順（VOICEVOX ENGINE API）:
  1. POST /audio_query?text=...&speaker=N  → 読み・アクセント等のクエリJSON
  2. POST /synthesis?speaker=N (body=クエリJSON) → WAV
"""
import os
import shutil
import subprocess

import requests

ENGINE_URL = os.getenv("VOICEVOX_URL", "http://127.0.0.1:50021")

# ずんだもん(ノーマル)=3。TTS_SPEAKER で差し替え可能（例: 四国めたん(ノーマル)=2）。
DEFAULT_SPEAKER = 3

# 標準速度だとショート動画では間延びするため、やや速める。
DEFAULT_SPEED = 1.15

# クレジット表記（VOICEVOX利用規約で必須）。話者を変えたらここも合わせること。
CREDIT = "VOICEVOX:ずんだもん"

# ffprobe が無い環境でのフォールバック用。日本語の読み上げ速度の実測概算（文字/秒）。
CHARS_PER_SECOND = 7.5


def _speaker() -> int:
    try:
        return int(os.getenv("TTS_SPEAKER", DEFAULT_SPEAKER))
    except ValueError:
        return DEFAULT_SPEAKER


def engine_available() -> bool:
    """VOICEVOXエンジンが起動しているか。"""
    try:
        resp = requests.get(f"{ENGINE_URL}/version", timeout=5)
        return resp.ok
    except Exception:
        return False


def synthesize(text: str, out_path: str) -> bool:
    """text を読み上げた WAV を out_path に書き出す。成功したら True。"""
    speaker = _speaker()
    try:
        query = requests.post(
            f"{ENGINE_URL}/audio_query",
            params={"text": text, "speaker": speaker},
            timeout=30,
        )
        if not query.ok:
            print(f"  ⚠ audio_query失敗 HTTP {query.status_code}: {query.text[:200]}")
            return False
        query_json = query.json()
        query_json["speedScale"] = float(os.getenv("TTS_SPEED", DEFAULT_SPEED))

        synthesis = requests.post(
            f"{ENGINE_URL}/synthesis",
            params={"speaker": speaker},
            json=query_json,
            timeout=120,
        )
        if not synthesis.ok:
            print(f"  ⚠ synthesis失敗 HTTP {synthesis.status_code}: {synthesis.text[:200]}")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(synthesis.content)
        return True
    except Exception as e:
        print(f"  ⚠ 音声合成例外: {e}")
        return False


def duration_seconds(audio_path: str, text: str = "") -> float:
    """WAVの長さ（秒）。シーンの尺をナレーションに合わせるために使う。
    ffprobe が無い環境では読み上げ文字数からの概算にフォールバックする。"""
    if shutil.which("ffprobe"):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
    return max(1.5, len(text) / CHARS_PER_SECOND)


def narrate_sections(sections: list, out_dir: str) -> bool:
    """各セクションの narration を音声化し、`audio`（ファイル名）と `durationSec` を
    その場で書き込む。1つでも失敗したら全体を無音扱いにする（一部だけ音が出る動画は
    かえって不自然なため）。"""
    if not engine_available():
        print(f"[tts] VOICEVOXエンジンに接続できません（{ENGINE_URL}）。ナレーションをスキップします")
        return False

    os.makedirs(out_dir, exist_ok=True)
    for i, section in enumerate(sections):
        text = section.get("narration") or ""
        if not text:
            return False
        filename = f"narration_{i}.wav"
        path = os.path.join(out_dir, filename)
        if not synthesize(text, path):
            return False
        section["audio"] = filename
        section["durationSec"] = duration_seconds(path, text)
    return True

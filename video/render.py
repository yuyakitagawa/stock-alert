"""
video/render.py

build_script.py が作った props JSON を Remotion に渡して mp4 を書き出す薄いラッパ。
Remotion のレンダリングは Chrome Headless Shell を使うため、初回実行時のみ
自動ダウンロード（~150MB）が走る。

ナレーション音声（tts.py が生成した mp3）は Remotion の staticFile() 経由でしか
参照できないため、video/remotion/public/ へコピーしてからレンダリングする。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video import se  # noqa: E402

# YouTube / TikTok の音量基準。両プラットフォームとも -14 LUFS 前後に正規化して再生し、
# これより大きい動画は下げるが、小さい動画は上げてくれない。VOICEVOXの生成音声は
# 素のままだと -25 LUFS 前後で、スマホの通常音量ではナレーションがほぼ聞こえないため、
# 書き出し後にここへ合わせる（2026-08-18: 声が聞こえないというオーナー指摘で追加）。
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11.0

REMOTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remotion")
PUBLIC_DIR = os.path.join(REMOTION_DIR, "public")
COMPOSITION_ID = "ArticleShort"

# Remotion に渡してはいけない（ShortProps に無い）補助キー。build_script.py が
# 投稿テキスト用に付けているもので、props として渡すと Remotion 側で未知プロパティになる。
NON_PROP_KEYS = ("articleId", "articleTitle")


def _stage_assets(props: dict, assets_dir: "str | None") -> list:
    """シーンが参照する音声ファイルと背景動画を remotion/public/ へコピーする。
    コピーしたファイルのパス一覧を返す（レンダリング後に削除するため）。"""
    staged = []
    if not assets_dir:
        props.pop("backgroundVideo", None)
        return staged
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    for scene in props.get("scenes", []):
        audio = scene.get("audio")
        if not audio:
            continue
        src = os.path.join(assets_dir, audio)
        dst = os.path.join(PUBLIC_DIR, audio)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            staged.append(dst)
        else:
            # 音声が見つからないシーンは無音で流す（動画自体は止めない）
            print(f"  ⚠ 音声ファイルが見つかりません: {src}（このシーンは無音になります）")
            scene.pop("audio", None)

    # 背景動画: 全編共通（トップレベル）とシーン別の両方に対応する
    bg_names = {props.get("backgroundVideo")} | {
        s.get("backgroundVideo") for s in props.get("scenes", [])
    }
    for bg in sorted(n for n in bg_names if n):
        src = os.path.join(assets_dir, bg)
        if os.path.exists(src):
            dst = os.path.join(PUBLIC_DIR, bg)
            shutil.copyfile(src, dst)
            staged.append(dst)
            continue
        # 見つからない背景はグラデーションにフォールバック
        print(f"  ⚠ 背景動画が見つかりません: {src}（グラデーション背景を使います）")
        if props.get("backgroundVideo") == bg:
            props.pop("backgroundVideo", None)
            props.pop("backgroundVideoDurationSec", None)
        for s in props.get("scenes", []):
            if s.get("backgroundVideo") == bg:
                s.pop("backgroundVideo", None)
                s.pop("backgroundVideoDurationSec", None)
    return staged


def _has_audio_stream(path: str) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _measure_loudness(path: str) -> "dict | None":
    """loudnorm の1パス目。測定値（measured_I等）のdictを返す。失敗時はNone。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path, "-af",
             f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=900,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ⚠ 音量測定に失敗しました（音量はそのまま）: {e}")
        return None
    # 測定結果は stderr の末尾にJSONで出る
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
    if not match:
        print("  ⚠ 音量測定結果を読み取れませんでした（音量はそのまま）")
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        print("  ⚠ 音量測定結果のJSONが壊れています（音量はそのまま）")
        return None


def normalize_loudness(path: str) -> bool:
    """書き出したmp4のナレーション音量を配信基準(-14 LUFS)へ揃える。
    映像は再エンコードせずコピーするso画質は変わらない。失敗しても元のmp4を残してFalseを返す
    （音量のために投稿そのものを止めない）。"""
    if not _has_audio_stream(path):
        return False
    stats = _measure_loudness(path)
    if stats is None:
        return False

    filt = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA={TARGET_LRA}"
        f":measured_I={stats.get('input_i')}:measured_TP={stats.get('input_tp')}"
        f":measured_LRA={stats.get('input_lra')}:measured_thresh={stats.get('input_thresh')}"
        f":offset={stats.get('target_offset', 0)}:linear=true"
    )
    tmp_path = f"{path}.norm.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-i", path, "-c:v", "copy",
             "-af", filt, "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", tmp_path],
            capture_output=True, text=True, timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ⚠ 音量正規化に失敗しました（音量はそのまま）: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

    if result.returncode != 0 or not os.path.exists(tmp_path):
        print("  ⚠ 音量正規化に失敗しました（音量はそのまま）")
        print(result.stderr[-1000:])
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False

    os.replace(tmp_path, path)
    print(f"  🔊 音量を {stats.get('input_i')} LUFS → {TARGET_LUFS} LUFS に正規化しました")
    return True


def has_broken_narration(props: dict) -> bool:
    """読み上げ文が文の途中で切れたまま残っていないか。build_script.py 側でも同じ検査を
    しているが、古い props JSON をそのままレンダリングする経路（手動実行・再実行）でも
    切れた字幕が投稿されないよう、書き出し直前にもう一度見る。"""
    for scene in props.get("scenes", []):
        if "…" in (scene.get("narration") or ""):
            return True
    return False


def render(props: dict, out_path: str, assets_dir: "str | None" = None) -> bool:
    """props で mp4 を書き出す。成功したら True。"""
    if not os.path.isdir(os.path.join(REMOTION_DIR, "node_modules")):
        print("[render] node_modules がありません。video/remotion で npm ci を実行してください")
        return False

    if has_broken_narration(props):
        print("[render] 読み上げ文が文の途中で切れています。台本を作り直してください")
        return False

    render_props = {k: v for k, v in props.items() if k not in NON_PROP_KEYS}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    staged = _stage_assets(render_props, assets_dir)

    # 効果音は毎回その場で生成する（外部素材を持たないのでライセンス確認が要らない）
    render_props["sfx"] = se.ensure_sound_effects(PUBLIC_DIR)
    if render_props["sfx"]:
        staged += [os.path.join(PUBLIC_DIR, name) for name in se.SE_FILENAMES]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(render_props, f, ensure_ascii=False)
        props_path = f.name

    cmd = [
        "npx", "remotion", "render", COMPOSITION_ID, os.path.abspath(out_path),
        f"--props={props_path}",
        "--log=error",
    ]
    try:
        result = subprocess.run(cmd, cwd=REMOTION_DIR, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("[render] レンダリングがタイムアウトしました（30分）")
        return False
    finally:
        os.unlink(props_path)
        for path in staged:
            if os.path.exists(path):
                os.unlink(path)

    if result.returncode != 0:
        print(f"[render] レンダリング失敗 (exit {result.returncode})")
        print(result.stderr[-2000:])
        return False

    if not os.path.exists(out_path):
        print(f"[render] mp4 が生成されませんでした: {out_path}")
        return False

    normalize_loudness(out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[render] ✅ {out_path} ({size_mb:.1f} MB)")
    return True


def main():
    if len(sys.argv) not in (3, 4):
        print("usage: python video/render.py <props.json> <out.mp4> [assets_dir]")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        props = json.load(f)
    assets_dir = sys.argv[3] if len(sys.argv) == 4 else None
    sys.exit(0 if render(props, sys.argv[2], assets_dir) else 1)


if __name__ == "__main__":
    main()

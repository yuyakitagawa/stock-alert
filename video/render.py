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
import shutil
import subprocess
import sys
import tempfile

REMOTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remotion")
PUBLIC_DIR = os.path.join(REMOTION_DIR, "public")
COMPOSITION_ID = "ArticleShort"

# Remotion に渡してはいけない（ShortProps に無い）補助キー。build_script.py が
# 投稿テキスト用に付けているもので、props として渡すと Remotion 側で未知プロパティになる。
NON_PROP_KEYS = ("articleId", "articleTitle")


def _stage_audio(props: dict, audio_dir: "str | None") -> list:
    """シーンが参照する音声ファイルを remotion/public/ へコピーする。
    コピーしたファイルのパス一覧を返す（レンダリング後に削除するため）。"""
    staged = []
    if not audio_dir:
        return staged
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    for scene in props.get("scenes", []):
        audio = scene.get("audio")
        if not audio:
            continue
        src = os.path.join(audio_dir, audio)
        dst = os.path.join(PUBLIC_DIR, audio)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            staged.append(dst)
        else:
            # 音声が見つからないシーンは無音で流す（動画自体は止めない）
            print(f"  ⚠ 音声ファイルが見つかりません: {src}（このシーンは無音になります）")
            scene.pop("audio", None)
    return staged


def render(props: dict, out_path: str, audio_dir: "str | None" = None) -> bool:
    """props で mp4 を書き出す。成功したら True。"""
    if not os.path.isdir(os.path.join(REMOTION_DIR, "node_modules")):
        print("[render] node_modules がありません。video/remotion で npm ci を実行してください")
        return False

    render_props = {k: v for k, v in props.items() if k not in NON_PROP_KEYS}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    staged = _stage_audio(render_props, audio_dir)

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

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[render] ✅ {out_path} ({size_mb:.1f} MB)")
    return True


def main():
    if len(sys.argv) not in (3, 4):
        print("usage: python video/render.py <props.json> <out.mp4> [audio_dir]")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        props = json.load(f)
    audio_dir = sys.argv[3] if len(sys.argv) == 4 else None
    sys.exit(0 if render(props, sys.argv[2], audio_dir) else 1)


if __name__ == "__main__":
    main()

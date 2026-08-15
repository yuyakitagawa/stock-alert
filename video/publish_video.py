"""
video/publish_video.py

自動動画投稿のオーケストレーター。1日1回、以下を順に実行する:
  1. build_script.py  … microCMSの新着記事×注目枠から1件選び、Claudeで縦動画の台本を作る
  2. render.py        … Remotion で 1080x1920 / 20秒の mp4 を書き出す
  3. youtube_client   … YouTube Shorts へアップロード
  4. tiktok_client    … TikTok へアップロード（既定は下書き。詳細は tiktok_client.py 参照）

対象記事が無い日は何も投稿せずに終了する（毎日必ず出す運用にはしない）。
片方のプラットフォームの認証情報が未設定・アップロード失敗でも、もう片方は続行する。

使い方:
  python video/publish_video.py                  # 台本→レンダリング→両方へ投稿
  python video/publish_video.py --dry-run        # 台本の生成まで（レンダリングも投稿もしない）
  python video/publish_video.py --render-only    # 台本→レンダリングまで（投稿しない）
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

from video import build_script, render, tts  # noqa: E402
from video import tiktok_client, youtube_client  # noqa: E402

OUT_DIR = os.path.join(os.path.expanduser("~/stock-alert"), "video", "out")


def run(dry_run: bool = False, render_only: bool = False, keep_video: bool = False) -> int:
    props = build_script.build(dry_run=dry_run)
    if props is None:
        print("[publish_video] 投稿対象がないため終了します")
        return 0

    if dry_run:
        print("[publish_video] --dry-run のためレンダリング・投稿は行いません")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUT_DIR, f"short_{props.get('stockCode', 'x')}_{stamp}.mp4")

    # ナレーション合成。失敗したら全編無音のまま続行する（音声のために投稿を止めない）。
    audio_dir = os.path.join(OUT_DIR, f"audio_{stamp}")
    if tts.narrate_sections(props["scenes"], audio_dir):
        print(f"[publish_video] ナレーション {len(props['scenes'])}本を合成しました")
    else:
        print("[publish_video] ナレーション無し（無音）で続行します")
        audio_dir = None

    if not render.render(props, video_path, audio_dir=audio_dir):
        print("[publish_video] レンダリングに失敗したため投稿を中止します")
        return 1

    with open(os.path.join(OUT_DIR, f"props_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False, indent=2)

    if render_only:
        print(f"[publish_video] --render-only のため投稿しません: {video_path}")
        return 0

    configured = [
        name for name, keys in (
            ("YouTube", ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")),
            ("TikTok", ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN")),
        )
        if all(os.getenv(k) for k in keys)
    ]

    posted = 0
    if youtube_client.upload(video_path, props):
        posted += 1
    if tiktok_client.upload(video_path, props):
        posted += 1

    print(f"[publish_video] {posted}プラットフォームへ投稿しました（対象記事: {props.get('articleTitle')}）")

    if not keep_video and posted > 0:
        os.remove(video_path)

    if not configured:
        # Secretsを1つも登録していない段階。動画は作れているので失敗扱いにはしない
        # （毎日ワークフローが赤くなって本当の失敗通知が埋もれるのを防ぐ）。
        print("[publish_video] 投稿先の認証情報が未登録のため、動画の生成のみで終了します")
        return 0
    # 認証情報があるのに1件も投稿できていない＝実際の失敗なので気付けるようにする
    return 0 if posted > 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="台本生成までで止める（レンダリング・投稿なし）")
    p.add_argument("--render-only", action="store_true", help="mp4の書き出しまでで止める（投稿なし）")
    p.add_argument("--keep-video", action="store_true", help="投稿後もmp4を削除しない")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run, render_only=args.render_only, keep_video=args.keep_video))


if __name__ == "__main__":
    main()

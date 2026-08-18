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

from video import background, build_script, line_notify, render, tts  # noqa: E402
from video import tiktok_client, youtube_client  # noqa: E402

OUT_DIR = os.path.join(os.path.expanduser("~/stock-alert"), "video", "out")


def run(dry_run: bool = False, render_only: bool = False, keep_video: bool = False,
        article_id: str = "") -> int:
    props = build_script.build(dry_run=dry_run, article_id=article_id)
    if props is None:
        print("[publish_video] 投稿対象がないため終了します")
        return 0

    if dry_run:
        print("[publish_video] --dry-run のためレンダリング・投稿は行いません")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUT_DIR, f"short_{props.get('stockCode', 'x')}_{stamp}.mp4")

    # 音声・背景動画などレンダリングに添える素材の置き場（Remotionのpublic/へ一時コピーされる）
    assets_dir = os.path.join(OUT_DIR, f"assets_{stamp}")

    # ナレーション合成。失敗したら全編無音のまま続行する（音声のために投稿を止めない）。
    has_audio = tts.narrate_sections(props["scenes"], assets_dir)
    if has_audio:
        print(f"[publish_video] ナレーション {len(props['scenes'])}本を合成しました")
    else:
        print("[publish_video] ナレーション無し（無音）で続行します")

    # 背景映像（Pexels、自然＋人物のプールからシーンごとにランダム割当）。
    # 取得できなければグラデーション背景で続行する。
    pool = background.fetch_pool(assets_dir)
    background.assign_backgrounds(props["scenes"], pool)

    if not render.render(props, video_path,
                         assets_dir=assets_dir if (has_audio or pool) else None):
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
    youtube_id = youtube_client.upload(video_path, props)
    if youtube_id:
        posted += 1
        # 動画公開をXへもクロス投稿する（X未認証・失敗でも動画投稿の成否には影響させない）。
        # requests_oauthlib未導入の環境でも動画パイプラインが落ちないよう遅延importにする。
        try:
            from web import x_client

            if x_client.post_video_tweet(props, youtube_id):
                print("  🐦 動画リンクをXへクロス投稿しました")
        except Exception as e:
            print(f"  ⚠ Xクロス投稿に失敗しましたが動画投稿は完了しています: {e}")
    tiktok_publish_id = tiktok_client.upload(video_path, props)
    if tiktok_publish_id:
        posted += 1

    print(f"[publish_video] {posted}プラットフォームへ投稿しました（対象記事: {props.get('articleTitle')}）")

    # TikTokの下書きはキャプションを運べないため、公開時に貼る文面をLINEでスマホへ送る
    line_notify.notify(props, tiktok_client.build_caption(props),
                       youtube_id=youtube_id, tiktok_publish_id=tiktok_publish_id)

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
    p.add_argument("--article-id", default="",
                   help="記事ID指定（通常の新着×注目枠選定を使わず、この記事を動画にする。記事URLでも可）")
    args = p.parse_args()
    sys.exit(run(dry_run=args.dry_run, render_only=args.render_only, keep_video=args.keep_video,
                 article_id=args.article_id))


if __name__ == "__main__":
    main()

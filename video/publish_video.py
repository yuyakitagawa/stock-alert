"""
video/publish_video.py

自動動画投稿のオーケストレーター。1日1回、以下を順に実行する:
  1. build_script.py  … microCMSの新着記事×注目枠から1件選び、Claudeで縦動画の台本を作る
  2. render.py        … Remotion で 1080x1920 の mp4 を書き出す
  3. youtube_client   … YouTube Shorts へアップロード

対象記事が無い日は何も投稿せずに終了する（毎日必ず出す運用にはしない）。
※TikTok投稿は2026-08-20に完全撤去（自アカウント用途はTikTokの本番審査ポリシー対象外のため。
  経緯は docs/tiktok_review.md）。

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

from video import background, build_script, line_notify, render, thumbnail, tts  # noqa: E402
from video import youtube_client, youtube_metrics  # noqa: E402

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

    # レンダリング前の認証チェック。トークンが死んでいる日に230秒かけて書き出しても
    # 動画の行き先が無い（2026-08-25はここで74.9MBが丸損した）。--render-only や
    # Secrets未登録のときは動画の生成自体が目的なので確認しない。
    if not render_only and youtube_client.is_configured() and not youtube_client.check_auth():
        print("[publish_video] YouTubeの認証が通らないため、レンダリング前に中止します")
        return 1

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

    posted = 0
    youtube_id = youtube_client.upload(video_path, props)
    if youtube_id:
        posted += 1
        # 公開した事実をその場でSupabaseに残す。当日のハートビートが「今日、動画が出たか」を
        # 数えられるようにするため（統計の収集は手動実行なので、それ待ちだと0本に見える）。
        youtube_metrics.record_upload(youtube_id, youtube_client.build_title(props))
        # 検索結果・チャンネルページ用のカスタムサムネイル（Canva台紙＋銘柄・金額）。
        # 失敗しても動画は公開済みなので成否は投稿数に影響させない
        thumb = thumbnail.compose(props, os.path.join(OUT_DIR, f"thumb_{stamp}.png"))
        if thumb:
            youtube_client.set_thumbnail(youtube_id, thumb)
        # 動画公開をXへもクロス投稿する（X未認証・失敗でも動画投稿の成否には影響させない）。
        # requests_oauthlib未導入の環境でも動画パイプラインが落ちないよう遅延importにする。
        try:
            from web import x_client

            if x_client.post_video_tweet(props, youtube_id):
                print("  🐦 動画リンクをXへクロス投稿しました")
        except Exception as e:
            print(f"  ⚠ Xクロス投稿に失敗しましたが動画投稿は完了しています: {e}")
    print(f"[publish_video] {posted}プラットフォームへ投稿しました（対象記事: {props.get('articleTitle')}）")

    # 投稿完了をLINEでスマホへ通知する
    line_notify.notify(props, youtube_id=youtube_id)

    if not keep_video and posted > 0:
        os.remove(video_path)

    if not youtube_client.is_configured():
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

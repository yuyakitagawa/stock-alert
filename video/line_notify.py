"""
video/line_notify.py

TikTokへ下書きを送信したあと、公開時に貼るキャプション文をLINEでスマホに届ける。

TikTokの下書き送信API（inbox）はキャプションを一緒に送れない仕様のため、
公開操作をするオーナーのスマホにコピペ用の文面を送っておく（TikTokの通知で下書きを開く
→ LINEからキャプションをコピペ → 公開、の30秒運用にする）。アプリ審査が通って
直接公開（TIKTOK_DIRECT_POST=1）に切り替えたら、この通知は公開報告として残す。

認証はブログ・市況アラートと同じLINE Messaging APIのチャネルを再利用する
（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）。未設定なら黙ってスキップし、
動画投稿そのものは止めない。
"""
import os

import requests

PUSH_URL = "https://api.line.me/v2/bot/message/push"


def build_message(props: dict, caption: str, youtube_id: "str | None",
                  tiktok_publish_id: "str | None") -> str:
    lines = [f"🎬 今日の動画: {props.get('articleTitle') or props.get('stockName', '')}"]
    if youtube_id:
        lines.append(f"▶️ YouTube公開済み: https://youtube.com/shorts/{youtube_id}")
    if tiktok_publish_id:
        lines.append("🎵 TikTokに下書きを送りました。アプリの通知から開いて、"
                     "下のキャプションを貼り付けて公開してください。")
        lines.append("")
        lines.append("--- キャプション（コピー用） ---")
        lines.append(caption)
    return "\n".join(lines)


def notify(props: dict, caption: str, youtube_id: "str | None" = None,
           tiktok_publish_id: "str | None" = None) -> bool:
    """投稿結果をLINEでプッシュする。認証情報未設定・失敗時はFalse（他処理は止めない）。"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id:
        print("[line_notify] LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 未設定のため通知をスキップします")
        return False
    if not youtube_id and not tiktok_publish_id:
        return False  # 何も投稿できていない日は通知しない

    message = build_message(props, caption, youtube_id, tiktok_publish_id)
    try:
        resp = requests.post(
            PUSH_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            json={"to": user_id, "messages": [{"type": "text", "text": message}]},
            timeout=15,
        )
        if resp.ok:
            print("[line_notify] 📱 LINE通知を送信しました")
            return True
        print(f"[line_notify] ⚠ LINE送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"[line_notify] ⚠ LINE送信例外: {e}")
        return False

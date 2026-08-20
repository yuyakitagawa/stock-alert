"""
video/line_notify.py

動画の投稿完了をLINEでスマホに通知する（毎日の実行結果の確認用）。

認証はブログ・市況アラートと同じLINE Messaging APIのチャネルを再利用する
（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）。未設定なら黙ってスキップし、
動画投稿そのものは止めない。
"""
import os

import requests

PUSH_URL = "https://api.line.me/v2/bot/message/push"


def build_message(props: dict, youtube_id: "str | None") -> str:
    lines = [f"🎬 今日の動画: {props.get('articleTitle') or props.get('stockName', '')}"]
    if youtube_id:
        lines.append(f"▶️ YouTube公開済み: https://youtube.com/shorts/{youtube_id}")
    return "\n".join(lines)


def notify(props: dict, youtube_id: "str | None" = None) -> bool:
    """投稿結果をLINEでプッシュする。認証情報未設定・失敗時はFalse（他処理は止めない）。"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id:
        print("[line_notify] LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 未設定のため通知をスキップします")
        return False
    if not youtube_id:
        return False  # 投稿できていない日は通知しない

    message = build_message(props, youtube_id)
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

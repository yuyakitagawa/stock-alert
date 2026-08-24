"""
video/line_notify.py

動画の投稿完了をLINEでスマホに通知する（毎日の実行結果の確認用）。

送信そのものは lib/notify.py（LINE Messaging APIの共通口）に任せる。認証は
ブログ・市況アラート・異常通知と同じチャネル（LINE_CHANNEL_ACCESS_TOKEN /
LINE_USER_ID）で、未設定なら黙ってスキップし動画投稿は止めない。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import notify as line_api  # noqa: E402

PUSH_URL = line_api.PUSH_URL


def build_message(props: dict, youtube_id: "str | None") -> str:
    lines = [f"🎬 今日の動画: {props.get('articleTitle') or props.get('stockName', '')}"]
    if youtube_id:
        lines.append(f"▶️ YouTube公開済み: https://youtube.com/shorts/{youtube_id}")
    return "\n".join(lines)


def notify(props: dict, youtube_id: "str | None" = None) -> bool:
    """投稿結果をLINEでプッシュする。認証情報未設定・失敗時はFalse（他処理は止めない）。
    投稿できなかった日は通知しない（欠落は tools/output_heartbeat.py が当日まとめて見る）。"""
    if not youtube_id:
        return False
    return line_api.push(build_message(props, youtube_id))

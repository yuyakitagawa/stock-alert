"""
lib/notify.py

パイプラインの異常をLINE（Messaging API push）でスマホに届ける共通モジュール。

なぜ必要か（2026-08-24の無言停止）:
  Anthropic APIが月次上限（HTTP 400 invalid_request_error）に達した状態で edinet_blog.yml が
  毎時回り続け、記事生成が全件失敗しても各ステップが continue-on-error のためワークフローは緑。
  記事が0件なので video_post.yml も「投稿対象がないため終了」で無言終了し、ブログも動画も
  丸一日止まっているのに通知は一切出なかった。唯一の見張りである日次ログレビュー
  （tools/daily_log_review.py）はClaude自身を使うため、同じ上限で一緒に落ちていた。

設計（見張りは壊れたものに依存させない）:
  - Claude・GitHub Actionsの成否判定に依存せず、LINE Messaging APIを直接叩くだけ。
  - 認証情報（LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID）が無ければ黙って False を返す。
  - 送信の失敗・例外は握りつぶす。通知の失敗で本処理を止めない。

使い方:
    from lib import notify
    notify.error("ブログ生成", "Anthropic APIの利用上限に到達", detail=str(e))
    notify.warn("動画", "投稿対象の記事が0件")

CLIから（GitHub Actions の `if: failure()` ステップ用）:
    python -m lib.notify "🚨 EDINET Blog Hourly が失敗しました" --url "$RUN_URL"
"""
import os
import sys

import requests

PUSH_URL = "https://api.line.me/v2/bot/message/push"

# LINEの1メッセージ上限は5,000文字。余裕を持って切る。
MAX_CHARS = 4_000
# エラー本文（スタックトレース等）はLINEでは読みにくいので先頭だけ載せる。
DETAIL_MAX_CHARS = 400


def is_configured() -> bool:
    """LINE通知の認証情報がそろっているか。"""
    return bool(os.getenv("LINE_CHANNEL_ACCESS_TOKEN") and os.getenv("LINE_USER_ID"))


def push(text: str) -> bool:
    """LINEへプッシュ送信する。未設定・失敗時は False（例外は投げない）。"""
    token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    user_id = os.getenv("LINE_USER_ID")
    if not token or not user_id:
        print("[notify] LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID 未設定のため通知をスキップします")
        return False
    if not (text or "").strip():
        return False
    body = text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"
    try:
        resp = requests.post(
            PUSH_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            json={"to": user_id, "messages": [{"type": "text", "text": body}]},
            timeout=15,
        )
        if resp.ok:
            print("[notify] 📱 LINE通知を送信しました")
            return True
        print(f"[notify] ⚠ LINE送信失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:  # ネットワーク断でも本処理は止めない
        print(f"[notify] ⚠ LINE送信例外: {e}")
        return False


def build_message(mark: str, where: str, message: str, detail: str = "",
                  url: str = "") -> str:
    """通知本文。1行目で「どこが」「何が」起きたかが分かるようにする。"""
    lines = [f"{mark} {where}", message.strip()]
    if detail:
        d = detail.strip().replace("\n", " ")
        lines += ["", d[:DETAIL_MAX_CHARS] + ("…" if len(d) > DETAIL_MAX_CHARS else "")]
    if url:
        lines += ["", url]
    return "\n".join(lines)


def error(where: str, message: str, detail: str = "", url: str = "") -> bool:
    """止まった・落ちたときの通知。"""
    return push(build_message("🚨", where, message, detail, url))


def warn(where: str, message: str, detail: str = "", url: str = "") -> bool:
    """落ちてはいないが期待どおりに出ていないときの通知。"""
    return push(build_message("⚠️", where, message, detail, url))


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="LINEへ1通プッシュする")
    p.add_argument("text", help="送信する本文")
    p.add_argument("--url", default="", help="末尾に付けるURL（Actionsの実行ログ等）")
    args = p.parse_args()
    body = args.text + (f"\n\n{args.url}" if args.url else "")
    return 0 if push(body) else 1


if __name__ == "__main__":
    sys.exit(main())

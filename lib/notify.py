"""
lib/notify.py

パイプラインの異常をLINE（Messaging API push）でスマホに届ける共通モジュール。

なぜ必要か（2026-08-24の無言停止）:
  Anthropic APIが月次上限（HTTP 400 invalid_request_error）に達した状態で edinet_blog.yml が
  毎時回り続け、記事生成が全件失敗しても各ステップが continue-on-error のためワークフローは緑。
  記事が0件なので video_post.yml も「投稿対象がないため終了」で無言終了し、ブログも動画も
  丸一日止まっているのに通知は一切出なかった。唯一の見張りだった日次ログレビュー
  （当時の tools/daily_log_review.py。2026-08-29に削除）はClaude自身を使うため、同じ上限で
  一緒に落ちていた。

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
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

PUSH_URL = "https://api.line.me/v2/bot/message/push"

# LINEの1メッセージ上限は5,000文字。余裕を持って切る。
MAX_CHARS = 4_000
# エラー本文（スタックトレース等）はLINEでは読みにくいので先頭だけ載せる。
DETAIL_MAX_CHARS = 400
# 同じ原因の通知を抑制する既定の時間幅。毎時のワークフローが同じ理由で落ち続けても
# 1日1通に収める（2026-08-24のAPI上限超過なら本来13通になっていた）。
DEDUPE_WINDOW_HOURS = 20.0


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


def _recent_send(dedupe_key: str, window_hours: float) -> "dict | None":
    """window_hours以内に同じdedupe_keyで送っていればその行を返す。
    判定できないとき（Supabase未設定・障害）は None＝送る側に倒す。
    見張りの通知は「多い」より「来ない」ほうが致命的なため。"""
    try:
        from lib import supabase_client as sb

        row = sb.select_one("notify_log", f"dedupe_key=eq.{dedupe_key}&select=*")
        if not row:
            return None
        # 窓が無限（once）なら行があること自体が「送信済み」。last_sent_at は見ない。
        if window_hours == math.inf:
            return row
        if not row.get("last_sent_at"):
            return None
        last = datetime.fromisoformat(str(row["last_sent_at"]).replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - last < timedelta(hours=window_hours):
            return row
        return None
    except Exception as e:
        print(f"[notify] ⚠ 重複判定に失敗したため送信します: {e}")
        return None


def _record_send(dedupe_key: str, text: str, prev: "dict | None") -> None:
    try:
        from lib import supabase_client as sb

        sb.upsert("notify_log", [{
            "dedupe_key": dedupe_key,
            "last_sent_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sent_count": int((prev or {}).get("sent_count") or 0) + 1,
            "last_text": text[:1000],
        }], on_conflict="dedupe_key")
    except Exception as e:
        print(f"[notify] ⚠ 送信履歴の記録に失敗（通知自体は送信済み）: {e}")


def push_once(dedupe_key: str, text: str, window_hours: float = DEDUPE_WINDOW_HOURS) -> bool:
    """同じdedupe_keyでwindow_hours以内に送っていなければ送る。

    毎時のワークフローが同じ原因で失敗し続けるとき（API上限超過なら復旧まで最大13便）、
    そのまま通知すると1日13通届いて通知疲れを起こす。抑制した回数は notify_log.sent_count に
    積んでいくので、後から「何便ぶん黙ったか」は追える。
    """
    prev = _recent_send(dedupe_key, window_hours)
    if prev is not None:
        span = "既に" if window_hours == math.inf else f"直近{window_hours:g}時間に"
        print(f"[notify] {span}同じ通知（{dedupe_key}）を送信済みのため抑制します")
        _record_send(dedupe_key, text, prev)  # 抑制した回数も数える
        return False
    if not push(text):
        return False
    _record_send(dedupe_key, text, None)
    return True


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


def error(where: str, message: str, detail: str = "", url: str = "",
          dedupe_key: str = "") -> bool:
    """止まった・落ちたときの通知。dedupe_keyを渡すと同じ原因の連投を抑える。"""
    body = build_message("🚨", where, message, detail, url)
    return push_once(dedupe_key, body) if dedupe_key else push(body)


def warn(where: str, message: str, detail: str = "", url: str = "",
         dedupe_key: str = "") -> bool:
    """落ちてはいないが期待どおりに出ていないときの通知。"""
    body = build_message("⚠️", where, message, detail, url)
    return push_once(dedupe_key, body) if dedupe_key else push(body)


def once(dedupe_key: str, text: str) -> bool:
    """同じ `dedupe_key` では二度と送らない通知。送ったら True。

    残枠50/80/100%のような「一度きりの状態」を毎時のジョブから鳴らすと、同じ内容が
    1日に何十通も届いてかえって見なくなる。窓を持たない `push_once`（=窓が無限）として
    実装しており、送信済みかどうかは同じ `notify_log` に残る
    （プロセス内フラグでは毎時の別プロセスをまたげない）。

    DBが引けないときは `push_once` と同じく送る側に倒す。通知が重複する不便より、
    鳴らないほうが危険。
    """
    return push_once(str(dedupe_key), text, window_hours=math.inf)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="LINEへ1通プッシュする")
    p.add_argument("text", help="送信する本文")
    p.add_argument("--url", default="", help="末尾に付けるURL（Actionsの実行ログ等）")
    p.add_argument("--dedupe-key", default="",
                   help="同じキーの通知を --dedupe-hours 以内は再送しない（毎時ワークフローの連投抑制）")
    p.add_argument("--dedupe-hours", type=float, default=DEDUPE_WINDOW_HOURS,
                   help=f"重複抑制の時間幅（既定{DEDUPE_WINDOW_HOURS:g}時間）")
    args = p.parse_args()
    body = args.text + (f"\n\n{args.url}" if args.url else "")
    if args.dedupe_key:
        # 抑制された場合も「送るべきものが無かった」だけなので正常終了にする
        push_once(args.dedupe_key, body, args.dedupe_hours)
        return 0
    return 0 if push(body) else 1


if __name__ == "__main__":
    sys.exit(main())

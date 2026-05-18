"""
Supabase の push_subscriptions を参照し、売りシグナルが出た保有株がある
サブスクリプションに Web Push 通知を送る（Step 6）。

実行条件: export_to_web.py が実行済みで web_rankings に本日データがある前提。
依存: requests, pywebpush, python-dotenv
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SITE_URL = os.getenv("SITE_URL", "https://stocksignal.jp")
INTERNAL_SEND_SECRET = os.getenv("INTERNAL_SEND_SECRET", "")
DRY_RUN = "--dry-run" in sys.argv

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[send_user_alerts] Supabase 環境変数未設定。スキップします。")
    sys.exit(0)


def main() -> None:
    today = date.today().isoformat()
    print(f"[send_user_alerts] {today} Web Push 送信開始 {'(dry-run)' if DRY_RUN else ''}")

    if DRY_RUN:
        # dry-run: /api/push/send を呼ばずにサブスクリプション数だけ表示
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/push_subscriptions?enabled=eq.true&select=id",
            headers=headers,
            timeout=10,
        )
        count = len(resp.json()) if resp.ok else "?"
        print(f"[send_user_alerts] [dry-run] 有効サブスクリプション: {count} 件。実際の送信はスキップ。")
        return

    # Next.js の /api/push/send を叩く（Web Push 送信ロジックは Next.js 側に集約）
    send_url = f"{SITE_URL}/api/push/send"
    headers = {"Content-Type": "application/json"}
    if INTERNAL_SEND_SECRET:
        headers["x-internal-secret"] = INTERNAL_SEND_SECRET

    resp = requests.post(send_url, headers=headers, json={}, timeout=60)
    if resp.ok:
        data = resp.json()
        print(
            f"[send_user_alerts] 完了 — 送信: {data.get('sent', '?')}件, "
            f"スキップ: {data.get('skipped', '?')}件"
        )
    else:
        print(f"[send_user_alerts] /api/push/send エラー: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()

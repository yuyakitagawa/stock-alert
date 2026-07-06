"""
web/ スクリプト共通ヘルパー（Supabase REST + LINE Push）
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")


def sb_get(path: str) -> list[dict]:
    """Supabase REST API からページネーションで全件取得する。"""
    import re
    # 既存の limit/offset パラメータを除去して重複を防ぐ
    path = re.sub(r'[?&]limit=\d+', '', path)
    path = re.sub(r'[?&]offset=\d+', '', path)
    path = path.rstrip('?&')
    sep = "&" if "?" in path else "?"
    base_url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    page_size = 1000
    out: list[dict] = []
    offset = 0
    while True:
        url = f"{base_url}{sep}limit={page_size}&offset={offset}"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
        out.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return out


def push_line(user_id: str, text: str) -> bool:
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定。送信スキップ。")
        return False
    text = text[:5000]
    try:
        resp = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            },
            json={"to": user_id, "messages": [{"type": "text", "text": text}]},
            timeout=15,
        )
        if resp.ok:
            return True
        print(f"LINE push failed ({user_id[:8]}...): {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"LINE push error ({user_id[:8]}...): {e}")
        return False

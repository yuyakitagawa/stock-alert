"""
generate_descriptions.py
全銘柄の会社説明を Claude Haiku で一括生成し、Supabase にキャッシュする。

- web_stock_meta(code/name/sector) を母集団とし、まだ会社説明
  (ai_analyses model_version=company-desc-v1) が無い銘柄だけ生成する（差分のみ）。
- 手動説明（スプシ由来）は既に存在するので上書きしない（差分生成なので自然に保護）。
- 並列生成＋バッチupsert。--limit で1回の生成数を制限（日次で少しずつ全銘柄を網羅）。

使い方:
  python3 web/generate_descriptions.py            # 既定 --limit 400 件まで生成
  python3 web/generate_descriptions.py --limit 50 # 50件だけ
  python3 web/generate_descriptions.py --all      # 未生成すべて
  python3 web/generate_descriptions.py --dry-run  # 生成せず対象数だけ表示
"""
import json
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")

MODEL       = "claude-haiku-4-5-20251001"   # description API と同じ
MODEL_VER   = "company-desc-v1"
CACHE_DATE  = "1970-01-01"
DEFAULT_LIMIT = 2000
# 組織レート上限(50 req/分)に収めるため、1回の呼び出しで複数銘柄をまとめて生成する。
BATCH_N     = 20    # 1リクエストあたりの銘柄数
WORKERS     = 3     # 並列数（BATCH_N×WORKERS で実効スループット）
MAX_RETRY   = 5

SYSTEM = (
    "あなたは日本株の企業情報に詳しいアナリストです。各企業について、"
    "どんな事業を行っている会社かを100〜150字の日本語で説明します。"
    "社名と業種から確実に言える範囲で簡潔に述べ、不確かな数値や固有の製品名は"
    "推測で書きません。"
)


def _headers() -> dict:
    return {
        "apikey":        SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",
    }


def _sb_get(table: str, query: str) -> list[dict]:
    base = f"{SUPABASE_URL}/rest/v1/{table}"
    out, page, off = [], 1000, 0
    while True:
        sep = "&" if query else ""
        r = requests.get(f"{base}?{query}{sep}limit={page}&offset={off}",
                         headers=_headers(), timeout=30)
        if not r.ok:
            print(f"[gen] {table} 読み出し失敗: {r.status_code} {r.text[:200]}")
            break
        rows = r.json()
        out.extend(rows)
        if len(rows) < page:
            break
        off += page
    return out


def _existing_codes() -> set[str]:
    rows = _sb_get("ai_analyses",
                   f"model_version=eq.{MODEL_VER}&select=code,summary")
    # summary が空のものは未生成扱いにして再生成対象に含める
    return {str(r["code"]) for r in rows if str(r.get("summary") or "").strip()}


def _targets(limit: int | None) -> list[dict]:
    meta = _sb_get("web_stock_meta", "select=code,name,sector")
    have = _existing_codes()
    todo = [m for m in meta
            if str(m.get("code")) not in have and str(m.get("name") or "").strip()]
    todo.sort(key=lambda m: str(m.get("code")))
    return todo if limit is None else todo[:limit]


def _parse_json(text: str) -> dict:
    """コードフェンス等を剥がして {コード: 説明} を取り出す。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def _generate_batch(client, stocks: list[dict]) -> list[dict]:
    """複数銘柄を1リクエストで生成。{コード:説明} のJSONを受け取り行リストを返す。"""
    import anthropic
    lines = "\n".join(
        f"- {str(s.get('code'))} {str(s.get('name') or '').strip()} "
        f"({str(s.get('sector') or '不明').strip()})"
        for s in stocks)
    user = (
        "以下の各企業について、どんな事業の会社かを100〜150字の日本語で説明してください。\n"
        '出力は {"銘柄コード": "説明文", ...} のJSONオブジェクトのみ。前置き・補足・'
        "コードフェンスは不要です。\n\n" + lines)
    by_code = {str(s.get("code")): s for s in stocks}

    for attempt in range(MAX_RETRY):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            text = (msg.content[0].text or "") if msg.content else ""
            parsed = _parse_json(text)
            rows = []
            for code, desc in parsed.items():
                code = str(code).strip()
                desc = str(desc).strip()
                if code in by_code and len(desc) >= 20:
                    rows.append({"code": code, "date": CACHE_DATE, "summary": desc,
                                 "bull_points": [], "bear_points": [],
                                 "model_version": MODEL_VER})
            return rows
        except anthropic.RateLimitError:
            wait = min(2 ** attempt * 5, 60)
            print(f"[gen] 429 rate limit. {wait}s 待機して再試行 ({attempt+1}/{MAX_RETRY})")
            time.sleep(wait)
        except Exception as e:
            print(f"[gen] バッチ生成失敗（{stocks[0].get('code')}〜）: {e}")
            return []
    return []


def _upsert(rows: list[dict]) -> None:
    if not rows:
        return
    r = requests.post(f"{SUPABASE_URL}/rest/v1/ai_analyses",
                      headers=_headers(), json=rows, timeout=30)
    if not r.ok:
        print(f"[gen] upsert失敗: {r.status_code} {r.text[:200]}")


def main() -> None:
    args = sys.argv[1:]
    dry  = "--dry-run" in args
    limit: int | None
    if "--all" in args:
        limit = None
    elif "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    else:
        limit = DEFAULT_LIMIT

    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        print("[gen] SUPABASE 未設定。中止。"); sys.exit(1)

    todo = _targets(limit)
    total_meta_msg = f"（今回 {len(todo)}件）"
    print(f"[gen] 会社説明 未生成の対象: {len(todo)}件 {total_meta_msg}")
    if dry or not todo:
        print("[gen] dry-run または対象なしで終了。"); return

    if not ANTHROPIC_API_KEY:
        print("[gen] ANTHROPIC_API_KEY 未設定。中止。"); sys.exit(1)
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    chunks = [todo[i:i + BATCH_N] for i in range(0, len(todo), BATCH_N)]
    print(f"[gen] {len(chunks)}バッチ（{BATCH_N}件/回・並列{WORKERS}）で生成開始")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_generate_batch, client, ch): i for i, ch in enumerate(chunks)}
        for fut in as_completed(futs):
            rows = fut.result()
            if rows:
                _upsert(rows); done += len(rows)
                print(f"[gen] {done}/{len(todo)} 生成・保存済み")
    print(f"[gen] 完了: {done}/{len(todo)} 件の会社説明を生成・保存")


if __name__ == "__main__":
    main()

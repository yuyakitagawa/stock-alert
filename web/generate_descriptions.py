"""
generate_descriptions.py
全銘柄の会社説明を生成し、Supabase にキャッシュする。

生成フロー（2段階）:
  Phase 1: Yahoo Finance JP の「特色」テキストをスクレイプ（会社四季報データ・事実確実）
  Phase 2: スクレイプ失敗分のみ Claude Haiku でバッチ生成（フォールバック）

- web_stock_meta(code/name/sector) を母集団とする。
- 差分生成（--refresh なしは未生成のみ）。
- 手動説明（スプシ由来）は既存のため自然に保護される。

使い方:
  python3 web/generate_descriptions.py            # 未生成分を最大2000件補完
  python3 web/generate_descriptions.py --limit 50 # 50件だけ
  python3 web/generate_descriptions.py --all      # 未生成すべて
  python3 web/generate_descriptions.py --refresh  # 全銘柄を再スクレイプ（事実精度向上）
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
WORKERS     = 3     # 並列数
MAX_RETRY   = 5

# Yahoo Finance JP スクレイプ設定
SCRAPE_UA      = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120"
SCRAPE_DELAY   = 1.2   # 1銘柄あたりの待機秒（3並列で実効3req/s）
SCRAPE_WORKERS = 3

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


def _targets(limit: int | None, refresh: bool = False) -> list[dict]:
    meta = _sb_get("web_stock_meta", "select=code,name,sector")
    valid = [m for m in meta if str(m.get("name") or "").strip()]
    if not refresh:
        have = _existing_codes()
        valid = [m for m in valid if str(m.get("code")) not in have]
    valid.sort(key=lambda m: str(m.get("code")))
    return valid if limit is None else valid[:limit]


# ─── Phase 1: Yahoo Finance JP スクレイプ ────────────────────────────────────

def _fetch_tokushoku(stock: dict) -> tuple[str, str | None]:
    """Yahoo Finance JPの「特色」テキストを取得。(code, text|None) を返す。"""
    code = str(stock.get("code"))
    url = f"https://finance.yahoo.co.jp/quote/{code}.T/profile"
    try:
        r = requests.get(url, headers={"User-Agent": SCRAPE_UA}, timeout=12)
        time.sleep(SCRAPE_DELAY)
        if not r.ok:
            return code, None
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        for h2 in soup.find_all("h2"):
            if h2.get_text(strip=True) == "特色":
                p = h2.find_next("p")
                if p:
                    text = p.get_text(strip=True)
                    text = re.sub(r"^【特色】\s*", "", text)
                    if len(text) >= 10:
                        return code, text
        return code, None
    except Exception:
        return code, None


def _scrape_phase(stocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Yahoo スクレイプを全銘柄に試みる。
    返り値: (scraped_rows, fallback_stocks)
      - scraped_rows: Supabase に直接 upsert できる行リスト
      - fallback_stocks: スクレイプ失敗 → Haiku フォールバックが必要
    """
    scraped_rows: list[dict] = []
    fallback:     list[dict] = []

    print(f"[gen] Phase1: Yahoo スクレイプ開始（{len(stocks)}件、{SCRAPE_WORKERS}並列）")
    done = 0
    with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as ex:
        futs = {ex.submit(_fetch_tokushoku, s): s for s in stocks}
        for fut in as_completed(futs):
            code, text = fut.result()
            done += 1
            if text:
                scraped_rows.append({
                    "code":          code,
                    "date":          CACHE_DATE,
                    "summary":       text,
                    "bull_points":   [],
                    "bear_points":   [],
                    "model_version": MODEL_VER,
                })
            else:
                fallback.append(futs[fut])
            if done % 100 == 0:
                print(f"[gen]   スクレイプ進捗: {done}/{len(stocks)} "
                      f"（取得 {len(scraped_rows)} / 失敗 {len(fallback)}）")

    print(f"[gen] Phase1 完了: 取得 {len(scraped_rows)}件 / フォールバック {len(fallback)}件")
    return scraped_rows, fallback


# ─── Phase 2: Claude Haiku バッチ生成（フォールバック） ──────────────────────

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


def _haiku_phase(stocks: list[dict]) -> list[dict]:
    """スクレイプ失敗分を Haiku でバッチ生成。"""
    if not stocks:
        return []
    if not ANTHROPIC_API_KEY:
        print("[gen] ANTHROPIC_API_KEY 未設定。Haiku フォールバックをスキップ。")
        return []
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chunks = [stocks[i:i + BATCH_N] for i in range(0, len(stocks), BATCH_N)]
    print(f"[gen] Phase2: Haiku バッチ生成（{len(stocks)}件、{len(chunks)}バッチ、{WORKERS}並列）")
    done = 0
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_generate_batch, client, ch): i for i, ch in enumerate(chunks)}
        for fut in as_completed(futs):
            batch_rows = fut.result()
            if batch_rows:
                rows.extend(batch_rows)
                done += len(batch_rows)
                print(f"[gen]   Haiku 進捗: {done}/{len(stocks)} 生成済み")
    print(f"[gen] Phase2 完了: {len(rows)}件 生成")
    return rows


# ─── Supabase upsert ──────────────────────────────────────────────────────────

def _upsert_bulk(rows: list[dict]) -> None:
    """最大500件ずつ upsert。"""
    if not rows:
        return
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        r = requests.post(f"{SUPABASE_URL}/rest/v1/ai_analyses",
                          headers=_headers(), json=chunk, timeout=60)
        if not r.ok:
            print(f"[gen] upsert 失敗: {r.status_code} {r.text[:200]}")
        else:
            print(f"[gen] upsert 完了: {len(chunk)}件 (offset {i})")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]
    dry     = "--dry-run" in args
    refresh = "--refresh" in args
    limit: int | None
    if "--all" in args or refresh:
        limit = None
    elif "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    else:
        limit = DEFAULT_LIMIT

    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        print("[gen] SUPABASE 未設定。中止。"); sys.exit(1)

    todo = _targets(limit, refresh=refresh)
    mode = "再スクレイプ(--refresh)" if refresh else "差分補完"
    print(f"[gen] {mode} 対象: {len(todo)}件")
    if dry or not todo:
        print("[gen] dry-run または対象なしで終了。"); return

    # Phase 1: Yahoo スクレイプ
    scraped_rows, fallback_stocks = _scrape_phase(todo)

    # Phase 2: Haiku フォールバック
    haiku_rows = _haiku_phase(fallback_stocks)

    # 保存（スクレイプ結果を先に、Haiku 結果で補完）
    all_rows = scraped_rows + haiku_rows
    print(f"[gen] 保存開始: 合計 {len(all_rows)}件（スクレイプ {len(scraped_rows)} + Haiku {len(haiku_rows)}）")
    _upsert_bulk(all_rows)
    print(f"[gen] 完了: {len(all_rows)}/{len(todo)} 件の会社説明を生成・保存")


if __name__ == "__main__":
    main()

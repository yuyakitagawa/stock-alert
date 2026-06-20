"""
qa_pages.py
QA: Webアプリの全ページを巡回し、開けない/エラー画面/空ページ/内容欠落を検知する。

固定ルート（/ /rankings /watchlist /review）＋ サンプル銘柄詳細ページ
（ウォッチリスト＋当日上位）を実際に HTTP 取得し、lib.data_sanity.check_pages で検査。
違反があれば QA アラートメールを送る（alert-only）。

使い方:
  python3 web/qa_pages.py                 # SITE_URL を巡回
  BASE_URL=http://localhost:3000 python3 web/qa_pages.py   # 任意URLを巡回
  python3 web/qa_pages.py --no-alert      # メール送信なし（確認用）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from lib.data_sanity import run_pages_gate

BASE_URL = (os.getenv("BASE_URL") or os.getenv("SITE_URL") or "").rstrip("/")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# 読み取りキー: anon優先、無ければservice（CIにはserviceのみ）
SUPABASE_KEY = (os.getenv("SUPABASE_ANON_KEY")
                or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
                or os.getenv("SUPABASE_SERVICE_KEY", ""))
UA = "Mozilla/5.0 (compatible; StockSignal-QA/1.0)"

# 固定ルート: (path, expect=本文に含まれるべき文言 or None)
STATIC_ROUTES = [
    ("/",          "注目銘柄"),
    ("/rankings",  None),
    ("/watchlist", "お得度"),
    ("/review",    None),
]


def _sample_stock_codes(limit_top: int = 3) -> list[str]:
    """サンプル銘柄詳細ページ = ウォッチリスト + 当日ランキング上位。"""
    codes: list[str] = []
    # ウォッチリスト
    import csv
    wl = os.path.join(BASE_DIR, "data", "pricing_power_watchlist.csv")
    try:
        with open(wl, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = str(row.get("code", "")).strip()
                if c:
                    codes.append(c)
    except Exception:
        pass
    # 当日ランキング上位（Supabase）
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            r = requests.get(
                f"{SUPABASE_URL}/rest/v1/gen_rankings?select=code&order=net.desc&limit={limit_top}",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=15)
            if r.ok:
                for row in r.json():
                    codes.append(str(row.get("code")))
        except Exception:
            pass
    # 重複排除（順序維持）
    seen, out = set(), []
    for c in codes:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out[:8]


def _fetch(route: str, expect=None) -> dict:
    url = BASE_URL + route
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        return {"route": route, "status": resp.status_code, "body": resp.text, "expect": expect}
    except Exception as e:
        return {"route": route, "error": str(e), "expect": expect}


def main() -> None:
    alert = "--no-alert" not in sys.argv
    if not BASE_URL:
        print("[qa_pages] BASE_URL / SITE_URL 未設定。スキップします。")
        return

    results = [_fetch(path, expect) for path, expect in STATIC_ROUTES]
    for code in _sample_stock_codes():
        results.append(_fetch(f"/stocks/{code}", expect="AIスコア"))

    print(f"[qa_pages] {BASE_URL} の {len(results)}ページを巡回")
    run_pages_gate(results, source="qa_pages", alert=alert)


if __name__ == "__main__":
    main()

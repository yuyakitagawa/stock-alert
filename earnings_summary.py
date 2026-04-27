import os
import re
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en;q=0.9",
}


def _parse_table(t, max_rows=4):
    """テーブルからデータ行を抽出"""
    trs = t.find_all("tr")
    headers = [c.get_text(strip=True) for c in trs[0].find_all(["th", "td"])]
    rows = []
    for tr in trs[1:]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
        if not cells or not cells[0]:
            continue
        if "前年" in cells[0] or cells[0] == "%":
            continue
        row = {}
        for i, h in enumerate(headers):
            if i < len(cells):
                row[h] = cells[i]
        if row.get("売上高", "").replace(",", "").isdigit():
            rows.append(row)
        if len(rows) >= max_rows:
            break
    return rows


def scrape_kabutan_earnings(code):
    url = "https://kabutan.jp/stock/finance?code=" + code
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        annual_table = None    # 年次（最終益あり・前年比なし）
        quarter_table = None   # 四半期累計（対通期進捗率あり）

        for t in soup.find_all("table"):
            first_row = t.find("tr")
            if not first_row:
                continue
            hcells = [c.get_text(strip=True) for c in first_row.find_all(["th", "td"])]
            if "決算期" not in hcells or "売上高" not in hcells or "営業益" not in hcells:
                continue
            if "対通期進捗率" in hcells and quarter_table is None:
                quarter_table = t
            elif "最終益" in hcells and "前年比" not in hcells and annual_table is None:
                annual_table = t

        result = []

        # 直近の四半期データ（最新4期分）
        if quarter_table:
            q_rows = _parse_table(quarter_table, max_rows=4)
            for r in q_rows:
                r["_type"] = "四半期"
            result = q_rows + result

        # 年次データ（直近2期分 - 末尾から取得して最新を使う）
        if annual_table:
            a_rows = _parse_table(annual_table, max_rows=10)
            a_rows = a_rows[-2:]  # 最新2期
            for r in a_rows:
                r["_type"] = "年次"
            result = result + a_rows

        return result if result else None

    except Exception as e:
        print("  [WARN] スクレイピング失敗 (%s): %s" % (code, e))
        return None


def summarize_with_claude(earnings_text):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            messages=[{
                "role": "user",
                "content": (
                    "以下の業績データについて、投資家向けに3点を簡潔に教えてください:\n"
                    "1. 直近の業績トレンド（増収増益か否か、前年比）\n"
                    "2. 主力事業・セグメントへの市場環境の影響（業種特性・外部環境）\n"
                    "3. 注目すべき点や懸念点\n"
                    "合計200文字以内で、箇条書きで回答してください。\n\n"
                    + earnings_text
                )
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print("  [WARN] API失敗: %s" % e)
        return None


CACHE_PATH = os.path.expanduser("~/stock-alert/earnings_cache.json")


def load_cache():
    if os.path.exists(CACHE_PATH):
        import json
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    import json
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_earnings_summary(code, name, sleep_sec=1.0, force=False):
    """
    force=False: キャッシュがあればスキップ（新規銘柄のみ取得）
    force=True: 強制再取得
    """
    from datetime import datetime, date
    import json

    cache = load_cache()

    # キャッシュにあって30日以内なら再利用
    if not force and code in cache:
        cached = cache[code]
        fetched = datetime.strptime(cached.get("date", "2000-01-01"), "%Y-%m-%d").date()
        days_old = (date.today() - fetched).days
        if days_old < 30:
            print("  (キャッシュ使用: %d日前取得)" % days_old)
            return cached

    rows = scrape_kabutan_earnings(code)
    time.sleep(sleep_sec)
    if not rows:
        return None

    text = "【%s（%s）の直近業績】\n" % (name, code)
    for row in rows:
        text += " / ".join("%s: %s" % (k, v) for k, v in row.items() if v) + "\n"

    summary = summarize_with_claude(text)
    result = {
        "rows": rows,
        "summary": summary,
        "raw_text": text,
        "date": str(date.today())
    }

    # キャッシュ保存
    cache[code] = result
    save_cache(cache)
    return result


if __name__ == "__main__":
    test_stocks = [("6098", "リクルート"), ("7203", "トヨタ自動車")]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    print("=" * 55)
    print("決算サマリー テスト")
    print("Claude API: %s" % ("✅ 設定済み" if api_key else "❌ 未設定"))
    print("=" * 55)
    for code, name in test_stocks:
        print("\n▶ %s（%s）" % (name, code))
        result = get_earnings_summary(code, name)
        if result is None:
            print("  データ取得失敗")
            continue
        print("  【業績データ】")
        for row in result["rows"]:
            print("  ", " / ".join("%s:%s" % (k, v) for k, v in row.items() if v))
        if result["summary"]:
            print("\n  【AIサマリー】\n  %s" % result["summary"])
        else:
            print("  APIサマリー: なし")

"""
kabutan_earnings.py — kabutan.jp から業績テーブルを取得・キャッシュ

取得データ（通期業績テーブル）:
  決算期 / 売上高 / 営業益 / 経常益 / 最終益 / 修正1株益(EPS) / 修正1株配(DPS) / 発表日

キャッシュ: tools/_kabutan_earnings_cache.json（7日で自動更新）
"""
import os, re, json
import requests
from datetime import date

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH  = os.path.join(BASE_DIR, "tools", "_kabutan_earnings_cache.json")
CACHE_DAYS  = 7   # 何日間キャッシュを使い回すか
HEADERS     = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", "").strip()


def _parse_tables(html: str) -> list[list[list[str]]]:
    clean = html.replace("\n", "").replace("\t", "")
    tables = re.findall(r"<table[^>]*>.*?</table>", clean)
    result = []
    for t in tables:
        rows = []
        for tr in re.findall(r"<tr[^>]*>.*?</tr>", t):
            cells = [_strip(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr)]
            if cells:
                rows.append(cells)
        result.append(rows)
    return result


def _find_earnings_table(tables: list[list[list[str]]]) -> list[list[str]] | None:
    """'決算期','売上高','営業益','最終益' を含むテーブルを返す。"""
    for rows in tables:
        if not rows:
            continue
        hdr = rows[0]
        if all(any(k in h for h in hdr) for k in ["決算期", "売上高", "営業益"]):
            return rows
    return None


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("－", "").replace("―", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _fy_label(cell: str) -> str | None:
    """'予2026.05' / '2025.05' → '2026-05' / '2025-05'"""
    m = re.search(r"(\d{4})\.(\d{2})", cell)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def fetch_kabutan_earnings(code: str | int, use_cache: bool = True) -> list[dict]:
    """
    kabutan.jp の通期業績を取得。

    戻り値: [
      { "fy_end": "2025-05", "is_forecast": False,
        "revenue": 309240, "op_profit": -1237, "ord_profit": -460,
        "net_income": -8658, "eps": -221.8, "dps": 75.0,
        "announce_date": "2025-07-15" },
      ...  # 直近5〜6期分（予想行含む）
    ]
    失敗時は []。
    """
    code = str(code)

    # ─── キャッシュ確認 ───────────────────────────────────────────
    cache: dict = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if use_cache and code in cache:
        entry = cache[code]
        cached_date = date.fromisoformat(entry.get("_fetched", "2000-01-01"))
        if (date.today() - cached_date).days < CACHE_DAYS:
            return entry.get("rows", [])

    # ─── kabutan.jp スクレイピング ────────────────────────────────
    try:
        r = requests.get(
            f"https://kabutan.jp/stock/finance/?code={code}",
            headers=HEADERS, timeout=12,
        )
        if r.status_code != 200:
            return []
    except Exception:
        return []

    tables = _parse_tables(r.text)
    tbl = _find_earnings_table(tables)
    if not tbl:
        return []

    # ヘッダー行解析
    hdr = tbl[0]
    col = {}
    for i, h in enumerate(hdr):
        if "決算期"  in h: col["fy"]    = i
        if "売上高"  in h: col["rev"]   = i
        if "営業益"  in h: col["op"]    = i
        if "経常益"  in h: col["ord"]   = i
        if "最終益"  in h: col["net"]   = i
        if "1株益"   in h: col["eps"]   = i
        if "1株配"   in h: col["dps"]   = i
        if "発表日"  in h: col["ann"]   = i

    rows_out = []
    for row in tbl[1:]:
        if not row or not row[0]:
            continue
        label = row[0]
        fy = _fy_label(label)
        if not fy:
            continue  # 前期比行など

        is_forecast = "予" in label

        def get(key: str) -> float | None:
            idx = col.get(key)
            return _to_float(row[idx]) if idx is not None and idx < len(row) else None

        # 発表日 YY/MM/DD → YYYY-MM-DD
        ann_raw = row[col["ann"]] if "ann" in col and col["ann"] < len(row) else ""
        m = re.search(r"(\d{2})/(\d{2})/(\d{2})", ann_raw)
        ann_date = f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

        rows_out.append({
            "fy_end":       fy,
            "is_forecast":  is_forecast,
            "revenue":      get("rev"),
            "op_profit":    get("op"),
            "ord_profit":   get("ord"),
            "net_income":   get("net"),
            "eps":          get("eps"),
            "dps":          get("dps"),
            "announce_date": ann_date,
        })

    # ─── キャッシュ保存 ───────────────────────────────────────────
    cache[code] = {"_fetched": date.today().isoformat(), "rows": rows_out}
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass

    return rows_out

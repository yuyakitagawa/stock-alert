"""
lib/company_info.py — 上場銘柄の会社HP URLと最新決算（Supabase `company_info`）

値の出どころは株探。
  - 銘柄ページ  kabutan.jp/stock/?code=XXXX         … 「会社サイト」欄 → website_url
  - 決算ページ  kabutan.jp/stock/finance/?code=XXXX … 通期業績（実績・会社予想）と
                                                       最新の累計決算（1Q/上期/3Q）
jquants_fin_summary（EDINET XBRL）は有報・半期報しか入らず、四半期報告書の廃止以降は
1Q/3Q が欠ける。決算短信ベースの最新値はこちらで持つ。金額は株探の表示単位（百万円）。

tools/fetch_company_info.py が古い順に巡回する（daily_alert.yml Step 2i）。
"""
import re
import time
from datetime import datetime, timedelta, timezone

import requests

import lib.supabase_client as sb
from lib.kabutan_earnings import HEADERS, _parse_tables, _to_float

TABLE = "company_info"
_TIMEOUT = 15

_WEBSITE_RE = re.compile(r"会社サイト</th>\s*<td>\s*<a href=\"(https?://[^\"]+)\"")
_CUM_LABEL_RE = re.compile(r"<h3>\s*([^<]*累計)決算【実績】\s*</h3>")
_FY_RE = re.compile(r"(\d{4}\.\d{2})")
_PERIOD_RE = re.compile(r"(\d{2}\.\d{2}-\d{2})")
_DATE_RE = re.compile(r"(\d{2})/(\d{2})/(\d{2})")


class RateLimited(Exception):
    """株探に続けて断られた。以降を取りに行っても無駄なので打ち切る。"""


# ── パース（ネットワークなし） ─────────────────────────────────────────

def parse_website(html: str) -> str | None:
    m = _WEBSITE_RE.search(html.replace("\n", "").replace("\t", ""))
    return m.group(1).strip() if m else None


def _date(cell: str) -> str | None:
    m = _DATE_RE.search(cell or "")
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _cols(hdr: list[str]) -> dict[str, int]:
    keys = {"sales": "売上高", "op": "営業益", "ord": "経常益", "np": "最終益",
            "eps": "1株益", "dps": "1株配", "progress": "進捗率", "ann": "発表日"}
    return {k: i for i, h in enumerate(hdr) for k, kw in keys.items() if kw in h}


def _cell(row: list[str], col: dict[str, int], key: str) -> str:
    i = col.get(key)
    return row[i] if i is not None and i < len(row) else ""


def _values(row: list[str], col: dict[str, int], prefix: str, keys: tuple[str, ...]) -> dict:
    out = {f"{prefix}_{k}": _to_float(_cell(row, col, k)) for k in keys}
    out[f"{prefix}_announced"] = _date(_cell(row, col, "ann"))
    return out


_FY_KEYS = ("sales", "op", "ord", "np", "eps", "dps")
_EMPTY_FY = {f"{p}_{k}": None for p in ("fy", "fc") for k in _FY_KEYS + ("period", "announced")}
_EMPTY_LATEST = {f"latest_{k}": None for k in
                 ("label", "period", "sales", "op", "ord", "np", "eps", "progress_pct", "announced")}


def parse_finance(html: str) -> dict:
    """決算ページから fy_*（通期実績）/ fc_*（会社予想）/ latest_*（最新の累計決算）を返す。

    表が見つからない項目は None（ETF・TOKYO PRO Market 等は決算表が無い）。
    """
    out = {**_EMPTY_FY, **_EMPTY_LATEST}
    tables = _parse_tables(html)

    # 通期業績: 「決算期 売上高 営業益 … 1株配 発表日」で、決算期が 2025.12 形式の最初の表
    for rows in tables:
        if not rows or not all(any(k in h for h in rows[0]) for k in ("決算期", "売上高", "1株配", "発表日")):
            continue
        body = [r for r in rows[1:] if r and _FY_RE.search(r[0])]
        if not body:
            continue
        col = _cols(rows[0])
        actual = [r for r in body if "予" not in r[0]]
        forecast = [r for r in body if "予" in r[0]]
        if actual:
            r = actual[-1]
            out.update(_values(r, col, "fy", _FY_KEYS), fy_period=_FY_RE.search(r[0]).group(1))
        if forecast:
            r = forecast[-1]
            out.update(_values(r, col, "fc", _FY_KEYS), fc_period=_FY_RE.search(r[0]).group(1))
        break

    # 最新の累計決算: 「決算期 … 進捗率 発表日」の表。最後の 26.01-06 形式の行が最新
    for rows in tables:
        if not rows or not all(any(k in h for h in rows[0]) for k in ("決算期", "進捗率", "発表日")):
            continue
        body = [r for r in rows[1:] if r and _PERIOD_RE.search(r[0])]
        if not body:
            continue
        col = _cols(rows[0])
        r = body[-1]
        v = _values(r, col, "latest", ("sales", "op", "ord", "np", "eps", "progress"))
        v["latest_progress_pct"] = v.pop("latest_progress")
        m = _CUM_LABEL_RE.search(html.replace("\n", "").replace("\t", ""))
        out.update(v, latest_period=_PERIOD_RE.search(r[0]).group(1),
                   latest_label=m.group(1).strip() if m else None)
        break
    return out


# ── 取得・保存 ─────────────────────────────────────────────────────────

def _get(url: str) -> str | None:
    """HTML。取得エラー（HTTP≠200・通信断）は None（＝保存しない。次回また取りに行く）。

    株探に載っていない銘柄（TOKYO PRO Market 等）は /stock/ へ302される。これは
    「値が無い」なので "" を返してNULLで保存する（エラー扱いにすると毎日取り直し、
    5xxA 台に固まっているため連続エラーでレート制限と誤判定して打ち切る）。
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=_TIMEOUT, allow_redirects=False)
    except Exception:
        return None
    if r.status_code in (301, 302):
        return ""
    return r.text if r.status_code == 200 else None


def fetch_one(code: str, with_website: bool) -> dict | None:
    """1銘柄ぶんの保存行。決算ページが取れなければ None。

    with_website=False のときは website_* キーを含めない（upsert で既存値を残す）。
    銘柄ページだけ失敗したときも website_* を含めない（NULLで上書きして90日放置しないため）。
    """
    fin = _get(f"https://kabutan.jp/stock/finance/?code={code}")
    if fin is None:
        return None
    now = datetime.now(timezone.utc).isoformat()
    row = {"code": code, **parse_finance(fin), "fetched_at": now}
    if with_website:
        top = _get(f"https://kabutan.jp/stock/?code={code}")
        if top is not None:
            row.update(website_url=parse_website(top), website_checked_at=now)
    return row


# 連続でこの回数断られたらレート制限とみなして打ち切る
MAX_CONSECUTIVE_ERRORS = 15


def fetch_many(codes: list[str], website_codes: set[str], sleep_sec: float = 1.0) -> list[dict]:
    """取れた銘柄の行だけ返す。レート制限なら RateLimited（それまでの分は e.args[0]）。"""
    out: list[dict] = []
    errors = 0
    for code in codes:
        row = fetch_one(code, code in website_codes)
        if row is None:
            errors += 1
            if errors >= MAX_CONSECUTIVE_ERRORS:
                raise RateLimited(out)
        else:
            out.append(row)
            errors = 0
        if sleep_sec:
            time.sleep(sleep_sec)
    return out


def save(rows: list[dict]) -> bool:
    return sb.upsert(TABLE, rows, on_conflict="code") if rows else True


def load() -> dict[str, dict]:
    """{code: {"fetched_at": str, "website_checked_at": str|None}}"""
    return {str(r["code"]): r for r in sb.select(TABLE, "select=code,fetched_at,website_checked_at")}


def target_codes() -> list[str]:
    """巡回対象: jpx_stock_list の普通株（ETF・REIT等の sector='-' と5桁の優先株を除く）。"""
    rows = sb.select("jpx_stock_list", "select=code,sector")
    return sorted(str(r["code"]) for r in rows
                  if len(str(r["code"])) == 4 and r.get("sector") and r["sector"] != "-")


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def plan(codes: list[str], saved: dict[str, dict], announced: dict[str, str],
         now: datetime, max_age_days: int, website_max_age_days: int) -> tuple[list[str], set[str]]:
    """(今回取る銘柄の優先順リスト, そのうち会社サイトも取り直す銘柄)。

    優先順: ①保存後に決算短信が出た銘柄 → ②未取得 → ③取得が max_age_days より古い順。
    announced は {code: 決算短信の開示日時}（TDnet）。
    """
    fresh_cut = now - timedelta(days=max_age_days)
    first, never, stale = [], [], []
    for c in codes:
        r = saved.get(c)
        fetched = _ts(r.get("fetched_at")) if r else None
        ann = _ts(announced.get(c))
        if fetched is None:
            (first if ann else never).append(c)
        elif ann and ann > fetched:
            first.append(c)
        elif fetched < fresh_cut:
            stale.append((fetched, c))
    todo = first + never + [c for _, c in sorted(stale)]

    web_cut = now - timedelta(days=website_max_age_days)
    website = {c for c in todo
               if (w := _ts((saved.get(c) or {}).get("website_checked_at"))) is None or w < web_cut}
    return todo, website


def recent_earnings_announcements(days_back: int = 2) -> dict[str, str]:
    """TDnet（やのしん）の直近の決算短信 {code: 開示日時}。取れなければ {}（優先付けが無くなるだけ）。"""
    from datetime import date
    from lib.tdnet import fetch_disclosures
    end = date.today()
    cond = f"{(end - timedelta(days=days_back)).strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    out: dict[str, str] = {}
    for d in fetch_disclosures(cond, limit=2000):
        if "決算短信" in d["title"]:
            # やのしんの pubdate は JST の "YYYY-MM-DD HH:MM:SS"
            ts = d["disclosed_at"].replace(" ", "T") + "+09:00"
            if ts > out.get(d["code"], ""):
                out[d["code"]] = ts
    return out

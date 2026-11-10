"""
lib/company_info.py — 上場銘柄の会社HP URLと最新決算（Supabase `company_info`）

値の出どころは株探。
  - 銘柄ページ  kabutan.jp/stock/?code=XXXX         … 「会社サイト」欄 → website_url
  - 決算ページ  kabutan.jp/stock/finance/?code=XXXX … 通期業績（実績・会社予想）と
                                                       最新の累計決算（1Q/上期/3Q）
jquants_fin_summary（EDINET XBRL）は有報・半期報しか入らず、四半期報告書の廃止以降は
1Q/3Q が欠ける。決算短信ベースの最新値はこちらで持つ。金額は株探の表示単位（百万円）。

最新の決算短信の URL（tanshin_*）は TDnet（ext_tdnet_disclosures、Step 2f が保存）から付け、
その短信の XBRL（tanshin_xbrl_url）から会社予想（tanshin_fc_*、百万円）を読む（lib/earnings_forecast.py の
パーサを使う）。TDnet の XBRL は約1か月で消えるので、読めるのは新しい短信だけ。
従業員数（employees_*）も同じテーブルにあるが、読み書きは lib/employees.py。

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
    """{code: {"fetched_at": str|None, "website_checked_at": str|None, "tanshin_url": str|None}}

    fetched_at が NULL の行（従業員数・短信URLだけ入っている行）は、決算は未取得として扱う。
    """
    return {str(r["code"]): r for r in sb.select(
        TABLE, "select=code,fetched_at,website_checked_at,tanshin_url,tanshin_fc_fetched_at")}


# やのしん経由の URL（webapi.yanoshin.jp/rd.php?https://www.release.tdnet.info/...）から TDnet 本体を取り出す
_RD_PREFIX = "rd.php?"


def tdnet_url(url: str | None) -> str | None:
    if not url:
        return None
    i = url.find(_RD_PREFIX)
    return url[i + len(_RD_PREFIX):] if i >= 0 else url


# 短信本体だけ: 「…決算短信」「…決算短信〔日本基準〕（連結）」「…決算短信（REIT）」。
# 「決算短信の開示延期に関するお知らせ」「決算短信補足説明資料」「決算説明資料（…決算短信）」は除く
_TANSHIN_RE = re.compile(r"決算短信\s*([〔［\[（(【].*)?$")


def pick_tanshin(rows: list[dict]) -> dict[str, dict]:
    """TDnet の開示から銘柄ごとの最新の決算短信 {code: tanshin_* の行}。訂正・お知らせ・URL無しは除く。"""
    out: dict[str, dict] = {}
    for r in rows:
        title = r.get("title") or ""
        url = tdnet_url(r.get("doc_url"))
        xbrl = tdnet_url(r.get("xbrl_url"))
        if not _TANSHIN_RE.search(title) or "訂正" in title or not url:
            continue
        code = str(r["code"])
        if str(r["disclosed_at"]) > str((out.get(code) or {}).get("tanshin_disclosed_at") or ""):
            out[code] = {"code": code, "tanshin_url": url, "tanshin_title": title,
                         "tanshin_disclosed_at": r["disclosed_at"], "tanshin_xbrl_url": xbrl}
    return out


def latest_tanshin() -> dict[str, dict]:
    """ext_tdnet_disclosures（Step 2f が保存。2026-06-24〜）の最新の決算短信。取れなければ {}。"""
    rows = sb.select("ext_tdnet_disclosures",
                     "title=like.*%E6%B1%BA%E7%AE%97%E7%9F%AD%E4%BF%A1*&select=code,disclosed_at,title,doc_url,xbrl_url")
    return pick_tanshin(rows)


def tanshin_updates(saved: dict[str, dict], latest: dict[str, dict]) -> list[dict]:
    """書き直す銘柄の行: 保存済みの URL と違う（新しい短信が出た・未保存）か、予想をまだ読めていない。"""
    return [row for c, row in sorted(latest.items())
            if (saved.get(c) or {}).get("tanshin_url") != row["tanshin_url"]
            or not (saved.get(c) or {}).get("tanshin_fc_fetched_at")]


_FC_EMPTY = {"tanshin_fc_fy_end": None, "tanshin_fc_sales": None, "tanshin_fc_op": None, "tanshin_fc_np": None}


def _mil(v: float | None) -> float | None:
    return None if v is None else round(v / 1e6, 3)


def tanshin_forecast(xbrl_url: str | None) -> dict | None:
    """短信の XBRL から tanshin_fc_* を作る。取得エラー（通信断・5xx）は None＝次回また読む。

    XBRL が消えている（4xx）・予想が載っていない短信は、値を NULL にして読んだ時刻だけ刻む
    （消えた URL を毎日取りに行かない）。金額は company_info の他の列に合わせて百万円。
    """
    from lib import earnings_forecast as ef
    now = datetime.now(timezone.utc).isoformat()
    if not xbrl_url:
        return {**_FC_EMPTY, "tanshin_fc_fetched_at": now}
    try:
        r = requests.get(xbrl_url, headers=ef.HEADERS, timeout=60)
    except Exception:
        return None
    if r.status_code >= 500:
        return None
    f = ef.parse_forecast(r.content) if r.status_code == 200 else None
    if not f:
        return {**_FC_EMPTY, "tanshin_fc_fetched_at": now}
    return {"tanshin_fc_fy_end": f["fy_end"], "tanshin_fc_sales": _mil(f.get("fsales")),
            "tanshin_fc_op": _mil(f.get("fop")), "tanshin_fc_np": _mil(f.get("fnp")),
            "tanshin_fc_fetched_at": now}


def with_tanshin_forecasts(rows: list[dict], max_reads: int, sleep_sec: float = 0.5) -> list[dict]:
    """tanshin_updates の各行に短信 XBRL の予想を付ける。新しい短信から max_reads 件まで読む。
    読めなかった行・上限を超えた行は予想の列を NULL、tanshin_fc_fetched_at も NULL にして返す
    （URL は保存し、予想は次回また読む）。決算発表の集中日は短信が1日数百件出るため上限を置く。"""
    out = []
    rows = sorted(rows, key=lambda r: str(r.get("tanshin_disclosed_at") or ""), reverse=True)
    for i, row in enumerate(rows):
        fc = tanshin_forecast(row.get("tanshin_xbrl_url")) if i < max_reads else None
        out.append(row | (fc or {**_FC_EMPTY, "tanshin_fc_fetched_at": None}))
        if sleep_sec and i + 1 < min(len(rows), max_reads):
            time.sleep(sleep_sec)
    return out


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
    from lib.tdnet import fetch_days
    end = date.today()
    out: dict[str, str] = {}
    for d in fetch_days(end - timedelta(days=days_back), end):
        if "決算短信" in d["title"]:
            # やのしんの pubdate は JST の "YYYY-MM-DD HH:MM:SS"
            ts = d["disclosed_at"].replace(" ", "T") + "+09:00"
            if ts > out.get(d["code"], ""):
                out[d["code"]] = ts
    return out

"""
lib/employees.py — 銘柄ごとの従業員数（Supabase `company_employees`）

値の出どころは Yahoo Finance の fullTimeEmployees（連結・現在値）。
通知のたびにYahooへ候補の数だけ問い合わせると、急落日（候補が数百）に数分かかり、
並列で取ると一部がレート制限で空振りして候補から漏れるため、テーブルに持つ。

  - tools/fetch_employees.py が古い順に少しずつ取り直す（daily_alert.yml Step 2h）
  - web/dip_buy_alert.py は get() で読み、無い銘柄だけYahooで補って書き戻す
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import lib.supabase_client as sb

TABLE = "company_employees"


class RateLimited(Exception):
    """Yahooに続けて断られた。以降を取りに行っても無駄なので打ち切る。"""


def fetch_one(code: str) -> tuple[bool, int | None]:
    """Yahooから1銘柄。(取れたか, 従業員数)。

    取得エラー（レート制限・通信断）は (False, None)。企業情報は返ってきたが従業員数が
    載っていないときだけ (True, None)。この2つを混ぜると、レート制限中に取った銘柄が
    「値が無い」として保存され、30日間取り直されなくなる（2026-09-19に2,432件で起きた）。
    """
    try:
        import yfinance as yf
        info = yf.Ticker(f"{code}.T").info
    except Exception:
        return False, None
    if not info or len(info) < 5:  # 断られたときは空に近い辞書が返る
        return False, None
    v = info.get("fullTimeEmployees")
    return True, (int(v) if v else None)


# 連続でこの回数断られたらレート制限とみなして打ち切る
MAX_CONSECUTIVE_ERRORS = 15


def fetch_many(codes: list[str], workers: int = 4) -> dict[str, int | None]:
    """取れた銘柄だけを返す（取得エラーの銘柄は含めない＝保存されず次回また取りに行く）。

    レート制限に当たったら RateLimited を送出する。それまでに取れた分は e.args[0]。
    """
    out: dict[str, int | None] = {}
    errors = 0
    for i in range(0, len(codes), workers * 5):
        chunk = codes[i:i + workers * 5]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(fetch_one, chunk))
        for c, (ok, v) in zip(chunk, res):
            if ok:
                out[c] = v
                errors = 0
            else:
                errors += 1
        if errors >= MAX_CONSECUTIVE_ERRORS:
            raise RateLimited(out)
    return out


def save(values: dict[str, int | None], today: date | None = None) -> bool:
    """None も「Yahooに値が無かった」として保存する（毎回取り直さないため）。
    渡すのは fetch_many が「取れた」と返した銘柄だけにすること。"""
    d = (today or date.today()).isoformat()
    rows = [{"code": c, "employees": v, "source": "yahoo", "fetched_date": d} for c, v in values.items()]
    return sb.upsert(TABLE, rows, on_conflict="code") if rows else True


def load(codes: list[str] | None = None) -> dict[str, dict]:
    """{code: {"employees": int|None, "fetched_date": str}}。codes 省略で全件。"""
    out: dict[str, dict] = {}
    if codes is None:
        for r in sb.select(TABLE, "select=code,employees,fetched_date"):
            out[str(r["code"])] = r
        return out
    for i in range(0, len(codes), 100):
        q = ",".join(f'"{c}"' for c in codes[i:i + 100])
        for r in sb.select(TABLE, f"code=in.({q})&select=code,employees,fetched_date"):
            out[str(r["code"])] = r
    return out


def get(codes: list[str]) -> dict[str, int | None]:
    """テーブルから読み、行が無い銘柄だけYahooで取って書き戻す。"""
    have = load(codes)
    missing = [c for c in codes if c not in have]
    try:
        got = fetch_many(missing) if missing else {}
    except RateLimited as e:
        got = e.args[0]
    if got:
        save(got)
    return {c: (have[c]["employees"] if c in have else got.get(c)) for c in codes}


def stale_codes(all_codes: list[str], rows: dict[str, dict], max_age_days: int, today: date) -> list[str]:
    """未取得の銘柄を先に、次に取得日が古い順。max_age_days 以内のものは含めない。"""
    never = [c for c in all_codes if c not in rows]
    old = sorted(
        (c for c in all_codes if c in rows
         and (today - date.fromisoformat(str(rows[c]["fetched_date"])[:10])).days > max_age_days),
        key=lambda c: str(rows[c]["fetched_date"]),
    )
    return never + old

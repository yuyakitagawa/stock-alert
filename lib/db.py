"""Supabase persistence layer for stock-alert.

Migrated from SQLite. All reads/writes go to Supabase REST API.
"""
import os
from datetime import date, timedelta

import lib.supabase_client as sb

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.getenv("STOCK_ALERT_HOME", _PROJECT_DIR)
DB_PATH = os.path.join(_BASE_DIR, "stock_alert.db")


def init_db():
    pass


# ── 移行後の共通ヘルパー（旧・直接sqlite3呼び出しの置換用） ──────────────

def get_latest_ranking_date():
    """gen_rankings の最新日付を返す。なければ None。"""
    row = sb.select_one("gen_rankings", "order=date.desc&select=date")
    return row["date"] if row else None


def get_ranking_by_date(date_str, select="*", order="net.desc"):
    """指定日のランキング全行を返す。"""
    return sb.select("gen_rankings", f"date=eq.{date_str}&order={order}&select={select}")


def get_ranking_dates_desc(limit=0):
    """gen_rankings の開示日を新しい順の重複なしで返す。"""
    rows = sb.select("gen_rankings", "order=date.desc&select=date")
    seen = []
    for r in rows:
        if r["date"] not in seen:
            seen.append(r["date"])
        if limit and len(seen) >= limit:
            break
    return seen


def get_price_cache_codes():
    """yahoo_price_cache に存在する銘柄コード一覧（重複なし）。"""
    rows = sb.select("yahoo_price_cache", "select=code")
    return sorted({r["code"] for r in rows})


def get_jquants_disc_dates():
    """jquants_fin_summary の disc_date 一覧（重複なし）。"""
    rows = sb.select("jquants_fin_summary", "select=disc_date")
    return {r["disc_date"] for r in rows}


def get_fundamentals_announce_dates(start, end):
    """kabutan_fundamentals の announce_date 一覧（範囲内・重複なし・昇順）。"""
    rows = sb.select(
        "kabutan_fundamentals",
        f"announce_date=gte.{start}&announce_date=lte.{end}"
        "&announce_date=not.is.null&order=announce_date.asc&select=announce_date"
    )
    seen = []
    for r in rows:
        if r["announce_date"] not in seen:
            seen.append(r["announce_date"])
    return seen


def get_all_yutai():
    """gen_stock_meta から優待情報を返す。"""
    return sb.select("gen_stock_meta", "select=code,has_yutai,yutai_month&has_yutai=eq.true")


# ── daily_ranking (→ gen_rankings) ────────────────────────────────────────

def save_daily_ranking(date_str, rows):
    """rows: list of dicts with keys: code, name, close, rise_prob, drop_prob, net, vol, recommend, rel20, stop_loss, per, pbr, piotroski, bps_growth, eps_surprise, pos52"""
    sb_rows = []
    for r in rows:
        sb_rows.append({
            "date": date_str,
            "code": r.get("code"),
            "name": r.get("name"),
            "close": r.get("close"),
            "rise_prob": r.get("rise_prob"),
            "drop_prob": r.get("drop_prob"),
            "net": r.get("net"),
            "vol": r.get("vol"),
            "recommend": r.get("recommend"),
            "rel20": r.get("rel20"),
            "stop_loss": r.get("stop_loss"),
            "per": r.get("per"),
            "pbr": r.get("pbr"),
            "piotroski": r.get("piotroski"),
            "bps_growth": r.get("bps_growth"),
            "eps_surprise": r.get("eps_surprise"),
            "pos52": r.get("pos52"),
        })
    sb.upsert("gen_rankings", sb_rows, on_conflict="date,code")


# ── earnings_cache (→ kabutan_earnings) ───────────────────────────────────────

CACHE_MISS = object()

def get_earnings_cache(code, today_str):
    """今日キャッシュ済みなら next_date(str or None)を返す。未キャッシュはCACHE_MISS。"""
    row = sb.select_one(
        "kabutan_earnings",
        f"code=eq.{code}&select=next_date,fetched_date"
    )
    if row and row.get("fetched_date") == today_str:
        return row.get("next_date")
    return CACHE_MISS


def set_earnings_cache(code, today_str, next_date_str):
    sb.upsert("kabutan_earnings", [{
        "code": str(code),
        "next_date": next_date_str,
        "fetched_date": today_str,
    }], on_conflict="code")


def get_yutai_cache(code, today_str):
    """gen_stock_metaから優待情報を返す。キャッシュ済みなら (has_yutai, yutai_month)。"""
    row = sb.select_one(
        "gen_stock_meta",
        f"code=eq.{code}&select=has_yutai,yutai_month,fetched_date"
    )
    if row and row.get("fetched_date") == today_str:
        return (bool(row.get("has_yutai")), row.get("yutai_month"))
    return CACHE_MISS


def set_yutai_cache(code, today_str, has_yutai, record_month):
    sb.upsert("gen_stock_meta", [{
        "code": str(code),
        "has_yutai": bool(has_yutai),
        "yutai_month": record_month,
        "fetched_date": today_str,
    }], on_conflict="code")


# ── kabutan_fundamentals ──────────────────────────────────────────────────

def upsert_fundamentals_annual(code, rows, today_str):
    """rows: list of dict(fy_end, announce_date, eps, dps, roe, bps)"""
    sb_rows = [{
        "code": str(code),
        "fy_end": r["fy_end"],
        "announce_date": r.get("announce_date"),
        "eps": r.get("eps"),
        "dps": r.get("dps"),
        "roe": r.get("roe"),
        "bps": r.get("bps"),
        "fetched_date": today_str,
    } for r in rows]
    sb.upsert("kabutan_fundamentals", sb_rows, on_conflict="code,fy_end")


def get_fundamentals_annual(code):
    """code の年度別ファンダを発表日昇順で返す。"""
    return sb.select(
        "kabutan_fundamentals",
        f"code=eq.{code}&announce_date=not.is.null&order=announce_date.asc"
        "&select=fy_end,announce_date,eps,dps,roe,bps"
    )


def load_all_fundamentals_annual():
    """全銘柄の年度別ファンダを {code: [rows...]} で返す"""
    rows = sb.select(
        "kabutan_fundamentals",
        "announce_date=not.is.null&order=code.asc,announce_date.asc"
        "&select=code,fy_end,announce_date,eps,dps,roe,bps"
    )
    out = {}
    for r in rows:
        out.setdefault(str(r["code"]), []).append(r)
    return out


def get_fundamentals_fetched_codes(today_str):
    """当日に取得済みの kabutan_fundamentals 銘柄コード集合。"""
    rows = sb.select(
        "kabutan_fundamentals",
        f"fetched_date=eq.{today_str}&select=code"
    )
    return {str(r["code"]) for r in rows}


def get_fundamentals_codes_count():
    rows = sb.select("kabutan_fundamentals", "select=code")
    return len(set(r["code"] for r in rows))


# ── sector_cache (→ gen_stock_meta) ───────────────────────────────────────

def get_all_sectors():
    rows = sb.select("gen_stock_meta", "select=code,sector")
    return {r["code"]: r["sector"] for r in rows if r.get("sector")}


def save_all_sectors(sector_map):
    today_str = date.today().isoformat()
    sb_rows = [{
        "code": code,
        "sector": sector,
        "fetched_date": today_str,
    } for code, sector in sector_map.items()]
    sb.upsert("gen_stock_meta", sb_rows, on_conflict="code")


# ── simulation_results ────────────────────────────────────────────────────

def save_simulation_results(run_date, rows):
    sb_rows = [{
        "run_date": run_date,
        "entry_date": r["entry_date"],
        "code": r["code"],
        "name": r.get("name"),
        "label": r.get("label"),
        "entry_price": r.get("entry_price"),
        "current_price": r.get("current_price"),
        "return_pct": r.get("return_pct"),
        "holding_days": r.get("holding_days"),
        "net_at_entry": r.get("net_at_entry"),
        "drop_prob_at_entry": r.get("drop_prob_at_entry"),
    } for r in rows]
    sb.upsert("simulation_results", sb_rows, on_conflict="run_date,entry_date,code")


def load_simulation_results(run_date=None):
    if run_date:
        return sb.select(
            "simulation_results",
            f"run_date=eq.{run_date}&order=entry_date.asc,return_pct.desc"
        )
    return sb.select(
        "simulation_results",
        "order=run_date.asc,entry_date.asc,return_pct.desc"
    )


# ── yahoo_price_cache ───────────────────────────────────────────────────────────

def get_price_cache_coverage(code):
    """キャッシュの (min_date_str, max_date_str) を返す。未キャッシュは None。"""
    mn_row = sb.select_one("yahoo_price_cache", f"code=eq.{code}&order=date.asc&select=date")
    mx_row = sb.select_one("yahoo_price_cache", f"code=eq.{code}&order=date.desc&select=date")
    if mn_row and mx_row:
        return mn_row["date"], mx_row["date"]
    return None


def get_price_cache(code, start_date_str, end_date_str):
    """キャッシュから DataFrame(Close, Volume) を返す。データ不足なら None。"""
    import pandas as pd
    from datetime import date as _date
    rows = sb.select(
        "yahoo_price_cache",
        f"code=eq.{code}&date=gte.{start_date_str}&date=lte.{end_date_str}"
        f"&order=date.asc&select=date,close,volume"
    )
    if len(rows) < 100:
        return None
    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]
    vols = [r["volume"] for r in rows]
    idx = [_date.fromisoformat(d) for d in dates]
    return pd.DataFrame({"Close": closes, "Volume": vols}, index=idx)


def save_price_cache(code, df):
    """DataFrame を yahoo_price_cache に INSERT IGNORE で保存。"""
    import math
    rows = []
    for idx, row in df.iterrows():
        d = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        cv = row.get("Close")
        vv = row.get("Volume")
        c = float(cv) if cv is not None and not (isinstance(cv, float) and math.isnan(cv)) else None
        v = int(vv) if vv is not None and not (isinstance(vv, float) and math.isnan(vv)) else None
        rows.append({"code": str(code), "date": d, "close": c, "volume": v})
    if rows:
        sb.insert_ignore("yahoo_price_cache", rows, on_conflict="code,date")



# ── tdnet_events ──────────────────────────────────────────────────────────

def upsert_tdnet_events(code: str, events: list):
    """events: list of {'announce_date', 'title', 'event_type'}"""
    today_str = date.today().isoformat()
    sb_rows = [{
        "code": str(code),
        "announce_date": e["announce_date"],
        "title": e["title"],
        "event_type": e["event_type"],
        "fetched_date": today_str,
    } for e in events]
    sb.upsert("tdnet_events", sb_rows, on_conflict="code,announce_date,title")


def tdnet_fetched_today(code: str, today_str: str) -> bool:
    """当日に code の tdnet をフェッチ済みか。"""
    row = sb.select_one(
        "tdnet_events",
        f"code=eq.{code}&fetched_date=eq.{today_str}&select=code"
    )
    return row is not None


def get_tdnet_events_recent(code: str, days: int = 60):
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    return sb.select(
        "tdnet_events",
        f"code=eq.{code}&announce_date=gte.{cutoff}&order=announce_date.desc"
        f"&select=announce_date,title,event_type"
    )


# ── edinet_large_holdings（大量保有報告書・5%ルール）───────────────────────

def upsert_edinet_large_holdings(records: list):
    if not records:
        return
    today_str = date.today().isoformat()
    seen = set()
    sb_rows = []
    for r in records:
        did = r["doc_id"]
        if did in seen:
            continue
        seen.add(did)
        sb_rows.append({
            "doc_id": did,
            "filer_name": r.get("filer_name"),
            "doc_type_code": r.get("doc_type_code"),
            "doc_description": r.get("doc_description"),
            "submit_date": r.get("submit_date"),
            "disc_date": r.get("disc_date"),
            "holding_ratio": r.get("holding_ratio"),
            "issuer_code": r.get("issuer_code"),
            "fetched_date": today_str,
        })
    sb.upsert("edinet_large_holdings", sb_rows, on_conflict="doc_id")


def get_edinet_large_holdings_recent(days: int = 30, codes: list | None = None):
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    q = f"disc_date=gte.{cutoff}&order=disc_date.desc,submit_date.desc"
    q += "&select=doc_id,filer_name,doc_type_code,doc_description,submit_date,disc_date,holding_ratio,issuer_code"
    if codes:
        code_list = ",".join(str(c) for c in codes)
        q += f"&issuer_code=in.({code_list})"
    return sb.select("edinet_large_holdings", q)


# ── yahoo_market_index ────────────────────────────────────────────────────

def get_market_index_latest_date(ticker: str):
    row = sb.select_one(
        "yahoo_market_index",
        f"ticker=eq.{ticker}&order=date.desc&select=date"
    )
    return row["date"] if row else None


def save_market_index_data(ticker: str, df):
    import math
    if df is None or len(df) == 0:
        return
    rows = []
    for idx, row in df.iterrows():
        d = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        c = row["Close"]
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        rows.append({"ticker": ticker, "date": d, "close": float(c)})
    if rows:
        sb.insert_ignore("yahoo_market_index", rows, on_conflict="ticker,date")


def load_market_index_data(ticker: str, days: int = 2200):
    import pandas as pd
    from datetime import date as _date
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = sb.select(
        "yahoo_market_index",
        f"ticker=eq.{ticker}&date=gte.{cutoff}&order=date.asc&select=date,close"
    )
    if not rows:
        return None
    dates = [r["date"] for r in rows]
    closes = [r["close"] for r in rows]
    idx = [_date.fromisoformat(d) for d in dates]
    return pd.DataFrame({"Close": closes}, index=idx)



# ── jquants_fin_summary ──────────────────────────────────────────────────

def bulk_upsert_jquants_fin_summary(rows: list):
    sb.upsert("jquants_fin_summary", rows, on_conflict="code,disc_date")


def get_jquants_fin_history(code: str, as_of_date: str, n: int = 4) -> list:
    return sb.select(
        "jquants_fin_summary",
        f"code=eq.{code}&disc_date=lte.{as_of_date}&order=disc_date.desc&limit={n}"
    )


def get_jquants_fin_history_fy(code: str, as_of_date: str, n: int = 3) -> list:
    return sb.select(
        "jquants_fin_summary",
        f"code=eq.{code}&disc_date=lte.{as_of_date}&doc_type=like.FY*"
        f"&order=disc_date.desc&limit={n}"
    )


def jquants_earnings_rows(code: str) -> list:
    """利益の質フィルター用の年次行を jquants_fin_summary から組み立てる（kabutan非依存）。
    各FY行を {fy_end, is_forecast, revenue, op_profit, net_income} に整形（fy_end昇順）。
    最新開示の会社予想(fop/fsales/fnp)があれば予想行を1件追加。"""
    rows = sb.select(
        "jquants_fin_summary",
        f"code=eq.{code}&doc_type=eq.FY&op=not.is.null"
        "&order=fy_end.asc&select=fy_end,sales,op,np,disc_date,fop,fsales,fnp"
    )
    out = []
    for r in rows:
        out.append({"fy_end": r["fy_end"], "is_forecast": False,
                    "revenue": r["sales"], "op_profit": r["op"], "net_income": r["np"]})
    fc = sb.select_one(
        "jquants_fin_summary",
        f"code=eq.{code}&fop=not.is.null&order=disc_date.desc&select=fop,fsales,fnp"
    )
    if fc and fc.get("fop") is not None:
        out.append({"fy_end": "forecast", "is_forecast": True,
                    "revenue": fc.get("fsales"), "op_profit": fc.get("fop"),
                    "net_income": fc.get("fnp")})
    return out


# ── top10 シミュレーション ────────────────────────────────────────────────

def get_top10_sim_active() -> list:
    return sb.select("gen_top10_sim", "status=eq.active&order=entry_date.desc")


def get_top10_sim_recent_exits(n: int = 20) -> list:
    return sb.select(
        "gen_top10_sim",
        f"status=eq.exited&order=exit_date.desc&limit={n}"
    )


def upsert_top10_sim_entry(entry_date: str, code: str, name: str,
                            entry_net: float, entry_price: float) -> None:
    sb.insert_ignore("gen_top10_sim", [{
        "entry_date": entry_date,
        "code": code,
        "name": name,
        "entry_net": entry_net,
        "entry_price": entry_price,
        "status": "active",
    }], on_conflict="entry_date,code")


def close_top10_sim_position(entry_date: str, code: str,
                              exit_date: str, exit_net: float,
                              exit_price: float) -> None:
    row = sb.select_one(
        "gen_top10_sim",
        f"entry_date=eq.{entry_date}&code=eq.{code}&select=entry_price"
    )
    if not row:
        return
    entry_price = row["entry_price"]
    return_pct = round((exit_price - entry_price) / entry_price * 100, 2) if entry_price else 0
    sb.upsert("gen_top10_sim", [{
        "entry_date": entry_date,
        "code": code,
        "exit_date": exit_date,
        "exit_net": exit_net,
        "exit_price": exit_price,
        "return_pct": return_pct,
        "status": "exited",
    }], on_conflict="entry_date,code")

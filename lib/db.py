"""SQLite persistence layer for stock-alert."""
import os
import sqlite3
from contextlib import contextmanager

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.getenv("STOCK_ALERT_HOME", _PROJECT_DIR)
DB_PATH = os.path.join(_BASE_DIR, "stock_alert.db")

_DDL = """
CREATE TABLE IF NOT EXISTS price_cache (
    code    TEXT NOT NULL,
    date    TEXT NOT NULL,
    close   REAL,
    volume  INTEGER,
    PRIMARY KEY (code, date)
);
CREATE TABLE IF NOT EXISTS simulation_results (
    run_date     TEXT NOT NULL,
    entry_date   TEXT NOT NULL,
    code         TEXT NOT NULL,
    name         TEXT,
    label        TEXT,
    entry_price  REAL,
    current_price REAL,
    return_pct   REAL,
    holding_days INTEGER,
    net_at_entry REAL,
    drop_prob_at_entry REAL,
    PRIMARY KEY (run_date, entry_date, code)
);
CREATE TABLE IF NOT EXISTS daily_ranking (
    date              TEXT NOT NULL,
    code              TEXT NOT NULL,
    name              TEXT,
    close             REAL,
    rise_prob         REAL,
    drop_prob         REAL,
    net               REAL,
    vol               REAL,
    recommend         TEXT,
    rel20             REAL,
    stop_loss         REAL,
    per               REAL,
    pbr               REAL,
    actual_return_63d REAL,
    PRIMARY KEY (date, code)
);
CREATE TABLE IF NOT EXISTS held_scores (
    date      TEXT NOT NULL,
    code      TEXT NOT NULL,
    name      TEXT,
    close     REAL,
    rise_prob REAL,
    drop_prob REAL,
    net       REAL,
    signal    TEXT,
    ret20     REAL,
    vol       REAL,
    rel20     REAL,
    PRIMARY KEY (date, code)
);
CREATE TABLE IF NOT EXISTS earnings_cache (
    code         TEXT PRIMARY KEY,
    next_date    TEXT,
    fetched_date TEXT
);
CREATE TABLE IF NOT EXISTS sector_cache (
    code         TEXT PRIMARY KEY,
    sector       TEXT,
    fetched_date TEXT
);
CREATE TABLE IF NOT EXISTS yutai_cache (
    code          TEXT PRIMARY KEY,
    record_month  INTEGER,
    has_yutai     INTEGER,
    fetched_date  TEXT
);
CREATE TABLE IF NOT EXISTS fundamentals_annual (
    code          TEXT NOT NULL,
    fy_end        TEXT NOT NULL,   -- 決算期末 YYYY-MM（例 2024-03）
    announce_date TEXT,            -- 発表日 YYYY-MM-DD（これ以降にEPS等が既知になる）
    eps           REAL,            -- 1株利益
    dps           REAL,            -- 1株配当（修正1株配）
    roe           REAL,            -- 自己資本利益率 %
    bps           REAL,            -- 1株純資産
    fetched_date  TEXT,
    PRIMARY KEY (code, fy_end)
);
CREATE TABLE IF NOT EXISTS earnings_sentiment (
    code         TEXT NOT NULL,
    fetched_date TEXT NOT NULL,    -- YYYY-MM-DD（当日キャッシュ）
    score        REAL NOT NULL,    -- -1.0〜+1.0（Claude Haiku 分析結果）
    PRIMARY KEY (code, fetched_date)
);
"""

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript(_DDL)


# ── daily_ranking ──────────────────────────────────────────────────────────

def save_daily_ranking(date_str, rows):
    """rows: list of dicts with keys: code, name, close, rise_prob, drop_prob, net, vol, recommend, rel20, stop_loss, per, pbr"""
    init_db()
    sql = """INSERT OR REPLACE INTO daily_ranking
             (date,code,name,close,rise_prob,drop_prob,net,vol,recommend,rel20,stop_loss,per,pbr)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    with _conn() as con:
        con.executemany(sql, [
            (date_str, r.get("code"), r.get("name"), r.get("close"),
             r.get("rise_prob"), r.get("drop_prob"), r.get("net"),
             r.get("vol"), r.get("recommend"), r.get("rel20"),
             r.get("stop_loss"), r.get("per"), r.get("pbr"))
            for r in rows
        ])


# ── held_scores ────────────────────────────────────────────────────────────

def save_held_scores(date_str, results):
    """results: list of dicts from _held_results_from_models"""
    init_db()
    sql = """INSERT OR REPLACE INTO held_scores
             (date,code,name,close,rise_prob,drop_prob,net,signal,ret20,vol,rel20)
             VALUES(?,?,?,?,?,?,?,?,?,?,?)"""
    with _conn() as con:
        con.executemany(sql, [
            (date_str, r["code"], r["name"], r["close"],
             r["prob"], r.get("drop_prob"), r["net"],
             r["signal"], r.get("ret20"), r.get("vol"), r.get("rel20"))
            for r in results
        ])


def load_prev_held_scores(today_str):
    """直前日の保有株スコアを {code: row_dict} で返す"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT MAX(date) AS d FROM held_scores WHERE date < ?", (today_str,)
        ).fetchone()
        if not row or not row["d"]:
            return {}
        rows = con.execute("SELECT * FROM held_scores WHERE date=?", (row["d"],)).fetchall()
        return {r["code"]: dict(r) for r in rows}


# ── earnings_cache ─────────────────────────────────────────────────────────

CACHE_MISS = object()

def get_earnings_cache(code, today_str):
    """今日キャッシュ済みなら next_date(str or None)を返す。未キャッシュはCACHE_MISS。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT next_date, fetched_date FROM earnings_cache WHERE code=?", (str(code),)
        ).fetchone()
    if row and row["fetched_date"] == today_str:
        return row["next_date"]  # "YYYY-MM-DD" or None
    return CACHE_MISS


def set_earnings_cache(code, today_str, next_date_str):
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO earnings_cache (code,next_date,fetched_date) VALUES(?,?,?)",
            (str(code), next_date_str, today_str)
        )


# ── yutai_cache ────────────────────────────────────────────────────────────

def get_yutai_cache(code, today_str):
    """今日キャッシュ済みなら (has_yutai, record_month) を返す。未キャッシュは CACHE_MISS。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT has_yutai, record_month, fetched_date FROM yutai_cache WHERE code=?", (str(code),)
        ).fetchone()
    if row and row["fetched_date"] == today_str:
        return (bool(row["has_yutai"]), row["record_month"])
    return CACHE_MISS


def set_yutai_cache(code, today_str, has_yutai, record_month):
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO yutai_cache (code,record_month,has_yutai,fetched_date) VALUES(?,?,?,?)",
            (str(code), record_month, int(has_yutai), today_str)
        )


# ── fundamentals_annual ──────────────────────────────────────────────────

def upsert_fundamentals_annual(code, rows, today_str):
    """rows: list of dict(fy_end, announce_date, eps, dps, roe, bps)。code単位でまとめてupsert。"""
    init_db()
    with _conn() as con:
        # dps 列が無い場合は ALTER TABLE で追加（既存DBの後方互換）
        cols = {r[1] for r in con.execute("PRAGMA table_info(fundamentals_annual)").fetchall()}
        if "dps" not in cols:
            con.execute("ALTER TABLE fundamentals_annual ADD COLUMN dps REAL")
        con.executemany(
            """INSERT OR REPLACE INTO fundamentals_annual
               (code, fy_end, announce_date, eps, dps, roe, bps, fetched_date)
               VALUES(?,?,?,?,?,?,?,?)""",
            [(str(code), r["fy_end"], r.get("announce_date"), r.get("eps"),
              r.get("dps"), r.get("roe"), r.get("bps"), today_str) for r in rows]
        )


def get_fundamentals_annual(code):
    """code の年度別ファンダを発表日昇順で返す。"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT fy_end, announce_date, eps, dps, roe, bps FROM fundamentals_annual
               WHERE code=? AND announce_date IS NOT NULL ORDER BY announce_date""",
            (str(code),)
        ).fetchall()
    return [dict(r) for r in rows]


def load_all_fundamentals_annual():
    """全銘柄の年度別ファンダを {code: [rows...]} で返す（バックテスト用バルク読込）。"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT code, fy_end, announce_date, eps, dps, roe, bps FROM fundamentals_annual
               WHERE announce_date IS NOT NULL ORDER BY code, announce_date"""
        ).fetchall()
    out = {}
    for r in rows:
        out.setdefault(str(r["code"]), []).append(dict(r))
    return out


def get_fundamentals_codes_count():
    init_db()
    with _conn() as con:
        return con.execute(
            "SELECT COUNT(DISTINCT code) FROM fundamentals_annual"
        ).fetchone()[0]


# ── sector_cache ───────────────────────────────────────────────────────────

def get_all_sectors():
    init_db()
    with _conn() as con:
        rows = con.execute("SELECT code, sector FROM sector_cache").fetchall()
        return {r["code"]: r["sector"] for r in rows}


def save_simulation_results(run_date, rows):
    """rows: list of dicts with keys matching simulation_results columns"""
    init_db()
    sql = """INSERT OR REPLACE INTO simulation_results
             (run_date,entry_date,code,name,label,entry_price,current_price,return_pct,holding_days,net_at_entry,drop_prob_at_entry)
             VALUES(?,?,?,?,?,?,?,?,?,?,?)"""
    with _conn() as con:
        con.executemany(sql, [
            (run_date, r["entry_date"], r["code"], r.get("name"), r.get("label"),
             r.get("entry_price"), r.get("current_price"), r.get("return_pct"),
             r.get("holding_days"), r.get("net_at_entry"), r.get("drop_prob_at_entry"))
            for r in rows
        ])


def load_simulation_results(run_date=None):
    """run_dateを指定すればその日の結果、Noneなら全件"""
    init_db()
    with _conn() as con:
        if run_date:
            rows = con.execute(
                "SELECT * FROM simulation_results WHERE run_date=? ORDER BY entry_date, return_pct DESC",
                (run_date,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM simulation_results ORDER BY run_date, entry_date, return_pct DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_holding_days(codes, today_str):
    """held_scores の MIN(date) から各コードの推定保有日数を返す。
    DBに記録がないコード（新規追加銘柄）はキーなしで返る。"""
    if not codes:
        return {}
    from datetime import datetime as _dt
    today = _dt.strptime(today_str, "%Y-%m-%d").date()
    init_db()
    result = {}
    with _conn() as con:
        for code in codes:
            row = con.execute(
                "SELECT MIN(date) AS first_date FROM held_scores WHERE code=?",
                (str(code),)
            ).fetchone()
            if row and row["first_date"]:
                first = _dt.strptime(row["first_date"], "%Y-%m-%d").date()
                result[str(code)] = (today - first).days
    return result


# ── price_cache ────────────────────────────────────────────────────────────

def get_price_cache_coverage(code):
    """キャッシュの (min_date_str, max_date_str) を返す。未キャッシュは None。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT MIN(date) AS mn, MAX(date) AS mx FROM price_cache WHERE code=?",
            (str(code),)
        ).fetchone()
    if row and row["mn"]:
        return row["mn"], row["mx"]
    return None


def get_price_cache(code, start_date_str, end_date_str):
    """キャッシュから DataFrame(Close, Volume) を返す。データ不足なら None。"""
    import pandas as pd
    from datetime import date as _date
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT date, close, volume FROM price_cache "
            "WHERE code=? AND date>=? AND date<=? ORDER BY date",
            (str(code), start_date_str, end_date_str)
        ).fetchall()
    if len(rows) < 100:
        return None
    dates  = [r["date"]  for r in rows]
    closes = [r["close"] for r in rows]
    vols   = [r["volume"] for r in rows]
    idx = [_date.fromisoformat(d) for d in dates]
    return pd.DataFrame({"Close": closes, "Volume": vols}, index=idx)


def save_price_cache(code, df):
    """DataFrame を price_cache に INSERT OR IGNORE で保存。"""
    import math
    init_db()
    rows = []
    for idx, row in df.iterrows():
        # 日付を YYYY-MM-DD 形式に統一（Timestamp の isoformat は時刻付きになるため）
        d = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        cv = row.get("Close")
        vv = row.get("Volume")
        c = float(cv) if cv is not None and not (isinstance(cv, float) and math.isnan(cv)) else None
        v = int(vv)   if vv is not None and not (isinstance(vv, float) and math.isnan(vv)) else None
        rows.append((str(code), d, c, v))
    if not rows:
        return
    with _conn() as con:
        con.executemany(
            "INSERT OR IGNORE INTO price_cache (code, date, close, volume) VALUES (?,?,?,?)",
            rows
        )


# ── earnings_sentiment ────────────────────────────────────────────────────

def get_earnings_sentiment(code: str, today_str: str):
    """今日キャッシュ済みならスコア(float)を返す。未キャッシュは None。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT score FROM earnings_sentiment WHERE code=? AND fetched_date=?",
            (str(code), today_str)
        ).fetchone()
    return float(row["score"]) if row else None


def set_earnings_sentiment(code: str, today_str: str, score: float):
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO earnings_sentiment (code, fetched_date, score) VALUES(?,?,?)",
            (str(code), today_str, float(score))
        )


def save_all_sectors(sector_map):
    from datetime import date
    today_str = date.today().isoformat()
    init_db()
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO sector_cache (code,sector,fetched_date) VALUES(?,?,?)",
            [(code, sector, today_str) for code, sector in sector_map.items()]
        )

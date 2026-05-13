"""SQLite persistence layer for stock-alert."""
import os
import sqlite3
from contextlib import contextmanager

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.getenv("STOCK_ALERT_HOME", _PROJECT_DIR)
DB_PATH = os.path.join(_BASE_DIR, "stock_alert.db")

_DDL = """
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
"""

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
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


# ── sector_cache ───────────────────────────────────────────────────────────

def get_all_sectors():
    init_db()
    with _conn() as con:
        rows = con.execute("SELECT code, sector FROM sector_cache").fetchall()
        return {r["code"]: r["sector"] for r in rows}


def save_all_sectors(sector_map):
    from datetime import date
    today_str = date.today().isoformat()
    init_db()
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO sector_cache (code,sector,fetched_date) VALUES(?,?,?)",
            [(code, sector, today_str) for code, sector in sector_map.items()]
        )

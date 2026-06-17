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
    piotroski         REAL,
    bps_growth        REAL,
    eps_surprise      REAL,
    pos52             REAL,
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
CREATE TABLE IF NOT EXISTS margin_data (
    code         TEXT NOT NULL,
    week_date    TEXT NOT NULL,    -- YYYY-MM-DD（週次確定日）
    buy_balance  REAL,             -- 信用買残（株）
    sell_balance REAL,             -- 信用売残（株）
    ratio        REAL,             -- 信用倍率
    fetched_date TEXT,
    PRIMARY KEY (code, week_date)
);
CREATE TABLE IF NOT EXISTS short_interest (
    code          TEXT NOT NULL,
    week_date     TEXT NOT NULL,   -- YYYY-MM-DD（TSE公開週）
    short_balance REAL,            -- 残高株数
    short_amount  REAL,            -- 残高金額（百万円）
    PRIMARY KEY (code, week_date)
);
CREATE TABLE IF NOT EXISTS tdnet_events (
    code         TEXT NOT NULL,
    announce_date TEXT NOT NULL,   -- YYYY-MM-DD
    title        TEXT,
    event_type   TEXT,             -- buyback | upward | downward | dividend | other
    fetched_date TEXT,
    PRIMARY KEY (code, announce_date, title)
);
CREATE TABLE IF NOT EXISTS market_index_cache (
    ticker  TEXT NOT NULL,  -- 'VIX' | 'SP500' | 'USDJPY'
    date    TEXT NOT NULL,  -- YYYY-MM-DD
    close   REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS jquants_fin_summary (
    code          TEXT NOT NULL,
    disc_date     TEXT NOT NULL,   -- YYYY-MM-DD（開示日 = point-in-time key）
    doc_type      TEXT,            -- FY / 1Q / 2Q / 3Q
    fy_end        TEXT,            -- 決算期末 YYYY-MM-DD
    np            REAL,            -- 当期純利益
    cfo           REAL,            -- 営業キャッシュフロー
    ta            REAL,            -- 総資産
    equity        REAL,            -- 純資産
    eps           REAL,            -- 1株利益
    bps           REAL,            -- 1株純資産
    div_ann       REAL,            -- 年間配当（1株）
    payout_ratio  REAL,            -- 配当性向
    sh_out        REAL,            -- 発行済株式数
    tr_sh         REAL,            -- 自己株式数
    fnp           REAL,            -- 通期会社予想・当期純利益（進捗率用）
    fop           REAL,            -- 通期会社予想・営業利益
    fsales        REAL,            -- 通期会社予想・売上高
    PRIMARY KEY (code, disc_date)
);
CREATE TABLE IF NOT EXISTS top10_sim (
    entry_date   TEXT NOT NULL,   -- 買いエントリー日
    code         TEXT NOT NULL,   -- 銘柄コード
    name         TEXT,            -- 銘柄名
    entry_net    REAL,            -- エントリー時ネットスコア(%)
    entry_price  REAL,            -- エントリー株価
    exit_date    TEXT,            -- 売却日（NULLなら保有中）
    exit_net     REAL,            -- 売却時ネットスコア(%)
    exit_price   REAL,            -- 売却株価
    return_pct   REAL,            -- 実現リターン(%)
    status       TEXT DEFAULT 'active',  -- 'active' | 'exited'
    PRIMARY KEY (entry_date, code)
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
    """rows: list of dicts with keys: code, name, close, rise_prob, drop_prob, net, vol, recommend, rel20, stop_loss, per, pbr, piotroski, bps_growth, eps_surprise, pos52"""
    init_db()
    # カラム追加（既存DBへのマイグレーション）
    with _conn() as con:
        existing = {r[1] for r in con.execute("PRAGMA table_info(daily_ranking)")}
        for col, typ in [("piotroski","REAL"),("bps_growth","REAL"),("eps_surprise","REAL"),("pos52","REAL")]:
            if col not in existing:
                con.execute(f"ALTER TABLE daily_ranking ADD COLUMN {col} {typ}")
    sql = """INSERT OR REPLACE INTO daily_ranking
             (date,code,name,close,rise_prob,drop_prob,net,vol,recommend,rel20,stop_loss,per,pbr,
              piotroski,bps_growth,eps_surprise,pos52)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    with _conn() as con:
        con.executemany(sql, [
            (date_str, r.get("code"), r.get("name"), r.get("close"),
             r.get("rise_prob"), r.get("drop_prob"), r.get("net"),
             r.get("vol"), r.get("recommend"), r.get("rel20"),
             r.get("stop_loss"), r.get("per"), r.get("pbr"),
             r.get("piotroski"), r.get("bps_growth"), r.get("eps_surprise"), r.get("pos52"))
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


# ── margin_data ──────────────────────────────────────────────────────────

def upsert_margin_data(code: str, week_date: str, buy: float, sell: float, ratio: float):
    init_db()
    from datetime import date
    today_str = date.today().isoformat()
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO margin_data
               (code, week_date, buy_balance, sell_balance, ratio, fetched_date)
               VALUES(?,?,?,?,?,?)""",
            (str(code), week_date, buy, sell, ratio, today_str)
        )


def get_margin_data_latest(code: str):
    """最新の信用倍率データを返す。なければ None。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT week_date, buy_balance, sell_balance, ratio FROM margin_data "
            "WHERE code=? ORDER BY week_date DESC LIMIT 1",
            (str(code),)
        ).fetchone()
    return dict(row) if row else None


def get_margin_data_history(code: str, n: int = 8):
    """過去 n 週の信用倍率履歴を返す（新しい順）。"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT week_date, buy_balance, sell_balance, ratio FROM margin_data "
            "WHERE code=? ORDER BY week_date DESC LIMIT ?",
            (str(code), n)
        ).fetchall()
    return [dict(r) for r in rows]


# ── short_interest ────────────────────────────────────────────────────────

def bulk_upsert_short_interest(rows):
    """rows: list of (code, week_date, short_balance, short_amount)"""
    init_db()
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO short_interest (code, week_date, short_balance, short_amount) VALUES(?,?,?,?)",
            rows
        )


def get_short_interest_latest(code: str):
    """最新の空売り残高データを返す。なければ None。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT week_date, short_balance, short_amount FROM short_interest "
            "WHERE code=? ORDER BY week_date DESC LIMIT 1",
            (str(code),)
        ).fetchone()
    return dict(row) if row else None


# ── tdnet_events ──────────────────────────────────────────────────────────

def upsert_tdnet_events(code: str, events: list):
    """events: list of {'announce_date', 'title', 'event_type'}"""
    init_db()
    from datetime import date
    today_str = date.today().isoformat()
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO tdnet_events (code, announce_date, title, event_type, fetched_date) VALUES(?,?,?,?,?)",
            [(str(code), e["announce_date"], e["title"], e["event_type"], today_str) for e in events]
        )


def get_tdnet_events_recent(code: str, days: int = 60):
    """直近 days 日以内の適時開示イベントを返す（新しい順）。"""
    init_db()
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _conn() as con:
        rows = con.execute(
            "SELECT announce_date, title, event_type FROM tdnet_events "
            "WHERE code=? AND announce_date >= ? ORDER BY announce_date DESC",
            (str(code), cutoff)
        ).fetchall()
    return [dict(r) for r in rows]


def save_all_sectors(sector_map):
    from datetime import date
    today_str = date.today().isoformat()
    init_db()
    with _conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO sector_cache (code,sector,fetched_date) VALUES(?,?,?)",
            [(code, sector, today_str) for code, sector in sector_map.items()]
        )


# ── market_index_cache ─────────────────────────────────────────────────────
# VIX / S&P500 / USD/JPY の日次終値をDBキャッシュ。毎回Yahoo取得をなくす。

def get_market_index_latest_date(ticker: str):
    """DBに保存されている最新日付文字列を返す。なければ None。"""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT MAX(date) AS mx FROM market_index_cache WHERE ticker=?",
            (ticker,)
        ).fetchone()
    return row["mx"] if row and row["mx"] else None


def save_market_index_data(ticker: str, df):
    """date-indexed DataFrame(Close列)を market_index_cache に一括保存。INSERT OR IGNORE。"""
    import math
    if df is None or len(df) == 0:
        return
    init_db()
    rows = []
    for idx, row in df.iterrows():
        d = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        c = row["Close"]
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        rows.append((ticker, d, float(c)))
    if rows:
        with _conn() as con:
            con.executemany(
                "INSERT OR IGNORE INTO market_index_cache (ticker, date, close) VALUES(?,?,?)",
                rows
            )


def load_market_index_data(ticker: str, days: int = 2200):
    """DBから直近 days 日分を date-indexed DataFrame(Close) で返す。なければ None。"""
    import pandas as pd
    from datetime import date, timedelta
    init_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with _conn() as con:
        rows = con.execute(
            "SELECT date, close FROM market_index_cache "
            "WHERE ticker=? AND date>=? ORDER BY date",
            (ticker, cutoff)
        ).fetchall()
    if not rows:
        return None
    dates  = [r["date"]  for r in rows]
    closes = [r["close"] for r in rows]
    from datetime import date as _date
    idx = [_date.fromisoformat(d) for d in dates]
    return pd.DataFrame({"Close": closes}, index=idx)


# ── margin_data / short_interest: point-in-time lookup ────────────────────

def get_margin_ratio_at(code: str, as_of_date: str) -> "float | None":
    """
    as_of_date（YYYY-MM-DD）時点で利用可能な最新の信用倍率を返す。
    週次データなので as_of_date 以前の最新値を返す。
    """
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT ratio FROM margin_data "
            "WHERE code=? AND week_date <= ? ORDER BY week_date DESC LIMIT 1",
            (str(code), as_of_date)
        ).fetchone()
    return float(row["ratio"]) if row and row["ratio"] is not None else None


def bulk_upsert_jquants_fin_summary(rows: list):
    """
    rows: list of dict with keys matching jquants_fin_summary columns
    (code, disc_date, doc_type, fy_end, np, cfo, ta, equity, eps, bps, div_ann, payout_ratio, sh_out, tr_sh)
    """
    init_db()
    with _conn() as con:
        # 既存DBへの予想カラム追加（マイグレーション）
        existing = {r[1] for r in con.execute("PRAGMA table_info(jquants_fin_summary)").fetchall()}
        for col in ("fnp", "fop", "fsales"):
            if col not in existing:
                con.execute(f"ALTER TABLE jquants_fin_summary ADD COLUMN {col} REAL")
        con.executemany(
            """INSERT OR REPLACE INTO jquants_fin_summary
               (code, disc_date, doc_type, fy_end, np, cfo, ta, equity, eps, bps,
                div_ann, payout_ratio, sh_out, tr_sh, fnp, fop, fsales)
               VALUES(:code,:disc_date,:doc_type,:fy_end,:np,:cfo,:ta,:equity,:eps,:bps,
                      :div_ann,:payout_ratio,:sh_out,:tr_sh,:fnp,:fop,:fsales)""",
            rows
        )


def get_jquants_fin_history(code: str, as_of_date: str, n: int = 4) -> list:
    """
    as_of_date 以前に開示された最新 n 件の財務サマリーを返す（新しい順）。
    FYのみに絞る場合は doc_type='FY' でフィルタ可能。
    """
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM jquants_fin_summary
               WHERE code=? AND disc_date <= ?
               ORDER BY disc_date DESC LIMIT ?""",
            (str(code), as_of_date, n)
        ).fetchall()
    return [dict(r) for r in rows]


def get_jquants_fin_history_fy(code: str, as_of_date: str, n: int = 3) -> list:
    """FYレポートのみ n 件（年次）"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM jquants_fin_summary
               WHERE code=? AND disc_date <= ? AND doc_type LIKE 'FY%'
               ORDER BY disc_date DESC LIMIT ?""",
            (str(code), as_of_date, n)
        ).fetchall()
    return [dict(r) for r in rows]


def get_short_balance_at(code: str, as_of_date: str) -> "dict | None":
    """
    as_of_date（YYYY-MM-DD）時点で利用可能な最新の空売り残高を返す。
    {'week_date': str, 'short_balance': float, 'short_amount': float | None}
    """
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT week_date, short_balance, short_amount FROM short_interest "
            "WHERE code=? AND week_date <= ? ORDER BY week_date DESC LIMIT 1",
            (str(code), as_of_date)
        ).fetchone()
    return dict(row) if row else None


# ── top10シミュレーション ─────────────────────────────────────────────────

def get_top10_sim_active() -> list:
    """現在アクティブなシミュレーションポジション一覧を返す。"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM top10_sim WHERE status='active' ORDER BY entry_date DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_top10_sim_recent_exits(n: int = 20) -> list:
    """直近 n 件の決済済みポジションを返す。"""
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM top10_sim WHERE status='exited' ORDER BY exit_date DESC LIMIT ?",
            (n,)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_top10_sim_entry(entry_date: str, code: str, name: str,
                            entry_net: float, entry_price: float) -> None:
    """新規エントリーを登録（既存なら無視）。"""
    init_db()
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO top10_sim
               (entry_date, code, name, entry_net, entry_price, status)
               VALUES (?, ?, ?, ?, ?, 'active')""",
            (entry_date, code, name, entry_net, entry_price)
        )


def close_top10_sim_position(entry_date: str, code: str,
                              exit_date: str, exit_net: float,
                              exit_price: float) -> None:
    """ポジションを決済済みに更新。"""
    init_db()
    with _conn() as con:
        con.execute(
            """UPDATE top10_sim SET
               exit_date=?, exit_net=?, exit_price=?,
               return_pct=ROUND((exit_price - entry_price) / entry_price * 100, 2),
               status='exited'
               WHERE entry_date=? AND code=?""",
            (exit_date, exit_net, exit_price, entry_date, code)
        )

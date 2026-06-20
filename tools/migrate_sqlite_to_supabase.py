#!/usr/bin/env python3
"""
tools/migrate_sqlite_to_supabase.py

既存の stock_alert.db（SQLite）の全テーブルを Supabase へ一括移行する一回限りのスクリプト。
GitHub Actions 上で DB キャッシュを復元してから実行する（実データはローカルではなくキャッシュ内）。

冪等: Supabase 側は upsert（PK衝突は上書き）なので何度流しても安全。
"""
import sys, os, sqlite3, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock_alert.db")

# SQLiteテーブル → (Supabaseテーブル, on_conflict列, カラム変換マップ)
# 値が None のマップは「列名そのまま」
TABLE_MAP = {
    "price_cache":         ("yahoo_price_cache",   "code,date", None),
    "simulation_results":  ("simulation_results",  "run_date,entry_date,code", None),
    "tdnet_events":        ("tdnet_events",        "code,announce_date,title", None),
    "market_index_cache":  ("yahoo_market_index",  "ticker,date", None),
    "jquants_fin_summary": ("jquants_fin_summary", "code,disc_date", None),
    "edinet_holdings":     ("edinet_large_holdings", "doc_id", None),
    "top10_sim":           ("gen_top10_sim",           "entry_date,code", None),
    # daily_ranking は gen_rankings に統合（rank列は後でexport_to_webが付与するためNULL可）
    "daily_ranking":       ("gen_rankings",        "date,code", None),
    # earnings_cache → kabutan_earnings, sector_cache → gen_stock_meta
    "earnings_cache":      ("kabutan_earnings",        "code", None),
    "sector_cache":        ("gen_stock_meta",      "code", None),
}


def clean_value(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def migrate_table(con, sqlite_table, only=None):
    sb_table, on_conflict, colmap = TABLE_MAP[sqlite_table]
    if only and sqlite_table not in only:
        return
    cur = con.execute(f"SELECT * FROM {sqlite_table}")
    cols = [d[0] for d in cur.description]
    total = 0
    batch = []
    BATCH = 1000
    for row in cur:
        rec = {cols[i]: clean_value(row[i]) for i in range(len(cols))}
        batch.append(rec)
        if len(batch) >= BATCH:
            sb.upsert(sb_table, batch, on_conflict=on_conflict)
            total += len(batch)
            batch = []
            print(f"  {sqlite_table} → {sb_table}: {total}行...")
    if batch:
        sb.upsert(sb_table, batch, on_conflict=on_conflict)
        total += len(batch)
    print(f"✓ {sqlite_table} → {sb_table}: 計{total}行")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="移行するSQLiteテーブル名（省略時は全テーブル）")
    args = p.parse_args()

    if not sb.is_configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY 未設定。中止。")
        sys.exit(1)
    if not os.path.exists(DB_PATH):
        print(f"{DB_PATH} が見つからない。DBキャッシュを復元してから実行せよ。")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for sqlite_table in TABLE_MAP:
        if sqlite_table not in existing:
            print(f"⏭ {sqlite_table}（SQLiteに無し・スキップ）")
            continue
        n = con.execute(f"SELECT COUNT(*) FROM {sqlite_table}").fetchone()[0]
        if n == 0:
            print(f"⏭ {sqlite_table}（0行・スキップ）")
            continue
        print(f"▶ {sqlite_table}（{n}行）を移行中...")
        migrate_table(con, sqlite_table, only=args.only)

    con.close()
    print("\n移行完了。")


if __name__ == "__main__":
    main()

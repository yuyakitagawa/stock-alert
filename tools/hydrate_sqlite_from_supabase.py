#!/usr/bin/env python3
"""
tools/hydrate_sqlite_from_supabase.py

Supabase から指定テーブルをローカル stock_alert.db（SQLite）へ流し込む。
重い分析ツール（catalyst_backtest 等の全銘柄 point-in-time クエリ）用の
読み取りアクセラレータ。Supabaseが真実源で、これは使い捨てローカルキャッシュ。

使い方:
  python tools/hydrate_sqlite_from_supabase.py yahoo_price_cache jquants_fin_summary
"""
import sys, os, sqlite3, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb
from lib.db import DB_PATH

# テーブル → (列定義SQL, PK)
SCHEMAS = {
    "yahoo_price_cache": (
        "code TEXT, date TEXT, close REAL, volume INTEGER, PRIMARY KEY(code,date)",
        ["code", "date", "close", "volume"],
    ),
    "jquants_fin_summary": (
        "code TEXT, disc_date TEXT, doc_type TEXT, fy_end TEXT, np REAL, cfo REAL, "
        "ta REAL, equity REAL, eps REAL, bps REAL, div_ann REAL, payout_ratio REAL, "
        "sh_out REAL, tr_sh REAL, fnp REAL, fop REAL, fsales REAL, op REAL, sales REAL, "
        "PRIMARY KEY(code,disc_date)",
        ["code", "disc_date", "doc_type", "fy_end", "np", "cfo", "ta", "equity", "eps",
         "bps", "div_ann", "payout_ratio", "sh_out", "tr_sh", "fnp", "fop", "fsales", "op", "sales"],
    ),
    "kabutan_fundamentals": (
        "code TEXT, fy_end TEXT, announce_date TEXT, eps REAL, dps REAL, roe REAL, "
        "bps REAL, fetched_date TEXT, PRIMARY KEY(code,fy_end)",
        ["code", "fy_end", "announce_date", "eps", "dps", "roe", "bps", "fetched_date"],
    ),
    "edinet_large_holdings": (
        "doc_id TEXT PRIMARY KEY, sec_code TEXT, filer_name TEXT, doc_type_code TEXT, "
        "doc_description TEXT, submit_date TEXT, disc_date TEXT, holding_ratio REAL, fetched_date TEXT",
        ["doc_id", "sec_code", "filer_name", "doc_type_code", "doc_description",
         "submit_date", "disc_date", "holding_ratio", "fetched_date"],
    ),
}


def hydrate(table: str, con):
    if table not in SCHEMAS:
        print(f"⏭ {table}: スキーマ未定義・スキップ")
        return
    ddl, cols = SCHEMAS[table]
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} ({ddl})")
    placeholders = ",".join("?" for _ in cols)
    insert = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"

    page = 1000
    offset = 0
    total = 0
    while True:
        rows = sb.select(table, f"order={cols[0]}.asc&limit={page}&offset={offset}")
        if not rows:
            break
        con.executemany(insert, [[r.get(c) for c in cols] for r in rows])
        con.commit()
        total += len(rows)
        offset += len(rows)
        if total % 50000 == 0 or len(rows) < page:
            print(f"  {table}: {total}行...")
        if len(rows) < page:
            break
    print(f"✓ {table}: {total}行をローカルSQLiteへ投入")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tables", nargs="+", help="流し込むテーブル名")
    args = p.parse_args()

    if not sb.is_configured():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY 未設定。中止。")
        sys.exit(1)

    con = sqlite3.connect(DB_PATH)
    for t in args.tables:
        hydrate(t, con)
    con.close()
    print("hydrate完了。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
tools/fetch_employees.py — 従業員数（company_employees）を古い順に更新する

daily_alert.yml Step 2h が毎日 --limit 400 で呼ぶ（約3,800銘柄を10日弱で一巡）。
従業員数は年1回（有価証券報告書）しか変わらないので、取得日から --max-age-days
以内の銘柄は取り直さない。未取得の銘柄（新規上場など）を先に取る。

使い方:
  python tools/fetch_employees.py                 # 古い順に400銘柄
  python tools/fetch_employees.py --limit 0       # 対象を全部（初回の一括投入）
"""
import argparse
import os
import sys
from datetime import date

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import employees
    from lib.price_store import rpc_codes

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="1回に取る銘柄数（0で全部）")
    ap.add_argument("--max-age-days", type=int, default=30)
    a = ap.parse_args()

    codes = rpc_codes()
    if not codes:
        print("[employees] 銘柄一覧が取れないため終了")
        return 1
    rows = employees.load()
    todo = employees.stale_codes(codes, rows, a.max_age_days, date.today())
    if a.limit:
        todo = todo[: a.limit]
    print(f"[employees] 銘柄 {len(codes)} / 保存済み {len(rows)} / 今回取得 {len(todo)}")
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        limited = False
        try:
            chunk = employees.fetch_many(batch)
        except employees.RateLimited as e:
            chunk, limited = e.args[0], True
        ok = employees.save(chunk)
        got = sum(v is not None for v in chunk.values())
        print(f"[employees] {i + len(batch)}/{len(todo)} 取得{len(chunk)} 値あり{got} 保存{'OK' if ok else '失敗'}", flush=True)
        if not ok:
            return 1
        if limited:
            # 取れなかった銘柄は保存していないので、次回また未取得として先頭に来る
            print("[employees] Yahooのレート制限に当たったため打ち切ります（残りは次回）")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

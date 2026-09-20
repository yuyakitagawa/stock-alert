#!/usr/bin/env python3
"""
tools/fetch_earnings_forecast.py — 決算短信（TDnet）の会社予想を取り込む

daily_alert.yml Step 2d2 が毎日 --days 3 で呼ぶ。J-Quants Free の提供終了（2026-04-24分まで）で
会社予想が入らなくなったため、決算短信のXBRLから取り直す。詳細は lib/earnings_forecast.py。

使い方:
  python tools/fetch_earnings_forecast.py                      # 直近3日
  python tools/fetch_earnings_forecast.py --days 14            # 直近14日
  python tools/fetch_earnings_forecast.py --from 2026-04-25 --to 2026-09-19   # 期間指定の取り直し
  python tools/fetch_earnings_forecast.py --days 1 --dry-run   # 保存せず件数だけ見る
"""
import argparse
import os
import sys
import time
from datetime import date, timedelta

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import earnings_forecast as ef
    import lib.supabase_client as sb

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--sleep", type=float, default=0.5, help="XBRL取得の間隔（TDnetへの負荷を抑える）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.date_from:
        start = date.fromisoformat(a.date_from)
        end = date.fromisoformat(a.date_to) if a.date_to else date.today()
    else:
        end = date.today()
        start = end - timedelta(days=a.days - 1)

    total_rows, total_docs, saved = 0, 0, 0
    d = start
    while d <= end:
        items = ef.list_summaries(d)
        total_docs += len(items)
        rows = []
        for it in items:
            row = ef.fetch_forecast(it)
            if row:
                rows.append(row)
            if a.sleep:
                time.sleep(a.sleep)
        total_rows += len(rows)
        if rows and not a.dry_run:
            if sb.upsert("jquants_fin_summary", rows, on_conflict="code,disc_date"):
                saved += len(rows)
            else:
                print(f"[forecast] {d} の保存に失敗")
                return 1
        print(f"[forecast] {d}: 決算短信{len(items)}件 → 予想{len(rows)}件", flush=True)
        d += timedelta(days=1)

    print(f"[forecast] 合計: 短信{total_docs}件 / 予想{total_rows}件 / 保存{saved}件"
          + ("（--dry-run のため保存なし）" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

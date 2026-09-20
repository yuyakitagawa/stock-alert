#!/usr/bin/env python3
"""
tools/fetch_company_profiles.py — 有報の事業情報を company_profile に取り込む

daily_alert.yml Step 2d3 が毎日 --days 3 で呼ぶ。EDINETに出た有価証券報告書（docTypeCode=120）
から「事業の内容」「対処すべき課題」「従業員の状況」を抜いて保存する。
押し目買いの通知に「どんな会社か」を1行添えるのが用途（詳細は lib/company_profile.py）。

使い方:
  python tools/fetch_company_profiles.py                          # 直近3日
  python tools/fetch_company_profiles.py --days 30                # 直近30日
  python tools/fetch_company_profiles.py --from 2026-06-01 --to 2026-07-31   # 期間指定
  python tools/fetch_company_profiles.py --days 1 --dry-run       # 保存せず件数だけ
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
    from lib import company_profile as cp
    from lib.edinet import fetch_documents_list, _fetch_xbrl_text

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--sleep", type=float, default=0.5, help="EDINETへの間隔")
    ap.add_argument("--force", action="store_true", help="保存済みの銘柄も取り直す")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.date_from:
        start = date.fromisoformat(a.date_from)
        end = date.fromisoformat(a.date_to) if a.date_to else date.today()
    else:
        end = date.today()
        start = end - timedelta(days=a.days - 1)

    have = set() if a.force else set(cp.load().keys())
    print(f"[profile] 保存済み {len(have)}銘柄 / 対象 {start}〜{end}")
    total_docs = saved = skipped = 0
    d = start
    while d <= end:
        docs = [x for x in fetch_documents_list(d)
                if x.get("docTypeCode") == cp.DOC_TYPE_ANNUAL and x.get("secCode")]
        rows = []
        for doc in docs:
            code = (doc.get("secCode") or "")[:4]
            if code in have:
                skipped += 1
                continue
            xbrl = _fetch_xbrl_text(doc["docID"])
            row = cp.row_from_document(doc, xbrl or "")
            if row:
                rows.append(row)
                have.add(code)
            if a.sleep:
                time.sleep(a.sleep)
        total_docs += len(docs)
        if rows and not a.dry_run:
            if not cp.save(rows):
                print(f"[profile] {d} の保存に失敗")
                return 1
            saved += len(rows)
        print(f"[profile] {d}: 有報{len(docs)}件 → 保存{len(rows)}件", flush=True)
        d += timedelta(days=1)

    print(f"[profile] 合計: 有報{total_docs}件 / 保存{saved}件 / 取得済みでスキップ{skipped}件"
          + ("（--dry-run のため保存なし）" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

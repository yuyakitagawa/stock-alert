#!/usr/bin/env python3
"""
tools/fetch_company_info.py — 会社HP URLと最新決算（company_info）を古い順に更新する

daily_alert.yml Step 2i が毎日 --limit 400 で呼ぶ（普通株 約3,800銘柄を10日弱で一巡）。
直近2日に決算短信を出した銘柄（TDnet）を先頭に回すので、決算は開示の翌営業日には入る。
会社サイトはほぼ変わらないので --website-max-age-days（既定90日）以内は取り直さない。
最新の決算短信の URL（tanshin_*）は株探を見に行かず、TDnet の保存分（ext_tdnet_disclosures）から
全銘柄ぶんを毎回付け直す（変わった銘柄だけ書く）。その短信の XBRL から会社予想（tanshin_fc_*）も読む。

使い方:
  python tools/fetch_company_info.py                 # 優先順に400銘柄
  python tools/fetch_company_info.py --limit 0       # 対象を全部（初回の一括投入・約2時間）
  python tools/fetch_company_info.py --dry-run --limit 3
"""
import argparse
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import company_info as ci

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="1回に取る銘柄数（0で全部）")
    ap.add_argument("--max-age-days", type=int, default=7)
    ap.add_argument("--website-max-age-days", type=int, default=90)
    ap.add_argument("--sleep", type=float, default=1.0, help="株探へのリクエスト間隔（秒）")
    ap.add_argument("--max-tanshin", type=int, default=300, help="1回に読む決算短信XBRLの上限（残りは次回）")
    ap.add_argument("--dry-run", action="store_true", help="DB保存しない")
    a = ap.parse_args()

    codes = ci.target_codes()
    if not codes:
        print("[company_info] 銘柄一覧が取れないため終了")
        return 1
    saved = ci.load()
    announced = ci.recent_earnings_announcements()
    todo, website = ci.plan(codes, saved, announced, datetime.now(timezone.utc),
                            a.max_age_days, a.website_max_age_days)
    if a.limit:
        todo = todo[: a.limit]
        website &= set(todo)
    print(f"[company_info] 対象 {len(codes)} / 保存済み {len(saved)} / 決算短信 {len(announced)} / "
          f"今回 {len(todo)}（うち会社サイト {len(website)}）")

    tanshin = ci.tanshin_updates(saved, {c: r for c, r in ci.latest_tanshin().items() if c in set(codes)})
    tanshin = ci.with_tanshin_forecasts(tanshin, a.max_tanshin)
    ok = True if a.dry_run else ci.save(tanshin)
    n_fc = sum(r.get("tanshin_fc_np") is not None for r in tanshin)
    n_retry = sum(r.get("tanshin_fc_fetched_at") is None for r in tanshin)
    print(f"[company_info] 決算短信 更新 {len(tanshin)}（予想あり {n_fc} / 読めず次回 {n_retry}） "
          f"保存{'OK' if ok else '失敗'}", flush=True)

    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        limited = False
        try:
            rows = ci.fetch_many(batch, website, a.sleep)
        except ci.RateLimited as e:
            rows, limited = e.args[0], True
        if a.dry_run:
            for r in rows:
                print(r)
            ok = True
        else:
            ok = ci.save(rows)
        got = sum(r.get("fy_period") is not None or r.get("latest_period") is not None for r in rows)
        print(f"[company_info] {i + len(batch)}/{len(todo)} 取得{len(rows)} 決算あり{got} "
              f"保存{'OK' if ok else '失敗'}", flush=True)
        if not ok:
            return 1
        if limited:
            # 取れなかった銘柄は保存していないので、次回また古い順で先頭側に来る
            print("[company_info] 株探に続けて断られたため打ち切ります（残りは次回）")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

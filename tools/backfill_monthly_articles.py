"""過去月の大量保有記事バックフィル（/monthly アーカイブの欠落月を埋める一時スクリプト）

- 対象: disc_date 2025-06-18 〜 2026-06-30（日次自動投稿が始まった2026-07より前）
- 各月 |holding_ratio| 上位 N 件のみ記事化（既存記事は already_published でスキップ）
- 株価チャートは「実行日から直近3ヶ月」を描く仕様のため、過去日付の記事では省略する
- X投稿はしない（publish_blog_articles.main() ではなく build_and_publish() を直接呼ぶ）
"""
import sys
import argparse
from datetime import date

sys.path.insert(0, "/Users/kitagawayuuya/stock-alert")

import web.publish_blog_articles as pub
from web.market_timing_alert import get_recent_large_holdings

CUTOFF_END = "2026-07-01"
START = date(2025, 6, 18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=None, help="対象月 YYYY-MM（省略時は全対象月）")
    ap.add_argument("--per-month", type=int, default=50)
    ap.add_argument("--max-articles", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    days = (date.today() - START).days + 1
    holdings = get_recent_large_holdings(days=days)
    past = [
        h for h in holdings
        if (h.get("disc_date") or "") < CUTOFF_END
        and h.get("issuer_code") and h.get("holding_ratio") is not None
    ]
    by_month: dict[str, list] = {}
    for h in past:
        by_month.setdefault(h["disc_date"][:7], []).append(h)

    curated = []
    for m in sorted(by_month):
        if args.months and m not in args.months:
            continue
        # 「その月に動いた量」= 比率変化幅で選ぶ。新規報告(prior無し)は取得比率そのもの。
        top = sorted(
            by_month[m],
            key=lambda h: abs(h["holding_ratio"] - (h.get("holding_ratio_prior") or 0)),
            reverse=True,
        )[: args.per_month]
        print(f"{m}: 候補{len(by_month[m])}件 → 上位{len(top)}件を対象")
        curated.extend(top)
    print(f"対象合計: {len(curated)}件")

    # build_and_publish が参照するデータ源を厳選済みリストに差し替え、過去記事ではチャートを省略
    pub.get_recent_large_holdings = lambda days=None, **kw: curated
    pub.build_price_chart_for_article = lambda code, name: None

    results = pub.build_and_publish(days=days, max_articles=args.max_articles, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(results)}件処理完了")


if __name__ == "__main__":
    main()

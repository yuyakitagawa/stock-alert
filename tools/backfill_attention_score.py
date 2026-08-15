#!/usr/bin/env python3
"""
tools/backfill_attention_score.py
既存のmicroCMS記事（クジラ注目度スコア追加前に投稿済みのもの）に、
attentionScore / attentionReasons を遡って計算・付与する一括更新スクリプト。

「売り」記事（tags に "売り" を含む）はスキャンスコア対象外
（lib.attention_score のスコアカードは「買い」開示の実績のみで較正済みのため）。

突き合わせ方法: 記事のstockCode/dealDateからedinet_large_holdingsの候補行を絞り込み、
1件ならそのまま使用、複数件（同日複数提出者）ならweb.publish_blog_articles.pyと同じ
金額概算(estimate_deal_amount_oku)で記事のdealAmountと突き合わせて特定する
（already_published()と同じ考え方の逆引き）。

Usage:
  python3 tools/backfill_attention_score.py            # 全件処理
  python3 tools/backfill_attention_score.py --dry-run   # 計算結果を表示するだけで更新しない
  python3 tools/backfill_attention_score.py --force     # 既にattentionScoreがある記事も再計算
"""
import os
import sys
import argparse
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb
from lib.attention_score import compute_attention_score
from web.publish_blog_articles import (
    _microcms_base_url, _microcms_headers, update_article,
    ratio_change_pct, estimate_deal_amount_oku, MicroCMSPermissionError,
)

load_dotenv()

AMOUNT_TOLERANCE = 0.05


def fetch_all_articles() -> list:
    articles = []
    offset = 0
    limit = 100
    while True:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "fields": "id,stockCode,dealDate,dealAmount,dealType,tags,attentionScore",
                "limit": limit,
                "offset": offset,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        articles.extend(data.get("contents", []))
        offset += limit
        if offset >= data.get("totalCount", 0):
            break
    return articles


def find_holding_row(code: str, disc_date: str, article_deal_amount: float) -> "dict | None":
    """記事に対応するedinet_large_holdings行を特定する（filer_nameが記事側に無いため逆引き）。"""
    rows = sb.select(
        "edinet_large_holdings",
        f"issuer_code=eq.{code}&disc_date=eq.{disc_date}&holding_ratio=not.is.null"
        "&select=filer_name,holding_ratio,holding_ratio_prior",
    )
    if not rows:
        return None
    if len(rows) == 1:
        row = rows[0]
        change = ratio_change_pct(code, row["filer_name"], row["holding_ratio"], disc_date)
        return {"holding_ratio": row["holding_ratio"], "change": change}

    for row in rows:
        change = ratio_change_pct(code, row["filer_name"], row["holding_ratio"], disc_date)
        estimated = estimate_deal_amount_oku(code, change, disc_date)
        if estimated is not None and abs(estimated - article_deal_amount) < AMOUNT_TOLERANCE:
            return {"holding_ratio": row["holding_ratio"], "change": change}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="計算結果を表示するだけで更新しない")
    p.add_argument("--force", action="store_true", help="既にattentionScoreがある記事も再計算する")
    args = p.parse_args()

    print("microCMS記事を全件取得中...")
    articles = fetch_all_articles()
    print(f"  {len(articles)}件\n")

    updated = skipped_sell = skipped_no_match = skipped_has_score = 0
    for a in articles:
        tags = a.get("tags") or ""
        if "売り" in tags:
            skipped_sell += 1
            continue
        if not args.force and a.get("attentionScore") is not None:
            skipped_has_score += 1
            continue

        code = a.get("stockCode")
        deal_date = str(a.get("dealDate", ""))[:10]
        deal_amount = a.get("dealAmount")
        deal_type = a.get("dealType")
        category = deal_type[0] if isinstance(deal_type, list) and deal_type else (deal_type or "その他")
        if not code or not deal_date or deal_amount is None:
            skipped_no_match += 1
            continue

        match = find_holding_row(code, deal_date, deal_amount)
        if match is None:
            print(f"  ⏭ {code} {deal_date} 推定{deal_amount}億円: 対応するEDINET開示が見つからず")
            skipped_no_match += 1
            continue

        attention = compute_attention_score(
            match["holding_ratio"], match["change"], deal_amount, category
        )
        print(f"  {code} {deal_date} 推定{deal_amount}億円 {category}: "
              f"score={attention['score']} {'★' * attention['stars']} reasons={attention['reasons']}")

        if args.dry_run:
            updated += 1
            continue

        payload = {
            "attentionScore": attention["score"],
            "attentionReasons": ",".join(attention["reasons"]),
        }
        try:
            if update_article(a["id"], payload):
                updated += 1
            time.sleep(0.2)
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため中断します: {e}")
            break

    print(f"\n更新: {updated}件 / 売りでスキップ: {skipped_sell}件 / "
          f"既にスコア有りでスキップ: {skipped_has_score}件 / 突合失敗でスキップ: {skipped_no_match}件")


if __name__ == "__main__":
    main()

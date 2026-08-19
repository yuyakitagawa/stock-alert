#!/usr/bin/env python3
"""公開済み記事のうち「X%を新規保有」が誤りのものを検出し、microCMSを訂正する。

背景（2026-08-19の監査）:
  web/publish_blog_articles.py の ratio_change_pct() は、開示に直前保有割合が無く、DBにも
  同一提出者の過去開示が無いとき「今回比率の全量が動いた」とみなしていた。変更報告書は提出者が
  既に5%以上を保有している届出なので、これは誤り。結果として

    - タイトルが「〇〇が7.6%を新規保有」（実際は既存保有の一部変更）
    - ratioChangePct が実態より大きい
    - dealAmount（変化幅×株価）が同じ倍率で過大

  になった記事が公開されている。生成側は同日の修正で塞いだが、既に出ている記事は残るため、
  このスクリプトで訂正する。

判定と訂正:
  1. タイトルに「を新規保有」を含む記事を対象にする。
  2. 記事の銘柄コード×開示日で edinet_large_holdings を引き、報告書種別が「変更」なら誤り。
  3. 開示に直前保有割合(holding_ratio_prior)があれば、正しい変化幅・推定金額・タイトルを
     再計算してPATCHする。
  4. 直前保有割合が無い開示は、正しい数字を作れない（誤った数字を別の誤った数字に
     置き換えることになる）ため自動訂正せず、一覧に出して手当ての判断を仰ぐ。

使い方:
  python3 tools/fix_new_holding_articles.py            # dry-run（既定。何も更新しない）
  python3 tools/fix_new_holding_articles.py --apply    # microCMSへPATCHする
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import lib.supabase_client as sb
from lib.edinet import disclosure_doc_label, disclosure_kind_label
from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, MicroCMSPermissionError,
    build_article_titles, estimate_deal_amount_oku, update_article,
)
from tools.scan_large_holdings import is_sell_disclosure

load_dotenv()

NEW_HOLDING_MARKER = "を新規保有"


def find_disclosure(code: str, disc_date: str) -> "dict | None":
    """記事の銘柄コード×開示日に対応するEDINET開示行を返す。提出者が複数ぶら下がる日は
    どの行が記事の元かを特定できないためNone（誤った行で訂正するより手を出さない）。"""
    rows = sb.select(
        "edinet_large_holdings",
        f"issuer_code=eq.{code}&disc_date=eq.{disc_date}"
        "&select=filer_name,doc_type_code,doc_description,holding_ratio,holding_ratio_prior",
    )
    rows = [r for r in rows if r.get("holding_ratio") is not None]
    filers = {r.get("filer_name") for r in rows}
    if len(filers) != 1:
        return None
    return rows[0]


def corrected_payload(article: dict, row: dict) -> "dict | None":
    """訂正後の title / ratioChangePct / dealAmount を組み立てる。
    直前保有割合が無く正しい変化幅を出せない場合はNone。"""
    prior = row.get("holding_ratio_prior")
    if prior is None:
        return None
    ratio = row["holding_ratio"]
    change = round(abs(ratio - prior), 2)
    if change <= 0:
        return None
    is_sell = is_sell_disclosure(row.get("doc_description") or "", ratio, prior)
    deal_amount = estimate_deal_amount_oku(
        str(article.get("stockCode")), change, article["dealDate"][:10]
    )
    if deal_amount is None:
        return None
    fact_sheet = {
        "stock_name": article.get("stockName") or "",
        "stock_code": str(article.get("stockCode")),
        "filer_name": row.get("filer_name") or "",
        "doc_type_label": disclosure_doc_label(
            row.get("doc_description"), row.get("doc_type_code", "")
        ),
        "holding_ratio": ratio,
        "direction": "sell" if is_sell else "buy",
        "ratio_change_pct": change,
        "prior_ratio": prior,
        "is_correction": False,
    }
    titles = build_article_titles(fact_sheet)
    return {
        "title": titles["title"],
        "titleEn": titles["titleEn"],
        "ratioChangePct": round(-change if is_sell else change, 2),
        "dealAmount": deal_amount,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="microCMSへ実際にPATCHする（既定はdry-run）")
    args = parser.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY が未設定です。")
        sys.exit(1)

    articles = fetch_all_articles()
    targets = [a for a in articles if NEW_HOLDING_MARKER in (a.get("title") or "")]
    print(f"記事総数: {len(articles)}件 / 「新規保有」タイトル: {len(targets)}件\n")

    fixed = 0
    correct_as_is = 0
    unresolved: list[tuple[str, str]] = []
    no_match: list[str] = []

    for a in targets:
        code = str(a.get("stockCode") or "")
        deal_date = (a.get("dealDate") or "")[:10]
        if not code or not deal_date:
            no_match.append(a["id"])
            continue
        row = find_disclosure(code, deal_date)
        if row is None:
            no_match.append(a["id"])
            continue
        if disclosure_kind_label(row.get("doc_description"), row.get("doc_type_code", "")) != "変更":
            correct_as_is += 1  # 大量保有報告書＝本当に新規保有。触らない
            continue

        payload = corrected_payload(a, row)
        if payload is None:
            unresolved.append((a["id"], f"{a.get('stockName')}({code}) {deal_date}"))
            continue

        print(f"  {a['id']} {a.get('stockName')}({code}) {deal_date}")
        print(f"    旧: {a.get('title')}")
        print(f"        変化幅 {a.get('ratioChangePct')}pt / 推定 {a.get('dealAmount')}億円")
        print(f"    新: {payload['title']}")
        print(f"        変化幅 {payload['ratioChangePct']}pt / 推定 {payload['dealAmount']}億円")
        if args.apply:
            try:
                if update_article(a["id"], payload):
                    fixed += 1
                else:
                    print("    ⚠ 更新に失敗しました")
            except MicroCMSPermissionError as e:
                print(f"  ✖ 権限エラーのため中断します: {e}")
                break
        else:
            fixed += 1

    headline = "訂正" if args.apply else "訂正対象（dry-run）"
    print(f"\n{headline}: {fixed}件 / 新規保有で正しい記事: {correct_as_is}件 / "
          f"開示を特定できず: {len(no_match)}件 / 前回比率不明で自動訂正不可: {len(unresolved)}件")
    if unresolved:
        print("\n前回比率が取れず自動訂正できない記事（手当ての判断が必要）:")
        for article_id, where in unresolved:
            print(f"  - {article_id} {where}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
tools/backfill_company_descriptions.py
kujira-watch.com の /stocks/[code] ページが「クロール済み-インデックス未登録」になる主因の
ひとつ、jpx_stock_list.description（会社情報カードの事業内容1文）が未設定のまま公開されて
いる既存銘柄を対象に、web.publish_blog_articles.get_company_description() で事業内容を
生成・キャッシュする。get_company_description自体は新規記事生成時から既に呼ばれているが
（jpx_stock_list.descriptionにキャッシュ）、対象銘柄の記事が生成された時点より前から
存在する銘柄には遡って入らないため、このツールで一括バックフィルする。
プロンプト・「不明なら空文字」ガードは新規記事生成時と完全に同一のものを流用する
（tools/rewrite_thin_blog_articles.pyと同じ方針）。

対象: edinet_large_holdings に提出実績があり（=/stocks/[code]ページが存在する）、
jpx_stock_list.description が未設定の銘柄。

Usage:
  python3 tools/backfill_company_descriptions.py --dry-run           # 対象件数の確認のみ
  python3 tools/backfill_company_descriptions.py --dry-run --limit 5
  python3 tools/backfill_company_descriptions.py                     # 実行
  python3 tools/backfill_company_descriptions.py --limit 500         # 件数を区切って分割実行
  python3 tools/backfill_company_descriptions.py --codes 7203,9984   # 銘柄コード指定
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb
from web.publish_blog_articles import ANTHROPIC_API_KEY, get_company_description

load_dotenv()


def stock_codes_with_pages() -> list:
    """/stocks/[code]ページが存在する銘柄コード一覧（edinet_large_holdingsに提出実績がある銘柄）。"""
    rows = sb.select("edinet_large_holdings", "select=issuer_code")
    return sorted({r["issuer_code"] for r in rows if r.get("issuer_code")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="生成せず対象件数・銘柄名の表示のみ")
    p.add_argument("--limit", type=int, default=None, help="処理件数の上限（動作確認・分割実行用）")
    p.add_argument("--codes", type=str, default=None, help="対象銘柄コードをカンマ区切りで明示指定（未設定判定を無視して強制対象にする）")
    args = p.parse_args()

    if not args.dry_run and not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY が未設定です。")
        sys.exit(1)

    meta_rows = sb.select("jpx_stock_list", "select=code,name,description")
    meta_by_code = {r["code"]: r for r in meta_rows}

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        all_codes = stock_codes_with_pages()
        print(f"/stocks/[code]ページが存在する銘柄: {len(all_codes)}件")
        codes = [c for c in all_codes if not (meta_by_code.get(c, {}).get("description") or "")]
    print(f"description未設定: {len(codes)}件\n")

    if args.limit is not None:
        codes = codes[: args.limit]

    generated = 0
    empty = []
    missing_name = []

    for code in codes:
        name = (meta_by_code.get(code) or {}).get("name")
        if not name:
            missing_name.append(code)
            continue

        if args.dry_run:
            print(f"  {code} {name}: (dry-run, スキップ)")
            continue

        description = get_company_description(code, name)
        if description:
            generated += 1
            print(f"  {code} {name}: {description}")
        else:
            empty.append(code)
            print(f"  {code} {name}: (空文字/不明)")
        time.sleep(0.3)

    print(f"\n生成: {generated}件 / 空文字（不明）: {len(empty)}件 / 会社名不明: {len(missing_name)}件")
    if missing_name:
        print(f"会社名不明のためスキップ: {missing_name[:20]}{'...' if len(missing_name) > 20 else ''}")
    if args.dry_run:
        print("\n--dry-run のため実際の生成・保存は行っていません。")


if __name__ == "__main__":
    main()

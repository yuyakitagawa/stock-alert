#!/usr/bin/env python3
"""
tools/strip_html_from_descriptions.py
jpx_stock_list.description に混入したHTMLタグを取り除く一括修正。

get_company_description() はClaudeのweb_searchで事業内容を裏取りするが、引用を伴う回答は
本文に <cite index="7-1">…</cite> を挟んで返すことがあり、保存前に除去していなかった。
kujira-watch の銘柄ページ・/trending・metadataのdescriptionにタグが文字としてそのまま
表示されていた（2026-08-25の点検で64銘柄）。生成側は同日に除去するよう修正済みで、
このスクリプトは既存データの後始末用（実行後は削除してよい）。

Usage:
  python3 tools/strip_html_from_descriptions.py --dry-run
  python3 tools/strip_html_from_descriptions.py
"""
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb

load_dotenv()

TAG_RE = re.compile(r"<[^>]+>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = sb.select("jpx_stock_list", "select=code,description&description=like.*<*")
    fixed = []
    for row in rows:
        cleaned = " ".join(TAG_RE.sub("", row.get("description") or "").split())
        if cleaned and cleaned != row.get("description"):
            fixed.append({"code": row["code"], "description": cleaned})

    print(f"対象 {len(rows)}件 / 修正 {len(fixed)}件")
    for f in fixed[:5]:
        print(f"  {f['code']}: {f['description'][:80]}")
    if args.dry_run:
        print("--dry-run のため更新していません")
        return
    if fixed:
        sb.upsert("jpx_stock_list", fixed, on_conflict="code")
        print(f"更新: {len(fixed)}件")


if __name__ == "__main__":
    main()

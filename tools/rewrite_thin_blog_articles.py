#!/usr/bin/env python3
"""
tools/rewrite_thin_blog_articles.py
Google Search Consoleで「クロール済み - インデックス未登録」と判定された既存記事のうち、
本文が薄い（可視文字数がTHIN_TEXT_THRESHOLD未満）ものを対象に、
web.publish_blog_articles.generate_article_body() の現行プロンプト（保有比率の変化幅を
事実として追加投入・本文目標650〜900字）で本文だけを再生成する。

対象の特定方法: Search Console APIへのアクセス手段が無いため、URLの直接指定ではなく
「可視文字数が閾値未満」という機械的な基準で候補を抽出する（旧プロンプト時代の記事は
本文が短く定型文比率が高いため、この基準で概ね一致する）。

再生成しないもの:
- タイトル（generate_article_body()は毎回新しいtitleも返すが、既存タイトルは
  アイキャッチ画像に焼き込み済みのため据え置く。新しいtitleは破棄する）
- 本文末尾の株価チャート<figure>（既存のものをそのまま新本文の末尾に付け替える。
  チャート画像自体の再生成・再アップロードは行わない）

事実の再構築: stockCode + dealDate（日付部分）でedinet_large_holdingsを逆引きし、
提出者を特定する（tools/reclassify_blog_articles.pyと同じ方式）。同一銘柄・同一開示日に
複数の提出者がいて一意に特定できない記事はスキップする。

更新はPUT（完全上書き）で行う（tools/reclassify_blog_articles.pyと同じ理由）。

Usage:
  python3 tools/rewrite_thin_blog_articles.py --dry-run           # 対象件数・内容の確認のみ
  python3 tools/rewrite_thin_blog_articles.py --dry-run --limit 5 # 先頭5件だけ確認
  python3 tools/rewrite_thin_blog_articles.py                     # 実際にmicroCMSを更新
  python3 tools/rewrite_thin_blog_articles.py --ids id1,id2       # 指定IDのみ対象
                                                                   # （Search Console提示のURL等、
                                                                   # 可視文字数の閾値ではなく明示指定したい場合）
"""
import os
import re
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb
from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, DOC_TYPE_LABELS,
    classify_filer, get_company_description, get_pit_ranking_snapshot, dp_level_label,
    ratio_change_pct, generate_article_body, update_article, MicroCMSPermissionError,
)

load_dotenv()

# 新プロンプトの本文目標(650〜900字)を大きく下回る水準を「薄い」とみなす閾値。
# HTMLタグと株価チャートの<figure>を除いた可視文字数で判定する。
THIN_TEXT_THRESHOLD = 600

_FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def visible_text_len(body_html: str) -> int:
    text = _FIGURE_RE.sub("", body_html or "")
    text = _TAG_RE.sub("", text)
    return len(text)


def find_filer_names(code: str, disc_date: str) -> set:
    rows = sb.select(
        "edinet_large_holdings",
        f"issuer_code=eq.{code}&disc_date=eq.{disc_date}&select=filer_name,doc_type_code,holding_ratio",
    )
    return rows


def build_patch_payload(new_body: str) -> dict:
    return {"body": new_body}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="更新せず対象件数・内容の表示のみ")
    p.add_argument("--limit", type=int, default=None, help="処理件数の上限（動作確認用）")
    p.add_argument("--ids", type=str, default=None, help="対象記事IDをカンマ区切りで明示指定（指定時は閾値判定を使わない）")
    args = p.parse_args()

    if not args.dry_run and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY が未設定です。")
        sys.exit(1)

    articles = fetch_all_articles()
    print(f"記事総数: {len(articles)}件")

    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        candidates = [a for a in articles if a["id"] in wanted]
        found_ids = {a["id"] for a in candidates}
        missing = wanted - found_ids
        if missing:
            print(f"指定IDのうち記事が見つからないもの: {sorted(missing)}")
        print(f"指定ID対象: {len(candidates)}件\n")
    else:
        candidates = [a for a in articles if visible_text_len(a.get("body", "")) < THIN_TEXT_THRESHOLD]
        print(f"薄い記事（可視文字数 < {THIN_TEXT_THRESHOLD}）: {len(candidates)}件\n")

    if args.limit is not None:
        candidates = candidates[: args.limit]

    rewritten = 0
    skipped_no_match = []
    skipped_ambiguous = []
    skipped_gen_failed = []

    for a in candidates:
        code = a.get("stockCode")
        disc_date = (a.get("dealDate") or "")[:10]
        name = a.get("stockName") or code
        old_len = visible_text_len(a.get("body", ""))

        if not code or not disc_date:
            skipped_no_match.append(a["id"])
            continue

        rows = find_filer_names(code, disc_date)
        filer_names = {r["filer_name"] for r in rows if r.get("filer_name")}
        if len(filer_names) != 1:
            if len(filer_names) > 1:
                skipped_ambiguous.append((a["id"], code, disc_date, filer_names))
            else:
                skipped_no_match.append(a["id"])
            continue
        filer_name = filer_names.pop()
        row = next(r for r in rows if r.get("filer_name") == filer_name)

        is_sell = "売り" in (a.get("tags") or "")
        change = ratio_change_pct(code, filer_name, row["holding_ratio"], disc_date)
        snapshot = get_pit_ranking_snapshot(code, disc_date)
        context_close = snapshot.get("close") if snapshot else None
        context_dp = snapshot.get("drop_prob") if snapshot else None
        filer_info = classify_filer(filer_name)
        company_description = get_company_description(code, name)

        fact_sheet = {
            "stock_name": name,
            "stock_code": code,
            "filer_name": filer_name,
            "doc_type_label": DOC_TYPE_LABELS.get(row.get("doc_type_code", ""), "大量保有関連報告書"),
            "holding_ratio": row["holding_ratio"],
            "disc_date": disc_date,
            "deal_amount_oku": a.get("dealAmount"),
            "direction": "sell" if is_sell else "buy",
            "deal_amount_label": "推定売却金額" if is_sell else "推定取得金額",
            "context_close": context_close,
            "context_dp_level": dp_level_label(context_dp) if context_dp is not None else None,
            "filer_description": filer_info.get("description") or "",
            "company_description": company_description,
            "ratio_change_pct": change,
        }
        generated = generate_article_body(fact_sheet)
        if generated is None:
            skipped_gen_failed.append(a["id"])
            continue

        new_body = generated["body"]
        old_figure = _FIGURE_RE.search(a.get("body") or "")
        if old_figure:
            new_body += old_figure.group(0)
        new_len = visible_text_len(new_body)

        rewritten += 1
        print(f"  {a['id']}: {name}({code}) {disc_date} 可視文字数 {old_len}→{new_len}")
        if args.dry_run:
            continue

        try:
            ok = update_article(a["id"], build_patch_payload(new_body))
            if not ok:
                print(f"  ⚠ {a['id']} の更新に失敗しました（詳細は上記ログ参照）")
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため中断: {e}")
            break

        time.sleep(0.5)

    print(
        f"\nリライト対象: {rewritten}件 / 提出者特定不能: {len(skipped_no_match)}件 / "
        f"複数提出者で特定不能: {len(skipped_ambiguous)}件 / 記事生成失敗: {len(skipped_gen_failed)}件"
    )
    if skipped_ambiguous:
        print("\n複数提出者で特定不能（手動確認が必要）:")
        for aid, code, d, names in skipped_ambiguous:
            print(f"  {aid}: {code} {d} candidates={names}")
    if args.dry_run:
        print("\n--dry-run のため実際の更新は行っていません。")


if __name__ == "__main__":
    main()

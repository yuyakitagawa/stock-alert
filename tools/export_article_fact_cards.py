#!/usr/bin/env python3
"""
tools/export_article_fact_cards.py
既存ブログ記事のリライト用に、1記事ぶんの「事実カード」をJSONで書き出す。

AdSense審査の不承認（2026-08-24「有用性の低いコンテンツ」）を受けた既存976記事の
リライトで使う。tools/rewrite_thin_blog_articles.py がAnthropic APIに書かせるのに対し、
こちらは事実の収集だけを行い、本文は人間（またはClaude Code）が書く運用のための入口。

書き出す内容は web/publish_blog_articles.py の fact_sheet と同じ事実（銘柄・提出者・
保有比率・変化幅・推定金額・事業内容）に、build_context_facts() が集める開示横断の
周辺事実（保有の積み上げ履歴・その投資家の他の保有銘柄・その銘柄の他の大株主・
開示日時点の指標）を足したもの。すべて disc_date 以前に絞った point-in-time。

Usage:
  python3 tools/export_article_fact_cards.py --limit 20 --out /tmp/cards.json
  python3 tools/export_article_fact_cards.py --ids id1,id2 --out /tmp/cards.json
  python3 tools/export_article_fact_cards.py --skip 100 --limit 20 --out /tmp/cards.json
"""
import os
import sys
import json
import argparse
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from tools.reclassify_blog_articles import fetch_all_articles
from tools.rewrite_thin_blog_articles import (
    THIN_TEXT_THRESHOLD, visible_text_len, find_filer_names,
)
from lib.edinet import disclosure_doc_label
from web.publish_blog_articles import (
    build_context_facts, format_context_facts, classify_filer,
    get_company_description, get_pit_ranking_snapshot, ratio_change_pct,
)

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tools/apply_rewritten_articles.py が反映のたびに追記する台帳。
# 書き直した本文も1,000字未満に収まることが多く、可視文字数の閾値だけでは
# 「もう直した記事」を候補から外せないため、IDで突き合わせて除外する。
DONE_LEDGER = os.path.join(REPO_ROOT, "logs", "rewritten_article_ids.txt")


def _norm(name: str) -> str:
    """提出者名の突合用の正規化。EDINETのXBRLは提出者名を全角（Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ…）で
    保持する一方、記事側のfilerNameや本文は半角で入ることがあるため、NFKC正規化して空白を落とす。"""
    return unicodedata.normalize("NFKC", name or "").replace(" ", "").replace("\u3000", "").lower()


def resolve_filer(article: dict, rows: list) -> "str | None":
    """同一銘柄・同一開示日に提出者が複数いる場合の一意化。

    銘柄コード×開示日だけでは絞れない記事が全体の18%（実測2026-08-25: 999件中182件）あり、
    そのままではリライトの材料が作れない。記事側が持つ情報で候補を絞る:
      1. microCMSのfilerNameが候補と一致すればそれを採る（154件がこの経路で解決する）
      2. 記事タイトルに候補の提出者名が含まれていればそれを採る
    どちらでも決まらなければNone（＝材料を作らずスキップ）。誤った提出者で記事を書き直すと
    別の投資家の取引として公開されることになるため、曖昧なままでは進めない。
    """
    names = {r["filer_name"] for r in rows if r.get("filer_name")}
    if len(names) == 1:
        return names.pop()
    if not names:
        return None

    by_norm = {_norm(n): n for n in names}
    filer_name = (article.get("filerName") or "").strip()
    if filer_name:
        hit = by_norm.get(_norm(filer_name))
        if hit:
            return hit

    title = _norm(article.get("title") or "")
    matches = [n for norm, n in by_norm.items() if norm and norm in title]
    if len(matches) == 1:
        return matches[0]
    return None


def load_done_ids() -> set:
    if not os.path.exists(DONE_LEDGER):
        return set()
    with open(DONE_LEDGER, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="書き出し先のJSONパス")
    p.add_argument("--limit", type=int, default=None, help="件数の上限")
    p.add_argument("--skip", type=int, default=0, help="先頭から読み飛ばす件数（分割実行用）")
    p.add_argument("--ids", default="", help="対象記事IDをカンマ区切りで明示指定")
    args = p.parse_args()

    ids = {i.strip() for i in args.ids.split(",") if i.strip()}
    articles = fetch_all_articles()
    if ids:
        targets = [a for a in articles if a["id"] in ids]
    else:
        done = load_done_ids()
        targets = [
            a for a in articles
            if a["id"] not in done and visible_text_len(a.get("body")) < THIN_TEXT_THRESHOLD
        ]
        print(f"リライト済み（台帳）: {len(done)}件 / 残り候補: {len(targets)}件")
    # 取引日の新しい順（読者が最初に見る記事から直す）
    targets.sort(key=lambda a: a.get("dealDate") or "", reverse=True)
    targets = targets[args.skip:]
    if args.limit:
        targets = targets[: args.limit]

    cards, skipped = [], []
    for a in targets:
        code = str(a.get("stockCode") or "")
        disc_date = (a.get("dealDate") or "")[:10]
        name = a.get("stockName") or code
        if not code or not disc_date:
            skipped.append((a["id"], "銘柄コードか取引日が無い"))
            continue

        rows = find_filer_names(code, disc_date)
        filer_name = resolve_filer(a, rows)
        if not filer_name:
            n = len({r["filer_name"] for r in rows if r.get("filer_name")})
            skipped.append((a["id"], f"提出者を一意に特定できない({n}件)"))
            continue
        row = next(r for r in rows if r.get("filer_name") == filer_name)

        is_sell = "売り" in (a.get("tags") or "")
        change = ratio_change_pct(code, filer_name, row["holding_ratio"], disc_date)
        snapshot = get_pit_ranking_snapshot(code, disc_date)
        context_facts = build_context_facts(code, filer_name, disc_date)

        cards.append({
            "id": a["id"],
            "title": a.get("title"),
            "stock_name": name,
            "stock_code": code,
            "filer_name": filer_name,
            "doc_type_label": disclosure_doc_label(row.get("doc_description"), row.get("doc_type_code", "")),
            "holding_ratio": row["holding_ratio"],
            "prior_ratio": row.get("holding_ratio_prior"),
            "ratio_change_pct": change,
            "disc_date": disc_date,
            "direction": "sell" if is_sell else "buy",
            "deal_amount_oku": a.get("dealAmount"),
            "close_at_disclosure": snapshot.get("close") if snapshot else None,
            "filer_description": classify_filer(filer_name).get("description") or "",
            "company_description": get_company_description(code, name) or "",
            "context_facts_text": format_context_facts(context_facts, name, filer_name),
            "current_len": visible_text_len(a.get("body")),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=1)
    print(f"書き出し: {len(cards)}件 → {args.out}")
    if skipped:
        print(f"スキップ: {len(skipped)}件")
        for aid, reason in skipped[:10]:
            print(f"  {aid}: {reason}")


if __name__ == "__main__":
    main()

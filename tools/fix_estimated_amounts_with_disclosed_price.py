"""公開済み記事のうち、開示原文に譲渡単価があるものの金額を実額へ差し替える。

短期大量譲渡（法第27条の25第2項）の開示には譲渡の相手方と単価が載っており、
単価×株数で実額が出せる。2026-08-26以前に投稿した記事は株価からの概算で書かれているため、
ファクトボックス（Supabaseの`short_term_transfers`を見て実額を表示する）と本文の数字が
食い違う。この差を埋めるため、本文中の概算金額の表記だけを実額へ置き換える。

置き換えるのは金額の数字と「推定◯◯金額」の"推定"だけで、本文の再生成（Claude呼び出し）は
行わない。本文に概算金額が1箇所も出てこない／2通り以上の表記で出てくる記事はスキップする
（機械的な置換で文意が壊れないことを確認できた記事だけ触る）。

実行:
  python3 tools/fix_estimated_amounts_with_disclosed_price.py --dry-run  # 対象と差分の確認
  python3 tools/fix_estimated_amounts_with_disclosed_price.py            # microCMSへPATCH
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb
from lib.edinet import summarize_disposals
from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import update_article


def _amount_patterns(amount: float) -> list:
    """「1274.9億円」「1,274.9億円」の両表記（末尾の.0は整数表記もありうる）。"""
    plain = f"{amount:g}"
    forms = {plain, f"{amount:,.1f}", f"{amount:,g}"}
    if amount == int(amount):
        forms.add(f"{int(amount)}")
        forms.add(f"{int(amount):,}")
    return sorted(forms, key=len, reverse=True)


def replace_amount(body: str, old_amount: float, new_amount: float) -> "str | None":
    """本文中の概算金額を実額へ置換する。安全に置換できないときは None。"""
    matched = [f for f in _amount_patterns(old_amount) if f"{f}億円" in body]
    if len(matched) != 1:
        return None  # 見つからない or 表記ゆれで複数該当（機械置換の対象外）
    new_text = f"{new_amount:,.1f}".rstrip("0").rstrip(".") if new_amount % 1 else f"{new_amount:,.0f}"
    out = body.replace(f"{matched[0]}億円", f"{new_text}億円")
    # 「推定売却金額」「推定取得金額」「推定金額」は実額になったので"推定"を外す
    out = re.sub(r"推定(売却金額|取得金額|金額)", r"\1", out)
    return out


def _fmt(value: float) -> str:
    """1.65 / 112.18 / 500 のように、無駄な小数点以下0を付けずに整形する。"""
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def build_targets() -> list:
    articles = fetch_all_articles()
    rows = sb.select(
        "edinet_large_holdings",
        "short_term_transfers=not.is.null&select=issuer_code,disc_date,filer_name,"
        "holding_ratio,holding_ratio_prior,short_term_transfers",
    )
    by_key = {}
    for r in rows:
        by_key.setdefault((str(r["issuer_code"]), r["disc_date"], r["filer_name"]), []).append(r)

    targets = []
    for a in articles:
        # 提出者が保存されていない旧記事は開示を一意に特定できないため触らない
        key = (str(a.get("stockCode")), (a.get("dealDate") or "")[:10], a.get("filerName") or "")
        candidates = by_key.get(key, [])
        # 同じ提出者が同じ日に複数の報告書を出すことがある（実例: 三井金属は2026-08-07に
        # 30.01%→11.16%と11.16%→3.86%の2通を提出）。記事の比率変化幅で開示を特定し、
        # 特定できない場合は触らない（別の開示の金額を書き込まないため）。
        if len(candidates) > 1:
            article_change = abs(a.get("ratioChangePct") or 0)
            candidates = [
                r for r in candidates
                if r.get("holding_ratio") is not None and r.get("holding_ratio_prior") is not None
                and abs(abs(r["holding_ratio"] - r["holding_ratio_prior"]) - article_change) < 0.05
            ]
            if len(candidates) != 1:
                continue
        for r in candidates:
            ratio, prior = r.get("holding_ratio"), r.get("holding_ratio_prior")
            change = abs(ratio - prior) if (ratio is not None and prior is not None) else None
            s = summarize_disposals(r["short_term_transfers"], change)
            if s["amount_oku"] is None or a.get("dealAmount") is None:
                continue
            if abs(a["dealAmount"] - s["amount_oku"]) < 0.05:
                continue
            new_body = replace_amount(a.get("body") or "", a["dealAmount"], s["amount_oku"])
            targets.append({
                "id": a["id"], "stockCode": a.get("stockCode"), "stockName": a.get("stockName"),
                "dealDate": (a.get("dealDate") or "")[:10], "filerName": a.get("filerName"),
                "old_amount": a["dealAmount"], "new_amount": s["amount_oku"],
                "counterparty": (s["counterparties"] or [None])[0],
                "old_body": a.get("body") or "", "new_body": new_body,
            })
            break
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="microCMSへ送らず対象と差分だけ表示")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    targets = build_targets()
    if args.limit:
        targets = targets[:args.limit]
    print(f"金額が食い違う公開記事: {len(targets)}件")
    updated, skipped = [], []
    for t in targets:
        head = (f"  {t['stockName']}({t['stockCode']}) {t['dealDate']} "
                f"{t['old_amount']}億円 → {t['new_amount']}億円 相手方={t['counterparty']}")
        if t["new_body"] is None:
            print(f"{head} ⏭ 本文に概算金額が一意に見つからずスキップ")
            skipped.append(t["id"])
            continue
        print(head)
        if args.dry_run:
            continue
        payload = {"body": t["new_body"], "dealAmount": t["new_amount"]}
        if update_article(t["id"], payload):
            updated.append(t["id"])
        else:
            print(f"    ✖ 更新失敗: {t['id']}")

    if updated:
        path = os.path.join("logs", f"amount_fix_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([t for t in targets if t["id"] in updated], f, ensure_ascii=False, indent=2)
        print(f"\n{len(updated)}件更新（バックアップ: {path}）")
    print(f"スキップ: {len(skipped)}件")


if __name__ == "__main__":
    main()

"""既存の「短期大量譲渡」開示に、譲渡の相手方・単価（short_term_transfers）を埋め直す。

短期大量譲渡（法第27条の25第2項）の変更報告書には、直近60日間の取得・処分が
「年月日／数量／割合／市場内外／取得又は処分の別／譲渡の相手方／単価」の表で載っている。
2026-08-26にこの取り込みを実装したため、それ以前の開示は列が空のまま。EDINET APIのみで
埋められる（Anthropic APIの課金は発生しない）。

実行:
  python3 tools/backfill_short_term_transfers.py            # 未取得のものを全件
  python3 tools/backfill_short_term_transfers.py --limit 20 # 件数を絞って試す
  python3 tools/backfill_short_term_transfers.py --dry-run  # 保存せず内容だけ表示
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb
from lib.edinet import _fetch_xbrl_text, parse_short_term_transfers, summarize_disposals

SLEEP_SEC = 0.3  # EDINETへの連続アクセスを抑える


def target_docs(limit: int = 0, refetch: bool = False) -> list:
    q = "doc_description=like.*短期大量譲渡*&order=disc_date.desc"
    q += "&select=doc_id,filer_name,issuer_code,issuer_name,disc_date,holding_ratio,holding_ratio_prior,short_term_transfers"
    if not refetch:
        q += "&short_term_transfers=is.null"
    rows = sb.select("edinet_large_holdings", q)
    return rows[:limit] if limit else rows


def backfill(limit: int = 0, dry_run: bool = False, refetch: bool = False) -> dict:
    docs = target_docs(limit=limit, refetch=refetch)
    print(f"対象: {len(docs)}件")
    stats = {"total": len(docs), "parsed": 0, "empty": 0, "failed": 0, "with_counterparty": 0}
    batch = []
    for i, d in enumerate(docs, 1):
        xbrl = _fetch_xbrl_text(d["doc_id"])
        if xbrl is None:
            stats["failed"] += 1
            print(f"  [{i}/{len(docs)}] ✖ XBRL取得失敗 {d['doc_id']}")
            time.sleep(SLEEP_SEC)
            continue
        rows = parse_short_term_transfers(xbrl)
        if not rows:
            stats["empty"] += 1
        else:
            stats["parsed"] += 1
            prior = d.get("holding_ratio_prior")
            ratio = d.get("holding_ratio")
            change = abs(ratio - prior) if (ratio is not None and prior is not None) else None
            s = summarize_disposals(rows, change)
            if s["counterparties"]:
                stats["with_counterparty"] += 1
            print(f"  [{i}/{len(docs)}] {d['disc_date']} {d.get('issuer_name') or d.get('issuer_code')}"
                  f" ← {d.get('filer_name')}: {len(rows)}行"
                  f" 相手方={'/'.join(s['counterparties'][:2]) or '—'}"
                  f" 実額={s['amount_oku'] if s['amount_oku'] is not None else '—'}億円")
            # issuer_code はNOT NULL。PostgreSQLはON CONFLICT解決の前にNOT NULLを見るため、
            # doc_idだけの部分upsertは既存行があっても23502で落ちる（既存列と同じ値を必ず送る）。
            batch.append({"doc_id": d["doc_id"], "issuer_code": d["issuer_code"],
                          "short_term_transfers": rows})
        if len(batch) >= 50 and not dry_run:
            if not sb.upsert("edinet_large_holdings", batch, on_conflict="doc_id"):
                print("  ✖ Supabaseへの保存に失敗したため中断します")
                return stats
            batch = []
        time.sleep(SLEEP_SEC)

    if batch and not dry_run and not sb.upsert("edinet_large_holdings", batch, on_conflict="doc_id"):
        print("  ✖ Supabaseへの保存に失敗しました")
    print(f"\n完了: 表あり{stats['parsed']}件（うち相手方が特定できたもの{stats['with_counterparty']}件） / "
          f"表なし{stats['empty']}件 / 取得失敗{stats['failed']}件")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="処理件数の上限（0で全件）")
    p.add_argument("--dry-run", action="store_true", help="Supabaseへ保存せず内容だけ表示")
    p.add_argument("--refetch", action="store_true", help="取得済みのものも取り直す")
    args = p.parse_args()
    backfill(limit=args.limit, dry_run=args.dry_run, refetch=args.refetch)


if __name__ == "__main__":
    main()

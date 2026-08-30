"""既存の edinet_large_holdings 各行のXBRLを引き直して、本表の項目を埋め直す。

用途は2つ:
  1. 保有目的・取得資金・保有株数（2026-08-30に追加した列）を過去の開示にも入れる
  2. 保有割合を「提出者＋共同保有者の合算」に直す
     （それ以前は保有者ごとのcontextを先勝ちで拾っていたため、共同保有の報告書で
      筆頭保有者の1枠だけを保有割合として出していた。実測1,101件中656件がズレていた）

EDINETのAPIは無料なので課金は発生しないが、1行につきXBRLのZIPを1本取るため
全件で数時間かかる。途中で落ちても `--only-missing` で続きから再開できる。

使い方:
  python3 tools/backfill_holding_details.py --only-missing   # 未取得の行だけ
  python3 tools/backfill_holding_details.py --all            # 全行（比率の直し込み用）
  python3 tools/backfill_holding_details.py --all --since 2025-06-18
  python3 tools/backfill_holding_details.py --all --dry-run  # 書き込まずに差分だけ数える
  python3 tools/backfill_holding_details.py --all --with-article  # 記事になった開示だけ先に直す
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import lib.supabase_client as sb  # noqa: E402
from lib.db import HOLDING_DETAIL_COLUMNS  # noqa: E402
from lib.edinet import fetch_xbrl_details  # noqa: E402

# 1回のupsertで送る行数。PostgRESTのpayload上限と、途中で落ちたときの
# やり直し量のバランス。
WRITE_BATCH = 200
# 保有割合が「変わった」と数える閾値（%pt）。丸め差を拾わないための幅。
RATIO_EPS = 0.05


def fetch_targets(only_missing: bool, since: str, limit: int,
                  with_article: bool = False) -> list:
    # issuer_code も必ず取る。PostgRESTのupsertは INSERT ... ON CONFLICT DO UPDATE なので、
    # 実際にはUPDATEになる行でも「INSERTしようとしたタプル」に対してNOT NULL制約が評価される。
    # payloadに issuer_code を含めないと、既存行の更新のはずが
    # `null value in column "issuer_code" ... violates not-null constraint` で
    # バッチごと400になる（2026-08-30に実測。25,000件を処理したのに1行も書けていなかった）。
    # このテーブルのNOT NULLは doc_id と issuer_code の2つだけ。
    query = "select=doc_id,disc_date,holding_ratio,issuer_code&order=disc_date.asc,doc_id.asc"
    if only_missing:
        query += "&xbrl_detail_fetched_date=is.null"
    if since:
        query += f"&disc_date=gte.{since}"
    if with_article:
        query += "&article_published_at=not.is.null"
    return sb.select("edinet_large_holdings", query, limit=limit)


def main() -> int:
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--only-missing", action="store_true",
                       help="xbrl_detail_fetched_date が未設定の行だけを対象にする（再開用）")
    group.add_argument("--all", action="store_true",
                       help="全行を対象にする（保有割合の直し込みはこちら）")
    p.add_argument("--since", type=str, default="", help="対象の開示日の下限 YYYY-MM-DD")
    p.add_argument("--with-article", action="store_true",
                   help="記事になった開示だけを対象にする（公開記事の是正を全件スイープの完走前に"
                        "始めるため。記事のある開示は2026-01-05以降しかないので数十分で終わる）")
    p.add_argument("--limit", type=int, default=0, help="対象行数の上限（0=無制限）")
    p.add_argument("--sleep", type=float, default=0.1, help="1件ごとの待機秒（EDINETへの配慮）")
    p.add_argument("--dry-run", action="store_true", help="書き込まずに差分だけ数える")
    args = p.parse_args()

    rows = fetch_targets(args.only_missing, args.since, args.limit, args.with_article)
    print(f"対象 {len(rows)}件"
          f"（{'未取得のみ' if args.only_missing else '全行'}"
          f"{'・記事のある開示のみ' if args.with_article else ''}"
          f"{'・' + args.since + '以降' if args.since else ''}"
          f"{'・dry-run' if args.dry_run else ''}）")
    if not rows:
        return 0

    pending: list[dict] = []
    # 書き込めなかった行数。1件でもあれば「完了」と言えないので最後に必ず出す。
    write_failed = [0]
    ratio_changed = 0
    ratio_examples: list[str] = []
    failed = 0
    written = 0

    def flush() -> None:
        nonlocal pending, written
        if not pending:
            return
        if not args.dry_run:
            # doc_id・issuer_codeと今回計算し直した列だけを送る。payloadに無い列は
            # 触られないので、記事の紐付け（article_published_at）などは保持される。
            if not sb.upsert("edinet_large_holdings", pending, on_conflict="doc_id"):
                write_failed[0] += len(pending)
                print(f"  ⚠ upsert失敗（{len(pending)}行）。この分は書けていない")
                pending = []
                return
        written += len(pending)
        pending = []

    for i, row in enumerate(rows, 1):
        doc_id = row["doc_id"]
        details = fetch_xbrl_details(doc_id)
        if not details.get("xbrl_detail_fetched_date"):
            # XBRL本文が取れなかった回。既存の値をNULLで潰さないよう、行ごと飛ばす。
            failed += 1
            continue

        # issuer_code は変更しないが、NOT NULL制約のためpayloadに必ず載せる（fetch_targets参照）。
        patch = {"doc_id": doc_id, "issuer_code": row.get("issuer_code")}
        for key in HOLDING_DETAIL_COLUMNS:
            patch[key] = details.get(key)

        new_ratio = details.get("holding_ratio")
        if new_ratio is not None:
            patch["holding_ratio"] = new_ratio
            old_ratio = row.get("holding_ratio")
            if old_ratio is None or abs(new_ratio - old_ratio) >= RATIO_EPS:
                ratio_changed += 1
                if len(ratio_examples) < 10:
                    ratio_examples.append(f"{doc_id} {old_ratio} → {new_ratio}")
        if details.get("holding_ratio_prior") is not None:
            patch["holding_ratio_prior"] = details["holding_ratio_prior"]

        pending.append(patch)
        if len(pending) >= WRITE_BATCH:
            flush()
            print(f"  {i}/{len(rows)}件 処理済み（比率変更 {ratio_changed} / 取得失敗 {failed}）")
        if args.sleep:
            time.sleep(args.sleep)

    flush()
    print(f"完了: 処理{len(rows)}件 / 書き込み{written}件 / 書き込み失敗{write_failed[0]}件 / "
          f"比率変更{ratio_changed}件 / 取得失敗{failed}件")
    for example in ratio_examples:
        print(f"  例 {example}")
    # 書き込みに失敗した行が残っているうちは成功扱いにしない。
    return 1 if write_failed[0] else 0


if __name__ == "__main__":
    sys.exit(main())

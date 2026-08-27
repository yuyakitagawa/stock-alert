#!/usr/bin/env python3
"""
tools/delete_articles_by_id.py
指定したIDのブログ記事をmicroCMSから削除する。

2026-08-24のAdSense審査「有用性の低いコンテンツ」対応で、既存記事のリライトを進める中で
「書き直そうにも中身が作れない記事」が見つかったため用意した。対象は次の3種類:

1. データが壊れていて売買を説明できない記事
   - 保有比率の変化幅が0pt（何がどれだけ動いたのか書けない）
   - 保有比率も変化幅も1%未満（「大量保有報告書」として示せる中身が無い）
   実例: 保有0.33%に対し推定金額898.2億円（正しければ時価総額27兆円になる桁違いの値）
2. 同一開示から二重に生成された重複記事
3. 提出者を特定できず事実を再構築できない記事

いずれも読者に誤った数字を見せる、または内容が薄いままになるため、noindexで隠すのではなく
サイトから消す（2026-08-25にオーナーが削除を選択）。

削除は取り消せないので、実行前に対象記事の全フィールドを logs/ のJSONへ保存する
（microCMSへ再投稿すれば復元できる。ただしidは変わる）。

Usage:
  python3 tools/delete_articles_by_id.py --ids-file logs/unwritable_article_ids.txt
  python3 tools/delete_articles_by_id.py --ids-file logs/unwritable_article_ids.txt --delete
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from tools.reclassify_blog_articles import fetch_all_articles
from tools.cleanup_duplicate_blog_articles import delete_article

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids-file", action="append", required=True,
                   help="削除対象の記事IDを1行1件で並べたファイル（複数指定可）")
    p.add_argument("--delete", action="store_true", help="実際に削除する（既定は確認のみ）")
    args = p.parse_args()

    ids = set()
    for path in args.ids_file:
        with open(path, encoding="utf-8") as f:
            ids |= {line.strip() for line in f if line.strip()}

    by_id = {a["id"]: a for a in fetch_all_articles()}
    targets = [by_id[i] for i in sorted(ids) if i in by_id]
    missing = sorted(ids - set(by_id))

    print(f"指定 {len(ids)}件 / microCMSに存在 {len(targets)}件 / 既に無い {len(missing)}件")
    for a in targets[:10]:
        print(f"  {a['id']}: {a.get('stockName')}({a.get('stockCode')}) {(a.get('dealDate') or '')[:10]} {a.get('title','')[:40]}")
    if len(targets) > 10:
        print(f"  … 他 {len(targets)-10}件")

    if not args.delete:
        print("\n--delete を付けると実際に削除します（今回は確認のみ）")
        return
    if not targets:
        print("削除対象がありません")
        return

    backup = os.path.join(REPO_ROOT, "logs",
                          f"deleted_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    with open(backup, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=1)
    print(f"\n削除前の全フィールドを保存: {backup}")

    deleted = 0
    for a in targets:
        if delete_article(a["id"]):
            deleted += 1
        else:
            print(f"  ⚠ {a['id']} の削除に失敗しました")
        time.sleep(0.3)
    print(f"削除: {deleted}/{len(targets)}件")


if __name__ == "__main__":
    main()

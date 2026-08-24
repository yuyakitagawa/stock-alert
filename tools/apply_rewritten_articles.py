#!/usr/bin/env python3
"""
tools/apply_rewritten_articles.py
人間（またはClaude Code）が書き直した本文をmicroCMSへ反映する。

tools/export_article_fact_cards.py が書き出した事実カードをもとに書いた本文を、
{"記事ID": "<p>...</p>形式のHTML本文"} のJSONで受け取り、PATCHで更新する。

安全策:
- 更新前に現行の本文を logs/ へバックアップする（--backup で出力先を変更可）。
- 本文末尾の株価チャート<figure>は既存のものをそのまま引き継ぐ（画像の再生成はしない）。
- タイトルは更新しない（アイキャッチ画像に焼き込み済みのため据え置く）。
- 本文がHTMLの<p>で始まらないもの、既存より短くなるものは反映せず警告する。

Usage:
  python3 tools/apply_rewritten_articles.py --in /tmp/bodies.json --dry-run
  python3 tools/apply_rewritten_articles.py --in /tmp/bodies.json
"""
import os
import re
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from tools.reclassify_blog_articles import fetch_all_articles
from tools.rewrite_thin_blog_articles import visible_text_len, _FIGURE_RE
from web.publish_blog_articles import update_article, MicroCMSPermissionError

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help='{"記事ID": "本文HTML"} のJSON')
    p.add_argument("--dry-run", action="store_true", help="更新せず内容確認のみ")
    p.add_argument("--backup", default=None, help="バックアップの出力先（既定: logs/rewrite_backup_<日時>.json）")
    args = p.parse_args()

    with open(args.infile, encoding="utf-8") as f:
        bodies = json.load(f)

    by_id = {a["id"]: a for a in fetch_all_articles()}
    targets, problems = [], []
    for aid, new_body in bodies.items():
        old = by_id.get(aid)
        if old is None:
            problems.append((aid, "microCMSに存在しない記事ID"))
            continue
        if not new_body.lstrip().startswith("<p"):
            problems.append((aid, "本文が<p>で始まっていない"))
            continue
        old_len = visible_text_len(old.get("body"))
        new_len = visible_text_len(new_body)
        if new_len <= old_len:
            problems.append((aid, f"新本文が既存より短い({old_len}→{new_len}字)"))
            continue
        # 既存の株価チャートを引き継ぐ
        figure = _FIGURE_RE.search(old.get("body") or "")
        body = new_body + (figure.group(0) if figure else "")
        targets.append((aid, old, body, old_len, new_len))

    for aid, old, _, old_len, new_len in targets:
        print(f"  {aid}: {old.get('stockName')}({old.get('stockCode')}) {old_len}→{new_len}字")
    if problems:
        print(f"\n反映しない記事 {len(problems)}件:")
        for aid, reason in problems:
            print(f"  {aid}: {reason}")

    if args.dry_run:
        print(f"\n--dry-run のため更新していません（対象 {len(targets)}件）")
        return

    if not targets:
        print("反映対象がありません")
        return

    backup_path = args.backup or os.path.join(
        REPO_ROOT, "logs", f"rewrite_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump({aid: old.get("body") for aid, old, _, _, _ in targets}, f, ensure_ascii=False, indent=1)
    print(f"\n現行本文をバックアップ: {backup_path}")

    updated = 0
    for aid, _, body, _, _ in targets:
        try:
            if update_article(aid, {"body": body}):
                updated += 1
            else:
                print(f"  ⚠ {aid} の更新に失敗しました")
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため中断: {e}")
            break
        time.sleep(0.3)
    print(f"更新: {updated}/{len(targets)}件")


if __name__ == "__main__":
    main()

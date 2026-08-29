"""過去に削除した記事URLのリダイレクト先を、削除時のバックアップJSONから復元する。

背景（2026-08-29のGSC実測）:
  検索結果に出ているURL 924件のうち194件が404で、そこに28日で25クリックが着地していた。
  うち124件は削除済みの記事URL。削除時に `logs/deleted_*.json` へ全フィールドを保存してあるので、
  そこから `id` と `stockCode` を拾って `/stocks/<code>` への引き継ぎ先を登録できる。

対象ファイル: `logs/deleted_*.json`（削除ツールのバックアップ）。
  編集前バックアップ（`*_fix_backup_*.json` 等）は記事が生きているので読まない。

実行:
    python3 tools/backfill_article_redirects.py            # dry-run（登録内容の確認）
    python3 tools/backfill_article_redirects.py --write    # Supabaseへ登録
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import article_redirects

LOG_GLOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "logs", "deleted_*.json")


def collect(paths: list) -> list:
    """バックアップJSONから {article_id, target_path, reason} を作る。
    同じidが複数のログに出たら先に見つかった方を採る（古い削除ログが先に来る想定）。"""
    rows, seen = [], set()
    for path in sorted(paths):
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            article_id = item.get("id")
            target = article_redirects.stock_target(item)
            if not article_id or not target or article_id in seen:
                continue
            seen.add(article_id)
            rows.append({"article_id": article_id, "target_path": target,
                         "reason": f"backfill:{os.path.basename(path)}"})
    return rows


def main():
    p = argparse.ArgumentParser(description="削除済み記事のリダイレクト先をバックアップJSONから復元する")
    p.add_argument("--write", action="store_true", help="実際にSupabaseへ登録する（無指定はdry-run）")
    p.add_argument("--limit", type=int, default=10, help="dry-runで表示する件数")
    args = p.parse_args()

    paths = glob.glob(LOG_GLOB)
    rows = collect(paths)
    print(f"バックアップ {len(paths)}ファイル から {len(rows)}件のリダイレクトを検出")
    for r in rows[:args.limit]:
        print(f"  /articles/{r['article_id']} → {r['target_path']}")
    if len(rows) > args.limit:
        print(f"  … 他 {len(rows) - args.limit}件")
    if not args.write:
        print("\n--write で登録します（今回は確認のみ）")
        return 0
    ok = article_redirects.record_many(rows)
    print(f"登録{'完了' if ok else 'に失敗'}: {len(rows)}件")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

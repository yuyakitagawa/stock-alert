"""ニュース価値の無い開示の記事をmicroCMSから削除する。

web.publish_blog_articles.is_indexable_article()（推定金額3億円以上 または
保有比率の変化1pt以上＝表示側のindex基準）を満たさない記事が対象。2026-08-18に生成側の
足切りを入れる前に公開された分（保有比率0.04%・推定額0億円の変更報告書など）を回収する。

**新規記事の足切り(is_worth_publishing、2026-08-29に5億円/1.5ptへ引き上げ)ではなく
index基準で判定する**。新しい足切りで消すと、既に順位が付いている既存記事まで
巻き込んで24%削ることになるため。

Googleに「/articles/テンプレートは低品質」と学習されるとテンプレート全体の新規記事が
クロールされなくなるため（GSC「検出 - インデックス未登録」の主因）、サイトから消す。

削除は取り消せないので、実行前に対象記事の全フィールドを --backup 先のJSONへ保存する
（microCMSへ再投稿すれば復元できる。ただしidは変わる）。

実行:
    python3 tools/delete_low_value_blog_articles.py              # dry-run（対象の表示のみ）
    python3 tools/delete_low_value_blog_articles.py --delete     # バックアップを取ってから削除
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.publish_blog_articles import (
    _microcms_base_url, _microcms_headers, is_indexable_article,
    MICROCMS_DOMAIN, MICROCMS_KEY,
)
from tools.cleanup_duplicate_blog_articles import delete_article


def fetch_all_articles() -> list:
    """全記事を本文込みで取得する（バックアップに使うので fields は絞らない）。"""
    articles, offset = [], 0
    while True:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={"limit": 100, "offset": offset, "orders": "-publishedAt"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        articles.extend(data.get("contents", []))
        offset += 100
        if offset >= data.get("totalCount", 0):
            return articles


def find_low_value(articles: list) -> list:
    """表示側のindex基準に掛からない記事を返す。"""
    return [
        a for a in articles
        if not is_indexable_article(a.get("dealAmount") or 0, a.get("ratioChangePct") or 0)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delete", action="store_true", help="実際に削除する（無指定はdry-run）")
    p.add_argument("--backup", default=None, help="バックアップJSONの出力先（既定: logs/deleted_low_value_articles_<日時>.json）")
    args = p.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("[delete_low_value_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return

    articles = fetch_all_articles()
    targets = find_low_value(articles)
    print(f"全{len(articles)}記事のうち、基準未満は{len(targets)}件")
    if not targets:
        return

    for a in targets[:20]:
        print(f"  {a['id']}: {a.get('stockName')}({a.get('stockCode')}) "
              f"{a.get('dealAmount')}億円 / {a.get('ratioChangePct')}pt — {a.get('title')}")
    if len(targets) > 20:
        print(f"  ...ほか{len(targets) - 20}件")

    if not args.delete:
        print("dry-run（--delete で実行）")
        return

    backup_path = args.backup or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs",
        f"deleted_low_value_articles_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json",
    )
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    print(f"バックアップ: {backup_path}")

    deleted, failed = 0, []
    for a in targets:
        if delete_article(a["id"]):
            deleted += 1
        else:
            failed.append(a["id"])
    print(f"削除完了: {deleted}件" + (f" / 失敗: {len(failed)}件 {failed}" if failed else ""))


if __name__ == "__main__":
    main()

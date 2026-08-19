#!/usr/bin/env python3
"""
tools/reclassify_blog_articles.py
既存のmicroCMSブログ「大口投資家の監視ブログ」（kujira-watch/）記事のdealType（投資家分類）を、
新しい13分類（edinet_filer_classificationマスター）に基づいて再分類する。

旧dealType（インサイダー買い/日系ファンド買い/外資系ファンド買い等）は現在のmicroCMS
セレクトフィールドの選択肢から外れており、既存記事だけが古い値のまま残っている状態を
解消するために使う（web/publish_blog_articles.py の classify_filer() 導入に伴う一括移行）。

各記事のstockCode+dealDateからedinet_large_holdingsを逆引きしてfiler_nameを特定し、
classify_filer()で新分類を判定してdealTypeを更新する。同一銘柄・同一開示日に複数の
提出者がいて一意に特定できない記事はスキップし、末尾にリストアップする（手動確認用）。

更新はPUT（完全上書き）で行う。PATCH権限が無いAPIキーでも動くようにするため、
記事は全フィールド取得し、dealTypeだけ書き換えて他のフィールドはそのまま送り返す。

Usage:
  python3 tools/reclassify_blog_articles.py --dry-run   # 変更内容の確認のみ（何も更新しない）
  python3 tools/reclassify_blog_articles.py             # 実際にmicroCMSを更新
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import requests

import lib.supabase_client as sb
from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, _microcms_base_url, _microcms_headers,
    classify_filer, update_article, MicroCMSPermissionError,
)

load_dotenv()


# PUTでの完全上書きに含めてはいけないmicroCMSのメタ情報フィールド
_METADATA_FIELDS = {"id", "createdAt", "updatedAt", "publishedAt", "revisedAt"}


def fetch_all_articles() -> list[dict]:
    """microCMSから全記事を全フィールド取得する（100件ずつページング）。
    PUTでの更新時に他フィールドをそのまま送り返せるよう、fieldsで絞り込まない。"""
    articles = []
    offset = 0
    limit = 100
    while True:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            # ordersを明示しないとページングの並びが安定せず、途中でPATCHが走ると
            # 一部の記事がどのページにも現れない（2026-08-19に是正ツールで取りこぼしを確認）。
            # createdAtは更新で変わらないので昇順で固定する。
            params={"limit": limit, "offset": offset, "orders": "createdAt"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        contents = data.get("contents", [])
        articles.extend(contents)
        if offset + limit >= data.get("totalCount", 0):
            break
        offset += limit
    return articles


def build_put_payload(article: dict, new_deal_type: str) -> dict:
    """記事の全フィールドからメタ情報を除いてdealTypeだけ書き換えたPUT用payloadを作る。"""
    payload = {k: v for k, v in article.items() if k not in _METADATA_FIELDS}
    payload["dealType"] = new_deal_type
    return payload


def find_filer_names(code: str, disc_date: str) -> set[str]:
    """stockCode+dealDateの日付部分に一致するedinet_large_holdingsの提出者名集合を返す。
    1件のみなら一意特定、複数あれば手動確認が必要（呼び出し側で判定）。"""
    rows = sb.select(
        "edinet_large_holdings",
        f"issuer_code=eq.{code}&disc_date=eq.{disc_date}&select=filer_name",
    )
    return {r["filer_name"] for r in rows if r.get("filer_name")}


def normalize_deal_type(value) -> "str | None":
    """microCMSのdealTypeは複数選択(配列)設定の記事が混在するため単一値に正規化する。
    空配列（値未設定）はNoneを返す（旧: value[0]が空配列でIndexErrorになっていたバグの修正）。"""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="更新せず変更内容を表示するのみ")
    args = p.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY が未設定です。")
        sys.exit(1)

    articles = fetch_all_articles()
    print(f"記事総数: {len(articles)}件\n")

    changed = 0
    unchanged = 0
    skipped_no_match: list[str] = []
    skipped_ambiguous: list[tuple] = []

    for a in articles:
        code = a.get("stockCode")
        deal_date = (a.get("dealDate") or "")[:10]
        if not code or not deal_date:
            skipped_no_match.append(a["id"])
            continue

        names = find_filer_names(code, deal_date)
        if len(names) != 1:
            if len(names) > 1:
                skipped_ambiguous.append((a["id"], code, deal_date, names))
            else:
                skipped_no_match.append(a["id"])
            continue
        filer_name = names.pop()

        info = classify_filer(filer_name)
        new_deal_type = info["category"]
        old_deal_type = normalize_deal_type(a.get("dealType"))

        if new_deal_type == old_deal_type:
            unchanged += 1
            continue

        changed += 1
        print(f"  {a['id']}: {code} {deal_date} 「{filer_name}」 {old_deal_type} → {new_deal_type}")
        if not args.dry_run:
            try:
                ok = update_article(a["id"], build_put_payload(a, new_deal_type))
                if not ok:
                    print(f"  ⚠ {a['id']} の更新に失敗しました（詳細は上記ログ参照）")
            except MicroCMSPermissionError as e:
                print(f"  ✖ 権限エラーのため中断: {e}")
                break

    print(
        f"\n変更対象: {changed}件 / 変更不要: {unchanged}件 / "
        f"提出者特定不能: {len(skipped_no_match)}件 / 複数提出者で特定不能: {len(skipped_ambiguous)}件"
    )
    if skipped_ambiguous:
        print("\n複数提出者で特定不能（手動確認が必要）:")
        for aid, code, d, names in skipped_ambiguous:
            print(f"  {aid}: {code} {d} candidates={names}")
    if args.dry_run:
        print("\n--dry-run のため実際の更新は行っていません。")


if __name__ == "__main__":
    main()

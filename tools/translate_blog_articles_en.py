#!/usr/bin/env python3
"""
tools/translate_blog_articles_en.py
既存のmicroCMSブログ「大口投資家の監視ブログ」（kujira-watch/、英語版は/en）記事のうち、
titleEn/bodyEn（英訳）が未設定のものにClaudeで英訳を生成し、PATCHで書き戻すバックフィル。

kujira-watchの/en配下は、titleEn/bodyEnが両方揃っている記事のみを一覧・詳細に表示する
（日英混在ページを検索エンジンに出さないため）。新規記事はweb/publish_blog_articles.pyの
generate_article_body()が投稿時に同時生成するが、それ以前に投稿済みの既存記事は
titleEn/bodyEnが無いため、本スクリプトで一括生成する。

冪等: titleEnが既にある記事はスキップするため、何度実行しても未翻訳分だけが処理される。
途中で止めても再実行すれば続きから進む。

Usage:
  python3 tools/translate_blog_articles_en.py --dry-run          # 対象件数の確認のみ
  python3 tools/translate_blog_articles_en.py --limit 10          # 先頭10件だけ試す
  python3 tools/translate_blog_articles_en.py                     # 全件処理
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
import requests

from web.publish_blog_articles import (
    MICROCMS_DOMAIN, MICROCMS_KEY, ANTHROPIC_API_KEY, CLAUDE_MODEL,
    _microcms_base_url, _microcms_headers, update_article, MicroCMSPermissionError,
)

load_dotenv()

# microCMSのレート制限・Claude APIコストを抑えるため、記事間に間隔を空ける。
SLEEP_BETWEEN_ARTICLES_SEC = 1.0


def fetch_untranslated_articles(limit: "int | None" = None) -> list[dict]:
    """titleEnが未設定（=英語版が無い）記事をid/title/bodyのみ取得する（100件ずつページング）。
    microCMSの`titleEn[exists]`フィルタは実データがあってもヒットしない既知不具合があるため
    （2026-08-14確認、begins_with等の他演算子は正常に動く一方でexistsだけ機能しない。原因は
    microCMS側未特定。kujira-watch側lib/microcms.tsでも同じ理由でクライアント側フィルタに
    切替済み）、全件取得してPython側でtitleEn有無を判定する。"""
    articles = []
    offset = 0
    page_size = 100
    while True:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "limit": page_size,
                "offset": offset,
                "fields": "id,title,body,titleEn",
                "orders": "-publishedAt",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        contents = [c for c in data.get("contents", []) if not c.get("titleEn")]
        articles.extend(contents)
        if limit is not None and len(articles) >= limit:
            return articles[:limit]
        total = data.get("totalCount", 0)
        if offset + page_size >= total:
            break
        offset += page_size
    return articles


def translate_article(title: str, body: str) -> "dict | None":
    """日本語のtitle/bodyを英訳したtitleEn/bodyEnをJSONで返す。パース失敗時はNone。
    事実の追加・削除はせず翻訳のみ行わせる（generate_article_body()と同じ「※推測:」規約を
    "*Speculation:"に対応づける）。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""以下は日本株の大量保有報告書に基づくブログ記事（日本語）です。
事実を追加・削除・変更せず、自然な英語（投資ニュース記事のトーン、直訳調は避ける）に翻訳してください。
本文はHTML（<p>タグ区切り）なので、タグ構造は保ったまま翻訳してください。
「※推測:」で始まる文がある場合は "*Speculation:" という接頭辞に置き換えてください。

タイトル: {title}
本文: {body}

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"titleEn": "英訳タイトル", "bodyEn": "<p>...</p>形式の英訳本文"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if not data.get("titleEn") or not data.get("bodyEn"):
            return None
        return data
    except Exception as e:
        print(f"    ⚠ 翻訳失敗: {e}")
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="対象件数の確認のみ（翻訳・更新はしない）")
    p.add_argument("--limit", type=int, default=None, help="処理する記事数の上限（未指定なら全件）")
    args = p.parse_args()

    if not MICROCMS_DOMAIN or not MICROCMS_KEY:
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY が未設定です。")
        sys.exit(1)
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY が未設定です。")
        sys.exit(1)

    articles = fetch_untranslated_articles(limit=args.limit)
    print(f"未翻訳の記事: {len(articles)}件")

    if args.dry_run:
        for a in articles[:10]:
            print(f"  - {a['id']}: {a['title']}")
        if len(articles) > 10:
            print(f"  ...他{len(articles) - 10}件")
        return

    ok_count = 0
    fail_count = 0
    for i, article in enumerate(articles, 1):
        title = article["title"]
        print(f"[{i}/{len(articles)}] {article['id']}: {title[:40]}")
        translated = translate_article(title, article["body"])
        if translated is None:
            print("    ⏭ 翻訳失敗のためスキップ")
            fail_count += 1
            time.sleep(SLEEP_BETWEEN_ARTICLES_SEC)
            continue

        try:
            success = update_article(article["id"], {
                "titleEn": translated["titleEn"],
                "bodyEn": translated["bodyEn"],
            })
        except MicroCMSPermissionError as e:
            print(f"    ✖ 権限エラーのため中断します: {e}")
            break

        if success:
            ok_count += 1
        else:
            fail_count += 1
        time.sleep(SLEEP_BETWEEN_ARTICLES_SEC)

    print(f"完了: 成功{ok_count}件 / 失敗{fail_count}件（全{len(articles)}件中）")


if __name__ == "__main__":
    main()

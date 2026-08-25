"""既存ブログ記事に解説図（web/article_figures.py）をバックフィルする（手動実行専用）。

2026-08-25以降に公開する記事には publish_blog_articles.attach_figures() が図を差し込むが、
それ以前の記事（約1,000件）は画像がアイキャッチ＋末尾の株価チャートの2枚しかない。
本ツールは本文を作り直さず（＝Anthropic APIを使わない）、Supabaseの既存データから図だけを
作って本文の該当段落の直後に差し込み、microCMSをPATCH更新する。

処理:
  1. microCMSから記事を新しい順に全件取得（body/bodyEn込み）
  2. 株価チャート以外の<figure>が既にある記事は処理済みとして除外
  3. 大量保有報告書の記事は build_context_facts() → build_article_figures()、
     自社株買いの記事は tdnet_buybacks の過去決議 → buyback_article_figures()
  4. 図をmicroCMSへアップロードし、insert_figures_into_body() で本文へ差し込む
     （株価チャートは末尾のまま。bodyEnにも同じ画像を英語キャプションで入れる）

まず --dry-run で対象件数と図の枚数を確認し、--limit で少しずつ流すこと
（記事1件あたり最大3枚の画像をmicroCMSのメディア領域に追加する）。

使い方:
    python3 tools/backfill_article_figures.py --dry-run
    python3 tools/backfill_article_figures.py --limit 50 --days 60
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

load_dotenv()

import web.publish_blog_articles as pb
from web.article_figures import (
    build_article_figures, buyback_article_figures, figure_html, insert_figures_into_body,
)
from web.publish_buyback_articles import DEAL_TYPE as BUYBACK_DEAL_TYPE, prior_buybacks

ARTICLE_FIELDS = "id,title,stockCode,stockName,dealType,dealDate,dealAmount,ratioChangePct,tags,filerName,body,bodyEn"

_FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.S)
# 株価チャートのaltは publish_blog_articles が「〇〇（コード）株価推移（直近3ヶ月）」で固定
_CHART_MARK = "株価推移"
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)%")


def deal_date_str(article: dict) -> str:
    return str(article.get("dealDate") or "")[:10]


def category_of(article: dict) -> str:
    dt = article.get("dealType")
    if isinstance(dt, list):
        return dt[0] if dt else ""
    return dt or ""


def is_buyback(article: dict) -> bool:
    tags = {t.strip() for t in (article.get("tags") or "").split(",")}
    return category_of(article) == BUYBACK_DEAL_TYPE or BUYBACK_DEAL_TYPE in tags


def has_explainer_figure(body: str) -> bool:
    """株価チャート以外の<figure>が既にあるか（＝バックフィル済み）。"""
    return any(_CHART_MARK not in f for f in _FIGURE_RE.findall(body or ""))


def merge_into_body(body: str, figures: list) -> str:
    """本文から<figure>を一旦外して解説図を段落間に差し込み、株価チャートを末尾へ戻す。"""
    charts = [f for f in _FIGURE_RE.findall(body or "") if _CHART_MARK in f]
    stripped = _FIGURE_RE.sub("", body or "")
    return insert_figures_into_body(stripped, figures) + "".join(charts)


def fetch_articles() -> list:
    """microCMSから全記事を新しい順に取得する（100件ずつページング）。"""
    articles, offset, limit = [], 0, 100
    while True:
        resp = requests.get(
            pb._microcms_base_url(),
            headers=pb._microcms_headers(),
            params={"limit": limit, "offset": offset, "orders": "-dealDate", "fields": ARTICLE_FIELDS},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        contents = data.get("contents", [])
        articles.extend(contents)
        if offset + limit >= data.get("totalCount", 0) or not contents:
            break
        offset += limit
    return articles


def select_candidates(articles: list, days: "int | None" = None, limit: "int | None" = None,
                      today: "str | None" = None) -> list:
    """図がまだ無く、銘柄コードと開示日が分かる記事だけを新しい順に返す。"""
    from datetime import date, timedelta

    cutoff = None
    if days:
        base = date.fromisoformat(today) if today else date.today()
        cutoff = (base - timedelta(days=days)).isoformat()
    out = []
    for a in articles:
        if not a.get("stockCode") or not deal_date_str(a):
            continue
        if cutoff and deal_date_str(a) < cutoff:
            continue
        if has_explainer_figure(a.get("body") or ""):
            continue
        out.append(a)
    out.sort(key=deal_date_str, reverse=True)
    return out[:limit] if limit else out


def ratio_of(article: dict) -> "float | None":
    """記事の保有比率。ratioChangePctは変化幅なので使えず、開示データ→タイトルの順に引く。"""
    row = pb.sb.select_one(
        "edinet_large_holdings",
        f"select=holding_ratio&issuer_code=eq.{article['stockCode']}"
        f"&disc_date=eq.{deal_date_str(article)}"
        f"&filer_name=eq.{requests.utils.quote(article.get('filerName') or '')}",
    ) if article.get("filerName") else None
    if row and row.get("holding_ratio") is not None:
        return float(row["holding_ratio"])
    m = _RATIO_RE.search(article.get("title") or "")
    return float(m.group(1)) if m else None


def figures_for(article: dict) -> list:
    """記事1件分の解説図（バイト列つき）を作る。作れなければ空リスト。"""
    code, disc_date = str(article["stockCode"]), deal_date_str(article)
    if is_buyback(article):
        return buyback_article_figures({
            "stock_name": article.get("stockName") or "",
            "disc_date": disc_date,
            "amount_oku": article.get("dealAmount"),
            "prior": prior_buybacks(code, f"{disc_date}T23:59:59"),
        })
    filer_name = article.get("filerName") or ""
    ratio = ratio_of(article)
    if not filer_name or ratio is None:
        return []
    return build_article_figures({
        "stock_name": article.get("stockName") or "",
        "stock_code": code,
        "filer_name": filer_name,
        "holding_ratio": ratio,
        "context_facts": pb.build_context_facts(code, filer_name, disc_date),
    })


def run(args) -> int:
    if not pb.MICROCMS_DOMAIN or not pb.MICROCMS_KEY:
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定")
        return 1

    all_articles = fetch_articles()
    targets = select_candidates(all_articles, days=args.days, limit=args.limit)
    print(f"全記事 {len(all_articles)}件 → 対象 {len(targets)}件"
          f"（days={args.days}, limit={args.limit}, dry_run={args.dry_run}）")

    done = skipped = failed = total_figures = 0
    for i, a in enumerate(targets, 1):
        label = f"[{i}/{len(targets)}] {a.get('stockName')}({a.get('stockCode')}) {deal_date_str(a)} {a['id']}"
        figures = figures_for(a)
        if not figures:
            print(f"{label} → スキップ（作れる図が無い）")
            skipped += 1
            continue
        if args.dry_run:
            print(f"{label} → 図{len(figures)}枚 {[f['filename'] for f in figures]}")
            done += 1
            total_figures += len(figures)
            continue

        ja, en = [], []
        for fig in figures:
            url = pb._upload_media(fig["bytes"], fig["filename"])
            if not url:
                continue
            ja.append({"html": figure_html(url, fig["alt"], fig["caption"]), "anchors": fig["anchors"]})
            en.append({"html": figure_html(url, fig["alt_en"], fig["caption_en"]), "anchors": []})
        if not ja:
            print(f"{label} → 失敗（アップロード）")
            failed += 1
            continue

        payload = {"body": merge_into_body(a.get("body") or "", ja)}
        if a.get("bodyEn"):
            payload["bodyEn"] = merge_into_body(a["bodyEn"], en)
        try:
            ok = pb.update_article(a["id"], payload)
        except pb.MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため中断: {e}")
            return 2
        if ok:
            print(f"{label} → OK 図{len(ja)}枚")
            done += 1
            total_figures += len(ja)
        else:
            print(f"{label} → 失敗（PATCH）")
            failed += 1
        if args.sleep:
            time.sleep(args.sleep)

    print(f"完了: 成功 {done}（図{total_figures}枚） / スキップ {skipped} / 失敗 {failed}")
    return 0 if failed == 0 else 2


def main():
    p = argparse.ArgumentParser(description="既存ブログ記事への解説図バックフィル")
    p.add_argument("--dry-run", action="store_true", help="対象と図の枚数を表示するだけ（アップロードしない）")
    p.add_argument("--limit", type=int, default=None, help="処理する最大件数（新しい順）")
    p.add_argument("--days", type=int, default=None, help="開示日が直近N日以内の記事に絞る")
    p.add_argument("--sleep", type=float, default=0.5, help="記事ごとの待ち秒数（microCMSへの連打を避ける）")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()

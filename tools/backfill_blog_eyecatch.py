"""既存ブログ記事にアイキャッチ画像をバックフィルする（手動実行専用）。

web/publish_blog_articles.py の build_eyecatch_for_article() が 2026-08-15〜08-22 の間
{"url": ...} オブジェクトを返していたため microCMS 側で「'eyecatch' has unexpected data type」
として除外され、全記事（2026-08-23時点で950件）が画像なしで公開されていた。
コミット 61d2d2a4 で文字列URLを返すよう修正済みなので新規記事には付くが、過去記事には
本ツールで後追いする。

処理:
  1. microCMS GET /api/v1/articles?filters=eyecatch[not_exists] で画像なし記事を新しい順に取得
  2. 記事ごとに generate_eyecatch_image() → upload_eyecatch() → update_article()（PATCH）
     で eyecatch にメディアURL文字列を設定
  3. 保有比率は Supabase edinet_large_holdings を stockCode+dealDate+filerName で引き、
     無ければ記事タイトルの「保有比率X%」「X%を新規保有」等から正規表現で取る

Pexels無料枠は200リクエスト/時・2万/月なので --max-per-hour（既定180）で間隔を空ける。
まず --limit 50 で動作確認し、--days 60 --index-only で直近のインデックス対象記事から回す。

--replace は逆に「画像が既に付いている記事」を対象にして画像を作り直す。2026-08-27に
search_pexels_photo() が常に photos[0] を返していたバグ（同じ投資家分類の記事が全部同じ写真に
なる）を直したため、それ以前に生成した既存記事の画像を差し替えるために使う。写真の選択は
提出者名+銘柄名+開示日のハッシュで決まるので、同じ記事を2回流しても同じ画像になる。

使い方:
    python3 tools/backfill_blog_eyecatch.py --dry-run --limit 50
    python3 tools/backfill_blog_eyecatch.py --limit 50 --days 60 --index-only
    python3 tools/backfill_blog_eyecatch.py --replace --max-per-hour 600
    python3 tools/backfill_blog_eyecatch.py --font "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
"""
import argparse
import os
import re
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

load_dotenv()

import lib.supabase_client as sb
import web.publish_blog_articles as pb

# kujira-watch/src/lib/articleIndexability.ts と同じ基準（--index-only）。
# 新規記事の足切り(pb.MIN_*)は2026-08-29に引き上げたが、index基準は据え置きなので
# そちらを参照する（引き上げた方を見ると、indexされている既存記事のアイキャッチを飛ばす）。
INDEXABLE_MIN_DEAL_AMOUNT_OKU = pb.INDEXABLE_MIN_DEAL_AMOUNT_OKU
INDEXABLE_MIN_RATIO_CHANGE_PT = pb.INDEXABLE_MIN_RATIO_CHANGE_PT

ARTICLE_FIELDS = "id,title,stockCode,stockName,dealType,dealDate,dealAmount,ratioChangePct,tags,filerName"

# タイトルから保有比率を拾う。「保有比率7.48%に引き上げ」「5.12%を新規保有」「保有比率を6.0%に訂正」
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)%")


def is_indexable(article: dict) -> bool:
    if (article.get("dealAmount") or 0) >= INDEXABLE_MIN_DEAL_AMOUNT_OKU:
        return True
    return abs(article.get("ratioChangePct") or 0) >= INDEXABLE_MIN_RATIO_CHANGE_PT


def deal_date_str(article: dict) -> str:
    return str(article.get("dealDate") or "")[:10]


def badge_label_for(article: dict) -> str:
    """publish側と同じバッジ判定を、記事の tags / タイトルから復元する。
    絵文字は generate_eyecatch_image() 側で ▲▼■ に置き換えられる。"""
    tags = article.get("tags") or ""
    tag_set = {t.strip() for t in tags.split(",")}
    if "訂正" in tag_set or "訂正" in (article.get("title") or ""):
        return "📝 訂正"
    if "売り" in tag_set:
        return "📉 売却"
    if "新規保有" in (article.get("title") or ""):
        return "📈 新規取得"
    return "📈 買い増し"


def ratio_from_title(title: str) -> "float | None":
    m = _RATIO_RE.search(title or "")
    return float(m.group(1)) if m else None


def select_candidates(articles: list, days: "int | None" = None, index_only: bool = False,
                      limit: "int | None" = None, today: "date | None" = None) -> list:
    """画像なし記事の一覧から処理対象を絞る。新しい順（dealDate降順）を保つ。"""
    today = today or date.today()
    cutoff = (today - timedelta(days=days)).isoformat() if days else None
    out = []
    for a in articles:
        if cutoff and deal_date_str(a) < cutoff:
            continue
        if index_only and not is_indexable(a):
            continue
        out.append(a)
    out.sort(key=deal_date_str, reverse=True)
    if limit:
        out = out[:limit]
    return out


def done_ids_from_logs(paths: list) -> set:
    """過去の実行ログから処理済みの記事IDを拾う。--replace は毎回microCMSから
    「画像あり記事」を取り直すため、中断して再開すると先頭からやり直しになる。
    やり直しても画像の中身は同じ（写真の選択が決定的）だが、microCMSに同じ画像の
    メディアが二重に積まれるだけなので、済んだ分は飛ばす。"""
    done = set()
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if "→ OK " not in line:
                        continue
                    # 例: [12/964] 銘柄(1234) 2026-08-27 <article_id> → OK https://...
                    head = line.split("→ OK ")[0].split()
                    if head:
                        done.add(head[-1])
        except OSError as e:
            print(f"⚠ スキップ用ログを読めません: {path} ({e})")
    return done


def fetch_articles_without_eyecatch(replace: bool = False) -> list:
    """microCMSから対象記事を新しい順に全件取得する（100件ずつページング）。
    既定は eyecatch 未設定の記事。replace=True では逆に eyecatch 設定済みの記事を返す
    （写真が全記事で重複していた頃の画像を作り直すため）。"""
    articles, offset, limit = [], 0, 100
    filters = "eyecatch[exists]" if replace else "eyecatch[not_exists]"
    while True:
        resp = requests.get(
            pb._microcms_base_url(),
            headers=pb._microcms_headers(),
            params={
                "limit": limit, "offset": offset, "orders": "-dealDate",
                "filters": filters, "fields": ARTICLE_FIELDS,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        contents = data.get("contents", [])
        articles.extend(contents)
        if offset + limit >= data.get("totalCount", 0) or not contents:
            break
        offset += limit
    return articles


def fetch_holding_ratios(articles: list) -> dict:
    """対象記事の (stockCode, dealDate, filerName) → holding_ratio を1クエリでまとめて引く。"""
    codes = sorted({str(a.get("stockCode")) for a in articles if a.get("stockCode")})
    dates = [deal_date_str(a) for a in articles if deal_date_str(a)]
    if not codes or not dates:
        return {}
    q = (
        f"issuer_code=in.({','.join(codes)})&disc_date=gte.{min(dates)}&disc_date=lte.{max(dates)}"
        "&select=issuer_code,disc_date,filer_name,holding_ratio&order=doc_id"
    )
    rows = sb.select("edinet_large_holdings", q)
    out = {}
    for r in rows:
        if r.get("holding_ratio") is None:
            continue
        out[(str(r["issuer_code"]), str(r["disc_date"])[:10], r.get("filer_name") or "")] = float(r["holding_ratio"])
    return out


def resolve_holding_ratio(article: dict, ratios: dict) -> "float | None":
    key = (str(article.get("stockCode")), deal_date_str(article), article.get("filerName") or "")
    if key in ratios:
        return ratios[key]
    return ratio_from_title(article.get("title") or "")


def build_card(article: dict, holding_ratio: float) -> dict:
    return {
        "filer_name": article.get("filerName") or "大口投資家",
        "stock_name": article.get("stockName") or "",
        "holding_ratio": holding_ratio,
        "badge_label": badge_label_for(article),
        "disc_date": deal_date_str(article),
    }


def category_of(article: dict) -> str:
    dt = article.get("dealType")
    if isinstance(dt, list):
        return dt[0] if dt else ""
    return dt or ""


def run(args) -> int:
    if args.font:
        pb.EYECATCH_FONT_PATH = args.font
    if not os.path.exists(pb.EYECATCH_FONT_PATH):
        print(f"フォントが見つかりません: {pb.EYECATCH_FONT_PATH}（--font か EYECATCH_FONT_PATH で指定）")
        return 1
    if not pb.MICROCMS_DOMAIN or not pb.MICROCMS_KEY:
        print("MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定")
        return 1
    if not args.dry_run and not pb.PEXELS_API_KEY:
        print("PEXELS_API_KEY 未設定")
        return 1

    all_articles = fetch_articles_without_eyecatch(replace=args.replace)
    done_ids = done_ids_from_logs(args.skip_log or [])
    if done_ids:
        before = len(all_articles)
        all_articles = [a for a in all_articles if a.get("id") not in done_ids]
        print(f"処理済み {len(done_ids)}件をログから除外（{before} → {len(all_articles)}件）")
    targets = select_candidates(all_articles, days=args.days, index_only=args.index_only, limit=args.limit)
    kind = "画像あり記事(差し替え)" if args.replace else "画像なし記事"
    print(f"{kind} {len(all_articles)}件 → 対象 {len(targets)}件"
          f"（days={args.days}, index_only={args.index_only}, limit={args.limit}, dry_run={args.dry_run}）")
    if not targets:
        return 0

    ratios = fetch_holding_ratios(targets)
    interval = 3600.0 / args.max_per_hour if args.max_per_hour > 0 else 0
    done = skipped = failed = 0
    last_call = 0.0
    for i, a in enumerate(targets, 1):
        ratio = resolve_holding_ratio(a, ratios)
        label = f"[{i}/{len(targets)}] {a.get('stockName')}({a.get('stockCode')}) {deal_date_str(a)} {a.get('id')}"
        if ratio is None and not args.replace:
            print(f"{label} → スキップ（保有比率が取れない）")
            skipped += 1
            continue
        # 差し替えでは比率が取れなくても写真の重複解消のために作り直す
        # （generate_eyecatch_image() 側が比率なしの表記に落とす）
        card = build_card(a, ratio or 0.0)
        if args.dry_run:
            ratio_label = f"{ratio:.2f}%" if ratio is not None else "比率なし"
            print(f"{label} → {card['badge_label']} {card['filer_name']} {ratio_label} [{category_of(a)}]")
            done += 1
            continue

        wait = interval - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.time()
        image = pb.generate_eyecatch_image(category_of(a), card)
        if not image:
            print(f"{label} → 失敗（画像生成）")
            failed += 1
            continue
        url = pb.upload_eyecatch(image)
        if not url:
            print(f"{label} → 失敗（アップロード）")
            failed += 1
            continue
        if patch_eyecatch(a["id"], url):
            print(f"{label} → OK {url}")
            done += 1
        else:
            # 失敗したURLを必ず出す。以前は出しておらず、50件中2件のHTTP 400を
            # ログから追えなかった（2026-08-30）。
            print(f"{label} → 失敗（PATCH） url={url}")
            failed += 1

    print(f"完了: 成功 {done} / スキップ {skipped} / 失敗 {failed}")
    return 0 if failed == 0 else 2


def patch_eyecatch(article_id: str, url: str, retries: int = 1, wait: float = 3.0) -> bool:
    """記事のeyecatchにメディアURLを貼る。1回だけリトライする。

    アップロード直後のメディアURLをコンテンツ側の検証が拒み
    HTTP 400 "'eyecatch' field invalid. Please set a valid URL." になることがある
    （2026-08-30の50件実行で2件。記事は旧画像のまま無傷で、再実行すると通った）。
    アップロードからPATCHまでの反映待ちと見て、作り直さず同じURLを貼り直す。"""
    for attempt in range(retries + 1):
        if pb.update_article(article_id, {"eyecatch": url}):
            return True
        if attempt < retries:
            time.sleep(wait)
    return False


def main():
    p = argparse.ArgumentParser(description="既存ブログ記事へのアイキャッチ画像バックフィル")
    p.add_argument("--dry-run", action="store_true", help="対象と画像に焼き込む内容を表示するだけ")
    p.add_argument("--limit", type=int, default=None, help="処理する最大件数（新しい順）")
    p.add_argument("--days", type=int, default=None, help="開示日が直近N日以内の記事に絞る")
    p.add_argument("--index-only", action="store_true",
                   help="インデックス対象（dealAmount>=3億円 or |ratioChangePct|>=1pt）のみ")
    p.add_argument("--replace", action="store_true",
                   help="画像が既に付いている記事を対象に画像を作り直す（写真重複バグの後始末）")
    # 候補リストをクエリ単位でキャッシュするようになったのでPexels検索APIは分類の数しか叩かない。
    # ここで律速しているのは写真CDNとmicroCMSへの連投なので、差し替え時は600程度まで上げてよい。
    p.add_argument("--skip-log", action="append", metavar="PATH",
                   help="過去の実行ログのパス。「→ OK」で終わった記事IDを対象から外す（中断からの再開用、複数指定可）")
    p.add_argument("--max-per-hour", type=int, default=180, help="画像生成の上限（件/時）")
    p.add_argument("--font", default=os.getenv("EYECATCH_FONT_PATH"),
                   help="日本語フォントのパス（既定: EYECATCH_FONT_PATH 環境変数）")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()

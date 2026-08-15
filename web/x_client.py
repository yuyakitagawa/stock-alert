"""
web/x_client.py
publish_blog_articles.py が投稿した新着記事のうち、ホームページの「注目」枠
（kujira-watch側 getFeaturedArticles()、直近プールをdealAmount降順に並べ直した
上位3件）に入っている記事だけをX(Twitter)へ自動投稿する。サイトで目立っていない
小粒な開示が「その日一番大きい」というだけでXに投稿される事態を避けるため、
「その日新規公開した記事」×「現在サイトで注目表示されている記事」の積集合を対象にする
（該当が無い地味な日は0件のこともある）。株価チャート画像
（publish_blog_articles.generate_price_chart_image、記事本文埋め込みと同じもの）を
添付し、テキストのみの投稿よりタイムライン上で目立つようにする（画像生成に
失敗した場合は画像無しでテキストのみ投稿する）。

認証はOAuth 1.0a User Context（X API v2 POST /2/tweetsの投稿、および画像アップロード用の
v1.1 media/uploadエンドポイントの両方に必須）。
X Developer Portalで「Read and Write」権限のAppを作成し、以下4つを取得する:
  X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
いずれか未設定の場合は投稿をスキップする（他のステップには影響しない）。
"""
import os

import requests
from requests_oauthlib import OAuth1

SITE_URL = "https://kujira-watch.com"
API_URL = "https://api.x.com/2/tweets"
MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"

# 1回の実行(1日1回)あたりの投稿上限（安全上限。実際にはfeatured_idsとの積集合が
# 3件を超えることは無い想定だが、念のためのキャップとして残す）。
TWEETS_PER_RUN = 3

# 無料/Basicプランのポスト上限(280字)に対する安全マージン。全角文字はX側の重み付けで
# 実質2字分として扱われるため、タイトルが長い場合の切り詰め基準を保守的に設定する。
TWEET_BODY_MAX_CHARS = 120


def _auth() -> "OAuth1 | None":
    key = os.getenv("X_API_KEY")
    key_secret = os.getenv("X_API_KEY_SECRET")
    token = os.getenv("X_ACCESS_TOKEN")
    token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    if not all([key, key_secret, token, token_secret]):
        return None
    return OAuth1(key, key_secret, token, token_secret)


def build_tweet_text(title: str, deal_amount_oku: float, is_sell: bool, article_id: str) -> str:
    """記事タイトル・金額規模からツイート本文を組み立てる。URLはX側でt.co短縮されるため
    文字数上限の計算には含めない。GA4で流入経路をXの自動投稿と識別できるようUTMを付与する。"""
    direction = "売却" if is_sell else "取得"
    body = f"{title}\n推定{direction}金額: {deal_amount_oku}億円"
    if len(body) > TWEET_BODY_MAX_CHARS:
        body = body[: TWEET_BODY_MAX_CHARS - 1] + "…"
    url = f"{SITE_URL}/articles/{article_id}?utm_source=x&utm_medium=social&utm_campaign=auto_post"
    return f"{body}\n{url}\n#EDINET #大量保有報告書"


def upload_media(image_bytes: bytes) -> "str | None":
    """画像(PNG)をv1.1 media/uploadへアップロードし、post_tweetのmedia_idに使う
    media_id_stringを返す。失敗時はNone（呼び出し側は画像無しで投稿を続行する）。"""
    auth = _auth()
    if auth is None:
        return None
    try:
        resp = requests.post(
            MEDIA_UPLOAD_URL, auth=auth,
            files={"media": ("chart.png", image_bytes, "image/png")}, timeout=30,
        )
        if not resp.ok:
            print(f"  ⚠ X画像アップロード失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("media_id_string")
    except Exception as e:
        print(f"  ⚠ X画像アップロード例外: {e}")
        return None


def post_tweet(text: str, media_id: "str | None" = None) -> bool:
    auth = _auth()
    if auth is None:
        return False
    payload: dict = {"text": text}
    if media_id:
        payload["media"] = {"media_ids": [media_id]}
    try:
        resp = requests.post(API_URL, auth=auth, json=payload, timeout=20)
        if resp.status_code not in (200, 201):
            print(f"  ⚠ X投稿失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  ⚠ X投稿例外: {e}")
        return False


# 日次サマリー「本日のクジラ」を投稿するUTC時（12時UTC=21時JST、edinet_blog.ymlの最終便）。
# 毎時バッチのうちこの1回だけ投稿することで、外部ストレージ無しで1日1回の重複ガードにする。
DAILY_SUMMARY_UTC_HOUR = 12


def _summary_line(article: dict, sign: str) -> str:
    """サマリーの「最大買い増し/最大売却」1行。提出者名が取れた記事は括弧書きで添える。"""
    filer = article.get("filerName") or ""
    filer_part = f"（{filer}）" if filer else ""
    return f"{article['stockName']} {sign}{article['dealAmount']}億円{filer_part}"


def build_daily_summary_text(articles: list, date_str: str, total_count: "int | None" = None) -> "str | None":
    """その日(dealDate)の記事一覧から「本日のクジラ」日次サマリー本文を組み立てる。
    0件の日はNone（投稿しない）。articlesはmicroCMSのdealAmount降順取得を前提とする。
    total_countはmicroCMSのtotalCount（100件超の日にarticlesが先頭100件に切れていても
    件数表示を正確にするため）。"""
    if not articles:
        return None
    count = total_count if total_count is not None else len(articles)
    total_oku = round(sum(a.get("dealAmount") or 0 for a in articles), 1)
    buys = [a for a in articles if "売り" not in (a.get("tags") or "")]
    sells = [a for a in articles if "売り" in (a.get("tags") or "")]

    month_day = f"{int(date_str[5:7])}/{int(date_str[8:10])}"
    lines = [f"🐋 本日のクジラ｜{month_day}の大量保有報告書", f"{count}件・合計{total_oku}億円", ""]
    if buys:
        lines += ["🟢 最大買い増し", _summary_line(buys[0], "+"), ""]
    if sells:
        lines += ["🔴 最大売却", _summary_line(sells[0], "-"), ""]
    url = f"{SITE_URL}/date/{date_str}?utm_source=x&utm_medium=social&utm_campaign=daily_summary"
    lines += [f"↓ 今日の全{count}件", url, "#EDINET #大量保有報告書"]
    return "\n".join(lines)


def fetch_articles_by_deal_date(date_str: str) -> "tuple[list, int]":
    """microCMSからdealDateが指定日の記事をdealAmount降順で取得する（最大100件）。
    (記事リスト, totalCount)を返す。失敗時は([], 0)。"""
    from web.publish_blog_articles import _microcms_base_url, _microcms_headers

    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"dealDate[equals]{date_str}T00:00:00.000Z",
                "orders": "-dealAmount",
                "limit": 100,
                "fields": "id,stockName,stockCode,dealAmount,tags,filerName",
            },
            timeout=20,
        )
        if not resp.ok:
            print(f"  ⚠ 日次サマリー用の記事取得失敗 HTTP {resp.status_code}")
            return [], 0
        data = resp.json()
        return data.get("contents", []), data.get("totalCount", 0)
    except Exception as e:
        print(f"  ⚠ 日次サマリー用の記事取得例外: {e}")
        return [], 0


def post_daily_summary(now_utc=None, force: bool = False) -> bool:
    """「本日のクジラ」日次サマリーを1日1回(21時JST=12時UTCの最終便のみ)Xへ投稿する。
    forceで時刻ガードを無視できる（手動実行用）。該当時刻以外・0件・認証未設定はFalse。"""
    from datetime import datetime, timedelta, timezone

    if _auth() is None:
        return False
    now = now_utc or datetime.now(timezone.utc)
    if not force and now.hour != DAILY_SUMMARY_UTC_HOUR:
        return False
    today_jst = (now + timedelta(hours=9)).strftime("%Y-%m-%d")
    articles, total_count = fetch_articles_by_deal_date(today_jst)
    text = build_daily_summary_text(articles, today_jst, total_count)
    if text is None:
        print("  [x_client] 本日の開示記事が無いため日次サマリーをスキップします")
        return False
    if post_tweet(text):
        print(f"  🐦 X日次サマリー投稿: {today_jst} 全{total_count}件")
        return True
    return False


def post_top_articles(published: list, featured_ids: set, top_n: int = TWEETS_PER_RUN) -> int:
    """publish_blog_articles.build_and_publish()が返すpublishedリスト（今日新規公開した
    記事）のうち、featured_ids（publish_blog_articles.get_featured_article_ids()、
    現在ホームページで「注目」表示されている記事id）にも含まれるものだけを、
    金額規模(dealAmount)が大きい順にXへ投稿する。積集合が無ければ0件（dry-run記事や
    featured_idsに入らない小粒な開示は対象外）。X認証情報が未設定の場合も0を返す。"""
    if _auth() is None:
        print("[x_client] X_API_KEY等が未設定のため投稿をスキップします")
        return 0

    from web.publish_blog_articles import generate_price_chart_image

    candidates = [a for a in published if a.get("id") and a["id"] in featured_ids]
    candidates.sort(key=lambda a: a.get("dealAmount", 0), reverse=True)

    posted = 0
    for article in candidates[:top_n]:
        is_sell = "売り" in (article.get("tags") or "")
        text = build_tweet_text(article["title"], article["dealAmount"], is_sell, article["id"])

        media_id = None
        image_bytes = generate_price_chart_image(article["stockCode"], article["stockName"])
        if image_bytes:
            media_id = upload_media(image_bytes)

        if post_tweet(text, media_id):
            print(f"  🐦 X投稿: {article['title']}" + ("（チャート付き）" if media_id else ""))
            posted += 1
    return posted

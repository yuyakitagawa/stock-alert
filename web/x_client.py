"""
web/x_client.py
publish_blog_articles.py が投稿した新着記事のうち、金額規模が大きい上位数件だけを
X(Twitter)へ自動投稿する。新規アカウントで全件投稿するとスパム的に見えるため、
質を優先し件数を絞る（TWEETS_PER_RUN）。

認証はOAuth 1.0a User Context（X API v2 POST /2/tweetsの投稿に必須）。
X Developer Portalで「Read and Write」権限のAppを作成し、以下4つを取得する:
  X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
いずれか未設定の場合は投稿をスキップする（他のステップには影響しない）。
"""
import os

import requests
from requests_oauthlib import OAuth1

SITE_URL = "https://kujira-watch.com"
API_URL = "https://api.x.com/2/tweets"

# 1回の実行(1日1回)あたりの投稿上限。新規アカウントでフォロワーが少ない段階では
# 全件投稿よりも金額規模の大きい注目度の高い開示に絞った方が質が保てる。
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
    文字数上限の計算には含めない。"""
    direction = "売却" if is_sell else "取得"
    body = f"{title}\n推定{direction}金額: {deal_amount_oku}億円"
    if len(body) > TWEET_BODY_MAX_CHARS:
        body = body[: TWEET_BODY_MAX_CHARS - 1] + "…"
    return f"{body}\n{SITE_URL}/articles/{article_id}\n#EDINET #大量保有報告書"


def post_tweet(text: str) -> bool:
    auth = _auth()
    if auth is None:
        return False
    try:
        resp = requests.post(API_URL, auth=auth, json={"text": text}, timeout=20)
        if resp.status_code not in (200, 201):
            print(f"  ⚠ X投稿失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  ⚠ X投稿例外: {e}")
        return False


def post_top_articles(published: list, top_n: int = TWEETS_PER_RUN) -> int:
    """publish_blog_articles.build_and_publish()が返すpublishedリストのうち、
    金額規模(dealAmount)が大きい順に上位top_n件をXへ投稿する。dry-run(id=None)の
    記事は対象外。X認証情報が未設定の場合は何もせず0を返す。"""
    if _auth() is None:
        print("[x_client] X_API_KEY等が未設定のため投稿をスキップします")
        return 0

    candidates = [a for a in published if a.get("id")]
    candidates.sort(key=lambda a: a.get("dealAmount", 0), reverse=True)

    posted = 0
    for article in candidates[:top_n]:
        is_sell = "売り" in (article.get("tags") or "")
        text = build_tweet_text(article["title"], article["dealAmount"], is_sell, article["id"])
        if post_tweet(text):
            print(f"  🐦 X投稿: {article['title']}")
            posted += 1
    return posted

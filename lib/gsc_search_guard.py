"""検索に出ている記事を削除対象から外すガード（記事を消す手動ツール共通）。

なぜ入れたか（2026-09-13）:
  8/18〜29の記事削除・/en除却で、GSCの過去3か月のクリック181件のうち38件（21%）を持っていたURLを
  自分たちで消していた。うち8/18の「低価値記事の削除」だけで15クリック分。同時期からGooglebotの巡回が
  1/4に落ち、9月の新規記事がほぼ「検出-インデックス未登録」になった。
  「基準未満」「PVゼロ」などサイト側の指標だけで消すと、Google検索で表示されている記事まで巻き込むので、
  削除の直前にGSCで表示回数を確かめ、表示が1回でもあった記事は残す。

GSCが取れないとき（鍵なし・権限なし・API障害）は None を返し、呼び出し側は削除を中止する（fail-closed）。
消した記事は戻せない（idが変わる）ので、確認できないまま消すより止まる方を選ぶ。
"""
import os
import re
import sys
import urllib.parse
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from lib import gcp_auth

SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
DEFAULT_SITE = "sc-domain:kujira-watch.com"
# GSCの画面・エクスポートの既定（過去3か月）に揃える。
LOOKBACK_DAYS = 90
MAX_ROWS = 25000

# 日本語 /articles/<id>、旧英語版 /en/articles/<id>、英語サブドメインの /articles/<id> のいずれも同じ記事id。
_ARTICLE_PATH = re.compile(r"^(?:/en)?/articles/([^/?#]+)/?$")


def article_id_from_url(url: str) -> "str | None":
    path = urllib.parse.urlparse(url).path
    m = _ARTICLE_PATH.match(path)
    return m.group(1) if m else None


def ids_with_impressions(rows: list) -> set:
    """GSCのpage次元の行から、表示回数が1回以上あった記事idを集める。"""
    ids = set()
    for r in rows:
        if (r.get("impressions") or 0) <= 0:
            continue
        aid = article_id_from_url((r.get("keys") or [""])[0])
        if aid:
            ids.add(aid)
    return ids


def searched_article_ids(days: int = LOOKBACK_DAYS) -> "set | None":
    """直近days日にGoogle検索で表示された記事id。取得できなければNone。"""
    site = os.getenv("GSC_SITE_URL", DEFAULT_SITE)
    try:
        token = gcp_auth.access_token(SCOPE)
        if not token:
            print("[gsc_search_guard] GCPの鍵が見つかりません")
            return None
        end = date.today()
        body = {"startDate": (end - timedelta(days=days)).isoformat(), "endDate": end.isoformat(),
                "dimensions": ["page"], "rowLimit": MAX_ROWS}
        resp = requests.post(
            f"https://searchconsole.googleapis.com/webmasters/v3/sites/"
            f"{urllib.parse.quote(site, safe='')}/searchAnalytics/query",
            headers={"Authorization": f"Bearer {token}"}, json=body, timeout=60)
        if resp.status_code != 200:
            print(f"[gsc_search_guard] GSC HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return ids_with_impressions(resp.json().get("rows") or [])
    except Exception as e:
        print(f"[gsc_search_guard] GSC取得失敗: {e}")
        return None


def split_protected(targets: list, protected: set) -> tuple:
    """(削除してよい記事, 検索に出ているので残す記事) に分ける。"""
    deletable = [a for a in targets if a["id"] not in protected]
    kept = [a for a in targets if a["id"] in protected]
    return deletable, kept


def filter_deletable(targets: list, days: int = LOOKBACK_DAYS) -> "list | None":
    """削除直前に呼ぶ。検索に出ている記事を除いた一覧を返す。GSCが取れなければNone（＝削除を中止）。"""
    protected = searched_article_ids(days)
    if protected is None:
        print("⛔ GSCで検索表示を確認できないため削除を中止します（検索に出ている記事を消さないため）")
        return None
    deletable, kept = split_protected(targets, protected)
    if kept:
        print(f"🛡 直近{days}日にGoogle検索で表示された記事 {len(kept)}件は削除しません")
        for a in kept[:10]:
            print(f"  {a['id']}: {a.get('stockName')}({a.get('stockCode')}) {a.get('title', '')[:40]}")
        if len(kept) > 10:
            print(f"  … 他 {len(kept) - 10}件")
    return deletable

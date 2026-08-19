"""
web/x_metrics.py

Xに投稿した各ポストのインプレッション等を日次で取得し、Supabaseの`x_posts`（最新値）と
`x_post_metrics`（日次スナップショット）に保存する。あわせてアカウントのフォロワー数を
`x_followers`に日次で記録する。

フォロワーは「インプレッション→プロフィールクリック→フォロー」の順にしか増えないので、
投稿単位では `user_profile_clicks`（プロフィールクリック）を、アカウント単位では日次の
フォロワー数を測る。この2つが無いと「どの投稿の型がフォロワーを連れてきたか」を判定できない。

投稿フォーマットを変えても「どの型が伸びたか」を測る手段が無い、というのが
docs/x_post_improvement_1000.md 施策1の指摘。x_clientがtweet_idを記録するようになった
ので、ここで数字を埋めて型ごとの比較ができるようにする。

実行:
  python3 web/x_metrics.py            # 直近30日の投稿のメトリクスとフォロワー数を更新
  python3 web/x_metrics.py --report   # 更新に加えてフォロワー推移と種別ごとの平均を表示
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

sys.path.insert(0, os.path.expanduser("~/stock-alert"))

import requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from lib import supabase_client as sb  # noqa: E402
from web.x_client import _auth  # noqa: E402

LOOKUP_URL = "https://api.x.com/2/tweets"
ME_URL = "https://api.x.com/2/users/me"
# X APIの1リクエストあたりのid上限
BATCH = 100
# non_public_metrics（インプレッション等）はAPI仕様上、投稿から30日以内の自分の投稿のみ取得できる。
METRICS_WINDOW_DAYS = 30


def fetch_target_tweet_ids() -> list:
    # ISO文字列の "+00:00" はクエリ文字列上でそのまま渡すと空白に解釈されて
    # PostgRESTが400を返すため、値をURLエンコードしてから渡す。
    since = quote((datetime.now(timezone.utc) - timedelta(days=METRICS_WINDOW_DAYS)).isoformat(), safe="")
    rows = sb.select("x_posts", f"posted_at=gte.{since}&select=tweet_id&order=posted_at.desc")
    return [r["tweet_id"] for r in rows if r.get("tweet_id")]


def fetch_metrics(tweet_ids: list) -> dict:
    """{tweet_id: {impressions, likes, ...}} を返す。non_public_metricsが取れない
    プラン・権限の場合はpublic_metricsだけで再取得する（impressionsはNoneになる）。"""
    auth = _auth()
    if auth is None:
        print("[x_metrics] X_API_KEY等が未設定のため取得をスキップします")
        return {}

    out: dict = {}
    for i in range(0, len(tweet_ids), BATCH):
        chunk = tweet_ids[i:i + BATCH]
        for fields in ("public_metrics,non_public_metrics", "public_metrics"):
            try:
                resp = requests.get(
                    LOOKUP_URL, auth=auth, timeout=30,
                    params={"ids": ",".join(chunk), "tweet.fields": fields},
                )
            except Exception as e:
                print(f"[x_metrics] 取得例外: {e}")
                break
            if resp.status_code == 200:
                for t in resp.json().get("data", []) or []:
                    pub = t.get("public_metrics") or {}
                    non_pub = t.get("non_public_metrics") or {}
                    out[t["id"]] = {
                        "impressions": non_pub.get("impression_count") or pub.get("impression_count"),
                        "likes": pub.get("like_count"),
                        "reposts": pub.get("retweet_count"),
                        "replies": pub.get("reply_count"),
                        "quotes": pub.get("quote_count"),
                        "bookmarks": pub.get("bookmark_count"),
                        # フォロワー獲得の中間指標。プロフィールを開かせない投稿はフォローに繋がらない。
                        "url_link_clicks": non_pub.get("url_link_clicks"),
                        "user_profile_clicks": non_pub.get("user_profile_clicks"),
                    }
                break
            print(f"[x_metrics] 取得失敗 HTTP {resp.status_code} ({fields}): {resp.text[:200]}")
    return out


def fetch_followers() -> dict:
    """アカウントのフォロワー数等を取得する。取得できなければ空dict。"""
    auth = _auth()
    if auth is None:
        return {}
    try:
        resp = requests.get(ME_URL, auth=auth, timeout=30,
                            params={"user.fields": "public_metrics"})
    except Exception as e:
        print(f"[x_metrics] フォロワー取得例外: {e}")
        return {}
    if resp.status_code != 200:
        print(f"[x_metrics] フォロワー取得失敗 HTTP {resp.status_code}: {resp.text[:200]}")
        return {}
    pm = ((resp.json().get("data") or {}).get("public_metrics") or {})
    if not pm:
        return {}
    return {
        "followers": pm.get("followers_count"),
        "following": pm.get("following_count"),
        "tweets": pm.get("tweet_count"),
        "listed": pm.get("listed_count"),
    }


def save_followers(snapshot: dict) -> None:
    if not snapshot:
        return
    sb.upsert("x_followers", [{"measured_on": date.today().isoformat(), **snapshot}],
              on_conflict="measured_on")
    print(f"[x_metrics] フォロワー {snapshot.get('followers')}人を記録しました")


def save(metrics: dict) -> int:
    if not metrics:
        return 0
    today = date.today().isoformat()
    now = datetime.now(timezone.utc).isoformat()
    sb.upsert("x_posts", [{"tweet_id": tid, **m, "metrics_updated_at": now}
                          for tid, m in metrics.items()], on_conflict="tweet_id")
    sb.upsert("x_post_metrics", [{"tweet_id": tid, "measured_on": today, **m}
                                 for tid, m in metrics.items()],
              on_conflict="tweet_id,measured_on")
    return len(metrics)


def follower_report() -> None:
    """フォロワー数の推移を表示する。増減の判断はこの表だけを見る。"""
    rows = sb.select("x_followers", "select=measured_on,followers&order=measured_on.desc&limit=40")
    rows = [r for r in rows if r.get("followers") is not None]
    if not rows:
        print("\n[x_metrics] フォロワーの記録がまだありません")
        return
    latest = rows[0]
    print(f"\nフォロワー {latest['followers']}人（{latest['measured_on']}時点）")
    by_day = {r["measured_on"]: r["followers"] for r in rows}
    for label, days in (("7日前比", 7), ("30日前比", 30)):
        past = date.fromisoformat(latest["measured_on"]) - timedelta(days=days)
        # 記録が飛んでいる日があるので、その日以前で最も新しい記録と比べる
        earlier = [d for d in sorted(by_day) if date.fromisoformat(d) <= past]
        if not earlier:
            print(f"  {label}: 記録がまだ足りません")
            continue
        base = by_day[earlier[-1]]
        print(f"  {label}: {latest['followers'] - base:+d}人（{earlier[-1]} {base}人）")


def report() -> None:
    """種別ごとの平均を表示する。フォーマット変更の効果はこの表で判断する。"""
    follower_report()
    rows = sb.select("x_posts", "select=kind,variant,has_media,impressions,likes,reposts,bookmarks,"
                                "url_link_clicks,user_profile_clicks")
    groups = defaultdict(list)
    for r in rows:
        if r.get("impressions") is not None or r.get("likes") is not None:
            groups[(r.get("kind"), r.get("variant"))].append(r)
    if not groups:
        print("[x_metrics] メトリクス付きの投稿がまだありません")
        return
    print(f"\n{'種別':<22}{'variant':<16}{'件数':>5}{'平均imp':>10}{'平均いいね':>12}{'平均保存':>10}"
          f"{'プロフclick':>12}{'プロフ率%':>10}")
    for (kind, variant), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        def avg(key):
            vals = [i[key] for i in items if i.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0
        imp = avg("impressions")
        prof = avg("user_profile_clicks")
        print(f"{kind or '-':<22}{variant or '-':<16}{len(items):>5}"
              f"{imp:>10.0f}{avg('likes'):>12.1f}{avg('bookmarks'):>10.1f}"
              f"{prof:>12.1f}{(prof / imp * 100 if imp else 0):>10.2f}")


def run(with_report: bool = False) -> int:
    if not sb.is_configured():
        print("[x_metrics] Supabase未設定のため実行しません")
        return 1
    save_followers(fetch_followers())
    ids = fetch_target_tweet_ids()
    if not ids:
        print("[x_metrics] 対象の投稿がありません")
        if with_report:
            report()
        return 0
    saved = save(fetch_metrics(ids))
    print(f"[x_metrics] {saved}/{len(ids)}件のメトリクスを更新しました")
    if with_report:
        report()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true", help="フォロワー推移と種別ごとの平均も表示する")
    args = p.parse_args()
    sys.exit(run(with_report=args.report))


if __name__ == "__main__":
    main()

"""
tools/x_benchmark.py

フォローした同ジャンルのアカウント（大量保有報告書・適時開示・日本株）の投稿を集めて、
「何が伸びているのか」を実データで見るためのツール。手動実行専用。

こちらのアカウントは直近30日33投稿で平均インプレッション0〜3しか無く、自分の投稿だけを
見ていても型の良し悪しを判定できない（母数がゼロなので差が出ない）。同じ題材を扱っていて
既に読者がいるアカウントの数字を借りて、伸びる型を先に決める。

出すもの:
  ①アカウント別の中央値（エンゲージメント・インプレッション・文字数・画像率・URL率）
  ②全アカウント横断のエンゲージメント上位投稿（本文つき。これが一番の材料）
  ③条件別の中央値: 画像の有無 / URLの有無 / 文字数帯 / ハッシュタグ数 / 投稿時間帯(JST)

平均ではなく**中央値**で比較する。1本のバズが平均を持ち上げると「たまたま伸びた1本の型」を
全体の傾向と誤認するため。

認証は web/x_client.py と同じ OAuth 1.0a（読み取りのみ）。

使い方:
  python3 tools/x_benchmark.py                       # 既定の同ジャンル9アカウント
  python3 tools/x_benchmark.py --usernames a,b,c
  python3 tools/x_benchmark.py --per-account 50 --top 30
"""
import argparse
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

from web.x_client import _auth

USERS_BY_URL = "https://api.x.com/2/users/by"
TIMELINE_URL = "https://api.x.com/2/users/{uid}/tweets"

# 既定の比較対象。2026-08-30にフォローした22件のうち、こちらと**同じ題材**（大量保有報告書・
# 適時開示・アクティビスト）を扱っていて、型を真似る意味があるものだけに絞っている。
# 週刊エコノミストのような一般メディアは題材も体力も違うので入れない。
DEFAULT_ACCOUNTS = [
    "activistw_app",    # アクティビストウォッチャー（大量保有追跡アプリ）
    "tairyo_hoyu",      # コバンザメ投資アラート（EDINET自動解析）
    "HoldingNavi",      # 大量保有報告書Navi
    "stpedia",          # M&A Online 大量保有速報/TOB速報
    "eyeR365X",         # あいあーる（適時開示・大量保有）
    "avesan_27",        # 決算をAIで速報
    "miraclemasui",     # 適時開示・市況速報
    "kiokunir",         # キオ君（IR資料要約。リーチ最大）
    "26ooo",            # じろ（アクティビスト・エンゲージメント・資本コスト）
]

JST = timezone(timedelta(hours=9))


def _get(url: str, params: dict = None) -> "tuple[dict | None, str]":
    auth = _auth()
    if auth is None:
        return None, "X_API_KEY等が未設定"
    try:
        resp = requests.get(url, auth=auth, params=params or {}, timeout=30)
    except Exception as e:
        return None, f"通信例外: {e}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json(), ""


def resolve(usernames: list) -> "tuple[list, str]":
    data, err = _get(USERS_BY_URL, {
        "usernames": ",".join(usernames),
        "user.fields": "public_metrics",
    })
    if err:
        return [], err
    return (data or {}).get("data", []), ""


def fetch_timeline(uid: str, limit: int) -> "tuple[list, str]":
    """本人の投稿だけを取る（リツイートとリプライは型の参考にならないので除く）。"""
    data, err = _get(TIMELINE_URL.format(uid=uid), {
        "max_results": min(limit, 100),
        "exclude": "retweets,replies",
        "tweet.fields": "public_metrics,created_at,entities,attachments",
    })
    if err:
        return [], err
    return (data or {}).get("data", []), ""


def summarize_tweet(tw: dict, username: str) -> dict:
    """1投稿を、型の比較に使える特徴量へ落とす。"""
    pm = tw.get("public_metrics", {}) or {}
    ent = tw.get("entities", {}) or {}
    text = tw.get("text", "") or ""
    created = tw.get("created_at", "")
    hour = None
    if created:
        try:
            hour = datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(JST).hour
        except ValueError:
            hour = None
    engagement = (pm.get("like_count", 0) + pm.get("retweet_count", 0)
                  + pm.get("reply_count", 0) + pm.get("quote_count", 0))
    return {
        "username": username,
        "text": text,
        "first_line": text.split("\n")[0][:60],
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "hashtags": len(ent.get("hashtags", []) or []),
        "has_url": bool(ent.get("urls", []) or []),
        "has_media": bool((tw.get("attachments", {}) or {}).get("media_keys")),
        "hour_jst": hour,
        "impressions": pm.get("impression_count", 0),
        "likes": pm.get("like_count", 0),
        "engagement": engagement,
        "created_at": created,
    }


def _median(values: list) -> float:
    return statistics.median(values) if values else 0.0


def _bucket_report(title: str, groups: dict) -> None:
    """条件別の中央値を出す。件数が少なすぎるグループは誤読の元なので伏せる。"""
    print(f"\n■ {title}")
    print(f"  {'条件':<16}{'件数':>5}{'中央値eng':>10}{'中央値imp':>11}")
    for key in sorted(groups, key=lambda k: -_median([t["engagement"] for t in groups[k]])):
        rows = groups[key]
        if len(rows) < 3:
            continue
        print(f"  {key:<16}{len(rows):>5}{_median([t['engagement'] for t in rows]):>10.1f}"
              f"{_median([t['impressions'] for t in rows]):>11.0f}")


def report(tweets: list, top: int) -> None:
    if not tweets:
        print("[x_benchmark] 対象の投稿が0件でした")
        return

    print(f"\n■ アカウント別（{len(tweets)}投稿）")
    print(f"  {'account':<18}{'投稿':>4}{'中央値eng':>10}{'中央値imp':>11}{'中央値字':>9}{'画像率':>7}{'URL率':>7}")
    by_user = {}
    for t in tweets:
        by_user.setdefault(t["username"], []).append(t)
    for user in sorted(by_user, key=lambda u: -_median([t["engagement"] for t in by_user[u]])):
        rows = by_user[user]
        print(f"  @{user:<17}{len(rows):>4}{_median([t['engagement'] for t in rows]):>10.1f}"
              f"{_median([t['impressions'] for t in rows]):>11.0f}"
              f"{_median([t['chars'] for t in rows]):>9.0f}"
              f"{sum(t['has_media'] for t in rows)/len(rows):>6.0%}"
              f"{sum(t['has_url'] for t in rows)/len(rows):>7.0%}")

    _bucket_report("画像の有無", _group(tweets, lambda t: "画像あり" if t["has_media"] else "画像なし"))
    _bucket_report("URLの有無", _group(tweets, lambda t: "URLあり" if t["has_url"] else "URLなし"))
    _bucket_report("文字数帯", _group(tweets, _chars_bucket))
    _bucket_report("ハッシュタグ数", _group(tweets, lambda t: f"{min(t['hashtags'], 3)}個" +
                                            ("以上" if t["hashtags"] >= 3 else "")))
    _bucket_report("投稿時間帯(JST)", _group(tweets, _hour_bucket))

    print(f"\n■ エンゲージメント上位{top}投稿（型の材料。本文は改行を / に置換）")
    for t in sorted(tweets, key=lambda x: -x["engagement"])[:top]:
        body = t["text"].replace("\n", " / ")[:200]
        print(f"\n  [eng {t['engagement']:>5} / imp {t['impressions']:>7} / {t['chars']:>3}字 / "
              f"{'画像' if t['has_media'] else '文のみ'} / {'URL' if t['has_url'] else 'URLなし'} / "
              f"{t['hour_jst']}時] @{t['username']}")
        print(f"  {body}")


def _group(tweets: list, keyfn) -> dict:
    groups = {}
    for t in tweets:
        groups.setdefault(keyfn(t), []).append(t)
    return groups


def _chars_bucket(t: dict) -> str:
    c = t["chars"]
    if c < 60:
        return "〜59字"
    if c < 100:
        return "60〜99字"
    if c < 140:
        return "100〜139字"
    return "140字〜"


def _hour_bucket(t: dict) -> str:
    h = t["hour_jst"]
    if h is None:
        return "不明"
    if h < 7:
        return "深夜0-6時"
    if h < 12:
        return "朝7-11時"
    if h < 16:
        return "昼12-15時"
    if h < 20:
        return "夕16-19時"
    return "夜20-23時"


def main() -> int:
    p = argparse.ArgumentParser(description="同ジャンルのXアカウントの投稿を集めて型を比較する")
    p.add_argument("--usernames", default=",".join(DEFAULT_ACCOUNTS), help="カンマ区切り")
    p.add_argument("--per-account", type=int, default=100, help="1アカウントあたりの取得件数(5-100)")
    p.add_argument("--top", type=int, default=25, help="本文を出す上位投稿の件数")
    args = p.parse_args()

    names = [n.strip().lstrip("@") for n in args.usernames.split(",") if n.strip()]
    users, err = resolve(names)
    if err:
        print(f"[x_benchmark] ユーザー解決に失敗 → {err}")
        return 1

    tweets = []
    for u in users:
        rows, err = fetch_timeline(u["id"], args.per_account)
        if err:
            print(f"[x_benchmark] @{u['username']} の取得に失敗 → {err}")
            continue
        tweets.extend(summarize_tweet(tw, u["username"]) for tw in rows)
        print(f"[x_benchmark] @{u['username']} {len(rows)}件")

    if not tweets:
        print("[x_benchmark] 1件も取得できませんでした")
        return 1
    report(tweets, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
tools/x_follow.py

Xアカウントのフォローを扱う。2026-08-30時点でアカウントは
フォロワー0・**フォロー0**・直近30日33投稿の平均インプレッション0〜3で、
投稿フォーマットをいくら変えても届く先が無い（docs/x_post_improvement_1000.md の
10施策はこの状態では効果検証すらできない）。ボトルネックは配信先が0であること
なので、まず「実際に日本株の話をしている生きたアカウント」をフォローして
タイムラインを作る。

2つのモードがある。
  discover : 直近7日の投稿を検索して、日本株・大量保有まわりで実際に発言している
             アカウントを抽出する（読み取りのみ。フォローはしない）。
  follow   : 明示的に渡したユーザー名だけをフォローする（--execute を付けたときだけ）。

**自動フォローの常時運用はしない。** Xの自動化ポリシーは大量・無差別な自動フォローと
フォロー/リムーブの反復を禁止しており、凍結リスクがある。このツールは手動実行
（workflow_dispatch）専用で、フォロー対象は毎回オーナーが承認したユーザー名を
明示的に渡す設計にしている。cronに載せてはいけない。

認証は web/x_client.py と同じ OAuth 1.0a User Context（X_API_KEY / X_API_KEY_SECRET /
X_ACCESS_TOKEN / X_ACCESS_TOKEN_SECRET）。フォローには書き込み権限が要る。

使い方:
  python3 tools/x_follow.py discover                      # 候補を探して一覧表示
  python3 tools/x_follow.py discover --min-followers 500
  python3 tools/x_follow.py follow --usernames a,b,c      # dry-run（何もしない）
  python3 tools/x_follow.py follow --usernames a,b,c --execute
"""
import argparse
import json
import sys
import time

import requests

from web.x_client import _auth

ME_URL = "https://api.x.com/2/users/me"
USERS_BY_URL = "https://api.x.com/2/users/by"
SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
FOLLOWING_URL = "https://api.x.com/2/users/{me_id}/following"

# 候補を探すときの検索クエリ。「大量保有報告書の話をしている人」を軸に、
# 隣接する日本株の話題まで広げる。リツイートと自分の投稿は除く。
DISCOVER_QUERIES = [
    "大量保有報告書 -is:retweet -is:reply lang:ja",
    "(アクティビスト OR 物言う株主) 株 -is:retweet -is:reply lang:ja",
    "(自社株買い OR 株主還元) -is:retweet -is:reply lang:ja",
    "(日本株 OR 個別株) 決算 -is:retweet -is:reply lang:ja",
]

# フォロー候補として妥当なフォロワー数の範囲。下限未満は届く先が無く、
# 上限超えは大手メディアや著名人でこちらの投稿を読む可能性が低い
# （※大手メディアはリプライ先としては有効なので、別途手で足す前提）。
DEFAULT_MIN_FOLLOWERS = 300
DEFAULT_MAX_FOLLOWERS = 200_000

# 1回の実行でフォローする上限。無差別な大量フォローと見なされないための安全弁。
MAX_FOLLOWS_PER_RUN = 50
# フォロー間隔（秒）。連打しない。
FOLLOW_INTERVAL_SEC = 3


def _get(url: str, params: dict = None) -> "tuple[dict | None, str]":
    """GETして (JSON, エラー文) を返す。成功時はエラー文が空。"""
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


def me_id() -> "tuple[str, str]":
    """自分のuser_idとusernameを返す。取れなければ ('','')。"""
    data, err = _get(ME_URL)
    if err:
        print(f"[x_follow] 自分のユーザー情報が取れません → {err}")
        return "", ""
    d = (data or {}).get("data", {})
    return d.get("id", ""), d.get("username", "")


def resolve_usernames(usernames: list) -> "tuple[list, str]":
    """ユーザー名のリストをユーザー情報に解決する。存在しない名前は落ちる。"""
    out = []
    # usernamesは1リクエストあたり100件まで
    for i in range(0, len(usernames), 100):
        chunk = usernames[i:i + 100]
        data, err = _get(USERS_BY_URL, {
            "usernames": ",".join(chunk),
            "user.fields": "public_metrics,description,verified,protected",
        })
        if err:
            return out, err
        out.extend((data or {}).get("data", []))
        for e in (data or {}).get("errors", []) or []:
            print(f"[x_follow] 解決できないユーザー名: {e.get('value')} ({e.get('detail','')[:80]})")
    return out, ""


def discover(min_followers: int, max_followers: int, per_query: int) -> list:
    """検索で候補アカウントを集める。ヒット数の多い順に返す。"""
    authors = {}
    for q in DISCOVER_QUERIES:
        data, err = _get(SEARCH_URL, {
            "query": q,
            "max_results": per_query,
            "tweet.fields": "created_at",
            "expansions": "author_id",
            "user.fields": "public_metrics,description,verified,protected",
        })
        if err:
            print(f"[x_follow] 検索失敗 「{q}」 → {err}")
            if "403" in err or "453" in err:
                print("[x_follow] 検索エンドポイントが今のAPIプランで使えない可能性があります。"
                      "その場合はdiscoverを諦め、followにユーザー名を手で渡してください。")
                return []
            continue
        users = {u["id"]: u for u in (data or {}).get("includes", {}).get("users", [])}
        for tw in (data or {}).get("data", []):
            u = users.get(tw.get("author_id"))
            if not u:
                continue
            rec = authors.setdefault(u["id"], {**u, "hits": 0, "queries": set()})
            rec["hits"] += 1
            rec["queries"].add(q.split()[0])
        print(f"[x_follow] 「{q}」 {len((data or {}).get('data', []))}件")

    return build_candidates(authors.values(), min_followers, max_followers)


def build_candidates(authors, min_followers: int, max_followers: int) -> list:
    """検索で集めたアカウントを候補に整形する。鍵アカとフォロワー数が範囲外のものを落とし、
    ヒット数（＝そのテーマで実際に何回発言しているか）の多い順に並べる。"""
    cands = []
    for rec in authors:
        pm = rec.get("public_metrics", {}) or {}
        followers = pm.get("followers_count", 0)
        if rec.get("protected"):
            continue
        if followers < min_followers or followers > max_followers:
            continue
        cands.append({
            "username": rec.get("username", ""),
            "name": rec.get("name", ""),
            "followers": followers,
            "tweets": pm.get("tweet_count", 0),
            "hits": rec.get("hits", 0),
            "queries": sorted(rec.get("queries", [])),
            "description": (rec.get("description", "") or "").replace("\n", " ")[:80],
        })
    cands.sort(key=lambda c: (-c["hits"], -c["followers"]))
    return cands


def follow(usernames: list, execute: bool) -> int:
    """ユーザー名のリストをフォローする。executeがFalseなら何もせず内容だけ出す。"""
    if len(usernames) > MAX_FOLLOWS_PER_RUN:
        print(f"[x_follow] 1回の上限{MAX_FOLLOWS_PER_RUN}件を超えています（{len(usernames)}件）。分けて実行してください")
        return 1
    users, err = resolve_usernames(usernames)
    if err:
        print(f"[x_follow] ユーザー解決に失敗 → {err}")
        return 1
    if not users:
        print("[x_follow] 解決できたユーザーが0件です")
        return 1

    if not execute:
        print(f"\n[x_follow] dry-run: 以下{len(users)}件をフォローします（--execute で実行）")
        for u in users:
            pm = u.get("public_metrics", {}) or {}
            print(f"  @{u['username']:<20} {pm.get('followers_count',0):>8}フォロワー  {u.get('name','')}")
        return 0

    my_id, my_name = me_id()
    if not my_id:
        return 1
    auth = _auth()
    ok = ng = 0
    for u in users:
        try:
            resp = requests.post(
                FOLLOWING_URL.format(me_id=my_id),
                auth=auth, json={"target_user_id": u["id"]}, timeout=30,
            )
        except Exception as e:
            print(f"[x_follow] @{u['username']} 通信例外: {e}")
            ng += 1
            continue
        if resp.status_code == 200:
            print(f"[x_follow] @{u['username']} フォローしました")
            ok += 1
        else:
            print(f"[x_follow] @{u['username']} 失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            ng += 1
            # プランや権限で弾かれている場合は続けても全部同じ結果になる
            if resp.status_code in (401, 403):
                print("[x_follow] 権限またはAPIプランの問題です。以降を中止します")
                break
        time.sleep(FOLLOW_INTERVAL_SEC)
    print(f"[x_follow] 完了: 成功{ok}件 / 失敗{ng}件（@{my_name}）")
    return 0 if ng == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Xのフォロー候補抽出とフォロー実行")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="候補アカウントを検索する（読み取りのみ）")
    d.add_argument("--min-followers", type=int, default=DEFAULT_MIN_FOLLOWERS)
    d.add_argument("--max-followers", type=int, default=DEFAULT_MAX_FOLLOWERS)
    d.add_argument("--per-query", type=int, default=100, help="1クエリあたりの取得件数(10-100)")
    d.add_argument("--limit", type=int, default=60, help="表示する候補の上限")

    f = sub.add_parser("follow", help="指定したユーザー名をフォローする")
    f.add_argument("--usernames", required=True, help="カンマ区切り。@は付けても付けなくてもよい")
    f.add_argument("--execute", action="store_true", help="実際にフォローする（無いとdry-run）")

    args = p.parse_args()

    if args.cmd == "discover":
        cands = discover(args.min_followers, args.max_followers, args.per_query)
        if not cands:
            print("[x_follow] 候補が0件でした")
            return 1
        print(f"\n[x_follow] 候補{len(cands)}件（上位{min(args.limit,len(cands))}件を表示）")
        print(f"{'username':<22}{'followers':>9}  {'hits':>4}  name / description")
        for c in cands[:args.limit]:
            print(f"@{c['username']:<21}{c['followers']:>9}  {c['hits']:>4}  {c['name']} | {c['description']}")
        print("\n--- JSON ---")
        print(json.dumps(cands[:args.limit], ensure_ascii=False))
        return 0

    names = [n.strip().lstrip("@") for n in args.usernames.split(",") if n.strip()]
    if not names:
        print("[x_follow] --usernames が空です")
        return 1
    return follow(names, args.execute)


if __name__ == "__main__":
    sys.exit(main())

"""
tools/traffic_report.py

`blog_crawler_log` の "Browser" 判定トラフィックから機械アクセスを除外し、
残りを「人が見た可能性のあるアクセス」として集計する（手動実行専用）。

なぜ必要か:
  proxy.ts の classifyVisitor() は「既知botのUAでなく、ブラウザのUAである」ものを
  すべて "Browser" として記録する。ヘッドレスブラウザを使うスクレイパーはこの条件を
  満たすため、実測では直近14日111,165PVのうち上位5IPだけで31.8%を占めていた。
  この状態のPVを見ても施策の前後比較はできない。

除外は4段階で行う（実データ2026-08-28の内訳を例に付す。Browser 3,573PVの内訳）:
  1. self          : 自サイトからの自己リクエスト（::1 / 127.0.0.1）。187PV
  2. bot_ua        : UAに bot/crawler/spider/externalagent/+URL を含むもの。1,384PV
                     crawlers.ts の BOT_PATTERNS に載っていない新顔クローラーはここで
                     拾う（実例: meta-externalagent 1,031PV / Amazonbot 337PV）。
                     気づいたら crawlers.ts 側にも追加してログの時点で分類する。
  3. heavy_ip      : 1IPあたりのPVが閾値超（--max-pv-per-ip）。1,421PV
  4. cookieless_ua : 同一UAでPVが十分あるのに visitor_id がPVとほぼ1:1のグループ。
                     クッキーを保持しない＝1リクエストごとに別人扱いになる機械。
                     実例: 汎用Chrome/148がOVHの62IPから327PV・visitor_id 312個。

判別に使えなかった指標（実データで確認済み、同じ検証を繰り返さないため記録する）:
  - クッキー(visitor_id)を「持っているか」: 再訪ありの1,268人が93,435PV＝1人74PV。
    クッキーを保持するクローラーが居るため、有無だけでは人間の判別にならない
    （上の cookieless_ua は「有無」ではなく「PVに対する発行数の比」で見ている）。
  - /api/counter の有無（JS実行）: JS実行ありの1,226人が1人75PV。ヘッドレスブラウザは
    JSを実行するため同じく判別にならない。

実行:
  python3 tools/traffic_report.py              # 直近14日
  python3 tools/traffic_report.py --days 1 --max-pv-per-ip 30
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402

# 1IPあたりこのPV数を超えたら、そのIPのアクセスは全部まとめて機械とみなす。
DEFAULT_MAX_PV_PER_IP = 100
DEFAULT_DAYS = 14
# 自サイト（SSR/ヘルスチェック）からの自己リクエスト。
SELF_IPS = {"::1", "127.0.0.1"}
# UAに自分がクローラーだと書いてあるパターン。crawlers.ts の BOT_PATTERNS は
# 個別名を列挙するので新顔を取りこぼす。こちらは名前を知らなくても拾える。
BOT_UA_RE = re.compile(r"bot|crawler|spider|externalagent|externalfetcher|\+http", re.I)
# 上のどれかを含むトークン全体（例: meta-externalagent/1.1, Amazonbot/0.1）。
BOT_NAME_RE = re.compile(
    r"[A-Za-z0-9_.-]*(?:bot|crawler|spider|externalagent|externalfetcher)[A-Za-z0-9_./-]*",
    re.I)
# cookieless_ua と判定する最小PV。少数のUAグループは「全員が1ページで離脱した実在の
# 人間」でも visitor_id とPVが1:1になるため、母数が小さいうちは機械と決めつけない。
DEFAULT_MIN_UA_PV = 20
# PVに対する visitor_id 数の比がこれ以上なら、クッキーを保持していない＝機械とみなす。
COOKIELESS_RATIO = 0.9
BUCKETS = ("self", "bot_ua", "heavy_ip", "cookieless_ua", "human")
JST = timezone(timedelta(hours=9))


def fetch_rows(days: int) -> list[dict]:
    """直近days日の "Browser" 判定アクセスを取得する。"""
    since = quote((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), safe="")
    return sb.select(
        "blog_crawler_log",
        f"bot_name=eq.Browser&occurred_at=gte.{since}"
        "&select=occurred_at,path,ip_address,user_agent,visitor_id&order=occurred_at.desc",
    )


def heavy_ips(rows: list[dict], max_pv: int) -> set:
    """PVが閾値を超えたIPの集合。ip_addressがNULLの行は1つのIPとしてまとめて数える
    （SQLのNOT INにNULLを混ぜると全件が消えるのと同じ穴を、集計側でも塞いでおく）。"""
    counts = Counter(r.get("ip_address") or "(null)" for r in rows)
    return {ip for ip, n in counts.items() if n > max_pv}


def is_bot_ua(user_agent) -> bool:
    """UAに自己申告のクローラー表記があるか。"""
    return bool(BOT_UA_RE.search(user_agent or ""))


def cookieless_uas(rows: list[dict], min_pv: int = DEFAULT_MIN_UA_PV) -> set:
    """visitor_id がPVとほぼ1:1のUAグループ＝クッキーを保持しない機械。
    visitor_id がNULLの行はそれぞれ別IDとして数える（保持していないのは同じ）。"""
    pv: Counter = Counter()
    vids: dict = defaultdict(set)
    nulls: Counter = Counter()
    for r in rows:
        ua = r.get("user_agent") or "(null)"
        pv[ua] += 1
        vid = r.get("visitor_id")
        if vid:
            vids[ua].add(vid)
        else:
            nulls[ua] += 1
    return {ua for ua, n in pv.items()
            if n >= min_pv and (len(vids[ua]) + nulls[ua]) / n >= COOKIELESS_RATIO}


def classify(rows: list[dict], max_pv: int, min_ua_pv: int = DEFAULT_MIN_UA_PV) -> dict:
    """行を BUCKETS の5分類に振り分ける。どの行も必ず1つの分類に入る。
    heavy_ip の判定は self / bot_ua を除いた残りで数える（機械のPVで底上げされたIPを
    共有する実在の人間まで巻き込まないため）。"""
    out: dict = {k: [] for k in BUCKETS}
    rest = []
    for r in rows:
        if (r.get("ip_address") or "(null)") in SELF_IPS:
            out["self"].append(r)
        elif is_bot_ua(r.get("user_agent")):
            out["bot_ua"].append(r)
        else:
            rest.append(r)

    heavy = heavy_ips(rest, max_pv)
    remainder = []
    for r in rest:
        if (r.get("ip_address") or "(null)") in heavy:
            out["heavy_ip"].append(r)
        else:
            remainder.append(r)

    cookieless = cookieless_uas(remainder, min_ua_pv)
    for r in remainder:
        key = "cookieless_ua" if (r.get("user_agent") or "(null)") in cookieless else "human"
        out[key].append(r)
    return out


def hour_histogram(rows: list[dict]) -> dict:
    """JSTの時間帯別PV。人間なら深夜が凹み、機械なら24時間平坦になる。"""
    hours: dict = defaultdict(int)
    for r in rows:
        raw = r.get("occurred_at")
        if not raw:
            continue
        # PostgRESTは "...+00:00" で返すが、末尾Zの表記も来うるので両方受ける
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        hours[ts.astimezone(JST).hour] += 1
    return dict(hours)


def article_rate(rows: list[dict]) -> float:
    """記事ページが全PVに占める割合(%)。入口が記事かトップページかを見る。"""
    if not rows:
        return 0.0
    articles = sum(1 for r in rows if (r.get("path") or "").startswith("/articles/"))
    return articles / len(rows) * 100


def repeat_visitors(rows: list[dict]) -> tuple[int, int, int]:
    """(2PV以上見た訪問者数, その訪問者のPV, 訪問者総数)。最も人間らしい層の規模。"""
    counts = Counter(r.get("visitor_id") for r in rows if r.get("visitor_id"))
    multi = {v: n for v, n in counts.items() if n > 1}
    return len(multi), sum(multi.values()), len(counts)


def top_paths(rows: list[dict], limit: int = 10) -> list:
    return Counter(r.get("path") or "-" for r in rows).most_common(limit)


def bot_ua_label(user_agent) -> str:
    """UAからクローラー名の部分だけを取り出す。自己申告はUAの末尾に付くことが多く、
    先頭70文字を出すとMozilla/Chromeの飾りだけが見えて名前が分からない。"""
    m = BOT_NAME_RE.search(user_agent or "")
    return m.group(0) if m else (user_agent or "(null)")[:70]


def top_bot_markers(rows: list[dict], limit: int = 5) -> list:
    """bot_ua で除外したクローラー名の内訳。crawlers.ts に追加すべき新顔を見つけるため。"""
    return Counter(bot_ua_label(r.get("user_agent")) for r in rows).most_common(limit)


def report(days: int, max_pv: int, min_ua_pv: int = DEFAULT_MIN_UA_PV) -> int:
    if not sb.is_configured():
        print("[traffic] Supabase未設定のため実行しません")
        return 1
    rows = fetch_rows(days)
    if not rows:
        print(f"[traffic] 直近{days}日のBrowser判定アクセスがありません")
        return 0

    buckets = classify(rows, max_pv, min_ua_pv)
    human = buckets["human"]
    human_ips = {r.get("ip_address") or "(null)" for r in human}
    labels = {
        "self": "自サイト（::1等）",
        "bot_ua": "UAがbot自己申告",
        "heavy_ip": f"1IPで{max_pv}PV超",
        "cookieless_ua": "クッキー不保持UA",
    }

    print(f"\n直近{days}日の Browser 判定アクセス: {len(rows):,}PV")
    for key, label in labels.items():
        n = len(buckets[key])
        ips = len({r.get("ip_address") or "(null)" for r in buckets[key]})
        print(f"  除外 {label:<18}: {ips:>4}IP / {n:>6,}PV ({n / len(rows) * 100:4.1f}%)")
    print(f"  残り（人の可能性）      : {len(human_ips):>4}IP / {len(human):>6,}PV "
          f"({len(human) / max(len(human_ips), 1):.1f}PV/IP)")

    if buckets["bot_ua"]:
        print("\n  UA自己申告で除外した内訳（crawlers.tsに未登録なら追加する）")
        for ua, n in top_bot_markers(buckets["bot_ua"]):
            print(f"    {n:>6}  {ua}")

    if not human:
        return 0

    multi, multi_pv, total_v = repeat_visitors(human)
    print(f"\n  残りの記事ページ率      : {article_rate(human):.1f}%")
    print(f"  残りの2PV以上の訪問者   : {multi}人 / {multi_pv}PV（訪問者総数{total_v}人）")

    hours = hour_histogram(human)
    if hours:
        lo = min(hours.values())
        hi = max(hours.values())
        print(f"\n  JST時間帯（最小{lo} 最大{hi} 比{hi / max(lo, 1):.1f}倍"
              f"／人なら深夜が凹む）")
        print("  " + " ".join(f"{h}:{hours.get(h, 0)}" for h in range(24)))

    print("\n  残りの人気path")
    for path, n in top_paths(human):
        print(f"    {n:>6}  {path}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--max-pv-per-ip", type=int, default=DEFAULT_MAX_PV_PER_IP,
                   help="1IPあたりこのPVを超えたら機械とみなす")
    p.add_argument("--min-ua-pv", type=int, default=DEFAULT_MIN_UA_PV,
                   help="クッキー不保持UAの判定に必要な最小PV")
    a = p.parse_args()
    sys.exit(report(a.days, a.max_pv_per_ip, a.min_ua_pv))


if __name__ == "__main__":
    main()

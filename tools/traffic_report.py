"""
tools/traffic_report.py

`blog_crawler_log` の "Browser" 判定トラフィックから大量アクセス元を除外し、
残りを「人が見た可能性のあるアクセス」として集計する（手動実行専用）。

なぜ必要か:
  proxy.ts の classifyVisitor() は「既知botのUAでなく、ブラウザのUAである」ものを
  すべて "Browser" として記録する。ヘッドレスブラウザを使うスクレイパーはこの条件を
  満たすため、実測では直近14日111,165PVのうち上位5IPだけで31.8%を占めていた。
  この状態のPVを見ても施策の前後比較はできない。

判別に使えなかった指標（実データで確認済み、同じ検証を繰り返さないため記録する）:
  - クッキー(visitor_id)の保持: 再訪ありの1,268人が93,435PV＝1人74PV。クッキーを保持する
    クローラーが居るため人間の判別にならない。
  - /api/counter の有無（JS実行）: JS実行ありの1,226人が1人75PV。ヘッドレスブラウザは
    JSを実行するため同じく判別にならない。

代わりに「1IPあたりの総PV」で切る。人が1つの回線から14日で数百PV見ることは稀で、
実測でも100PV超の167IPを除くと残りは1IPあたり13PVに落ち、深夜が凹み昼夜の波が出る
（3時 357PV / 12時 1,373PV）。閾値は --max-pv-per-ip で変えられる。

実行:
  python3 tools/traffic_report.py              # 直近14日
  python3 tools/traffic_report.py --days 30 --max-pv-per-ip 200
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402

# 1IPあたりこのPV数を超えたら、そのIPのアクセスは全部まとめて機械とみなす。
DEFAULT_MAX_PV_PER_IP = 100
DEFAULT_DAYS = 14
JST = timezone(timedelta(hours=9))


def fetch_rows(days: int) -> list[dict]:
    """直近days日の "Browser" 判定アクセスを取得する。"""
    since = quote((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), safe="")
    return sb.select(
        "blog_crawler_log",
        f"bot_name=eq.Browser&occurred_at=gte.{since}"
        "&select=occurred_at,path,ip_address&order=occurred_at.desc",
    )


def heavy_ips(rows: list[dict], max_pv: int) -> set:
    """PVが閾値を超えたIPの集合。ip_addressがNULLの行は1つのIPとしてまとめて数える
    （SQLのNOT INにNULLを混ぜると全件が消えるのと同じ穴を、集計側でも塞いでおく）。"""
    counts = Counter(r.get("ip_address") or "(null)" for r in rows)
    return {ip for ip, n in counts.items() if n > max_pv}


def split(rows: list[dict], max_pv: int) -> tuple[list[dict], list[dict]]:
    """(人が見た可能性のあるアクセス, 大量アクセス) に分ける。"""
    heavy = heavy_ips(rows, max_pv)
    human, machine = [], []
    for r in rows:
        (machine if (r.get("ip_address") or "(null)") in heavy else human).append(r)
    return human, machine


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


def top_paths(rows: list[dict], limit: int = 10) -> list:
    return Counter(r.get("path") or "-" for r in rows).most_common(limit)


def report(days: int, max_pv: int) -> int:
    if not sb.is_configured():
        print("[traffic] Supabase未設定のため実行しません")
        return 1
    rows = fetch_rows(days)
    if not rows:
        print(f"[traffic] 直近{days}日のBrowser判定アクセスがありません")
        return 0

    human, machine = split(rows, max_pv)
    heavy = heavy_ips(rows, max_pv)
    human_ips = {r.get("ip_address") or "(null)" for r in human}

    print(f"\n直近{days}日の Browser 判定アクセス: {len(rows):,}PV")
    print(f"  除外（1IPで{max_pv}PV超）: {len(heavy)}IP / {len(machine):,}PV "
          f"({len(machine) / len(rows) * 100:.1f}%)")
    print(f"  残り（人の可能性）      : {len(human_ips)}IP / {len(human):,}PV "
          f"({len(human) / max(len(human_ips), 1):.1f}PV/IP)")
    print(f"  残りの記事ページ率      : {article_rate(human):.1f}%"
          f"（除外分は{article_rate(machine):.1f}%）")

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
    a = p.parse_args()
    sys.exit(report(a.days, a.max_pv_per_ip))


if __name__ == "__main__":
    main()

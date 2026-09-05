"""
tools/en_crawl_report.py

英語版（en.kujira-watch.com）にクローラーが来ているかを見るレポート（手動実行専用）。

背景:
  英語版は /en 配下で運用していた時期（〜2026-08-29）に検索流入がほぼ無く廃止したが、
  「ディレクトリではなくサブドメインなら別サイトとして巡回されるのか」を確かめるため、
  microCMSに残っている英訳済み記事だけをサブドメインで配信し直した（2026-09-04）。
  巡回の実態は blog_crawler_log（kujira-watch/src/proxy.ts が記録）にしか無く、
  同テーブルの host 列で日英を区別する（host が NULL の行は列追加前＝日本語版）。

出すもの（すべて前期間との差分付き）:
  1. 英語版ホストのクローラー別ヒット数（Browser も1行として出す＝人の目安）
  2. 同じクローラーの日本語版ヒット数との比較（英語版の比率）
  3. 英語版の日別ヒット数（クローラー合計 / Browser）… 来始めた日が分かる
  4. 英語版で踏まれているパスTOP
  5. 英語版で存在しないURLに当たっているパス（rewrite漏れ・リンク切れの発見用）

実行:
  python3 tools/en_crawl_report.py              # 直近14日 vs その前の14日
  python3 tools/en_crawl_report.py --days 7 --limit 20
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402
from tools.ga4_clicks import delta  # noqa: E402

DEFAULT_DAYS = 14
DEFAULT_LIMIT = 15
# kujira-watch/src/lib/en.ts の EN_HOST と揃える。
EN_HOST = os.getenv("EN_HOST", "en.kujira-watch.com")
JST = timezone(timedelta(hours=9))

# 英語版に実在するページ。kujira-watch/src/app/(en)/en 配下と1対1。ページを増減したらここも直す。
EN_EXACT_PATHS = {"/", "/about", "/privacy", "/robots.txt", "/sitemap-en.xml"}
EN_PREFIX_PATTERNS = [
    re.compile(r"^/articles/[^/]+$"),
    # ルート直下の共通アセット（proxy.ts の SHARED_ROOT_PATHS）と Next の内部パス。
    re.compile(r"^/(?:icon|apple-icon|logo|ads\.txt|manifest\.webmanifest|favicon\.ico)"),
    re.compile(r"^/(?:_next|api)/"),
]
# next.config.ts の redirects() で / に寄せているもの（404ではない）。
EN_REDIRECT_PATTERNS = [re.compile(r"^/en(?:/|$)"), re.compile(r"^/articles$")]


def en_path_status(path: str) -> str:
    """"ok" / "redirect" / "missing"。クエリ・末尾スラッシュは落として判定する。"""
    clean = unquote(path.split("?")[0])
    if len(clean) > 1:
        clean = clean.rstrip("/")
    if clean in EN_EXACT_PATHS or clean == "":
        return "ok"
    if any(p.search(clean) for p in EN_REDIRECT_PATTERNS):
        return "redirect"
    if any(p.search(clean) for p in EN_PREFIX_PATTERNS):
        return "ok"
    return "missing"


def fetch_rows(start: datetime, end: datetime) -> list[dict]:
    """期間内の全ログ（日英とも）。bot_name が付いている行しか無いので数万行で収まる。"""
    return sb.select(
        "blog_crawler_log",
        f"occurred_at=gte.{quote(start.isoformat(), safe='')}"
        f"&occurred_at=lt.{quote(end.isoformat(), safe='')}"
        "&select=occurred_at,path,host,bot_name&order=occurred_at.desc",
    )


def is_en(row: dict) -> bool:
    return (row.get("host") or "") == EN_HOST


def print_counter(now: Counter, before: Counter, limit: int, unit: str = "回") -> None:
    if not now:
        print("  （データなし）")
        return
    for key, count in now.most_common(limit):
        print(f"  {count:>7,}{unit}  {key}{delta(count, before.get(key, 0))}")


def report(days: int, limit: int) -> None:
    now_end = datetime.now(timezone.utc)
    now_start = now_end - timedelta(days=days)
    prev_start = now_start - timedelta(days=days)

    now_rows = fetch_rows(now_start, now_end)
    prev_rows = fetch_rows(prev_start, now_start)
    en_now = [r for r in now_rows if is_en(r)]
    en_prev = [r for r in prev_rows if is_en(r)]
    ja_now = [r for r in now_rows if not is_en(r)]

    print(f"■ 英語版({EN_HOST})のクローラー別ヒット数")
    if not en_now and not en_prev:
        print("  （英語版へのアクセス記録がありません。Vercelのドメイン設定・DNS・"
              "proxy.ts のデプロイを確認）")
        return
    bots_now = Counter(r["bot_name"] for r in en_now)
    bots_prev = Counter(r["bot_name"] for r in en_prev)
    print_counter(bots_now, bots_prev, limit)

    print("\n■ 同じクローラーの日本語版ヒット数と英語版の比率")
    ja_bots = Counter(r["bot_name"] for r in ja_now)
    for bot, en_count in bots_now.most_common(limit):
        ja_count = ja_bots.get(bot, 0)
        share = en_count / (en_count + ja_count) * 100 if (en_count + ja_count) else 0
        print(f"  {bot:<22} 英語 {en_count:>7,}  日本語 {ja_count:>8,}  英語比率 {share:>5.1f}%")

    print("\n■ 英語版の日別ヒット数（JST）: クローラー / Browser")
    by_day: dict = defaultdict(Counter)
    for r in en_now:
        day = datetime.fromisoformat(r["occurred_at"]).astimezone(JST).date().isoformat()
        by_day[day]["Browser" if r["bot_name"] == "Browser" else "crawler"] += 1
    for day in sorted(by_day):
        c = by_day[day]
        print(f"  {day}  クローラー {c['crawler']:>6,}  Browser {c['Browser']:>5,}")

    print("\n■ 英語版で踏まれているパスTOP（クローラーのみ）")
    crawler_now = [r for r in en_now if r["bot_name"] != "Browser"]
    crawler_prev = [r for r in en_prev if r["bot_name"] != "Browser"]
    print_counter(Counter(r["path"] for r in crawler_now),
                  Counter(r["path"] for r in crawler_prev), limit)

    print("\n■ 英語版で存在しないURLに当たっているパス（404の可能性）")
    missing = Counter(r["path"] for r in en_now if en_path_status(r["path"]) == "missing")
    print_counter(missing, Counter(), limit)


def main() -> int:
    parser = argparse.ArgumentParser(description="英語版サブドメインのクローラー巡回レポート")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"集計日数（既定{DEFAULT_DAYS}）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"表示行数（既定{DEFAULT_LIMIT}）")
    args = parser.parse_args()

    if not sb.is_configured():
        print("[supabase] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です")
        return 1
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=args.days)
    print(f"英語版クローラーレポート: {start}〜{end}（比較: その前の{args.days}日）\n")
    report(args.days, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

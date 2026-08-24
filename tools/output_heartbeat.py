"""
tools/output_heartbeat.py

「今日、出るはずのものが出たか」を成果物そのもの（microCMSの記事・Supabaseのx_posts）で
数え、欠けていればLINEへ通知する当日ハートビート。

なぜワークフローの成否ではなく成果物を見るのか（2026-08-24の無言停止）:
  edinet_blog.yml は各ステップが continue-on-error のため、記事生成がAnthropic APIの
  利用上限で全件失敗しても run は success。記事が0件なので video_post.yml も
  「投稿対象がないため終了」で正常終了する。GitHub Actions は全部緑、成果物だけがゼロ、
  という状態を検知できるのは成果物側から数える見張りだけ。Claudeを一切使わないので、
  API上限やモデル障害の最中でも動く。

判定（平日想定。ops.yml から 13:00 UTC = 22:00 JST に実行）:
  🚨 素材（当日のEDINET大量保有開示 or 自社株買い決定）があるのにブログ記事が0件
  🚨 X投稿が0件
  ⚠️ ブログ記事は出ているのに動画のクロス投稿が0本
  正常時は送らない（--always で毎日1通送る）

終了コードは既定で常に0（異常は通知で伝える。集計自体に失敗したときだけ例外で落ちる →
ワークフローが赤くなり failure 通知が飛ぶ）。手元で異常を終了コードで見たいときは --strict。

使い方:
  python tools/output_heartbeat.py              # 異常があればLINE通知
  python tools/output_heartbeat.py --always     # 正常でも1通送る
  python tools/output_heartbeat.py --dry-run    # 送らずに本文を表示
  python tools/output_heartbeat.py --strict     # 異常があれば終了コード1
  python tools/output_heartbeat.py --date 2026-08-24
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from lib import notify  # noqa: E402
from lib import supabase_client as sb  # noqa: E402

JST = timezone(timedelta(hours=9))


def jst_today(date_str: "str | None" = None) -> str:
    return date_str or datetime.now(JST).date().isoformat()


def _day_start_utc(date_str: str) -> str:
    """JSTのその日の0時をUTCのISO8601（末尾Z）で返す。"""
    start = datetime.fromisoformat(date_str).replace(tzinfo=JST)
    return start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def count_blog_articles(date_str: str) -> int:
    """microCMSに当日公開された記事数。取得できなければ -1（不明）。"""
    domain = os.getenv("MICROCMS_SERVICE_DOMAIN", "")
    key = os.getenv("MICROCMS_API_KEY", "")
    if not domain or not key:
        return -1
    try:
        resp = requests.get(
            f"https://{domain}.microcms.io/api/v1/articles",
            headers={"X-MICROCMS-API-KEY": key},
            params={"limit": 0, "fields": "id",
                    "filters": f"publishedAt[greater_than]{_day_start_utc(date_str)}"},
            timeout=20,
        )
        if not resp.ok:
            print(f"[heartbeat] ⚠ microCMS HTTP {resp.status_code}: {resp.text[:200]}")
            return -1
        return int(resp.json().get("totalCount", 0))
    except Exception as e:
        print(f"[heartbeat] ⚠ microCMS取得失敗: {e}")
        return -1


def _count_rows(table: str, query: str) -> int:
    try:
        return len(sb.select(table, query))
    except Exception as e:
        print(f"[heartbeat] ⚠ {table} 取得失敗: {e}")
        return -1


def collect(date_str: str) -> dict:
    """当日の素材と成果物の件数を集める。取得できなかった項目は -1。"""
    day_start = _day_start_utc(date_str)
    return {
        "date": date_str,
        "holdings": _count_rows("edinet_large_holdings",
                                f"disc_date=eq.{date_str}&select=doc_id"),
        "buybacks": _count_rows("tdnet_buybacks",
                                f"disclosed_at=gte.{date_str}T00:00:00"
                                f"&disclosed_at=lt.{date_str}T23:59:59&select=code"),
        "articles": count_blog_articles(date_str),
        "x_posts": _count_rows("x_posts", f"posted_at=gte.{day_start}&select=tweet_id"),
        "videos": _count_rows("x_posts",
                              f"posted_at=gte.{day_start}&kind=eq.video&select=tweet_id"),
    }


def judge(counts: dict) -> list[str]:
    """異常メッセージのリスト（空なら正常）。件数不明(-1)は判定しない。"""
    problems = []
    material = max(counts["holdings"], 0) + max(counts["buybacks"], 0)
    if counts["articles"] == 0 and material > 0:
        problems.append(
            f"ブログ記事が0件（当日の素材: 大量保有{counts['holdings']}件 / "
            f"自社株買い{counts['buybacks']}件）"
        )
    if counts["x_posts"] == 0:
        problems.append("X投稿が0件")
    if counts["videos"] == 0 and counts["articles"] > 0:
        problems.append(f"動画の投稿が0本（当日の記事は{counts['articles']}件）")
    return problems


def _n(v: int) -> str:
    return "不明" if v < 0 else str(v)


def build_message(counts: dict, problems: list[str]) -> str:
    md = f"{int(counts['date'][5:7])}/{int(counts['date'][8:10])}"
    body = (f"ブログ{_n(counts['articles'])}件 / X{_n(counts['x_posts'])}件 / "
            f"動画{_n(counts['videos'])}本\n"
            f"素材: 大量保有{_n(counts['holdings'])}件 / 自社株買い{_n(counts['buybacks'])}件")
    if not problems:
        return notify.build_message("✅", f"{md} 自動投稿", body)
    return notify.build_message("🚨", f"{md} 自動投稿が欠けています",
                                "・" + "\n・".join(problems) + "\n\n" + body)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="対象日（JST、YYYY-MM-DD。既定は今日）")
    p.add_argument("--always", action="store_true", help="正常でもLINEに1通送る")
    p.add_argument("--dry-run", action="store_true", help="送らずに本文を表示する")
    p.add_argument("--strict", action="store_true", help="異常があれば終了コード1で終わる")
    args = p.parse_args()

    counts = collect(jst_today(args.date))
    problems = judge(counts)
    message = build_message(counts, problems)
    print(message)

    if not args.dry_run and (problems or args.always):
        notify.push(message)
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())

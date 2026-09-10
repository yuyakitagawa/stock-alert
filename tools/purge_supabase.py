"""Supabaseの保持期間を適用する（日次）。

Free枠はDB 500MB。放っておくと blog_crawler_log が月120MB、gen_rankings が月30MB、
yahoo_price_cache が月8MB増え続け、2026-09-07に544MBまで育って枠を超えた。
消す条件はDB側の関数に書いてあるので、ここはそれを順に呼ぶだけ。

  python3 tools/purge_supabase.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402


def main() -> int:
    if not sb.is_configured():
        print("[purge] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定。スキップします。")
        return 0

    failed = []

    buckets = sb.rpc("purge_blog_crawler_log", {})
    if buckets is None:
        failed.append("purge_blog_crawler_log")
    else:
        for b in buckets:
            print(f"[purge] blog_crawler_log {b['bucket']}: {b['deleted']:,}行")

    for fn, label in (("purge_gen_rankings", "gen_rankings"),
                      ("purge_yahoo_price_cache", "yahoo_price_cache")):
        n = sb.rpc(fn, {})
        if n is None:
            failed.append(fn)
        else:
            print(f"[purge] {label}: {n:,}行")

    if failed:
        print(f"[purge] 失敗: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

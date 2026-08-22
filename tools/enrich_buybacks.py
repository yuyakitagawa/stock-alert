"""
tools/enrich_buybacks.py

自社株買い「決定」開示（ext_tdnet_disclosures）の原文PDFから取得枠を抽出して tdnet_buybacks に保存する。
daily_alert.yml の Step 2f（fetch_tdnet.py）の直後に走る。

    python3 tools/enrich_buybacks.py              # 直近3日
    python3 tools/enrich_buybacks.py --days 60    # バックフィル
    python3 tools/enrich_buybacks.py --dry-run    # 保存しない
    python3 tools/enrich_buybacks.py --days 30 --retry-failed  # 抽出失敗行の再試行
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from lib.buyback import enrich  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="処理件数の上限（0=無制限）")
    p.add_argument("--retry-failed", action="store_true", help="抽出失敗（extract_ok=false）の行も読み直す")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    enrich(days=args.days, dry_run=args.dry_run, limit=args.limit, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()

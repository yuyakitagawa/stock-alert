"""
tools/fetch_jquants_history.py
J-Quants API v2 から信用残高・空売り残高の全期間データを一括取得して DB に保存。

必要な環境変数:
    JQUANTS_API_KEY=<J-Quants Standard プランの API キー>

使い方:
    python3 tools/fetch_jquants_history.py                     # 信用残高＋空売り残高（全期間）
    python3 tools/fetch_jquants_history.py --margin            # 信用残高のみ
    python3 tools/fetch_jquants_history.py --short             # 空売り残高のみ
    python3 tools/fetch_jquants_history.py --start 20220101    # 開始日を指定
    python3 tools/fetch_jquants_history.py --dry-run           # カラム確認のみ（10日分）
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from lib.db import init_db


def check_api_key():
    key = os.environ.get("JQUANTS_API_KEY", "")
    if not key:
        print("ERROR: JQUANTS_API_KEY が設定されていません。")
        print()
        print("設定方法:")
        print("  export JQUANTS_API_KEY='your-api-key-here'")
        print("  # または .env ファイルに追記して GitHub Actions secrets にも追加")
        sys.exit(1)
    print(f"  APIキー確認: {'*' * (len(key)-4)}{key[-4:]}")


def dry_run_check():
    """カラム名確認用：10日分だけ取得して構造を出力"""
    from jquantsapi import ClientV2
    cli = ClientV2()
    today = date.today().strftime("%Y%m%d")
    # 10日前
    from datetime import timedelta
    start = (date.today() - timedelta(days=10)).strftime("%Y%m%d")

    print(f"\n--- 信用残高 (margin-interest) カラム確認 ---")
    try:
        df = cli.get_mkt_margin_interest(from_yyyymmdd=start, to_yyyymmdd=today)
        print(f"カラム: {list(df.columns)}")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"ERROR: {e}")

    print(f"\n--- 空売り残高 (short-sale-report) カラム確認 ---")
    try:
        df = cli.get_mkt_short_sale_report(disclosed_date_from=start, disclosed_date_to=today)
        print(f"カラム: {list(df.columns)}")
        print(df.head(3).to_string())
    except Exception as e:
        print(f"ERROR: {e}")


def fetch_margin(start: str, end: str):
    """信用残高（信用取引週末残高）を一括取得 → kabutan_jquants_margin テーブルへ"""
    from lib.jquants import fetch_margin_history
    print(f"\n信用残高 取得中: {start} → {end} ...")
    saved = fetch_margin_history(start=start, end=end)
    print(f"  完了: {saved:,} 行保存")
    return saved


def fetch_short(start: str, end: str):
    """空売り残高報告を一括取得 → jpx_short_interest テーブルへ"""
    from lib.jquants import fetch_short_history
    print(f"\n空売り残高 取得中: {start} → {end} ...")
    saved = fetch_short_history(start=start, end=end)
    print(f"  完了: {saved:,} 行保存")
    return saved


def main():
    parser = argparse.ArgumentParser(description="J-Quants 信用残高・空売り残高 一括取得")
    parser.add_argument("--margin",  action="store_true", help="信用残高のみ取得")
    parser.add_argument("--short",   action="store_true", help="空売り残高のみ取得")
    parser.add_argument("--start",   default="20200101",  help="取得開始日 YYYYMMDD（デフォルト: 20200101）")
    parser.add_argument("--dry-run", action="store_true", help="カラム確認のみ（保存なし）")
    args = parser.parse_args()

    print("=" * 60)
    print(f"J-Quants 歴史データ一括取得")
    print(f"  開始日: {args.start}")
    print(f"  終了日: {date.today().strftime('%Y%m%d')} (今日)")
    print("=" * 60)

    check_api_key()
    init_db()

    if args.dry_run:
        dry_run_check()
        return

    end = date.today().strftime("%Y%m%d")
    do_margin = args.margin or (not args.margin and not args.short)
    do_short  = args.short  or (not args.margin and not args.short)

    total = 0
    if do_margin:
        total += fetch_margin(args.start, end)
    if do_short:
        total += fetch_short(args.start, end)

    print(f"\n✅ 完了: 合計 {total:,} 行保存")
    print("次: python3 core/rf_train_v3.py  ← 再学習（信用・空売り特徴量込み）")


if __name__ == "__main__":
    main()

"""
tools/fetch_edinet_financials.py
EDINET API v2 から決算書類(有報/四半期報)のXBRLを取得し、
財務データを jquants_fin_summary テーブルに保存する。

J-Quants Free プラン期限切れ後の代替データソース。

使い方:
    python3 tools/fetch_edinet_financials.py                # 直近30日をスキャン
    python3 tools/fetch_edinet_financials.py --days 90      # 直近90日
    python3 tools/fetch_edinet_financials.py --start 2026-03-18  # 指定日以降を全取得
    python3 tools/fetch_edinet_financials.py --dry-run      # DB保存なし（テスト用）
    python3 tools/fetch_edinet_financials.py --verify       # APIキー確認のみ

必要: EDINET_API_KEY 環境変数
所要時間: 1日あたり約10-30秒（XBRL取得+解析）
"""
import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(description="EDINET決算XBRL取得→DB保存")
    parser.add_argument("--days", type=int, default=30, help="遡る日数（デフォルト30）")
    parser.add_argument("--start", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="DB保存しない")
    parser.add_argument("--verify", action="store_true", help="APIキー確認のみ")
    parser.add_argument("--sleep", type=float, default=1.0, help="XBRL取得間のスリープ秒")
    args = parser.parse_args()

    load_env()

    if args.verify:
        from lib.edinet import verify_api
        result = verify_api()
        print(f"EDINET API: {'OK' if result['ok'] else 'NG'}")
        print(f"  理由: {result['reason']}")
        print(f"  日付: {result['date']}")
        print(f"  書類数: {result['total']}")
        print(f"  大量保有: {result['large']}")
        return

    from lib.edinet_financials import scan_financial_reports

    print("=" * 60)
    print("EDINET 決算XBRL取得パイプライン")
    if args.start:
        print(f"  期間: {args.start} ～ 今日")
    else:
        print(f"  期間: 直近{args.days}日")
    print(f"  DB保存: {'OFF (dry-run)' if args.dry_run else 'ON'}")
    print("=" * 60)

    records = scan_financial_reports(
        days_back=args.days,
        persist=not args.dry_run,
        start_date=args.start,
        skip_weekends=True,
        sleep_sec=args.sleep,
    )

    print(f"\n{'=' * 60}")
    print(f"完了: {len(records)}件の財務データ取得")
    if records:
        codes = set(r["code"] for r in records)
        print(f"  銘柄数: {len(codes)}")
        doc_types = {}
        for r in records:
            dt = r.get("doc_type", "?")
            doc_types[dt] = doc_types.get(dt, 0) + 1
        print(f"  内訳: {doc_types}")

        # サンプル表示
        print(f"\n  サンプル（先頭5件）:")
        for r in records[:5]:
            print(f"    {r['code']} {r['doc_type']} {r['disc_date']}: "
                  f"sales={r.get('sales')} op={r.get('op')} np={r.get('np')} "
                  f"eps={r.get('eps')} div={r.get('div_ann')}")


if __name__ == "__main__":
    main()

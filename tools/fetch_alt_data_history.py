"""
tools/fetch_alt_data_history.py
オルタナティブデータの履歴を一括収集するバッチツール。

収集対象:
  1. TDnet適時開示（kabutan IR + TDnet）

【実行タイミング】
  - 週次で実行推奨（毎週月曜 or 平日 1回）

Usage:
  python3 tools/fetch_alt_data_history.py --tdnet [--top N]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import requests
import pandas as pd
import io
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.utils import HEADERS
from lib.db import init_db, upsert_tdnet_events
from lib.alt_data import get_tdnet_events


def get_tse_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()
        code_col = next(c for c in df.columns if "コード" in c)
        name_col = next(c for c in df.columns if "銘柄名" in c)
        result = df[[code_col, name_col]].copy()
        result.columns = ["code", "name"]
        result["code"] = result["code"].str.strip()
        result = result[result["code"].str.match(r'^\d{4}$')]
        return result.reset_index(drop=True)
    except Exception as e:
        print(f"銘柄リスト取得失敗: {e}")
        return None


def fetch_tdnet_batch(codes_names: list, workers: int = 3):
    """TDnet適時開示を一括取得"""
    print(f"\nTDnet適時開示バッチ取得: {len(codes_names)} 銘柄...")
    success = 0
    errors = 0

    def _fetch_one(code_name):
        code, name = code_name
        try:
            events = get_tdnet_events(str(code), days=90)
            return bool(events)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=workers) as exc:
        futures = {exc.submit(_fetch_one, cn): cn for cn in codes_names}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success += 1
            else:
                errors += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(codes_names)} 完了 (成功:{success} エラー:{errors})")
            time.sleep(1.0)

    print(f"  完了: 成功{success} / エラー{errors}")
    return success


def main():
    parser = argparse.ArgumentParser(description="オルタナティブデータ一括収集")
    parser.add_argument("--tdnet",   action="store_true", help="TDnet適時開示を取得")
    parser.add_argument("--all",     action="store_true", help="全て取得")
    parser.add_argument("--top",     type=int, default=0, help="上位N銘柄のみ（0=全銘柄）")
    args = parser.parse_args()

    if not (args.tdnet or args.all):
        parser.print_help()
        return

    init_db()
    print("="*60)
    print("オルタナティブデータ一括収集  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("="*60)

    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    if args.top > 0:
        stock_list = stock_list.head(args.top)
    codes_names = list(zip(stock_list["code"].tolist(), stock_list["name"].tolist()))
    print(f"対象: {len(codes_names)} 銘柄")

    if args.all or args.tdnet:
        fetch_tdnet_batch(codes_names)

    print("\n✅ 完了")


if __name__ == "__main__":
    main()

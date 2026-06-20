"""
tools/fetch_alt_data_history.py
オルタナティブデータの履歴を一括収集するバッチツール。

収集対象:
  1. 信用倍率（kabutan.jp/stock/credit）
  2. 空売り残高（JPX 週次公開データ） ← 最新版のみ（履歴は自動蓄積）
  3. Googleトレンド（pytrends、上位銘柄のみ）

【実行タイミング】
  - 週次で実行推奨（毎週月曜 or 平日 1回）
  - 空売りデータは TSE が金曜に翌週月曜公開するため、月曜実行が最新
  - 信用倍率は週次（月曜更新）

【次のモデル再学習で特徴量として追加予定の項目】
  - shinyo_f:    信用倍率（log正規化）
  - short_pct_f: 空売り比率
  - tdnet_buyback_days_f: 自社株買い発表からの日数

Usage:
  python3 tools/fetch_alt_data_history.py [--margin] [--short] [--tdnet] [--top N]
  python3 tools/fetch_alt_data_history.py --all  # 全て実行
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
from lib.db import init_db, upsert_margin_data, bulk_upsert_short_interest, upsert_tdnet_events
from lib.alt_data import _load_tse_short_data, get_tdnet_events

_KABUTAN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en;q=0.9",
}


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


def fetch_margin_batch(codes: list, workers: int = 8):
    """信用倍率を一括取得"""
    import re
    today_str = date.today().isoformat()
    success = 0
    errors = 0

    def _fetch_one(code):
        import re
        try:
            resp = requests.get(
                f"https://kabutan.jp/stock/credit?code={code}",
                headers=_KABUTAN_HEADERS, timeout=8
            )
            if resp.status_code != 200:
                return None
            text = resp.text.replace("\n", "").replace("\t", "").replace(" ", "")
            ratio_m = re.search(r'信用倍率[^<]{0,20}<td[^>]*>([\d,.]+)倍?</td>', text)
            buy_m   = re.search(r'信用買残[^<]{0,20}<td[^>]*>([\d,]+)', text)
            sell_m  = re.search(r'信用売残[^<]{0,20}<td[^>]*>([\d,]+)', text)
            if not ratio_m:
                return None
            ratio = float(ratio_m.group(1).replace(",", ""))
            buy  = float(buy_m.group(1).replace(",", ""))  if buy_m  else 0.0
            sell = float(sell_m.group(1).replace(",", "")) if sell_m else 0.0
            date_m = re.search(r'(\d{4}/\d{2}/\d{2})', text[:3000])
            week_date = date_m.group(1).replace("/", "-") if date_m else today_str
            return (code, week_date, buy, sell, ratio)
        except Exception:
            return None

    print(f"\n信用倍率バッチ取得: {len(codes)} 銘柄...")
    with ThreadPoolExecutor(max_workers=workers) as exc:
        futures = {exc.submit(_fetch_one, c): c for c in codes}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                code, week_date, buy, sell, ratio = result
                upsert_margin_data(code, week_date, buy, sell, ratio)
                success += 1
            else:
                errors += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(codes)} 完了 (成功:{success} エラー:{errors})")
            time.sleep(0.3)  # Rate limiting

    print(f"  完了: 成功{success} / エラー{errors}")
    return success


def fetch_short_data():
    """TSE公開の空売り残高週次データをダウンロード"""
    print("\n空売り残高データ取得中（TSE週次Excel）...")
    ok = _load_tse_short_data()
    if ok:
        print("  ✅ 空売り残高データ取得成功（DB保存済み）")
    else:
        print("  ❌ 空売り残高データ取得失敗")
    return ok


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
            time.sleep(1.0)  # TDnet rate limiting

    print(f"  完了: 成功{success} / エラー{errors}")
    return success


def main():
    parser = argparse.ArgumentParser(description="オルタナティブデータ一括収集")
    parser.add_argument("--margin",  action="store_true", help="信用倍率を取得")
    parser.add_argument("--short",   action="store_true", help="空売り残高を取得")
    parser.add_argument("--tdnet",   action="store_true", help="TDnet適時開示を取得")
    parser.add_argument("--all",     action="store_true", help="全て取得")
    parser.add_argument("--top",     type=int, default=0, help="上位N銘柄のみ（0=全銘柄）")
    args = parser.parse_args()

    if not (args.margin or args.short or args.tdnet or args.all):
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
    codes = stock_list["code"].tolist()
    codes_names = list(zip(stock_list["code"].tolist(), stock_list["name"].tolist()))
    print(f"対象: {len(codes)} 銘柄")

    if args.all or args.short:
        fetch_short_data()

    if args.all or args.margin:
        fetch_margin_batch(codes)

    if args.all or args.tdnet:
        fetch_tdnet_batch(codes_names)

    print("\n✅ 完了")


if __name__ == "__main__":
    main()

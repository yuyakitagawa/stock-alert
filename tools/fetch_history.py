#!/usr/bin/env python3
"""
fetch_history.py — 全銘柄の株価履歴を10年分 Yahoo Finance から取得して yahoo_price_cache に保存

使い方:
  python3 tools/fetch_history.py          # 全銘柄を10年分取得（初回: 数時間かかる）
  python3 tools/fetch_history.py --years 5  # 5年分だけ
  python3 tools/fetch_history.py --resume   # 取得済みをスキップして続きから

注意: Yahoo Finance のレート制限を避けるため1リクエストごとに1秒待機します。
      中断しても --resume で続きから再開できます。
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from lib.db import save_price_cache, get_price_cache_coverage, init_db

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

parser = argparse.ArgumentParser()
parser.add_argument("--years",  type=int, default=10, help="取得する年数（デフォルト: 10）")
parser.add_argument("--resume", action="store_true",  help="取得済み銘柄をスキップ")
parser.add_argument("--sleep",  type=float, default=0.8, help="リクエスト間隔（秒）")
args = parser.parse_args()

TARGET_DAYS = args.years * 365


def fetch_yahoo(code, days):
    """Yahoo Finance から最大 days 日分の日足データを取得"""
    ticker   = f"{code}.T"
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        data   = r.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        ts        = result[0].get("timestamp", [])
        adjcloses = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        raw_c     = result[0].get("indicators", {}).get("quote",    [{}])[0].get("close",    [])
        volumes   = result[0].get("indicators", {}).get("quote",    [{}])[0].get("volume",   [])
        if not ts or not adjcloses:
            return None
        closes = [a if a is not None else r for a, r in zip(adjcloses, raw_c)]
        df = pd.DataFrame(
            {"Close": closes, "Volume": volumes},
            index=pd.to_datetime(ts, unit="s", utc=True)
        )
        df.index = df.index.tz_convert("Asia/Tokyo").normalize().date
        df.index = pd.to_datetime(df.index)
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def get_all_codes():
    """yahoo_price_cache 既存コード + JPX 銘柄リストを合わせた全コード"""
    from lib.db import get_price_cache_codes
    codes_from_db = get_price_cache_codes()
    if codes_from_db:
        return codes_from_db

    # DB が空の場合は JPX から取得
    print("yahoo_price_cache が空のため JPX から銘柄リストを取得...")
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        r = requests.get(url, timeout=30)
        df = pd.read_excel(pd.io.common.BytesIO(r.content), dtype=str)
        df.columns = df.columns.str.strip()
        mc = [c for c in df.columns if "市場・商品区分" in c]
        cc = [c for c in df.columns if "コード" in c]
        if mc and cc:
            mask = df[mc[0]].str.contains("内国株式", na=False)
            return df.loc[mask, cc[0]].str.strip().tolist()
    except Exception as e:
        print(f"[WARN] JPX 取得失敗: {e}")
    return []


def main():
    init_db()
    codes = get_all_codes()
    if not codes:
        print("ERROR: 銘柄コードが取得できません。先に screener.py を実行してください。")
        sys.exit(1)

    target_start = (date.today() - timedelta(days=TARGET_DAYS)).isoformat()
    print(f"対象銘柄: {len(codes)}本  取得期間: {target_start} 〜 今日  ({args.years}年分)")
    print(f"リクエスト間隔: {args.sleep}秒  推定時間: {len(codes)*args.sleep/3600:.1f}時間")
    print("Ctrl+C で中断可能。--resume フラグで再開できます。")
    print("=" * 60)

    done = skip = fail = 0
    for i, code in enumerate(codes, 1):
        # --resume: すでに十分なデータがあればスキップ
        if args.resume:
            cov = get_price_cache_coverage(code)
            if cov and cov[0] <= target_start:
                skip += 1
                if skip % 500 == 0:
                    print(f"  スキップ中... {skip} 件")
                continue

        df = fetch_yahoo(code, TARGET_DAYS + 30)  # 少し余裕を持って取得
        if df is not None and len(df) > 50:
            save_price_cache(code, df)
            done += 1
        else:
            fail += 1

        # 進捗表示
        if i % 100 == 0 or i == len(codes):
            elapsed = i / len(codes) * 100
            print(f"  [{i:4d}/{len(codes)}] {elapsed:.0f}%  取得: {done}  スキップ: {skip}  失敗: {fail}")

        time.sleep(args.sleep)

    print("=" * 60)
    print(f"完了: 取得={done}  スキップ={skip}  失敗={fail}")
    print("次のステップ: python3 tools/backtest.py --start 2020-01-01 --end 2020-06-30 で任意期間検証できます")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fetch_history.py — 全銘柄の株価履歴を Yahoo Finance から取得して yahoo_price_cache に保存

daily_alert.yml Step 0 が毎日呼び出し、yahoo_price_cache を最新日まで差分更新する
（J-Quants Free プランは直近データを取得できないため、価格取得は Yahoo Finance のみを使う）。
insert_ignore で保存するため、既存の(code,date)は上書きされず新規日付のみ追加される。

使い方:
  python3 tools/fetch_history.py          # 全銘柄を10年分取得（初回: 数時間かかる）
  python3 tools/fetch_history.py --years 5  # 5年分だけ
  python3 tools/fetch_history.py --years 1  # 日次更新用（daily_alert.yml Step 0）
  python3 tools/fetch_history.py --resume   # 取得済みをスキップして続きから（初回backfill専用。
                                             # 古い方の日付しか見ないため日次更新には使わないこと）

yfinance ライブラリで100銘柄ずつバッチ取得します（生のchart APIは現在crumb/cookie必須で
単発requestsでは取得できないため）。バッチ間はレート制限を避けるため --sleep 秒待機します。
中断しても --resume で続きから再開できます。
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from lib.db import save_price_cache, get_price_cache_coverage, init_db

BATCH_SIZE = 100

parser = argparse.ArgumentParser()
parser.add_argument("--years",  type=int, default=10, help="取得する年数（デフォルト: 10）")
parser.add_argument("--resume", action="store_true",  help="取得済み銘柄をスキップ")
parser.add_argument("--sleep",  type=float, default=2.0, help="バッチ間隔（秒）")
args = parser.parse_args()

TARGET_DAYS = args.years * 365


def fetch_yahoo_batch(codes: list, days: int) -> dict:
    """yfinance で複数銘柄をまとめて取得し {code: DataFrame(Close, Volume)} を返す。
    生のchart APIはcrumb/cookie必須になり単発requestsでは取得できなくなったため、
    Yahoo側の認証をハンドリングするyfinanceライブラリ経由にしている。
    """
    tickers = [f"{c}.T" for c in codes]
    try:
        data = yf.download(tickers, period=f"{days}d", auto_adjust=True,
                            progress=False, group_by="ticker", threads=True)
    except Exception as e:
        print(f"  [WARN] バッチ取得失敗: {e}")
        return {}

    result = {}
    for code, ticker in zip(codes, tickers):
        try:
            sub = data if len(tickers) == 1 else data[ticker]
            sub = sub[["Close", "Volume"]].dropna(subset=["Close"])
            if len(sub) == 0:
                continue
            sub = sub.copy()
            sub.index = pd.to_datetime(sub.index.date)
            result[code] = sub
        except Exception:
            continue
    return result


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

    # --resume: すでに十分なデータがある銘柄は対象から除外
    skip = 0
    if args.resume:
        filtered = []
        for code in codes:
            cov = get_price_cache_coverage(code)
            if cov and cov[0] <= target_start:
                skip += 1
            else:
                filtered.append(code)
        codes = filtered

    n_batches = (len(codes) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"対象銘柄: {len(codes)}本（スキップ{skip}件）  取得期間: {target_start} 〜 今日  ({args.years}年分)")
    print(f"バッチサイズ: {BATCH_SIZE}銘柄/回  バッチ間隔: {args.sleep}秒  推定バッチ数: {n_batches}")
    print("=" * 60)

    done = fail = 0
    days = TARGET_DAYS + 30  # 少し余裕を持って取得
    for b in range(n_batches):
        batch_codes = codes[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        fetched = fetch_yahoo_batch(batch_codes, days)
        for code in batch_codes:
            df = fetched.get(code)
            if df is not None and len(df) > 50:
                save_price_cache(code, df)
                done += 1
            else:
                fail += 1

        elapsed = (b + 1) / n_batches * 100
        print(f"  [batch {b+1:3d}/{n_batches}] {elapsed:.0f}%  取得: {done}  失敗: {fail}")

        if b < n_batches - 1:
            time.sleep(args.sleep)

    print("=" * 60)
    print(f"完了: 取得={done}  スキップ={skip}  失敗={fail}")
    print("次のステップ: python3 tools/backtest.py --start 2020-01-01 --end 2020-06-30 で任意期間検証できます")


if __name__ == "__main__":
    main()

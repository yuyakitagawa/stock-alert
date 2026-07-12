#!/usr/bin/env python3
"""
update_price_cache.py — J-Quants で直近 N 日分の株価を取得して yahoo_price_cache を更新

使い方:
  python3 tools/update_price_cache.py           # 直近 30 営業日
  python3 tools/update_price_cache.py --days 7  # 直近 7 営業日
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=30, help="取得する直近営業日数（デフォルト30）")
args = parser.parse_args()

api_key = os.environ.get("JQUANTS_API_KEY", "")
if not api_key:
    print("ERROR: JQUANTS_API_KEY が設定されていません")
    sys.exit(1)

from jquantsapi import ClientV2
from lib.db import save_price_cache, init_db

init_db()
cli = ClientV2(api_key=api_key)

# 対象日付を計算（土日を除いた直近 N 営業日）
target_dates = []
d = date.today()
while len(target_dates) < args.days:
    if d.weekday() < 5:  # 月〜金
        target_dates.append(d.isoformat())
    d -= timedelta(days=1)
target_dates.reverse()

print(f"[update_price_cache] {target_dates[0]} 〜 {target_dates[-1]} ({len(target_dates)}日) の価格を取得中...")

total_codes = 0
errors = 0

for date_str in target_dates:
    date_yyyymmdd = date_str.replace("-", "")
    try:
        df = cli.get_eq_bars_daily(date_yyyymmdd=date_yyyymmdd)
        if df is None or len(df) == 0:
            print(f"  {date_str}: データなし（休場日？）")
            continue

        # 列名の正規化（新API: Code, AdjClose or Close, Volume）
        df = df.rename(columns={
            "Code": "code", "AdjClose": "close", "Volume": "volume",
            "Close": "close_raw",
        })
        # AdjClose が使えればそちらを優先（split-adjusted）
        if "close" not in df.columns and "close_raw" in df.columns:
            df = df.rename(columns={"close_raw": "close"})

        # コードを4桁に正規化（J-Quantsは5桁末尾0: "13010" → "1301"）
        df["code"] = df["code"].astype(str).str.strip().apply(
            lambda c: c[:4] if len(c) == 5 else c.zfill(4)
        )
        df = df[["code", "close", "volume"]].dropna(subset=["close"])
        df["date"] = date_str

        # コード別DataFrameを作ってsave_price_cache
        count = 0
        for code, grp in df.groupby("code"):
            price_df = pd.DataFrame(
                {"Close": grp["close"].values, "Volume": grp["volume"].values},
                index=pd.to_datetime([date_str] * len(grp))
            )
            save_price_cache(str(code), price_df)
            count += 1
        total_codes += count
        print(f"  {date_str}: {count} 銘柄 保存完了")

    except Exception as e:
        msg = str(e)
        if "subscription covers" in msg or "429" in msg:
            print(f"  {date_str}: サブスクリプション対象外またはレート制限（スキップ）")
            continue
        print(f"  {date_str}: ERROR {e}")
        errors += 1

print(f"\n[update_price_cache] 完了: {total_codes} 銘柄件数 / エラー {errors} 日")

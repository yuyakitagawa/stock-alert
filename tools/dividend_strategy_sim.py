"""
配当落ち前後トレード戦略シミュレーション

戦略:
  A) 配当落ち前N日に買い → 前日売り（配当権利取り相場の上昇を狙う）
  B) 配当落ち後M日に買い → K日後売り（配当落ち後の戻しを狙う）
  C) A+B 両方つなげる複合戦略

日本株の配当落ち日: 3月末・9月末の最終営業日の翌営業日
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "stock_alert.db")

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--yutai", action="store_true", help="株主優待銘柄のみ対象")
_args, _ = _parser.parse_known_args()

# シミュレーションするパラメータ組み合わせ
PRE_DAYS_LIST  = [5, 10, 20]     # 配当落ち前N日に買い
POST_DAYS_LIST = [1, 3, 5]       # 配当落ち後M日に買い直し
HOLD_DAYS_LIST = [10, 20, 30]    # 戻し買い後K日保有
# ──────────────────────────────────────────────────────────


def load_all_prices() -> dict[str, pd.Series]:
    """price_cacheから全銘柄の価格をまとめて読み込む"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT code, date, close FROM price_cache WHERE date >= '2023-06-01' ORDER BY code, date",
        conn
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    result = {}
    for code, grp in df.groupby("code"):
        s = grp.set_index("date")["close"].dropna()
        if len(s) > 50:
            result[str(code)] = s
    return result


def load_yutai_from_sheets() -> dict[str, list[int]]:
    """スプシ「📋 銘柄ファンダメンタル」から優待確定月を持つ銘柄を取得。
    戻り値: {code: [month, ...]}（複数月の場合あり）
    """
    import os, sys
    sys.path.insert(0, BASE_DIR)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(
        os.path.join(BASE_DIR, "gcp_key.json"),
        scopes=["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.getenv("SPREADSHEET_ID", ""))
    ws = sh.worksheet("📋 銘柄ファンダメンタル")
    rows = ws.get_all_values()

    header = rows[0]
    col_code  = header.index("コード")
    col_yutai = header.index("優待確定月")

    result: dict[str, list[int]] = {}
    for row in rows[1:]:
        code = str(row[col_code]).strip()
        yutai_str = row[col_yutai].strip()
        if not yutai_str or yutai_str in ("なし", "—", "-", ""):
            continue
        # "3月,9月" や "3月" を parse
        months = []
        for part in yutai_str.split(","):
            part = part.strip().replace("月", "")
            if part.isdigit():
                months.append(int(part))
        if months:
            result[code] = months
    return result


def build_ex_dates_from_months(code_months: dict[str, list[int]], biz_days: list) -> dict[str, list[date]]:
    """
    銘柄ごとの優待/配当確定月リストから、過去の配当落ち日を算出。
    日本株の配当落ち日 ≈ 確定月の月末から2営業日前。
    """
    import calendar
    # 対象年（price_cacheの範囲: 2023年以降）
    target_years = [2023, 2024, 2025]

    result: dict[str, list[date]] = {}
    for code, months in code_months.items():
        ex_dates = []
        for year in target_years:
            for month in months:
                last_day = calendar.monthrange(year, month)[1]
                month_end = pd.Timestamp(year, month, last_day)
                # 月末以前の営業日から2営業日前を配当落ち日とする
                before = [d for d in biz_days if d <= month_end]
                if len(before) < 3:
                    continue
                ex_date = before[-2].date()
                if ex_date <= date.today():
                    ex_dates.append(ex_date)
        if ex_dates:
            result[code] = ex_dates
    return result


def load_biz_days() -> list:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT date FROM price_cache WHERE code='7203' ORDER BY date", conn)
    conn.close()
    return pd.to_datetime(df["date"]).sort_values().tolist()


def nearest_price(series: pd.Series, target: date, direction: str = "nearest") -> tuple[date, float] | None:
    """target日前後で最も近い営業日の価格を返す"""
    if series is None or series.empty:
        return None
    target_ts = pd.Timestamp(target)
    if direction == "before":
        sub = series[series.index <= target_ts]
        if sub.empty:
            return None
        return sub.index[-1].date(), float(sub.iloc[-1])
    elif direction == "after":
        sub = series[series.index >= target_ts]
        if sub.empty:
            return None
        return sub.index[0].date(), float(sub.iloc[0])
    else:
        if target_ts in series.index:
            return target, float(series[target_ts])
        diffs = abs(series.index - target_ts)
        idx = diffs.argmin()
        return series.index[idx].date(), float(series.iloc[idx])


def sim_pre_trade(series: pd.Series, ex_date: date, pre_days: int) -> dict | None:
    """配当落ち前N営業日に買い → 配当落ち前日に売るトレード"""
    biz_days = series[series.index < pd.Timestamp(ex_date)]
    if len(biz_days) < pre_days + 1:
        return None
    buy_day  = biz_days.index[-pre_days].date()
    buy_px   = float(biz_days.iloc[-pre_days])
    sell_day = biz_days.index[-1].date()
    sell_px  = float(biz_days.iloc[-1])
    ret = (sell_px - buy_px) / buy_px * 100
    return {"buy_day": buy_day, "buy_px": buy_px,
            "sell_day": sell_day, "sell_px": sell_px, "ret_pct": ret}


def sim_post_trade(series: pd.Series, ex_date: date, post_days: int, hold_days: int) -> dict | None:
    """配当落ち後M営業日に買い → K営業日後に売るトレード"""
    biz_after = series[series.index >= pd.Timestamp(ex_date)]
    if len(biz_after) < post_days + hold_days:
        return None
    buy_day  = biz_after.index[post_days].date()
    buy_px   = float(biz_after.iloc[post_days])
    # 配当落ち日の価格（基準）
    ex_px    = float(biz_after.iloc[0])
    sell_day = biz_after.index[post_days + hold_days].date()
    sell_px  = float(biz_after.iloc[post_days + hold_days])
    ret = (sell_px - buy_px) / buy_px * 100
    drop_from_ex = (buy_px - ex_px) / ex_px * 100  # 配当落ち日比で何%下で買えたか
    return {"buy_day": buy_day, "buy_px": buy_px, "ex_px": ex_px,
            "drop_from_ex": drop_from_ex,
            "sell_day": sell_day, "sell_px": sell_px,
            "ret_pct": ret, "hold_days": hold_days}


def run_simulation():
    print("price_cacheから全銘柄データ読み込み中...")
    all_prices = load_all_prices()
    biz_days   = load_biz_days()

    print("スプシから優待/配当確定月を取得中...")
    yutai_map = load_yutai_from_sheets()   # code -> [month, ...]
    print(f"  優待銘柄: {len(yutai_map)}銘柄")

    if _args.yutai:
        # 優待確定月ベースでex-dateを算出
        ex_dates_map = build_ex_dates_from_months(yutai_map, biz_days)
        label = "株主優待銘柄"
    else:
        # 全銘柄: スプシの全銘柄の配当確定月を使う（優待がない銘柄も含む）
        # ここでは優待のある銘柄のex-datesを使う（全銘柄版は従来通り）
        ex_dates_map = build_ex_dates_from_months(yutai_map, biz_days)
        label = "全銘柄（スプシベース）"

    codes = [c for c in ex_dates_map if c in all_prices]
    print(f"対象銘柄: {len(codes)}銘柄 ({label})（各銘柄の実際の優待確定月ごとにシミュレーション）\n")

    pre_results  = []
    post_results = []

    for i, code in enumerate(codes):
        series = all_prices[code]
        if (i + 1) % 200 == 0:
            print(f"  進捗: {i+1}/{len(codes)}銘柄...")

        for ex_date in ex_dates_map.get(code, []):
            if ex_date > date.today():
                continue

            # A) 前入り戦略
            for pre_days in PRE_DAYS_LIST:
                r = sim_pre_trade(series, ex_date, pre_days)
                if r:
                    pre_results.append({
                        "code": code, "ex_date": ex_date,
                        "pre_days": pre_days, **r
                    })

            # B) 戻し買い戦略
            for post_days in POST_DAYS_LIST:
                for hold_days in HOLD_DAYS_LIST:
                    r = sim_post_trade(series, ex_date, post_days, hold_days)
                    if r:
                        post_results.append({
                            "code": code, "ex_date": ex_date,
                            "post_days": post_days, **r
                        })

    return pd.DataFrame(pre_results), pd.DataFrame(post_results)


def print_summary(pre_df: pd.DataFrame, post_df: pd.DataFrame):
    print("\n" + "="*60)
    print("【A】配当落ち前N日買い → 前日売り（権利取り相場）")
    print("="*60)
    if pre_df.empty:
        print("データなし")
    else:
        g = pre_df.groupby("pre_days")["ret_pct"].agg(
            件数="count", 平均リターン="mean", 中央値="median",
            勝率=lambda x: (x > 0).mean() * 100,
            最大="max", 最小="min"
        ).round(2)
        print(g.to_string())

    print("\n" + "="*60)
    print("【B】配当落ち後M日買い → K日後売り（戻し買い）")
    print("="*60)
    if post_df.empty:
        print("データなし")
    else:
        g = post_df.groupby(["post_days", "hold_days"])["ret_pct"].agg(
            件数="count", 平均リターン="mean", 中央値="median",
            勝率=lambda x: (x > 0).mean() * 100,
            平均drop_from_ex=lambda x: post_df.loc[x.index, "drop_from_ex"].mean()
        ).round(2)
        print(g.to_string())

        print("\n--- 配当落ち後の平均価格変動（post_days別）---")
        drop_g = post_df.groupby("post_days")["drop_from_ex"].agg(
            平均下落率="mean", 中央値="median"
        ).round(2)
        print(drop_g.to_string())

    print("\n" + "="*60)
    print("【C】複合戦略: A + B をつなげた場合の合算リターン試算")
    print("="*60)
    if pre_df.empty or post_df.empty:
        print("データなし")
    else:
        best_pre  = pre_df[pre_df["pre_days"] == 20].groupby(["code","ex_date"])["ret_pct"].first()
        best_post = post_df[(post_df["post_days"] == 3) & (post_df["hold_days"] == 20)]\
                        .groupby(["code","ex_date"])["ret_pct"].first()
        combined  = (best_pre + best_post).dropna()
        print(f"  前20日買い + 落ち後3日買い/20日保有")
        print(f"  平均合算リターン : {combined.mean():.2f}%")
        print(f"  中央値           : {combined.median():.2f}%")
        print(f"  勝率             : {(combined > 0).mean()*100:.1f}%")
        print(f"  件数             : {len(combined)}")

    print("\n" + "="*60)
    print("【参考】決算月別の戻し買い成績（post_days=5, hold=30）")
    print("="*60)
    if not post_df.empty:
        sub = post_df[(post_df["post_days"] == 5) & (post_df["hold_days"] == 30)].copy()
        sub["ex_month"] = pd.to_datetime(sub["ex_date"]).dt.month
        g = sub.groupby("ex_month")["ret_pct"].agg(
            件数="count", 平均リターン="mean", 勝率=lambda x: (x > 0).mean() * 100
        ).round(2)
        print(g.to_string())


if __name__ == "__main__":
    print("配当落ち前後トレード戦略シミュレーション（各銘柄の実際の決算日ベース）")
    print()

    pre_df, post_df = run_simulation()

    print_summary(pre_df, post_df)

    # CSV保存
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "simulations")
    pre_df.to_csv(os.path.join(out_dir, "dividend_pre_sim.csv"), index=False)
    post_df.to_csv(os.path.join(out_dir, "dividend_post_sim.csv"), index=False)
    print(f"\nCSV保存: simulations/dividend_pre_sim.csv, dividend_post_sim.csv")

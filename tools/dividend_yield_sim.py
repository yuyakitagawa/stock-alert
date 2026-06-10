"""
配当利回り別の戻し買いシミュレーション

スプシの優待銘柄リストを使い、各銘柄の配当利回りを
Yahoo Finance から取得して利回り帯別に成績を比較する。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, sqlite3, calendar
import requests
import numpy as np
import pandas as pd
from datetime import date, datetime
from lib.utils import HEADERS

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(BASE_DIR, "stock_alert.db")
DIV_CACHE = os.path.join(BASE_DIR, "tools", "_div_cache.json")

# ── パラメータ ─────────────────────────────────────────
POST_DAYS = 5    # 配当落ち後N日で買い
HOLD_DAYS = 20   # 保有日数
YIELD_BINS = [(0, 1.5, "低利回り(<1.5%)"),
              (1.5, 3.0, "中利回り(1.5-3%)"),
              (3.0, 5.0, "高利回り(3-5%)"),
              (5.0, 99,  "超高利回り(5%+)")]
# ──────────────────────────────────────────────────────


def load_div_cache() -> dict:
    if os.path.exists(DIV_CACHE):
        with open(DIV_CACHE) as f:
            return json.load(f)
    return {}


def save_div_cache(cache: dict):
    with open(DIV_CACHE, "w") as f:
        json.dump(cache, f)


def fetch_dividends(code: str) -> dict[str, float]:
    """Yahoo Finance から過去2年の配当履歴 {日付str: 1株配当額} を返す"""
    ticker = f"{code}.T"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=3y&events=div"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        result = r.json().get("chart", {}).get("result", [])
        if not result:
            return {}
        divs = result[0].get("events", {}).get("dividends", {})
        return {
            datetime.fromtimestamp(int(ts)).date().isoformat(): d["amount"]
            for ts, d in divs.items()
        }
    except Exception:
        return {}


def get_all_dividends(codes: list[str]) -> dict[str, dict[str, float]]:
    """全銘柄の配当履歴を取得（キャッシュ利用）"""
    cache = load_div_cache()
    need = [c for c in codes if c not in cache]
    print(f"  配当データ: キャッシュ済み {len(cache)}件 / 取得中 {len(need)}件")
    for i, code in enumerate(need):
        cache[code] = fetch_dividends(code)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(need)} 取得中...")
            save_div_cache(cache)
        time.sleep(0.25)
    save_div_cache(cache)
    return cache


def load_all_prices() -> dict[str, pd.Series]:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT code, date, close FROM price_cache WHERE date >= '2023-01-01' ORDER BY code, date",
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


def load_biz_days() -> list:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT date FROM price_cache WHERE code='7203' ORDER BY date", conn)
    conn.close()
    return pd.to_datetime(df["date"]).sort_values().tolist()


def load_yutai_from_sheets() -> dict[str, list[int]]:
    import sys
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
        months = []
        for part in yutai_str.split(","):
            part = part.strip().replace("月", "")
            if part.isdigit():
                months.append(int(part))
        if months:
            result[code] = months
    return result


def build_ex_dates(code_months: dict[str, list[int]], biz_days: list) -> dict[str, list[date]]:
    result: dict[str, list[date]] = {}
    for code, months in code_months.items():
        ex_dates = []
        for year in [2023, 2024, 2025]:
            for month in months:
                last_day = calendar.monthrange(year, month)[1]
                month_end = pd.Timestamp(year, month, last_day)
                before = [d for d in biz_days if d <= month_end]
                if len(before) < 3:
                    continue
                ex_date = before[-2].date()
                if ex_date <= date.today():
                    ex_dates.append(ex_date)
        if ex_dates:
            result[code] = ex_dates
    return result


def nearest_div(div_hist: dict[str, float], ex_date: date, window_days: int = 10) -> float | None:
    """ex_date前後±window_days以内の配当を探す"""
    for delta in range(-window_days, window_days + 1):
        key = (ex_date + pd.Timedelta(days=delta)).isoformat()
        if key in div_hist:
            return div_hist[key]
    return None


def run():
    print("データ読み込み中...")
    all_prices = load_all_prices()
    biz_days   = load_biz_days()

    print("スプシから優待確定月を取得...")
    yutai_map  = load_yutai_from_sheets()
    ex_map     = build_ex_dates(yutai_map, biz_days)
    codes      = [c for c in ex_map if c in all_prices]
    print(f"  対象: {len(codes)}銘柄")

    print("配当データをYahoo Financeから取得...")
    div_data   = get_all_dividends(codes)

    print("\n各銘柄×配当落ち日でシミュレーション中...")
    records = []
    for code in codes:
        series   = all_prices[code]
        div_hist = div_data.get(code, {})

        for ex_date in ex_map[code]:
            # 配当落ち後POST_DAYS営業日目に買い
            biz_after = series[series.index >= pd.Timestamp(ex_date)]
            if len(biz_after) < POST_DAYS + HOLD_DAYS + 1:
                continue

            ex_px   = float(biz_after.iloc[0])
            buy_px  = float(biz_after.iloc[POST_DAYS])
            sell_px = float(biz_after.iloc[POST_DAYS + HOLD_DAYS])
            ret     = (sell_px - buy_px) / buy_px * 100

            # 配当額と利回り
            div_amt = nearest_div(div_hist, ex_date)
            if div_amt is None or div_amt <= 0:
                continue
            # 配当落ち前日の株価で利回り計算
            biz_before = series[series.index < pd.Timestamp(ex_date)]
            if biz_before.empty:
                continue
            pre_px = float(biz_before.iloc[-1])
            div_yield = div_amt / pre_px * 100

            records.append({
                "code":      code,
                "ex_date":   ex_date,
                "ex_month":  ex_date.month,
                "div_yield": div_yield,
                "div_amt":   div_amt,
                "pre_px":    pre_px,
                "buy_px":    buy_px,
                "sell_px":   sell_px,
                "ret_pct":   ret,
            })

    df = pd.DataFrame(records)
    print(f"  有効サンプル: {len(df)}件（配当データあり）")
    return df


def print_results(df: pd.DataFrame):
    print("\n" + "="*65)
    print(f"【配当利回り別 戻し買い成績】落ち後{POST_DAYS}日買い/{HOLD_DAYS}日保有")
    print("="*65)

    # 利回りビンを付ける
    def yield_bin(y):
        for lo, hi, label in YIELD_BINS:
            if lo <= y < hi:
                return label
        return "超高利回り(5%+)"

    df["yield_bin"] = df["div_yield"].apply(yield_bin)
    df["yield_bin"] = pd.Categorical(df["yield_bin"],
        categories=[lb for _, _, lb in YIELD_BINS], ordered=True)

    g = df.groupby("yield_bin", observed=True)["ret_pct"].agg(
        件数="count",
        平均リターン="mean",
        中央値="median",
        勝率=lambda x: (x > 0).mean() * 100,
        最大="max",
        最小="min",
    ).round(2)
    print(g.to_string())

    print("\n--- 利回り帯別の配当利回り分布 ---")
    dg = df.groupby("yield_bin", observed=True)["div_yield"].agg(
        平均利回り="mean", 中央値="median"
    ).round(2)
    print(dg.to_string())

    print("\n--- 利回りを連続値で見た相関 ---")
    corr = df[["div_yield", "ret_pct"]].corr().iloc[0, 1]
    print(f"  配当利回り vs 戻しリターン 相関係数: {corr:.3f}")

    print("\n--- 月別×利回り別（件数10以上のみ）---")
    g2 = df.groupby(["ex_month", "yield_bin"], observed=True)["ret_pct"].agg(
        件数="count", 平均="mean", 勝率=lambda x: (x > 0).mean() * 100
    ).round(2)
    g2 = g2[g2["件数"] >= 10]
    print(g2.to_string())

    # CSV保存
    out = os.path.join(BASE_DIR, "simulations", "dividend_yield_sim.csv")
    df.to_csv(out, index=False)
    print(f"\nCSV保存: simulations/dividend_yield_sim.csv")


if __name__ == "__main__":
    df = run()
    print_results(df)

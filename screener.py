import requests
import pandas as pd
import numpy as np
import time
import io
import os
import argparse
from datetime import datetime, timedelta

MIN_MOMENTUM     = 5.0
MAX_MOMENTUM     = 30.0
MIN_VOLATILITY   = 20.0
MAX_VOLATILITY   = 50.0
MIN_MOMENTUM_20D = -3.0
MIN_PRICE        = 300
MIN_VOL_RATIO    = 1.0
MIN_REL_STRENGTH = 0.05

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_tse_stock_list():
    """JPXから全上場銘柄を取得（ETF・REIT除外）"""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    print("銘柄リストを取得中...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        print(f"   ステータス: {resp.status_code} / サイズ: {len(resp.content)} bytes")
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()

        code_col = [c for c in df.columns if "コード" in c][0]
        name_col = [c for c in df.columns if "銘柄名" in c][0]
        result = df[[code_col, name_col]].copy()
        result.columns = ["code", "name"]
        result["code"] = result["code"].str.strip()

        market_col = [c for c in df.columns if "市場" in c or "商品区分" in c]
        if market_col:
            market_filter = df[market_col[0]].str.contains("内国株式", na=False)
            result = result[market_filter]
            result = result[result["code"].str.match(r"^\d{4}$")]
            print(f"   市場区分フィルター適用")
        else:
            result = result[result["code"].str.match(r"^\d{4}$")]
            result = result[~result["name"].str.contains(
                "ETF|REIT|投信|上場投資|ファンド", na=False
            )]
            print(f"   銘柄名フィルター適用（市場区分列なし）")

        print(f"   → ETF除外後: {len(result)} 銘柄")
        return result.reset_index(drop=True)

    except Exception as e:
        print(f"銘柄リスト取得失敗: {e}")
        return None


def get_prices(code, days=180):
    """Yahoo Finance から過去N日の終値・出来高を取得"""
    ticker = f"{code}.T"
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = (
            result[0].get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose", [])
        )
        volumes = (
            result[0].get("indicators", {})
            .get("quote", [{}])[0]
            .get("volume", [])
        )
        if not timestamps or not closes:
            return None
        idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo")
        df = pd.DataFrame({"Date": idx, "Close": closes, "Volume": volumes}).dropna(subset=["Close"])
        return df
    except Exception:
        return None


def get_nikkei_3m_return():
    """日経225の3ヶ月リターンを取得"""
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=180)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        closes = (
            result[0].get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose", [])
        )
        closes = [c for c in closes if c is not None]
        if len(closes) < 63:
            return None
        n3 = min(63, len(closes) - 1)
        return (closes[-1] - closes[-n3 - 1]) / closes[-n3 - 1]
    except Exception:
        return None


def calc_metrics(df, nikkei_return_3m=None):
    """モメンタム・ボラティリティ・出来高・相対強度・スコアを計算"""
    if df is None or len(df) < 30:
        return None

    prices = df["Close"].values
    n = len(prices)

    coef  = np.polyfit(np.arange(n), prices, 1)
    slope = coef[0]

    n3  = min(63,  n - 1)
    n20 = min(20,  n - 1)
    momentum_20d = (prices[-1] - prices[-n20 - 1]) / prices[-n20 - 1] * 100
    momentum_3m  = (prices[-1] - prices[-n3  - 1]) / prices[-n3  - 1] * 100

    vol = (np.diff(prices) / prices[:-1]).std() * np.sqrt(252) * 100

    vr2060 = 1.0
    if "Volume" in df.columns and len(df) >= 60:
        vols = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).values
        vol20 = vols[-20:].mean()
        vol60 = vols[-60:].mean()
        if vol60 > 0:
            vr2060 = vol20 / vol60

    rel_strength_3m = (momentum_3m / 100 - nikkei_return_3m) if nikkei_return_3m is not None else 0.0
    score = momentum_3m * vr2060 if slope > 0 else 0

    return {
        "momentum":        round(momentum_3m, 2),
        "momentum_20d":    round(momentum_20d, 2),
        "vol":             round(vol, 2),
        "vr2060":          round(vr2060, 3),
        "rel_strength_3m": round(rel_strength_3m, 4),
        "score":           round(score, 2),
        "close":           round(float(prices[-1]), 1),
        "slope_up":        bool(slope > 0),
    }


def apply_screener_v1(universe_df):
    mask = (
        (universe_df["momentum"]          >= MIN_MOMENTUM)
        & (universe_df["momentum"]        <= MAX_MOMENTUM)
        & (universe_df["momentum_20d"]    >= MIN_MOMENTUM_20D)
        & (universe_df["vol"]             >= MIN_VOLATILITY)
        & (universe_df["vol"]             <= MAX_VOLATILITY)
        & (universe_df["close"]           >= MIN_PRICE)
        & (universe_df["slope_up"])
        & (universe_df["vr2060"]          >= MIN_VOL_RATIO)
        & (universe_df["rel_strength_3m"] >= MIN_REL_STRENGTH)
    )
    return universe_df[mask].copy()


def _format_output(df):
    out = df[["code", "name", "close", "vr2060", "momentum", "vol", "score"]].rename(columns={
        "code":     "銘柄コード",
        "name":     "銘柄名",
        "close":    "直近株価(円)",
        "vr2060":   "出来高比(20d/60d)",
        "momentum": "3ヶ月モメンタム(%)",
        "vol":      "ボラティリティ(%)",
        "score":    "総合スコア",
    }).sort_values("総合スコア", ascending=False).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def _print_top10(out_df):
    print(f"\n上位10銘柄:")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<22}  {'3ヶ月モメンタム(%)':>14}")
    print("-" * 58)
    for _, r in out_df.head(10).iterrows():
        print(f"{int(r['順位']):>4}  {r['銘柄コード']:>6}  {r['銘柄名']:<22}  {r['3ヶ月モメンタム(%)']:>+13.1f}%")


def main():
    parser = argparse.ArgumentParser(description="TSE株価スクリーナー（v1）")
    parser.add_argument("--test", action="store_true", help="5銘柄のみ処理するテストモード")
    args = parser.parse_args()

    print("=" * 55)
    print("TSE 株価スクリーナー  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    if args.test:
        print("【テストモード: 5銘柄のみ】")
    print("=" * 55)

    print("日経225の3ヶ月リターン取得中...")
    nikkei_return_3m = get_nikkei_3m_return()
    if nikkei_return_3m is not None:
        print(f"  日経225 3ヶ月リターン: {nikkei_return_3m*100:+.1f}%")
    else:
        print("  日経225データ取得失敗 → 相対強度条件をスキップ")

    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    if args.test:
        stock_list = stock_list.head(5)

    total  = len(stock_list)
    rows   = []
    errors = 0

    print(f"\n{total} 銘柄をスクリーニング中...")

    for i, row in stock_list.iterrows():
        code = row["code"]
        name = row["name"]

        df     = get_prices(code, days=180)
        result = calc_metrics(df, nikkei_return_3m)

        if result is None:
            errors += 1
            if args.test:
                print(f"  ❓ {name}({code}): データ取得失敗")
            continue

        rows.append({
            "code":            code,
            "name":            name,
            "momentum":        result["momentum"],
            "momentum_20d":    result["momentum_20d"],
            "vol":             result["vol"],
            "vr2060":          result["vr2060"],
            "rel_strength_3m": result["rel_strength_3m"],
            "score":           result["score"],
            "close":           result["close"],
            "slope_up":        result["slope_up"],
        })

        if args.test:
            print(
                f"  {name}({code}): "
                f"モメンタム={result['momentum']:+.1f}% / "
                f"出来高比={result['vr2060']:.2f} / "
                f"相対強度={result['rel_strength_3m']*100:+.1f}%"
            )

        if not args.test and (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} 処理済み... (収集: {len(rows)}銘柄)")

        time.sleep(0.1)

    if not rows:
        print("\nERROR: データ取得失敗")
        return

    universe_df = pd.DataFrame(rows)
    date_str    = datetime.now().strftime("%Y%m%d")

    filtered = apply_screener_v1(universe_df)
    if not filtered.empty:
        out      = _format_output(filtered)
        out_path = f"screener_v1_{date_str}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        out.to_csv(f"screener_{date_str}.csv", index=False, encoding="utf-8-sig")
        print(f"\n{len(filtered)} 銘柄通過 → {out_path}")
        _print_top10(out)
    else:
        print("\n通過銘柄なし")

    print(f"\n{'='*55}")
    print(f"完了: 対象 {total} 銘柄 / 取得失敗 {errors} 銘柄")


if __name__ == "__main__":
    main()

import requests
import pandas as pd
import numpy as np
import time
import io
import sys
from datetime import datetime, timedelta

R2_THRESHOLD = 0.65       # 0.70→0.65: 押し目のある上昇トレンドも取り込む
MIN_MOMENTUM = 5.0
MAX_VOLATILITY = 50.0
MIN_MOMENTUM_20D = -3.0   # 20日モメンタム下限（直近失速銘柄を除外）
MIN_PRICE = 300           # 低位株フィルター（300円未満除外）

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_tse_stock_list():
    """JPXから全上場銘柄を取得（ETF・REIT除外、1000番台含む）"""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    print("銘柄リストを取得中...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        print(f"   ステータス: {resp.status_code} / サイズ: {len(resp.content)} bytes")
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()

        # コード列・銘柄名列を探す
        code_col = [c for c in df.columns if "コード" in c][0]
        name_col = [c for c in df.columns if "銘柄名" in c][0]
        result = df[[code_col, name_col]].copy()
        result.columns = ["code", "name"]
        result["code"] = result["code"].str.strip()

        # 市場区分列でETF・REIT除外（内国株式のみ残す）
        market_col = [c for c in df.columns if "市場" in c or "商品区分" in c]
        if market_col:
            market_filter = df[market_col[0]].str.contains("内国株式", na=False)
            result = result[market_filter]
            result = result[result["code"].str.match(r"^\d{4}$")]
            print(f"   市場区分フィルター適用")
        else:
            # フォールバック：ETF・REITっぽい銘柄名を除外
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
    """Yahoo Finance から過去N日の終値を取得"""
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
        if not timestamps or not closes:
            return None
        idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo")
        df = pd.DataFrame({"Date": idx, "Close": closes}).dropna()
        return df
    except Exception:
        return None


def calc_score(df):
    """R²・モメンタム・ボラティリティ・スコアを計算"""
    if df is None or len(df) < 30:
        return None

    prices = df["Close"].values
    n = len(prices)
    x = np.arange(n)

    # R²（線形トレンドの安定度）
    x_mean = x.mean()
    y_mean = prices.mean()
    ss_tot = ((prices - y_mean) ** 2).sum()
    ss_res = ((prices - (np.polyval(np.polyfit(x, prices, 1), x))) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 傾き（上昇トレンドか）
    slope = np.polyfit(x, prices, 1)[0]

    # 3ヶ月モメンタム（過去63営業日）
    n_mom = min(63, n - 1)
    n_mom20 = min(20, n - 1)
    momentum_20d = (prices[-1] - prices[-n_mom20 - 1]) / prices[-n_mom20 - 1] * 100
    momentum = (prices[-1] - prices[-n_mom - 1]) / prices[-n_mom - 1] * 100

    # ボラティリティ（年率換算%）
    daily_returns = np.diff(prices) / prices[:-1]
    vol = daily_returns.std() * np.sqrt(252) * 100

    # 総合スコア（R² × モメンタム、上昇トレンドのみ）
    score = r2 * momentum if slope > 0 else 0

    return {
        "r2": round(r2, 3),
        "momentum": round(momentum, 2),
        "vol": round(vol, 2),
        "score": round(score, 2),
        "close": round(float(prices[-1]), 1),
        "momentum_20d": round(momentum_20d, 2),
        "slope_up": slope > 0
    }


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "test"

    print("=" * 55)
    print("TSE 株価スクリーナー  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"条件: R²≥{R2_THRESHOLD} / モメンタム≥{MIN_MOMENTUM}% / ボラ≤{MAX_VOLATILITY}%")
    if test_mode:
        print("【テストモード: 5銘柄のみ】")
    print("=" * 55)

    # 銘柄リスト取得
    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    if test_mode:
        stock_list = stock_list.head(5)

    total = len(stock_list)
    passed = []
    errors = 0

    print(f"\n{total} 銘柄をスクリーニング中...")

    for i, row in stock_list.iterrows():
        code = row["code"]
        name = row["name"]

        df = get_prices(code, days=180)
        result = calc_score(df)

        if result is None:
            errors += 1
            if test_mode:
                print(f"  ❓ {name}({code}): データ取得失敗")
            continue

        # スクリーニング条件
        if (
            result["r2"] >= R2_THRESHOLD
            and result["momentum"] >= MIN_MOMENTUM
            and result["momentum_20d"] >= MIN_MOMENTUM_20D
            and result["vol"] <= MAX_VOLATILITY
            and result["close"] >= MIN_PRICE
            and result["slope_up"]
        ):
            passed.append({
                "銘柄コード": code,
                "銘柄名": name,
                "直近株価(円)": result["close"],
                "R²（安定度）": result["r2"],
                "3ヶ月モメンタム(%)": result["momentum"],
                "ボラティリティ(%)": result["vol"],
                "総合スコア": result["score"],
            })
            if test_mode:
                print(
                    f"  ✅ {name}({code}): "
                    f"R²={result['r2']} / "
                    f"モメンタム={result['momentum']:+.1f}% / "
                    f"ボラ={result['vol']:.1f}% / "
                    f"スコア={result['score']:.2f}"
                )
        else:
            if test_mode:
                print(
                    f"  ✗  {name}({code}): "
                    f"R²={result['r2']} / "
                    f"モメンタム={result['momentum']:+.1f}% / "
                    f"ボラ={result['vol']:.1f}%"
                )

        # 進捗表示（本番モード）
        if not test_mode and (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} 処理済み... (通過: {len(passed)}銘柄)")

        time.sleep(0.1)

    # 結果保存
    if passed:
        result_df = pd.DataFrame(passed)
        result_df = result_df.sort_values("総合スコア", ascending=False).reset_index(drop=True)
        result_df.insert(0, "順位", range(1, len(result_df) + 1))

        date_str = datetime.now().strftime("%Y%m%d")
        out_path = f"screener_{date_str}.csv"
        result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n{'='*55}")
        print(f"スクリーニング完了")
        print(f"  対象: {total} 銘柄 / 通過: {len(passed)} 銘柄 / 取得失敗: {errors} 銘柄")
        print(f"  保存: {out_path}")
        print(f"\n上位10銘柄:")
        print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<22}  {'モメンタム':>10}  {'スコア':>8}")
        print("-" * 60)
        for _, r in result_df.head(10).iterrows():
            print(
                f"{int(r['順位']):>4}  {r['銘柄コード']:>6}  "
                f"{r['銘柄名']:<22}  "
                f"{r['3ヶ月モメンタム(%)']:>+9.1f}%  "
                f"{r['総合スコア']:>8.2f}"
            )
    else:
        print("\n通過銘柄なし")


if __name__ == "__main__":
    main()
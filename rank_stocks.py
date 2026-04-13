import requests
import pandas as pd
import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib

FORECAST = 63
RISE_THRESHOLD = 15.0
TOP_SHOW = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_prices(code, days=400):
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
        df = pd.DataFrame(
            {"Close": closes},
            index=pd.to_datetime(timestamps, unit="s", utc=True)
        )
        return df.dropna()
    except Exception:
        return None


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


def extract_features(p):
    """最新時点の特徴量を返す（rf_predict.pyと同じ計算）"""
    if len(p) < 91:
        return None
    current = p[-1]
    if current == 0:
        return None

    ret5  = (current - p[-6])  / p[-6]  if len(p) >= 6  else 0
    ret20 = (current - p[-21]) / p[-21] if len(p) >= 21 else 0
    ret60 = (current - p[-61]) / p[-61] if len(p) >= 61 else 0
    ret90 = (current - p[-91]) / p[-91]

    ma5  = p[-5:].mean()
    ma25 = p[-25:].mean() if len(p) >= 25 else p.mean()
    ma75 = p[-75:].mean() if len(p) >= 75 else p.mean()
    ma5_25  = ma5  / ma25 - 1 if ma25 > 0 else 0
    ma25_75 = ma25 / ma75 - 1 if ma75 > 0 else 0

    rsi = calc_rsi(p, period=14)

    dr20 = np.diff(p[-21:]) / p[-21:-1] if len(p) >= 21 else np.array([0])
    vol20 = dr20.std() * np.sqrt(252) * 100

    dr60 = np.diff(p[-61:]) / p[-61:-1] if len(p) >= 61 else np.array([0])
    vol60 = dr60.std() * np.sqrt(252) * 100

    week52 = p[-252:] if len(p) >= 252 else p
    hi, lo = week52.max(), week52.min()
    pos52 = (current - lo) / (hi - lo) if hi > lo else 0.5

    return [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52]


def main():
    print("=" * 55)
    print("スクリーナー × RF ランキング  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"スクリーナー通過銘柄に上昇確率スコアをつけてランキング")
    print("=" * 55)

    # モデル読み込み
    model_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    if not os.path.exists(model_path):
        print("ERROR: rf_model.pkl が見つかりません。先に rf_predict.py を実行してください")
        return
    model = joblib.load(model_path)
    print(f"\nモデル読み込み: {model_path}")

    # スクリーナーCSV読み込み
    files = glob.glob(os.path.expanduser("~/stock-alert/screener_*.csv"))
    if not files:
        print("ERROR: screener CSVが見つかりません")
        return
    screener_df = pd.read_csv(max(files, key=os.path.getmtime))
    codes = screener_df["銘柄コード"].astype(str).tolist()
    names = dict(zip(screener_df["銘柄コード"].astype(str), screener_df["銘柄名"]))
    print(f"スクリーナー通過銘柄: {len(codes)} 銘柄")
    print(f"\n確率スコア計算中...")

    # 各銘柄の確率を計算
    results = []
    for i, code in enumerate(codes):
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            continue
        feat = extract_features(prices["Close"].values)
        if feat is None:
            continue
        prob = float(model.predict_proba([feat])[0][1])
        close = float(prices["Close"].iloc[-1])
        results.append({
            "銘柄コード": code,
            "銘柄名": names.get(code, ""),
            "直近株価(円)": round(close, 1),
            "上昇確率(%)": round(prob * 100, 1),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)} 処理済み...")
        time.sleep(0.2)

    # ランキング
    result_df = pd.DataFrame(results).sort_values("上昇確率(%)", ascending=False).reset_index(drop=True)
    result_df.index += 1
    result_df.insert(0, "順位", result_df.index)

    # 表示
    print(f"\n{'='*55}")
    print(f"上位{TOP_SHOW}銘柄ランキング（上昇確率順）")
    print(f"{'='*55}")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<20}  {'株価':>8}  {'上昇確率':>8}")
    print("-" * 55)
    for _, row in result_df.head(TOP_SHOW).iterrows():
        print(
            f"{int(row['順位']):>4}  {row['銘柄コード']:>6}  "
            f"{row['銘柄名']:<20}  "
            f"{row['直近株価(円)']:>8,.0f}円  "
            f"{row['上昇確率(%)']:>6.1f}%"
        )

    # CSV保存
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.expanduser(f"~/stock-alert/ranking_{date_str}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")
    print("完了")


if __name__ == "__main__":
    main()

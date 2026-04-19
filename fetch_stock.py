import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

STOCKS = {
    "9434": "ソフトバンク",
    "4689": "LY Corporation(旧ヤフー)",
    "6098": "リクルート",
    "8001": "伊藤忠商事",
    "8058": "三菱商事",
    "7203": "トヨタ",
    "7201": "日産",
    "4452": "花王"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

LOOKBACK       = 60    # 過去60日を入力として使う
FORECAST_DAYS  = 5     # 5日後を予測
DROP_THRESHOLD = -3.0  # 下落アラート閾値（%）

def get_historical_prices(code, days=600):
    """過去N日分の終値を取得"""
    ticker = f"{code}.T"
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = (result[0].get("indicators", {})
                           .get("adjclose", [{}])[0]
                           .get("adjclose", []))
        df = pd.DataFrame({"Date": pd.to_datetime(timestamps, unit="s"),
                           "Close": closes}).dropna()
        return df
    except Exception as e:
        print(f"  取得エラー: {e}")
        return None

def prepare_data(df):
    """LSTMの入力データと正解ラベルを作成"""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(LOOKBACK, len(scaled) - FORECAST_DAYS):
        X.append(scaled[i - LOOKBACK:i, 0])
        future_return = ((df["Close"].iloc[i + FORECAST_DAYS] - df["Close"].iloc[i])
                         / df["Close"].iloc[i] * 100)
        y.append(1 if future_return <= DROP_THRESHOLD else 0)

    return np.array(X), np.array(y), scaler

def build_model():
    """LSTMモデルを構築"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def predict_drop(code, name):
    """1銘柄について5日以内の-3%超下落確率を予測"""
    print(f"  {name}({code}): データ取得中...", end=" ", flush=True)
    df = get_historical_prices(code)
    if df is None or len(df) < LOOKBACK + FORECAST_DAYS + 50:
        print("データ不足")
        return None
    print(f"{len(df)}日分取得 → 学習中...", end=" ", flush=True)

    X, y, scaler = prepare_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split   = int(len(X) * 0.8)
    X_train = X[:split];  y_train = y[:split]
    X_test  = X[split:];  y_test  = y[split:]

    model = build_model()
    model.fit(X_train, y_train,
              epochs=30,
              batch_size=32,
              verbose=0,
              validation_data=(X_test, y_test))

    # 直近60日で予測
    scaler2 = MinMaxScaler()
    scaler2.fit_transform(df[["Close"]])
    scaled_all = scaler.transform(df[["Close"]])
    X_latest   = scaled_all[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    prob       = float(model.predict(X_latest, verbose=0)[0][0])

    print(f"完了 → 下落確率: {prob*100:.1f}%")
    return {
        "code":             code,
        "name":             name,
        "latest_close":     round(df["Close"].iloc[-1], 1),
        "drop_probability": round(prob * 100, 1),
        "is_alert":         prob >= 0.5
    }

# ===== 実行 =====
print("=" * 50)
print(f"LSTM 下落予測  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"予測: {FORECAST_DAYS}日以内に{DROP_THRESHOLD}%超下落する確率")
print("=" * 50)

results = []
for code, name in STOCKS.items():
    result = predict_drop(code, name)
    if result:
        results.append(result)
    time.sleep(1)

print()
print("=" * 50)
print("【予測結果】")
for r in results:
    mark = "⚠️ " if r["is_alert"] else "✅ "
    print(f"{mark} {r['name']}({r['code']}): "
          f"下落確率 {r['drop_probability']}%  終値 {r['latest_close']}円")

alerts = [r for r in results if r["is_alert"]]
print()
if alerts:
    print(f"⚠️  {len(alerts)}件で下落シグナルが出ています")
else:
    print("✅  下落シグナルなし")
print("=" * 50)
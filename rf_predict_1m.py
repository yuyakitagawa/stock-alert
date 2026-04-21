import requests
import pandas as pd
import numpy as np
import time
import os
import io
import random
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

FORECAST = 21
RISE_THRESHOLD = 7.0
SELL_SIGNAL_PROB = 0.4
SAMPLE_N = 500
RANDOM_SEED = 42

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

HELD_STOCKS = {
    "9434": "ソフトバンク",
    "4689": "LY Corporation",
    "6098": "リクルート",
    "8001": "伊藤忠商事",
    "8058": "三菱商事",
    "7203": "トヨタ",
    "7201": "日産",
    "4452": "花王"
}

FEATURE_NAMES = [
    "ret5", "ret20", "ret60", "ret90",
    "ma5_25ratio", "ma25_75ratio",
    "rsi14", "vol20", "vol60", "position52w"
]


def get_tse_stock_list():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()
        code_col = [c for c in df.columns if "コード" in c][0]
        name_col = [c for c in df.columns if "銘柄名" in c][0]
        result = df[[code_col, name_col]].copy()
        result.columns = ["code", "name"]
        result["code"] = result["code"].str.strip()
        result = result[result["code"].str.match(r"^[2-9]\d{3}$")]
        return result.reset_index(drop=True)
    except Exception as e:
        print(f"  銘柄リスト取得失敗: {e}")
        return None


def get_prices(code, days=800):
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
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def compute_features(prices_arr):
    """1銘柄の全時点の特徴量とラベルを生成"""
    X_train, y_train = [], []
    X_test, y_test = [], []
    n = len(prices_arr)
    split_idx = int(n * 0.8)

    for i in range(90, n - FORECAST):
        p = prices_arr[:i + 1]
        current = p[-1]
        if current == 0:
            continue

        # リターン
        ret5  = (current - p[-6])  / p[-6]  if len(p) >= 6  else 0
        ret20 = (current - p[-21]) / p[-21] if len(p) >= 21 else 0
        ret60 = (current - p[-61]) / p[-61] if len(p) >= 61 else 0
        ret90 = (current - p[-91]) / p[-91] if len(p) >= 91 else 0

        # 移動平均比率
        ma5  = p[-5:].mean()
        ma25 = p[-25:].mean() if len(p) >= 25 else p.mean()
        ma75 = p[-75:].mean() if len(p) >= 75 else p.mean()
        ma5_25  = ma5  / ma25 - 1 if ma25 > 0 else 0
        ma25_75 = ma25 / ma75 - 1 if ma75 > 0 else 0

        # RSI
        rsi = calc_rsi(p, period=14)

        # ボラティリティ（年率換算%）
        if len(p) >= 21:
            dr20 = np.diff(p[-21:]) / p[-21:-1]
            vol20 = dr20.std() * np.sqrt(252) * 100
        else:
            vol20 = 30.0
        if len(p) >= 61:
            dr60 = np.diff(p[-61:]) / p[-61:-1]
            vol60 = dr60.std() * np.sqrt(252) * 100
        else:
            vol60 = 30.0

        # 52週レンジ内のポジション
        week52 = p[-252:] if len(p) >= 252 else p
        hi, lo = week52.max(), week52.min()
        pos52 = (current - lo) / (hi - lo) if hi > lo else 0.5

        feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52]

        # ラベル（上昇・下落それぞれ）
        future = prices_arr[i + FORECAST]
        change = (future - current) / current * 100
        label_rise = 1 if change >= RISE_THRESHOLD else 0
        label_drop = 1 if change <= -RISE_THRESHOLD else 0

        if i < split_idx:
            X_train.append(feat)
            y_train.append((label_rise, label_drop))
        else:
            X_test.append(feat)
            y_test.append((label_rise, label_drop))

    y_train_rise = [y[0] for y in y_train]
    y_train_drop = [y[1] for y in y_train]
    y_test_rise  = [y[0] for y in y_test]
    y_test_drop  = [y[1] for y in y_test]
    return X_train, y_train_rise, y_train_drop, X_test, y_test_rise, y_test_drop


def predict_latest(model, prices_arr):
    """最新時点の特徴量で上昇確率を返す"""
    p = prices_arr
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

    if len(p) >= 21:
        dr20 = np.diff(p[-21:]) / p[-21:-1]
        vol20 = dr20.std() * np.sqrt(252) * 100
    else:
        vol20 = 30.0
    if len(p) >= 61:
        dr60 = np.diff(p[-61:]) / p[-61:-1]
        vol60 = dr60.std() * np.sqrt(252) * 100
    else:
        vol60 = 30.0

    week52 = p[-252:] if len(p) >= 252 else p
    hi, lo = week52.max(), week52.min()
    pos52 = (current - lo) / (hi - lo) if hi > lo else 0.5

    feat = [[ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52]]
    return float(model.predict_proba(feat)[0][1])


def main():
    print("=" * 55)
    print("XGBoost 上昇・下落予測  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"学習: 全銘柄からランダム{SAMPLE_N}銘柄")
    print(f"特徴量: リターン・移動平均・RSI・ボラティリティ・52週レンジ")
    print(f"予測: {FORECAST}営業日後に+{RISE_THRESHOLD}%以上上昇する確率")
    print("=" * 55)

    # Step 1: 銘柄リスト取得
    print(f"\n[Step 1] 全銘柄リストを取得中...")
    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    sampled = stock_list.sample(n=min(SAMPLE_N, len(stock_list)), random_state=RANDOM_SEED)
    codes = sampled["code"].tolist()
    print(f"  全{len(stock_list)}銘柄 → ランダム{len(codes)}銘柄を学習に使用")

    # Step 2: 特徴量生成
    print(f"\n[Step 2] データ取得・特徴量生成中...")
    all_X_train, all_yr_train, all_yd_train = [], [], []
    all_X_test,  all_yr_test,  all_yd_test  = [], [], []
    success = 0

    for i, code in enumerate(codes):
        prices = get_prices(code, days=800)
        if prices is None or len(prices) < 90 + FORECAST + 10:
            continue
        Xtr, ytr_r, ytr_d, Xte, yte_r, yte_d = compute_features(prices["Close"].values)
        if len(Xtr) == 0:
            continue
        all_X_train.extend(Xtr)
        all_yr_train.extend(ytr_r)
        all_yd_train.extend(ytr_d)
        all_X_test.extend(Xte)
        all_yr_test.extend(yte_r)
        all_yd_test.extend(yte_d)
        success += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)} 処理済み (学習サンプル: {len(all_X_train)})")
        time.sleep(0.3)

    X_train   = np.array(all_X_train)
    yr_train  = np.array(all_yr_train)
    yd_train  = np.array(all_yd_train)
    X_test    = np.array(all_X_test)
    yr_test   = np.array(all_yr_test)
    yd_test   = np.array(all_yd_test)

    print(f"\n  取得成功: {success}/{len(codes)} 銘柄")
    print(f"  学習: {len(X_train)} サンプル / テスト: {len(X_test)} サンプル")
    print(f"  上昇ラベル率: {yr_train.mean()*100:.1f}% / 下落ラベル率: {yd_train.mean()*100:.1f}%")

    def train_model(X_tr, y_tr, X_te, y_te, label_name):
        print(f"\n[Step 3] {label_name}モデル学習中 (XGBoost)...")
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = neg / pos if pos > 0 else 1.0  # クラス不均衡補正
        print(f"  pos={int(pos)}, neg={int(neg)}, scale_pos_weight={spw:.2f}")
        m = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        y_prob = m.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        print(f"  Accuracy: {acc:.3f} / AUC: {auc:.3f}")
        return m

    rise_model = train_model(X_train, yr_train, X_test, yr_test, "上昇")
    drop_model = train_model(X_train, yd_train, X_test, yd_test, "下落")

    rise_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    drop_path = os.path.expanduser("~/stock-alert/rf_drop_model.pkl")
    joblib.dump(rise_model, rise_path)
    joblib.dump(drop_model, drop_path)
    print(f"\n  上昇モデル保存: {rise_path}")
    print(f"  下落モデル保存: {drop_path}")

    # Step 4: チェック銘柄への適用
    csv_path = os.path.expanduser("~/stock-alert/watch_list.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        wl = pd.read_csv(csv_path, dtype=str)
        held = dict(zip(wl["コード"].str.strip(), wl["銘柄名"].str.strip()))
    else:
        held = HELD_STOCKS

    print("\n[Step 4] チェック銘柄 判定")
    print("-" * 55)
    for code, name in held.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        rise_prob = predict_latest(rise_model, prices["Close"].values)
        drop_prob = predict_latest(drop_model, prices["Close"].values)
        if rise_prob is None:
            continue
        rise_pct = rise_prob * 100
        drop_pct = drop_prob * 100 if drop_prob is not None else 0
        net = rise_pct - drop_pct
        if net >= 15:
            signal = "🟢 強気買い  "
        elif net >= 5:
            signal = "🔵 やや強気  "
        elif net >= -5:
            signal = "🟡 中立      "
        elif net >= -15:
            signal = "🟠 やや弱気  "
        else:
            signal = "🔴 売り検討  "
        print(f"  {signal} {name}({code}): 上昇{rise_pct:5.1f}% 下落{drop_pct:5.1f}% ネット{net:+.1f}%")
        time.sleep(0.3)

    print("\n完了")


if __name__ == "__main__":
    main()
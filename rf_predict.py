import requests
import pandas as pd
import numpy as np
import time
import os
import io
import random
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

FORECAST = 63
RISE_THRESHOLD = 15.0
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

        # ラベル
        future = prices_arr[i + FORECAST]
        label = 1 if (future - current) / current * 100 >= RISE_THRESHOLD else 0

        if i < split_idx:
            X_train.append(feat)
            y_train.append(label)
        else:
            X_test.append(feat)
            y_test.append(label)

    return X_train, y_train, X_test, y_test


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
    print("Random Forest 上昇予測  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
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
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    success = 0

    for i, code in enumerate(codes):
        prices = get_prices(code, days=800)
        if prices is None or len(prices) < 90 + FORECAST + 10:
            continue
        Xtr, ytr, Xte, yte = compute_features(prices["Close"].values)
        if len(Xtr) == 0:
            continue
        all_X_train.extend(Xtr)
        all_y_train.extend(ytr)
        all_X_test.extend(Xte)
        all_y_test.extend(yte)
        success += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)} 処理済み (学習サンプル: {len(all_X_train)})")
        time.sleep(0.3)

    X_train = np.array(all_X_train)
    y_train = np.array(all_y_train)
    X_test  = np.array(all_X_test)
    y_test  = np.array(all_y_test)

    pos_rate = y_train.mean() * 100
    print(f"\n  取得成功: {success}/{len(codes)} 銘柄")
    print(f"  学習: {len(X_train)} サンプル / テスト: {len(X_test)} サンプル")
    print(f"  上昇ラベル率: {pos_rate:.1f}%  (30〜50%が理想)")

    # Step 3: 学習
    print("\n[Step 3] Random Forest 学習中...")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes.astype(int), weights))

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight=cw,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  学習完了")

    # Step 4: 精度評価
    print("\n[Step 4] テスト精度評価...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Accuracy : {acc:.3f}")
    print(f"  AUC      : {auc:.3f}  (0.5=ランダム / 0.7以上で実用的)")
    print(classification_report(y_test, y_pred, target_names=["下落(0)", "上昇(1)"]))

    # 特徴量の重要度
    print("  特徴量の重要度:")
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"    {name:15s} {imp:.3f} {bar}")

    model_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  モデル保存: {model_path}")

    # Step 5: 保有株への適用
    print("\n[Step 5] 保有株 売りシグナル判定")
    print("-" * 45)
    for code, name in HELD_STOCKS.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        prob = predict_latest(model, prices["Close"].values)
        if prob is None:
            continue
        signal = "⚠️  売り検討" if prob < SELL_SIGNAL_PROB else "✅ 保持"
        print(f"  {signal}  {name}({code}): 上昇確率 {prob * 100:.1f}%")
        time.sleep(0.3)

    print("\n完了")


if __name__ == "__main__":
    main()
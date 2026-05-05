"""
analyze_alpha.py
スクリーナーとモデルそれぞれのアルファ貢献度を分離して分析する。

スクリーナー通過銘柄を BACKTEST_DATE 時点のモデルスコアで4グループに分け、
実際のリターンを比較することで
  - スクリーナー単体のアルファ（ランダムグループ vs 日経）
  - モデルの追加価値（上位グループ vs ランダムグループ）
を定量化する。
"""

import os
import glob
import time
import random
import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date, timedelta

from utils import calc_rsi, compute_seq_features, HEADERS, SEQ_DAYS

import sys as _sys

BACKTEST_DATE = date(2026, 2, 3)
TODAY         = date.today()

if len(_sys.argv) > 1 and _sys.argv[1] == "bear":
    BACKTEST_DATE = date(2024, 7, 1)
    TODAY         = date(2024, 10, 1)
    print("【下落相場テストモード: 2024年8月クラッシュ期】")

RANDOM_SEED = 42
# BACKTEST_DATE の252営業日前（≈365日）のデータが必要なため十分な日数を確保
FETCH_DAYS  = max(500, (date.today() - BACKTEST_DATE).days + 400)


# ── 株価取得 ──────────────────────────────────
def fetch_yahoo(ticker):
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=FETCH_DAYS)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        data   = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        ts      = result[0].get("timestamp", [])
        closes  = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        volumes = result[0].get("indicators", {}).get("quote",    [{}])[0].get("volume",   [])
        if not ts or not closes:
            return None
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Tokyo").normalize()
        df  = pd.DataFrame({"Close": closes, "Volume": volumes}, index=idx)
        df  = df.dropna(subset=["Close"])
        df.index = df.index.date
        return df
    except Exception:
        return None


# ── BACKTEST_DATE 時点の81次元特徴量計算 ─────────
def extract_features_at(hist, target_date, nk_rets=None):
    close  = hist["Close"].dropna()
    volume = hist["Volume"].dropna()
    close.index  = pd.to_datetime(close.index).date
    volume.index = pd.to_datetime(volume.index).date

    past_c = close[close.index <= target_date]
    past_v = volume[volume.index <= target_date]
    if len(past_c) < 91:
        return None

    p = past_c.values
    v = past_v.values
    c = p[-1]
    if c == 0:
        return None

    ret5  = (c - p[-6])  / p[-6]  if len(p) >= 6  else 0
    ret20 = (c - p[-21]) / p[-21] if len(p) >= 21 else 0
    ret60 = (c - p[-61]) / p[-61] if len(p) >= 61 else 0
    ret90 = (c - p[-91]) / p[-91]

    ma5  = p[-5:].mean()
    ma25 = p[-25:].mean() if len(p) >= 25 else p.mean()
    ma75 = p[-75:].mean() if len(p) >= 75 else p.mean()
    ma5_25  = ma5  / ma25 - 1 if ma25 > 0 else 0
    ma25_75 = ma25 / ma75 - 1 if ma75 > 0 else 0

    rsi   = calc_rsi(p)
    vol20 = (np.diff(p[-21:]) / p[-21:-1] if len(p) >= 21 else np.array([0])).std() * np.sqrt(252) * 100
    vol60 = (np.diff(p[-61:]) / p[-61:-1] if len(p) >= 61 else np.array([0])).std() * np.sqrt(252) * 100

    week52 = p[-252:] if len(p) >= 252 else p
    hi, lo = week52.max(), week52.min()
    pos52  = (c - lo) / (hi - lo) if hi > lo else 0.5

    rhi        = p[-60:].max() if len(p) >= 60 else p.max()
    drawdown60 = (c - rhi) / rhi
    hi52       = p[-252:].max() if len(p) >= 252 else p.max()
    from_hi52  = (c - hi52) / hi52
    stk = 0
    for j in range(1, min(21, len(p))):
        if p[-j] < p[-j - 1]: stk += 1
        else: break
    down_streak    = stk / 20.0
    momentum_accel = ret5 - (ret20 / 4)
    ma5_5ago  = p[-10:-5].mean() if len(p) >= 10 else ma5
    ma25_5ago = p[-30:-5].mean() if len(p) >= 30 else ma25
    cross_prev   = ma5_5ago / ma25_5ago - 1 if ma25_5ago > 0 else 0
    ma_cross_dir = ma5_25 - cross_prev

    if v is not None and len(v) >= 20:
        va     = np.array([x if x is not None else np.nan for x in v], dtype=float)
        va5    = np.nanmean(va[-5:])  if len(va) >= 5  else 1
        va20   = np.nanmean(va[-20:]) if len(va) >= 20 else 1
        va60   = np.nanmean(va[-60:]) if len(va) >= 60 else va20
        vr520  = va5  / va20 if va20 > 0 else 1.0
        vr2060 = va20 / va60 if va60 > 0 else 1.0
        vsurge = va[-1] / va20 if va20 > 0 and not np.isnan(va[-1]) else 1.0
    else:
        vr520, vr2060, vsurge = 1.0, 1.0, 1.0

    nk5  = nk_rets[0] if nk_rets is not None else 0.0
    nk20 = nk_rets[1] if nk_rets is not None else 0.0
    nk60 = nk_rets[2] if nk_rets is not None else 0.0

    seq_raw = (np.clip(np.diff(p[-(SEQ_DAYS+1):]) / p[-(SEQ_DAYS+1):-1], -0.2, 0.2)
               if len(p) >= SEQ_DAYS + 1 else np.zeros(SEQ_DAYS))

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + compute_seq_features(seq_raw)

    if any(np.isnan(feat[:10])) or any(np.isinf(feat[:10])):
        return None
    return feat


def group_stats(gdf, label, nk_return):
    avg  = gdf["実績リターン(%)"].mean()
    med  = gdf["実績リターン(%)"].median()
    wins = (gdf["実績リターン(%)"] > 0).sum()
    vs_nk = (avg - nk_return) if nk_return is not None else float("nan")
    print(f"\n【{label}】  n={len(gdf)}")
    print(f"  ネットスコア平均 : {gdf['ネット(%)'].mean():+.1f}%")
    print(f"  実績リターン 平均: {avg:+.2f}%  中央値: {med:+.2f}%")
    print(f"  勝率             : {wins}/{len(gdf)} = {wins/len(gdf)*100:.0f}%")
    if nk_return is not None:
        print(f"  vs 日経225       : {vs_nk:+.2f}%")
    for _, row in gdf.iterrows():
        mark = "✅" if row["実績リターン(%)"] >= 15 else ("🔵" if row["実績リターン(%)"] > 0 else "❌")
        print(f"    {row['コード']}  {str(row['銘柄名']):<20}  "
              f"ネット{row['ネット(%)']:>+6.1f}%  実績{row['実績リターン(%)']:>+7.2f}%  {mark}")
    return avg


def main():
    print("=" * 60)
    print("  Alpha分析: スクリーナー vs モデル貢献度")
    print(f"  予測基準日: {BACKTEST_DATE}  実績評価日: {TODAY}")
    print(f"  FETCH_DAYS: {FETCH_DAYS}")
    print("=" * 60)

    save_dir = os.path.expanduser("~/stock-alert")

    # モデル読み込み
    rise_model = joblib.load(os.path.join(save_dir, "rf_model.pkl"))
    drop_model = joblib.load(os.path.join(save_dir, "rf_drop_model.pkl"))
    print("モデル読み込み完了")

    # 最新スクリーナーCSV
    files = sorted(glob.glob(os.path.join(save_dir, "screener_*.csv")))
    if not files:
        print("ERROR: screener_*.csv が見つかりません")
        return
    csv_path = files[-1]
    print(f"スクリーナーCSV: {os.path.basename(csv_path)}")
    sc_df    = pd.read_csv(csv_path, dtype=str)
    code_col = [c for c in sc_df.columns if "コード" in c][0]
    name_col = [c for c in sc_df.columns if "銘柄名" in c][0]
    stocks   = list(zip(sc_df[code_col].str.strip(), sc_df[name_col].str.strip()))
    print(f"スクリーナー通過銘柄: {len(stocks)}件")

    # 日経225データ取得
    print("\n日経225データ取得中...")
    nk_hist    = fetch_yahoo("%5EN225")
    nk_rets_bt = None
    nk_return  = None
    if nk_hist is not None:
        nk_c = nk_hist["Close"].dropna()
        nk_c.index = pd.to_datetime(nk_c.index).date
        nk_past = nk_c[nk_c.index <= BACKTEST_DATE]
        nk_now  = nk_c[nk_c.index <= TODAY]
        if len(nk_past) >= 61:
            p = nk_past.values
            nk_rets_bt = (
                (p[-1]-p[-6]) /p[-6]  if len(p) >= 6  else 0,
                (p[-1]-p[-21])/p[-21] if len(p) >= 21 else 0,
                (p[-1]-p[-61])/p[-61] if len(p) >= 61 else 0,
            )
        if not nk_past.empty and not nk_now.empty:
            nk_return = (float(nk_now.iloc[-1]) - float(nk_past.iloc[-1])) / float(nk_past.iloc[-1]) * 100
            print(f"日経225: {BACKTEST_DATE} → {TODAY}  {nk_return:+.2f}%")

    # 各銘柄のスコアと実績取得
    print(f"\n{BACKTEST_DATE}時点の特徴量・スコア計算中...")
    results = []
    for i, (code, name) in enumerate(stocks):
        hist = fetch_yahoo(f"{code}.T")
        if hist is None:
            time.sleep(0.3)
            continue

        feat = extract_features_at(hist, BACKTEST_DATE, nk_rets_bt)
        if feat is None:
            time.sleep(0.3)
            continue

        # rank_stocks.py と同じフィルター
        if feat[12] > 0.15 or feat[10] < -0.15:
            time.sleep(0.3)
            continue

        close = hist["Close"].dropna()
        close.index = pd.to_datetime(close.index).date
        past    = close[close.index <= BACKTEST_DATE]
        present = close[close.index <= TODAY]
        if past.empty or present.empty:
            time.sleep(0.3)
            continue

        price_then = float(past.iloc[-1])
        price_now  = float(present.iloc[-1])
        actual_ret = (price_now - price_then) / price_then * 100

        rise_prob = float(rise_model.predict_proba([feat])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat])[0][1]) * 100
        net       = rise_prob - drop_prob

        results.append({
            "コード":         code,
            "銘柄名":         name,
            "ネット(%)":      round(net, 1),
            "上昇確率(%)":    round(rise_prob, 1),
            "下落確率(%)":    round(drop_prob, 1),
            "実績リターン(%)": round(actual_ret, 2),
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(stocks)} 処理済み...")
        time.sleep(0.3)

    if not results:
        print("ERROR: データ取得失敗")
        return

    df = pd.DataFrame(results).sort_values("ネット(%)", ascending=False).reset_index(drop=True)
    print(f"有効銘柄数: {len(df)}")

    # グループ分け（ネットスコア順）
    n         = len(df)
    top_df    = df.head(10)
    bottom_df = df.tail(10)
    mid_start = max(0, n // 2 - 5)
    mid_df    = df.iloc[mid_start:mid_start + 10]
    rng       = random.Random(RANDOM_SEED)
    rand_idx  = rng.sample(range(n), min(10, n))
    rand_df   = df.iloc[sorted(rand_idx)]

    print("\n" + "=" * 60)
    print("  グループ別パフォーマンス比較")
    print("=" * 60)

    top_avg  = group_stats(top_df,    "モデル上位10（スクリーナー＋モデル両方で選抜）", nk_return)
    rand_avg = group_stats(rand_df,   "ランダム10（スクリーナー効果のみ）",             nk_return)
    mid_avg  = group_stats(mid_df,    "中位10（モデルスコア中間）",                     nk_return)
    bot_avg  = group_stats(bottom_df, "モデル下位10（スクリーナー通過・モデル低評価）", nk_return)

    print("\n" + "=" * 60)
    print("  サマリー")
    print("=" * 60)
    nk_str = f"{nk_return:+.2f}%" if nk_return is not None else "N/A"
    print(f"  日経225             : {nk_str}")
    print(f"  ランダム10          : {rand_avg:+.2f}%")
    print(f"  中位10              : {mid_avg:+.2f}%")
    print(f"  モデル上位10        : {top_avg:+.2f}%")
    print(f"  モデル下位10        : {bot_avg:+.2f}%")
    if nk_return is not None:
        screener_alpha = rand_avg - nk_return
        model_alpha    = top_avg  - rand_avg
        total_alpha    = top_avg  - nk_return
        print(f"\n  スクリーナーアルファ : {screener_alpha:+.2f}%  (ランダム vs 日経)")
        print(f"  モデル追加価値      : {model_alpha:+.2f}%  (上位 vs ランダム)")
        print(f"  合計アルファ        : {total_alpha:+.2f}%  (上位 vs 日経)")

    out_path = os.path.join(save_dir, f"alpha_analysis_{BACKTEST_DATE}_{TODAY}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")
    print("完了")


if __name__ == "__main__":
    main()

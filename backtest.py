"""
backtest.py
3ヶ月前（2026年1月末）時点のデータでモデルを動かし、
現在の株価と比較してバックテストを行う。
日経225との比較も出力する。

注意: モデルの学習データに含まれる期間のため、
      過学習の影響を受ける可能性がある。参考値として見ること。
"""

import os
import io
import time
import glob
import random
import requests
import numpy as np
import pandas as pd
import joblib
import argparse as _argparse
from datetime import datetime, date, timedelta
from lib.utils import get_nikkei_returns, calc_rsi, compute_seq_features, add_cs_rank_features, IsotonicCalibrated, HEADERS, SEQ_DAYS

# ── パラメータ ──────────────────────────────
BACKTEST_DATE  = date(2026, 2, 3)    # 予測基準日（約3ヶ月前=63営業日前）
TODAY          = date.today()         # 現在日（動的）
BEAR_START     = date(2024, 7,  1)    # 下落相場テスト（2024年8月円キャリー崩壊期）
BEAR_END       = date(2024, 10, 1)    # 下落相場テスト終了
Q2_2025_START  = date(2025, 5, 14)   # 1年前の3ヶ月テスト
Q2_2025_END    = date(2025, 8, 14)   # 1年前の3ヶ月テスト終了

# 実行モード: python3 backtest.py [bear|1year|q2_2025] [--start DATE] [--end DATE] [--top-n N] ...
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("mode", nargs="?", choices=["bear", "1year", "q2_2025"], default=None)
_parser.add_argument("--top-n",   type=int,   default=10,   help="上位N銘柄を選択（デフォルト: 10）")
_parser.add_argument("--net-min", type=float, default=None, help="ネットスコア最低閾値（例: 15）")
_parser.add_argument("--compare",  action="store_true", help="保有数×閾値を一括比較")
_parser.add_argument("--screened", action="store_true", help="スクリーナー特化モデルを使用")
_parser.add_argument("--start",   type=str,   default=None, help="開始日 YYYY-MM-DD")
_parser.add_argument("--end",     type=str,   default=None, help="終了日 YYYY-MM-DD")
_args, _ = _parser.parse_known_args()

if _args.mode == 'bear':
    BACKTEST_DATE = BEAR_START
    TODAY         = BEAR_END
    print('【下落相場テストモード: 2024年8月クラッシュ期】')
elif _args.mode == '1year':
    BACKTEST_DATE = date(2025, 5, 5)
    print('【1年バックテストモード: 2025-05-05 → 今日】')
elif _args.mode == 'q2_2025':
    BACKTEST_DATE = Q2_2025_START
    TODAY         = Q2_2025_END
    print('【1年前3ヶ月テストモード: 2025-05-14 → 2025-08-14】')

if _args.start:
    BACKTEST_DATE = date.fromisoformat(_args.start)
if _args.end:
    TODAY = date.fromisoformat(_args.end)
if _args.start or _args.end:
    print(f'【カスタム期間: {BACKTEST_DATE} → {TODAY}】')

RISE_THRESHOLD = 15.0
NET_THRESHOLD  = 5.0
TOP_N          = _args.top_n
NET_MIN        = _args.net_min
COMPARE_MODE   = _args.compare
NIKKEI_CODE    = "^N225"
SAMPLE_N       = 0     # 0 = 全銘柄
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

# BACKTEST_DATEから91営業日前（≈130日）のデータが必要なため動的計算
# 例: bear mode (2024-07-01) → (2026-05-04 - 2024-07-01) + 180 ≈ 853日
FETCH_DAYS = max(800, (date.today() - BACKTEST_DATE).days + 180)

# ── スクリーナー定数（v1） ───────────────────────
_SC_MIN_MOMENTUM     = 8.0
_SC_MAX_MOMENTUM     = 30.0
_SC_MIN_VOLATILITY   = 22.0
_SC_MAX_VOLATILITY   = 50.0
_SC_MIN_MOMENTUM_20D = 0.0
_SC_MIN_PRICE        = 300
_SC_MIN_RSI          = 45.0
_SC_MAX_RSI          = 70.0

def _fetch_yahoo(ticker, days=800):
    """Yahoo Finance APIから株価DataFrameを取得"""
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = (result[0].get("indicators", {})
                          .get("adjclose", [{}])[0]
                          .get("adjclose", []))
        volumes = (result[0].get("indicators", {})
                           .get("quote", [{}])[0]
                           .get("volume", []))
        if not timestamps or not closes:
            return None
        idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").normalize()
        df = pd.DataFrame({"Close": closes, "Volume": volumes}, index=idx)
        df = df.dropna(subset=["Close"])
        df.index = df.index.date
        return df
    except Exception:
        return None


def get_hist_for_features(code):
    """特徴量計算用の全履歴を取得"""
    ticker = f"{code}.T"
    return _fetch_yahoo(ticker, days=FETCH_DAYS)


def compute_screener_at(hist, target_date, nikkei_return_3m=None):
    """target_date時点のスクリーナーフィルタ用メトリクスを計算"""
    close = hist["Close"].dropna()
    close.index = pd.to_datetime(close.index).date
    past = close[close.index <= target_date]

    if len(past) < 30:
        return None

    p = past.values
    n = len(p)

    n3  = min(63,  n - 1)
    n20 = min(20,  n - 1)

    return_3m    = (p[-1] - p[-n3  - 1]) / p[-n3  - 1]
    momentum_20d = (p[-1] - p[-n20 - 1]) / p[-n20 - 1] * 100
    vol          = (np.diff(p) / p[:-1]).std() * np.sqrt(252) * 100
    close_price  = float(p[-1])
    rsi          = calc_rsi(p)

    coef     = np.polyfit(np.arange(n), p, 1)
    slope_up = bool(coef[0] > 0)

    # 出来高比（v1用）
    vr2060 = 1.0
    if "Volume" in hist.columns:
        volume = hist["Volume"].dropna()
        volume.index = pd.to_datetime(volume.index).date
        past_vol = volume[volume.index <= target_date]
        if len(past_vol) >= 60:
            vols  = pd.to_numeric(past_vol, errors="coerce").fillna(0).values
            vol20 = vols[-20:].mean()
            vol60 = vols[-60:].mean()
            if vol60 > 0:
                vr2060 = vol20 / vol60

    # 日経比相対強度（v1用）
    rel_strength_3m = (return_3m - nikkei_return_3m) if nikkei_return_3m is not None else 0.0

    return {
        "return_3m":       return_3m,
        "momentum_20d":    momentum_20d,
        "vol":             vol,
        "rsi":             rsi,
        "vr2060":          vr2060,
        "rel_strength_3m": rel_strength_3m,
        "slope_up":        slope_up,
        "close":           close_price,
    }


def _get_screener_pass_codes(raw_screener):
    """v1スクリーナー条件を満たす銘柄コードのセットを返す"""
    sc_df = pd.DataFrame([s for s in raw_screener if s is not None and s.get("close") is not None])
    if sc_df.empty:
        return set()
    mask = (
        (sc_df["return_3m"] * 100   >= _SC_MIN_MOMENTUM)
        & (sc_df["return_3m"] * 100 <= _SC_MAX_MOMENTUM)
        & (sc_df["momentum_20d"]    >= _SC_MIN_MOMENTUM_20D)
        & (sc_df["vol"]             >= _SC_MIN_VOLATILITY)
        & (sc_df["vol"]             <= _SC_MAX_VOLATILITY)
        & (sc_df["close"]           >= _SC_MIN_PRICE)
        & (sc_df["slope_up"])
        & (sc_df.get("vr2060", pd.Series(1.0, index=sc_df.index)) >= 1.0)
        & (sc_df.get("rel_strength_3m", pd.Series(0.0, index=sc_df.index)) >= 0.05)
        & (sc_df.get("rsi", pd.Series(50.0, index=sc_df.index)) >= _SC_MIN_RSI)
        & (sc_df.get("rsi", pd.Series(50.0, index=sc_df.index)) <= _SC_MAX_RSI)
    )
    return set(sc_df.loc[mask, "code"].astype(str))


def get_nikkei_prices():
    """日経225の株価を取得"""
    return _fetch_yahoo("%5EN225", days=FETCH_DAYS)


# ── 特徴量計算 ──────────────────────────────

def extract_features_at(hist, target_date, nk_rets=None):
    """target_date時点の特徴量を計算"""
    close = hist["Close"].dropna()
    close.index = pd.to_datetime(close.index).date
    volume = hist["Volume"].dropna()
    volume.index = pd.to_datetime(volume.index).date

    past_close = close[close.index <= target_date]
    past_vol   = volume[volume.index <= target_date]

    if len(past_close) < 91:
        return None

    p = past_close.values
    v = past_vol.values

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

    rsi = calc_rsi(p)

    dr20 = np.diff(p[-21:]) / p[-21:-1] if len(p) >= 21 else np.array([0])
    vol20 = dr20.std() * np.sqrt(252) * 100

    dr60 = np.diff(p[-61:]) / p[-61:-1] if len(p) >= 61 else np.array([0])
    vol60 = dr60.std() * np.sqrt(252) * 100

    week52 = p[-252:] if len(p) >= 252 else p
    hi, lo = week52.max(), week52.min()
    pos52 = (current - lo) / (hi - lo) if hi > lo else 0.5

    SEQ_DAYS = 60
    if len(p) >= SEQ_DAYS + 1:
        seq_raw = np.clip(np.diff(p[-(SEQ_DAYS + 1):]) / p[-(SEQ_DAYS + 1):-1], -0.2, 0.2)
    else:
        seq_raw = np.zeros(SEQ_DAYS)
    seq = compute_seq_features(seq_raw)

    # トレンド反転5特徴量
    rhi = p[-60:].max() if len(p) >= 60 else p.max()
    drawdown60 = (current - rhi) / rhi
    hi52 = p[-252:].max() if len(p) >= 252 else p.max()
    from_hi52 = (current - hi52) / hi52
    stk = 0
    for j in range(1, min(21, len(p))):
        if p[-j] < p[-j - 1]: stk += 1
        else: break
    down_streak = stk / 20.0
    momentum_accel = ret5 - (ret20 / 4)
    ma5_5ago  = p[-10:-5].mean() if len(p) >= 10 else ma5
    ma25_5ago = p[-30:-5].mean() if len(p) >= 30 else ma25
    cross_prev = ma5_5ago / ma25_5ago - 1 if ma25_5ago > 0 else 0
    ma_cross_dir = ma5_25 - cross_prev

    # 出来高3特徴量
    if v is not None and len(v) >= 20:
        va = np.array([x if x is not None else np.nan for x in v], dtype=float)
        va5  = np.nanmean(va[-5:])  if len(va) >= 5  else 1
        va20 = np.nanmean(va[-20:]) if len(va) >= 20 else 1
        va60 = np.nanmean(va[-60:]) if len(va) >= 60 else va20
        vr520  = va5  / va20 if va20 > 0 else 1.0
        vr2060 = va20 / va60 if va60 > 0 else 1.0
        vsurge = va[-1] / va20 if va20 > 0 and not np.isnan(va[-1]) else 1.0
    else:
        vr520, vr2060, vsurge = 1.0, 1.0, 1.0

    # 日経マクロ3特徴量（呼び出し元から渡す）
    nk5 = nk_rets[0] if nk_rets is not None else 0.0
    nk20 = nk_rets[1] if nk_rets is not None else 0.0
    nk60 = nk_rets[2] if nk_rets is not None else 0.0

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + seq
    if any(np.isnan(feat[:10])) or any(np.isinf(feat[:10])):
        return None
    return feat


# ── スクリーナー銘柄リスト取得 ──────────────
def fetch_tse_codes():
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    try:
        r = requests.get(url, timeout=30)
        df = pd.read_excel(pd.io.common.BytesIO(r.content), dtype=str)
        df.columns = df.columns.str.strip()
        market_col = [c for c in df.columns if "市場・商品区分" in c]
        code_col   = [c for c in df.columns if "コード" in c]
        name_col   = [c for c in df.columns if "銘柄名" in c]
        if market_col and code_col:
            mask = df[market_col[0]].str.contains("内国株式", na=False)
            codes = df.loc[mask, code_col[0]].str.strip().tolist()
            names = df.loc[mask, name_col[0]].str.strip().tolist()
            return list(zip(codes, names))
    except Exception as e:
        print(f"[WARN] 銘柄リスト取得失敗: {e}")
    return []


# ── メイン ──────────────────────────────────
def main():
    print("=" * 60)
    print(f"  バックテスト: {BACKTEST_DATE} → {TODAY}")
    print(f"  予測上位{TOP_N}銘柄を「買い」として検証")
    print("=" * 60)

    # モデル読み込み（--screened フラグでスクリーナー特化モデルを使用）
    model_suffix = "_screened" if getattr(_args, "screened", False) else ""
    rise_path = os.path.join(BASE_DIR, f"rf_model{model_suffix}.pkl")
    drop_path = os.path.join(BASE_DIR, f"rf_drop_model{model_suffix}.pkl")
    if not os.path.exists(rise_path):
        print(f"ERROR: {rise_path} が見つかりません")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    print(f"モデル読み込み完了{'（スクリーナー特化）' if model_suffix else ''}")

    print("TSE全銘柄リストを取得中（バイアスなし）...")
    all_stocks = fetch_tse_codes()
    if SAMPLE_N > 0:
        random.seed(42)
        stocks = random.sample(all_stocks, min(SAMPLE_N, len(all_stocks)))
    else:
        stocks = all_stocks
    print(f"バックテスト対象: {len(stocks)}銘柄")

    # 日経225データを事前取得（特徴量計算用）
    print("\n日経225データ取得中（特徴量用）...")
    nikkei_hist = get_nikkei_prices()

    # 日経225の3ヶ月リターンをBACKTEST_DATE時点で計算（相対強度用）
    nikkei_return_3m_sc = None
    if nikkei_hist is not None:
        nk_c = nikkei_hist["Close"].dropna()
        nk_c.index = pd.to_datetime(nk_c.index).date
        nk_past = nk_c[nk_c.index <= BACKTEST_DATE]
        if len(nk_past) >= 64:
            nkp = nk_past.values
            n3  = min(63, len(nkp) - 1)
            nikkei_return_3m_sc = (nkp[-1] - nkp[-n3 - 1]) / nkp[-n3 - 1]
            print(f"日経225 3ヶ月リターン({BACKTEST_DATE}時点): {nikkei_return_3m_sc*100:+.1f}%")

    # フェーズ1: 全銘柄の特徴量・スクリーナーメトリクスを収集
    print(f"\n{BACKTEST_DATE}時点の特徴量でスコア計算中...")
    raw_feats, raw_meta, raw_screener = [], [], []
    for i, (code, name) in enumerate(stocks):
        hist = get_hist_for_features(code)
        if hist is None:
            time.sleep(0.3); continue

        # スクリーナーメトリクスをBACKTEST_DATE時点で計算
        sc_metrics = compute_screener_at(hist, BACKTEST_DATE, nikkei_return_3m_sc)

        nk_rets_bt = None
        if nikkei_hist is not None:
            nk_c = nikkei_hist["Close"].dropna()
            nk_c.index = pd.to_datetime(nk_c.index).date
            nk_past_all = nk_c[nk_c.index <= BACKTEST_DATE]
            if len(nk_past_all) >= 61:
                nkp = nk_past_all.values
                nk5_v  = (nkp[-1]-nkp[-6]) /nkp[-6]  if len(nkp)>=6  else 0
                nk20_v = (nkp[-1]-nkp[-21])/nkp[-21] if len(nkp)>=21 else 0
                nk60_v = (nkp[-1]-nkp[-61])/nkp[-61] if len(nkp)>=61 else 0
                nk_rets_bt = (nk5_v, nk20_v, nk60_v)
        feat = extract_features_at(hist, BACKTEST_DATE, nk_rets_bt)
        if feat is None:
            time.sleep(0.3); continue
        if feat[12] > 0.15 or feat[10] < -0.15:
            time.sleep(0.3); continue
        close = hist["Close"].dropna()
        close.index = pd.to_datetime(close.index).date
        past    = close[close.index <= BACKTEST_DATE]
        present = close[close.index <= TODAY]
        if past.empty or present.empty:
            time.sleep(0.3); continue
        raw_feats.append(feat)
        raw_meta.append((code, name, float(past.iloc[-1]), float(present.iloc[-1])))
        raw_screener.append({"code": code, **(sc_metrics if sc_metrics else {})})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(stocks)} 取得済み...")
        time.sleep(0.3)

    # フェーズ1.5: スクリーナーフィルター適用
    if raw_screener:
        pass_codes = _get_screener_pass_codes(raw_screener)
        keep = [j for j, m in enumerate(raw_meta) if m[0] in pass_codes]
        raw_feats    = [raw_feats[j]    for j in keep]
        raw_meta     = [raw_meta[j]     for j in keep]
        raw_screener = [raw_screener[j] for j in keep]
        print(f"スクリーナー v1: {len(pass_codes)} 銘柄通過 / {len(raw_feats)} 件処理")

    sc_map = {s["code"]: s for s in raw_screener}

    # フェーズ2: クロスセクショナルランク特徴量を付加
    if not raw_feats:
        print("ERROR: データが取得できませんでした"); return
    feats_aug = add_cs_rank_features(np.array(raw_feats, dtype=float))

    # フェーズ3: モデルスコアと実績リターンを集計
    results = []
    for idx, (code, name, price_then, price_now) in enumerate(raw_meta):
        feat_aug = feats_aug[idx]
        actual_return = (price_now - price_then) / price_then * 100
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100 if drop_model else 0
        net = rise_prob - drop_prob
        sc = sc_map.get(code, {})
        results.append({
            "コード": code,
            "銘柄名": name,
            f"{BACKTEST_DATE}株価": round(price_then, 1),
            f"{TODAY}株価": round(price_now, 1),
            "実績リターン(%)": round(actual_return, 2),
            "上昇確率(%)": round(rise_prob, 1),
            "下落確率(%)": round(drop_prob, 1),
            "ネット(%)": round(net, 1),
            "モメンタム3M(%)": round(sc["return_3m"] * 100, 1) if sc.get("return_3m") is not None else None,
            "モメンタム20d(%)": round(sc["momentum_20d"], 1) if sc.get("momentum_20d") is not None else None,
            "ボラ(%)": round(sc["vol"], 1) if sc.get("vol") is not None else None,
            "RSI": round(sc["rsi"], 1) if sc.get("rsi") is not None else None,
            "相対強度3M(%)": round(sc["rel_strength_3m"] * 100, 1) if sc.get("rel_strength_3m") is not None else None,
            "予測正解": int(actual_return >= RISE_THRESHOLD),
        })

    if not results:
        print("ERROR: データが取得できませんでした")
        return

    df = pd.DataFrame(results)

    # ── 日経225の同期間リターン ──
    print("\n日経225のリターンを取得中...")
    nikkei_df = get_nikkei_prices()
    nikkei_close = nikkei_df["Close"] if nikkei_df is not None else pd.Series(dtype=float)
    nk_past    = nikkei_close[nikkei_close.index <= BACKTEST_DATE] if len(nikkei_close) > 0 else pd.Series(dtype=float)
    nk_present = nikkei_close[nikkei_close.index <= TODAY] if len(nikkei_close) > 0 else pd.Series(dtype=float)
    nikkei_return = None
    if not nk_past.empty and not nk_present.empty:
        nikkei_return = (float(nk_present.iloc[-1]) - float(nk_past.iloc[-1])) / float(nk_past.iloc[-1]) * 100
        print(f"日経225: {BACKTEST_DATE} ¥{nk_past.iloc[-1]:,.0f} → {TODAY} ¥{nk_present.iloc[-1]:,.0f} ({nikkei_return:+.2f}%)")

    # ── 全銘柄の統計 ──
    print(f"\n{'='*60}")
    print(f"【全{len(df)}銘柄の実績】")
    print(f"  平均リターン: {df['実績リターン(%)'].mean():+.2f}%")
    print(f"  中央値リターン: {df['実績リターン(%)'].median():+.2f}%")
    wins = (df["実績リターン(%)"] > 0).sum()
    print(f"  勝率（プラスリターン）: {wins}/{len(df)} = {wins/len(df)*100:.1f}%")

    def _print_group(group_df, label, nk_ret):
        """銘柄グループの成績サマリーを表示"""
        n = len(group_df)
        if n == 0:
            print(f"\n【{label}】該当なし")
            return
        avg = group_df["実績リターン(%)"].mean()
        wins_0  = (group_df["実績リターン(%)"] > 0).sum()
        wins_15 = (group_df["実績リターン(%)"] >= RISE_THRESHOLD).sum()
        print(f"\n{'='*60}")
        print(f"【{label} ({n}銘柄)】")
        print(f"  平均リターン: {avg:+.2f}%")
        print(f"  勝率（+0%以上）: {wins_0}/{n} = {wins_0/n*100:.1f}%")
        print(f"  大勝率（+15%以上）: {wins_15}/{n} = {wins_15/n*100:.1f}%")
        if nk_ret is not None:
            diff = avg - nk_ret
            alpha_wins = (group_df["実績リターン(%)"] > nk_ret).sum()
            print(f"  vs 日経225: {avg:+.2f}% vs {nk_ret:+.2f}% → アルファ {diff:+.2f}%")
            print(f"  日経アルファ勝率: {alpha_wins}/{n} = {alpha_wins/n*100:.1f}%")

    def _print_detail(group_df, label):
        print(f"\n【{label}詳細】")
        print(f"{'コード':>6}  {'銘柄名':<20}  {'予測ネット':>10}  {'実績':>8}  結果")
        print("-" * 60)
        for _, row in group_df.sort_values("実績リターン(%)", ascending=False).iterrows():
            mark = "✅" if row["実績リターン(%)"] >= RISE_THRESHOLD else ("🔵" if row["実績リターン(%)"] > 0 else "❌")
            print(
                f"{row['コード']:>6}  {str(row['銘柄名']):<20}  "
                f"ネット{row['ネット(%)']:>+6.1f}%  "
                f"実績{row['実績リターン(%)']:>+7.2f}%  {mark}"
            )

    if COMPARE_MODE:
        # ── 比較モード: 保有数×閾値を一括比較 ──
        print(f"\n{'='*60}")
        print("【比較モード: 保有数 × ネットスコア閾値】")
        nk = nikkei_return if nikkei_return is not None else 0
        header = f"{'設定':<22} {'銘柄数':>5} {'平均':>8} {'アルファ':>9} {'日経勝率':>8} {'勝率':>7} {'大勝率':>7}"
        print(header)
        print("-" * 70)
        for net_min in [None, 10.0, 15.0, 20.0]:
            for top_n in [5, 10, 15, 30]:
                if net_min is not None:
                    cand = df[df["ネット(%)"] >= net_min].nlargest(top_n, "ネット(%)")
                    tag  = f"net≥{net_min:.0f} top{top_n}"
                else:
                    cand = df.nlargest(top_n, "ネット(%)")
                    tag  = f"top{top_n}"
                n = len(cand)
                if n == 0:
                    print(f"  {tag:<20} {'0':>5} {'N/A':>8}")
                    continue
                avg      = cand["実績リターン(%)"].mean()
                alpha    = avg - nk
                a_wins   = (cand["実績リターン(%)"] > nk).sum()
                wins_0   = (cand["実績リターン(%)"] > 0).sum()
                wins_15  = (cand["実績リターン(%)"] >= RISE_THRESHOLD).sum()
                print(
                    f"  {tag:<20} {n:>5} {avg:>+7.2f}% {alpha:>+8.2f}%"
                    f"  {a_wins}/{n}={a_wins/n*100:.0f}%"
                    f"  {wins_0/n*100:.0f}%  {wins_15/n*100:.0f}%"
                )
        # 詳細は通常モードのTOP_N設定で表示
        top_df = df.nlargest(TOP_N, "ネット(%)")
    else:
        # ── 通常モード ──
        if NET_MIN is not None:
            top_df = df[df["ネット(%)"] >= NET_MIN].nlargest(TOP_N, "ネット(%)")
            label  = f"ネットスコア≥{NET_MIN} 上位{len(top_df)}"
        else:
            top_df = df.nlargest(TOP_N, "ネット(%)")
            label  = f"ネットスコア上位{TOP_N}"
        _print_group(top_df, label, nikkei_return)

    # ── 詳細表示 ──
    _print_detail(top_df, f"上位{len(top_df)}")

    # CSV保存
    df["日経リターン(%)"] = nikkei_return
    out_dir = os.path.join(BASE_DIR, "simulations", "backtests")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"backtest_{BACKTEST_DATE}_{TODAY}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")
    print("完了")


if __name__ == "__main__":
    main()

"""
backtest.py
3ヶ月前（2026年1月末）時点のデータでモデルを動かし、
現在の株価と比較してバックテストを行う。
日経225との比較も出力する。

注意: モデルの学習データに含まれる期間のため、
      過学習の影響を受ける可能性がある。参考値として見ること。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
_parser.add_argument("--top-n",   type=int,   default=5,    help="下落確率が低い上位N銘柄を選択（デフォルト: 5）")
_parser.add_argument("--drop-max", type=float, default=None, help="下落確率の上限閾値（例: 8）")
_parser.add_argument("--compare",  action="store_true", help="保有数×閾値を一括比較")
_parser.add_argument("--screened",    action="store_true", help="スクリーナー特化モデルを使用")
_parser.add_argument("--no-screener", action="store_true", help="スクリーナーをスキップして全銘柄をモデルで評価")
_parser.add_argument("--start",   type=str,   default=None, help="開始日 YYYY-MM-DD")
_parser.add_argument("--end",     type=str,   default=None, help="終了日 YYYY-MM-DD")
_parser.add_argument("--model-cutoff", type=str, default=None, help="使用するモデルのcutoff日 YYYY-MM-DD")
_parser.add_argument("--rolling",       action="store_true",  help="ローリング5日バックテストモード")
_parser.add_argument("--forecast-days", type=int, default=5,  help="保有日数（ローリングモード用、デフォルト5）")
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

FORECAST_DAYS  = _args.forecast_days if hasattr(_args, 'forecast_days') else 21
RISE_THRESHOLD = 5.0 if FORECAST_DAYS <= 30 else 15.0   # 21日モデル=5%, 63日モデル=15%
BIG_WIN_THRESHOLD = 8.0 if FORECAST_DAYS <= 30 else 15.0  # 21日で+8%=大勝
TOP_N          = _args.top_n
DROP_MAX       = _args.drop_max
COMPARE_MODE   = _args.compare
NO_SCREENER    = _args.no_screener
NIKKEI_CODE    = "^N225"
SAMPLE_N       = 0     # 0 = 全銘柄
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

# BACKTEST_DATEから91営業日前（≈130日）のデータが必要なため動的計算
# 例: bear mode (2024-07-01) → (2026-05-04 - 2024-07-01) + 180 ≈ 853日
FETCH_DAYS = max(800, (date.today() - BACKTEST_DATE).days + 180)

# ── point-in-time ファンダメンタル（lib/fundamentals に集約）─────────────────
from lib.fundamentals import pit_fundamental_features
from lib.utils import extract_features as _extract_features

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
    """Supabaseから株価DataFrameを取得（個別株・市場指数共用）"""
    if "N225" in ticker or "%5EN225" in ticker:
        from lib.db import load_market_index_data
        return load_market_index_data("N225", days=days)
    code = ticker.replace(".T", "")
    from lib.db import get_price_df
    return get_price_df(code, days=days)


def get_hist_for_features(code):
    """特徴量計算用の全履歴を取得（DBキャッシュ優先）"""
    from lib.db import get_price_cache_coverage, get_price_cache, save_price_cache
    code_str = str(code)
    required_start = (date.today() - timedelta(days=FETCH_DAYS)).isoformat()
    required_end   = TODAY.isoformat()

    coverage = get_price_cache_coverage(code_str)
    if coverage and coverage[0] <= required_start and coverage[1] >= required_end:
        return get_price_cache(code_str, required_start, date.today().isoformat())

    ticker = f"{code}.T"
    df = _fetch_yahoo(ticker, days=FETCH_DAYS)
    if df is not None:
        save_price_cache(code_str, df)
    return df


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

def extract_features_at(hist, target_date, nk_rets=None, code=None):
    """target_date時点の特徴量を計算（extract_features()に委譲）。
    nk_rets: (nk5, nk20, nk60) フラクション単位（rf_train_v3と同単位）。
    code を渡すと point-in-time ファンダ（VIX/FX欠損 → 0.0デフォルト）を再構成。
    """
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

    # fundamentals dict を組み立て（pit_fundamental_featuresはdictを返すよう変更済み）
    if code is not None:
        fundamentals = pit_fundamental_features(code, target_date, current)
    else:
        fundamentals = {"month": target_date.month}

    return _extract_features(p, v, nk_rets, fundamentals=fundamentals)


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

    # モデル読み込み（下落モデルのみ。--screened フラグでスクリーナー特化モデルを使用）
    model_suffix = "_screened" if getattr(_args, "screened", False) else ""
    model_cutoff_tag = f"_{_args.model_cutoff}" if getattr(_args, "model_cutoff", None) else ""
    drop_path = os.path.join(BASE_DIR, f"rf_drop_model{model_suffix}{model_cutoff_tag}.pkl")
    if not os.path.exists(drop_path):
        print(f"ERROR: {drop_path} が見つかりません")
        return
    drop_model = joblib.load(drop_path)
    print(f"モデル読み込み完了{f' (cutoff:{_args.model_cutoff})' if model_cutoff_tag else ''}{'（スクリーナー特化）' if model_suffix else ''}")

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
        feat = extract_features_at(hist, BACKTEST_DATE, nk_rets_bt, code=code)
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
    if NO_SCREENER:
        print(f"スクリーナースキップ（--no-screener）: {len(raw_feats)} 件全評価")
    elif raw_screener:
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
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100
        sc = sc_map.get(code, {})
        results.append({
            "コード": code,
            "銘柄名": name,
            f"{BACKTEST_DATE}株価": round(price_then, 1),
            f"{TODAY}株価": round(price_now, 1),
            "実績リターン(%)": round(actual_return, 2),
            "下落確率(%)": round(drop_prob, 1),
            "モメンタム3M(%)": round(sc["return_3m"] * 100, 1) if sc.get("return_3m") is not None else None,
            "モメンタム20d(%)": round(sc["momentum_20d"], 1) if sc.get("momentum_20d") is not None else None,
            "ボラ(%)": round(sc["vol"], 1) if sc.get("vol") is not None else None,
            "RSI": round(sc["rsi"], 1) if sc.get("rsi") is not None else None,
            "相対強度3M(%)": round(sc["rel_strength_3m"] * 100, 1) if sc.get("rel_strength_3m") is not None else None,
            "上昇達成": int(actual_return >= RISE_THRESHOLD),
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
        wins_big = (group_df["実績リターン(%)"] >= BIG_WIN_THRESHOLD).sum()
        print(f"\n{'='*60}")
        print(f"【{label} ({n}銘柄)】")
        print(f"  平均リターン: {avg:+.2f}%")
        print(f"  勝率（+0%以上）: {wins_0}/{n} = {wins_0/n*100:.1f}%")
        print(f"  大勝率（+{BIG_WIN_THRESHOLD:.0f}%以上）: {wins_big}/{n} = {wins_big/n*100:.1f}%")
        if nk_ret is not None:
            diff = avg - nk_ret
            alpha_wins = (group_df["実績リターン(%)"] > nk_ret).sum()
            print(f"  vs 日経225: {avg:+.2f}% vs {nk_ret:+.2f}% → アルファ {diff:+.2f}%")
            print(f"  日経アルファ勝率: {alpha_wins}/{n} = {alpha_wins/n*100:.1f}%")

    def _print_detail(group_df, label):
        print(f"\n【{label}詳細】")
        print(f"{'コード':>6}  {'銘柄名':<20}  {'予測下落確率':>10}  {'実績':>8}  結果")
        print("-" * 60)
        for _, row in group_df.sort_values("実績リターン(%)", ascending=False).iterrows():
            mark = "✅" if row["実績リターン(%)"] >= BIG_WIN_THRESHOLD else ("🔵" if row["実績リターン(%)"] > 0 else "❌")
            print(
                f"{row['コード']:>6}  {str(row['銘柄名']):<20}  "
                f"下落確率{row['下落確率(%)']:>6.1f}%  "
                f"実績{row['実績リターン(%)']:>+7.2f}%  {mark}"
            )

    if COMPARE_MODE:
        # ── 比較モード: 保有数×下落確率閾値を一括比較 ──
        print(f"\n{'='*60}")
        print("【比較モード: 保有数 × 下落確率上限】")
        nk = nikkei_return if nikkei_return is not None else 0
        header = f"{'設定':<22} {'銘柄数':>5} {'平均':>8} {'アルファ':>9} {'日経勝率':>8} {'勝率':>7} {'大勝率':>7}"
        print(header)
        print("-" * 70)
        for drop_max in [None, 15.0, 10.0, 8.0]:
            for top_n in [5, 10, 15, 30]:
                if drop_max is not None:
                    cand = df[df["下落確率(%)"] <= drop_max].nsmallest(top_n, "下落確率(%)")
                    tag  = f"drop≤{drop_max:.0f} top{top_n}"
                else:
                    cand = df.nsmallest(top_n, "下落確率(%)")
                    tag  = f"top{top_n}"
                n = len(cand)
                if n == 0:
                    print(f"  {tag:<20} {'0':>5} {'N/A':>8}")
                    continue
                avg      = cand["実績リターン(%)"].mean()
                alpha    = avg - nk
                a_wins   = (cand["実績リターン(%)"] > nk).sum()
                wins_0   = (cand["実績リターン(%)"] > 0).sum()
                wins_big = (cand["実績リターン(%)"] >= BIG_WIN_THRESHOLD).sum()
                print(
                    f"  {tag:<20} {n:>5} {avg:>+7.2f}% {alpha:>+8.2f}%"
                    f"  {a_wins}/{n}={a_wins/n*100:.0f}%"
                    f"  {wins_0/n*100:.0f}%  {wins_big/n*100:.0f}%"
                )
        # 詳細は通常モードのTOP_N設定で表示
        top_df = df.nsmallest(TOP_N, "下落確率(%)")
    else:
        # ── 通常モード ──
        if DROP_MAX is not None:
            top_df = df[df["下落確率(%)"] <= DROP_MAX].nsmallest(TOP_N, "下落確率(%)")
            label  = f"下落確率≤{DROP_MAX} 下位{len(top_df)}"
        else:
            top_df = df.nsmallest(TOP_N, "下落確率(%)")
            label  = f"下落確率が低い上位{TOP_N}"
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


def run_rolling_main():
    """ローリング N 日バックテスト: 期間内を FORECAST_DAYS ごとに分割して評価"""
    fd = FORECAST_DAYS
    print("=" * 60)
    print(f"  ローリング{fd}日バックテスト: {BACKTEST_DATE} → {TODAY}")
    print(f"  上位{TOP_N}銘柄 × 各ラウンド")
    print("=" * 60)

    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(drop_path):
        print(f"ERROR: {drop_path} が見つかりません"); return
    drop_model = joblib.load(drop_path)
    print(f"モデル読み込み完了")

    print("TSE全銘柄リストを取得中...")
    all_stocks = fetch_tse_codes()

    print("日経225データ取得中...")
    nikkei_hist = get_nikkei_prices()
    if nikkei_hist is None:
        print("ERROR: 日経データ取得失敗"); return
    nk_c = nikkei_hist["Close"].dropna()
    nk_c.index = pd.to_datetime(nk_c.index).date
    trading_dates = sorted(d for d in nk_c.index if BACKTEST_DATE <= d <= TODAY)
    if len(trading_dates) < fd + 1:
        print("ERROR: 期間内の営業日が不足"); return
    entry_dates = trading_dates[::fd]

    # 株価データを一括取得
    print(f"株価データ取得中（{len(all_stocks)}銘柄）...")
    stocks_hist = {}
    for i, (code, name) in enumerate(all_stocks):
        h = get_hist_for_features(code)
        if h is not None:
            stocks_hist[code] = (h, name)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_stocks)} 取得済み...")
        time.sleep(0.05)

    from lib.utils import classify_market_regime

    # ① 品質フィルター（常に適用: 株価≥300円・データ十分）
    quality_filtered = {c: (h, name) for c, (h, name) in stocks_hist.items()
                        if len(h) >= 91 and float(h["Close"].dropna().iloc[-1]) >= 300}
    print(f"品質フィルター通過: {len(quality_filtered)} 銘柄")

    # 品質フィルターのみ使用（スクリーナー判定は行わない）

    per_round_avgs = []
    per_round_nk   = []
    all_trades = []

    for ri, entry_date in enumerate(entry_dates):
        ei = trading_dates.index(entry_date)
        if ei + fd >= len(trading_dates):
            break
        exit_date = trading_dates[ei + fd]

        # 日経リターン（エントリー日時点）
        nk_at = nk_c[nk_c.index <= entry_date]
        if len(nk_at) < 61:
            continue

        nk_vals  = nk_at.values
        round_stocks = quality_filtered  # 常に全銘柄（品質フィルターのみ）
        nkp = nk_at.values
        nkr = ((nkp[-1]-nkp[-6])/nkp[-6] if len(nkp)>=6 else 0,
                (nkp[-1]-nkp[-21])/nkp[-21] if len(nkp)>=21 else 0,
                (nkp[-1]-nkp[-61])/nkp[-61] if len(nkp)>=61 else 0)

        raw_feats, raw_meta = [], []
        for code, (h, name) in round_stocks.items():
            feat = extract_features_at(h, entry_date, nkr, code=code)
            if feat is None:
                continue
            cl = h["Close"].dropna()
            cl.index = pd.to_datetime(cl.index).date
            ep = cl[cl.index <= entry_date]
            xp = cl[cl.index <= exit_date]
            if ep.empty or xp.empty:
                continue
            raw_feats.append(feat)
            raw_meta.append((code, name, float(ep.iloc[-1]), float(xp.iloc[-1])))

        if not raw_feats:
            continue

        fa = add_cs_rank_features(np.array(raw_feats, dtype=float))
        scores = []
        for idx, (code, name, pe, px) in enumerate(raw_meta):
            dp = float(drop_model.predict_proba([fa[idx]])[0][1]) * 100
            ret = (px - pe) / pe * 100
            scores.append((code, name, dp, ret))

        scores.sort(key=lambda x: x[2])  # 下落確率が低い順
        top = scores[:TOP_N]
        rets = [r[3] for r in top]
        avg_r = float(np.mean(rets))
        per_round_avgs.append(avg_r)
        # 上位TOP_N銘柄の成績（バックテスト用）
        for code, name, dp, ret in top:
            all_trades.append({"ラウンド": ri+1, "entry": str(entry_date),
                                "exit": str(exit_date), "code": code,
                                "銘柄名": name, "drop_prob": round(dp,1), "return": round(ret,2),
                                "selected": 1})
        # メタ学習用: スクリーナー通過全銘柄のスコアも記録（TOP_N以外はselected=0）
        top_codes = {s[0] for s in top}
        for code, name, dp, ret in scores:
            if code not in top_codes:
                all_trades.append({"ラウンド": ri+1, "entry": str(entry_date),
                                    "exit": str(exit_date), "code": code,
                                    "銘柄名": name, "drop_prob": round(dp,1), "return": round(ret,2),
                                    "selected": 0})

        # 日経同期間リターン
        nk_entry = nk_c[nk_c.index <= entry_date]
        nk_exit  = nk_c[nk_c.index <= exit_date]
        nk_ret = (float(nk_exit.iloc[-1]) - float(nk_entry.iloc[-1])) / float(nk_entry.iloc[-1]) * 100 \
                 if not nk_entry.empty and not nk_exit.empty else 0
        per_round_nk.append(nk_ret)
        wins_r = sum(1 for r in rets if r > 0)
        print(f"  R{ri+1:02d} [{entry_date}→{exit_date}] avg={avg_r:+.2f}%  "
              f"日経={nk_ret:+.2f}%  勝率={wins_r}/{len(rets)}")

    if not per_round_avgs:
        print("ERROR: 有効ラウンドなし"); return

    n_rounds = len(per_round_avgs)
    avg  = float(np.mean(per_round_avgs))
    wins = sum(1 for x in per_round_avgs if x > 0)
    bigs = sum(1 for x in per_round_avgs if x >= BIG_WIN_THRESHOLD)
    nk_avg = float(np.mean(per_round_nk)) if per_round_nk else 0.0
    alpha  = avg - nk_avg

    print(f"\n{'='*60}")
    print(f"【ローリング{fd}日 複合結果: {n_rounds}ラウンド】")
    print(f"  平均リターン: {avg:+.2f}%")
    print(f"  勝率（+0%以上）: {wins}/{n_rounds} = {wins/n_rounds*100:.1f}%")
    print(f"  大勝率（+{BIG_WIN_THRESHOLD:.0f}%以上）: {bigs}/{n_rounds} = {bigs/n_rounds*100:.1f}%")
    print(f"  日経平均リターン: {nk_avg:+.2f}%")
    print(f"  日経アルファ: {alpha:+.2f}%")

    # 日経ベンチマーク（期間合計）
    nk_s = nk_c[nk_c.index <= BACKTEST_DATE]
    nk_e = nk_c[nk_c.index <= TODAY]
    if not nk_s.empty and not nk_e.empty:
        nk_total = (float(nk_e.iloc[-1]) - float(nk_s.iloc[-1])) / float(nk_s.iloc[-1]) * 100
        print(f"  日経225 期間合計: {nk_total:+.2f}% ({BACKTEST_DATE}→{TODAY})")

    # CSV保存
    if all_trades:
        out_dir = os.path.join(BASE_DIR, "simulations", "backtests")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rolling{fd}d_{BACKTEST_DATE}_{TODAY}.csv")
        pd.DataFrame(all_trades).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n全結果保存: {out_path}")
    print("完了")


if __name__ == "__main__":
    if getattr(_args, 'rolling', False):
        run_rolling_main()
    else:
        main()

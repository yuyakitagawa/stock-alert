import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import threading
import pandas as pd
import numpy as np
import time
import os
import glob
import requests as _requests
from datetime import datetime, timedelta, date as _date
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from lib.utils import get_prices, get_nikkei_returns, calc_rsi, extract_features, add_cs_rank_features, get_fundamentals, IsotonicCalibrated, HEADERS, SEQ_DAYS, recommend_from_scores, classify_market_regime, get_market_index_df_cached
from config import BASE_DIR, BEAR_MARKET_THRESHOLD, FORECAST, RISE_THRESHOLD, MAX_BUY_VOL20, \
                   MARKET_TIMING_ENABLED, MARKET_TIMING_20D_THRESH
from core.screener import get_tse_stock_list

TOP_SHOW = 10
MIN_LIQUIDITY_M  = 50.0   # 20日平均売買代金(百万円)
BULL_SMA_PERIOD  = 20     # 強気判定に使うSMA期間（日）
BETA_MIN_BULL    = 0.4    # 強気相場時の最低β（日経との連動性）

# 米国セクターETFリードラグフィルター（US前日リターンが負なら降格）
SECTOR_TO_ETF = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
}
# 相関係数 > 0.15 の強相関セクターのみフィルター対象（2023-2026 21,416サンプル検証済み）
STRONG_EFFECT_ETFS = {"XLK", "XLF", "XLI", "XLB", "XLV", "XLY"}

_SECTOR_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sector_map.json")
_sector_cache: dict = {}


def _load_sector_cache():
    global _sector_cache
    if os.path.exists(_SECTOR_CACHE_PATH):
        try:
            with open(_SECTOR_CACHE_PATH, "r") as f:
                _sector_cache = json.load(f)
        except Exception:
            _sector_cache = {}


def _save_sector_cache():
    try:
        os.makedirs(os.path.dirname(_SECTOR_CACHE_PATH), exist_ok=True)
        with open(_SECTOR_CACHE_PATH, "w") as f:
            json.dump(_sector_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_sector_etf(code: str) -> "str | None":
    """JPX銘柄コードを米国セクターETFティッカーに変換（sector_map.jsonでキャッシュ）"""
    import yfinance as yf
    if code in _sector_cache:
        return _sector_cache[code]
    try:
        info = yf.Ticker(f"{code}.T").info
        sector = info.get("sector", "")
        etf = SECTOR_TO_ETF.get(sector)
        _sector_cache[code] = etf
        return etf
    except Exception:
        _sector_cache[code] = None
        return None


def fetch_us_sector_etf_returns() -> dict:
    """前営業日の米国セクターETF Close-to-Close リターン(%)を返す"""
    import yfinance as yf
    etfs = sorted(set(SECTOR_TO_ETF.values()))
    try:
        data = yf.download(etfs, period="5d", auto_adjust=True, progress=False)["Close"]
        result = {}
        for e in etfs:
            col = data[e] if e in data.columns else None
            if col is None:
                continue
            vals = col.dropna()
            if len(vals) >= 2:
                result[e] = float((vals.iloc[-1] - vals.iloc[-2]) / vals.iloc[-2] * 100)
        return result
    except Exception as ex:
        print(f"  米国ETF取得失敗: {ex}")
        return {}


YUTAI_SKIP_DAYS = 21  # 権利落ち日N日前からS買いを除外


def _get_yutai_record_month(code: str):
    """kabutan.jp から株主優待の権利確定月を取得。優待なし→None、取得失敗→None。"""
    from lib.db import get_yutai_cache, set_yutai_cache, CACHE_MISS
    today_str = datetime.now().strftime("%Y-%m-%d")
    cached = get_yutai_cache(code, today_str)
    if cached is not CACHE_MISS:
        has_yutai, record_month = cached
        return record_month if has_yutai else None
    try:
        resp = _requests.get(f"https://kabutan.jp/stock/yutai?code={code}",
                             headers=_KABUTAN_HEADERS, timeout=8)
        record_month = None
        has_yutai = False
        if resp.status_code == 200:
            m = re.search(r'権利確定月は(\d{1,2})月', resp.text)
            if m:
                record_month = int(m.group(1))
                has_yutai = True
        set_yutai_cache(code, today_str, has_yutai, record_month)
        return record_month if has_yutai else None
    except Exception:
        return None


def _days_to_yutai_record(code: str, today=None) -> "int | None":
    """権利落ち日（権利確定月の最終営業日-1）までの日数。優待なし→None。"""
    record_month = _get_yutai_record_month(code)
    if record_month is None:
        return None
    if today is None:
        today = datetime.now().date()
    year = today.year
    # 権利確定月の月末日を求め、そこから2営業日前を権利落ち日と近似
    import calendar
    last_day = calendar.monthrange(year, record_month)[1]
    ex_date = _date(year, record_month, last_day) - timedelta(days=2)
    # 来年分も考慮
    delta = (ex_date - today).days
    if delta < -7:
        last_day2 = calendar.monthrange(year + 1, record_month)[1]
        ex_date2 = _date(year + 1, record_month, last_day2) - timedelta(days=2)
        delta = (ex_date2 - today).days
    return delta


def passes_buy_filter(feat, close, volumes, nk20=None, ret_504=None, r2_504=None):
    """最小限の品質フィルター（スクリーナーを廃止、モデルスコアで選別）
    残す条件: 株価・流動性・急落中の除外のみ
    """
    if close < 300:               return False  # 株価 < 300円（低位株除外）
    if feat[10] < -0.20:          return False  # drawdown60 < -20%（急落中は除外）
    if feat[12] > 0.20:           return False  # down_streak > 4日（連続下落中は除外）
    if feat[6] >= 80.0:            return False  # RSI ≥ 80（過熱域のみ除外）
    if volumes and len(volumes) >= 20:
        valid = [v for v in volumes[-20:] if v is not None and not np.isnan(v)]
        if valid:
            va20 = np.mean(valid)
            if va20 * close / 1e6 < MIN_LIQUIDITY_M:
                return False  # 流動性なし（売買代金 < 50百万円）
    return True












def _is_nk225_bull(nk_closes, sma_period=BULL_SMA_PERIOD):
    """N225終値系列から短期SMAで強気判定。N225 > SMA(period) なら True。"""
    if nk_closes is None or len(nk_closes) < sma_period:
        return False
    return float(nk_closes[-1]) >= float(np.mean(nk_closes[-sma_period:]))


def _calc_nk225_beta(stock_prices, nk_closes, window=60):
    """直近window日の株価 vs N225のβ値を計算。"""
    if stock_prices is None or nk_closes is None:
        return None
    s = stock_prices[-window-1:] if len(stock_prices) >= window+1 else stock_prices
    n = nk_closes[-window-1:] if len(nk_closes) >= window+1 else nk_closes
    min_len = min(len(s), len(n))
    if min_len < 21:
        return None
    s_ret = np.diff(s[-min_len:]) / s[-min_len:-1]
    n_ret = np.diff(n[-min_len:]) / n[-min_len:-1]
    var_n = np.var(n_ret)
    if var_n == 0:
        return None
    return float(np.cov(s_ret, n_ret)[0][1] / var_n)




def main():
    print("=" * 55)
    print("スクリーナー × RF ランキング  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"スクリーナー通過銘柄に上昇確率スコアをつけてランキング")
    print("=" * 55)

    # モデル読み込み（上昇・下落 × 絶対・相対 = 4モデル）
    rise_path = os.path.join(BASE_DIR, "rf_model.pkl")
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    alpha_rise_path = os.path.join(BASE_DIR, "rf_alpha_model.pkl")
    alpha_drop_path = os.path.join(BASE_DIR, "rf_alpha_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pkl が見つかりません。先に rf_train_v3.py を実行してください")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    alpha_rise_model = joblib.load(alpha_rise_path) if os.path.exists(alpha_rise_path) else None
    alpha_drop_model = joblib.load(alpha_drop_path) if os.path.exists(alpha_drop_path) else None
    print(f"\n上昇モデル読み込み: {rise_path}")
    if drop_model:        print(f"下落モデル読み込み: {drop_path}")
    if alpha_rise_model:  print(f"α上昇モデル読み込み: {alpha_rise_path}")
    if alpha_drop_model:  print(f"α下落モデル読み込み: {alpha_drop_path}")

    # 日経225リターン取得 + 相場レジーム判定
    print("\n日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD

    # 相場レジーム判定（SMA63/200ベース）
    regime = 'uncertain'
    try:
        from lib.utils import get_nikkei_returns as _gnr
        import requests as _req
        from datetime import timedelta as _td
        _url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
                f"?interval=1d&period1={int((__import__('datetime').datetime.now()-_td(days=400)).timestamp())}"
                f"&period2={int(__import__('datetime').datetime.now().timestamp())}")
        _resp = _req.get(_url, headers=HEADERS, timeout=15)
        _data = _resp.json()
        _result = _data.get("chart", {}).get("result", [])
        if _result:
            _closes = [c for c in _result[0].get("indicators",{}).get("adjclose",[{}])[0].get("adjclose",[]) if c]
            regime = classify_market_regime(_closes)
    except Exception:
        pass

    # レジーム別 動的銘柄数（Daniel & Moskowitz 2016: 強気時は拡大、弱気時は縮小）
    regime_top_n = {'bull': 10, 'uncertain': 5, 'bear': 3}
    dynamic_top_n = regime_top_n.get(regime, 5)

    # ── VIX恐怖指数・S&P500・USD/JPY（クロスアセット）取得 ─────────────────────
    print("\nマクロデータ取得中（VIX・S&P500・USD/JPY）...")
    _live_macro = {"vix": None, "us5": None, "us20": None}
    _live_jpy   = {"jpy5": None, "usdjpy_closes": None}
    try:
        _vix_df = get_market_index_df_cached("VIX",    "%5EVIX",    days=60)
        if _vix_df is not None and len(_vix_df) > 0:
            _live_macro["vix"] = float(_vix_df["Close"].iloc[-1])
            print(f"  VIX: {_live_macro['vix']:.1f}")
        _sp5_df = get_market_index_df_cached("SP500",  "%5EGSPC",   days=60)
        if _sp5_df is not None and len(_sp5_df) >= 21:
            _p = _sp5_df["Close"].values
            _live_macro["us5"]  = round((_p[-1] - _p[-6])  / _p[-6]  * 100, 2) if len(_p) >= 6  else 0.0
            _live_macro["us20"] = round((_p[-1] - _p[-21]) / _p[-21] * 100, 2) if len(_p) >= 21 else 0.0
            print(f"  S&P500: 5日{_live_macro['us5']:+.2f}% / 20日{_live_macro['us20']:+.2f}%")
        _jpy_df = get_market_index_df_cached("USDJPY", "USDJPY%3DX", days=120)
        if _jpy_df is not None and len(_jpy_df) >= 6:
            _jc = _jpy_df["Close"].values
            _live_jpy["jpy5"] = round((_jc[-1] - _jc[-6]) / _jc[-6] * 100, 2) if len(_jc) >= 6 else 0.0
            _live_jpy["usdjpy_closes"] = _jc
            print(f"  USD/JPY: 5日{_live_jpy['jpy5']:+.2f}%")
    except Exception as _e:
        print(f"  マクロデータ取得失敗: {_e}")

    # VIX レジーム調整: VIX > 30 は恐怖相場 → top_n を -1（最小1）
    vix_val = _live_macro.get("vix")
    if vix_val is not None and vix_val > 30:
        dynamic_top_n = max(1, dynamic_top_n - 1)
        print(f"  ⚠️ 高VIX({vix_val:.1f} > 30): 推奨銘柄数を {dynamic_top_n + 1}→{dynamic_top_n}に縮小")

    # N225終値系列を取得して強気判定（βフィルター用）
    _nk225_closes = None
    _nk225_bull = False
    try:
        _nk_df = get_market_index_df_cached("N225", "%5EN225", 400)
        if _nk_df is not None and len(_nk_df) >= BULL_SMA_PERIOD:
            _nk225_closes = _nk_df["Close"].values
            _nk225_bull = _is_nk225_bull(_nk225_closes, BULL_SMA_PERIOD)
    except Exception:
        pass

    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        regime_label = {'bull': '📈強気', 'bear': '📉弱気', 'uncertain': '🔶中立'}.get(regime, regime)
        bull_label = "強気(SMA20超)" if _nk225_bull else "非強気"
        print(f"  相場レジーム: {regime_label}  →  推奨銘柄数: {dynamic_top_n}銘柄")
        print(f"  βフィルター: {bull_label} → {'β>={:.1f}の銘柄のみ💎対象'.format(BETA_MIN_BULL) if _nk225_bull else 'フィルターなし'}")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）")
    else:
        print("  日経225: 取得失敗（相対リターンはN/A）")
        is_bear = False
        dynamic_top_n = 5

    # ── 市場状況の警告のみ（停止しない — 最終判断はユーザーが行う）
    if nk20 is not None and nk20 < MARKET_TIMING_20D_THRESH:
        print(f"\n⚠️ 下落注意（日経20日: {nk20:+.1f}%）— 相場判断はご自身で。シグナルは継続出力します。")

    # 全TSE銘柄リスト取得（JPX直読み）
    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return
    codes = stock_list["code"].tolist()
    names = dict(zip(stock_list["code"], stock_list["name"]))
    print(f"全銘柄スキャン: {len(codes)} 銘柄")
    print(f"\n確率スコア計算中（並列処理）...")

    # フェーズ1: 全銘柄の特徴量を収集（並列）
    nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
    raw_data = []
    lock = threading.Lock()
    done_count = [0]
    fund_map = {}   # code -> {"PER","PBR","ROE"}（全銘柄、pit eps/bps から算出）
    total = len(codes)

    def fetch_one(code, _macro=_live_macro, _jpy=_live_jpy):
        from lib.utils import _days_to_nearest_event
        from datetime import date as _date
        prices = get_prices(code, days=400)
        with lock:
            done_count[0] += 1
            if done_count[0] % 500 == 0:
                print(f"  {done_count[0]}/{total} 取得済み... (有効: {len(raw_data)}銘柄)")
        if prices is None or len(prices) < 91:
            time.sleep(0.1)
            return None

        # ファンダメンタル取得（PER/PBR/ROE/決算まで日数/権利落ち後経過日数）
        from lib.fundamentals import get_pit_fundamentals as _get_pit
        fd_raw    = get_fundamentals(code)
        today     = _date.today()
        pit       = _get_pit(code, today) or {}
        per_live = fd_raw.get("PER"); pbr_live = fd_raw.get("PBR")
        # PER/PBR は J-Quants の eps/bps から算出。
        # ライブ取得(yfinance)は日本株でNoneが多いのでフォールバックに留める。
        from lib.fundamentals import get_pit_valuation as _get_val
        _val = _get_val(code, today)
        _close_px = float(prices["Close"].iloc[-1]) if len(prices) > 0 else None
        _eps = _val.get("eps"); _bps = _val.get("bps")
        per_calc = (round(_close_px / _eps, 1) if _eps and _eps > 0 and _close_px else per_live)
        pbr_calc = (round(_close_px / _bps, 2) if _bps and _bps > 0 and _close_px else pbr_live)
        roe_calc = fd_raw.get("ROE") if fd_raw.get("ROE") is not None else pit.get("roe")
        with lock:
            fund_map[code] = {
                "PER": per_calc, "PBR": pbr_calc, "ROE": roe_calc,
                "piotroski":    pit.get("piotroski"),
                "bps_growth":   pit.get("bps_growth"),
                "eps_surprise": pit.get("eps_surprise"),
            }
        # 配当利回り: ライブ株価 × PBR/PER から配当を逆算 or pit.dps使用
        _dps = pit.get("dps"); _close = prices["Close"].iloc[-1] if len(prices) > 0 else None
        div_yield_live = (_dps / _close * 100) if _dps and _dps > 0 and _close else None
        # USD/JPY ベータ（ライブ: 直近60日の株価 vs USD/JPY のベータ）
        _fx_beta_live = None
        _jpy_closes = _jpy.get("usdjpy_closes")
        if _jpy_closes is not None and len(prices) >= 61:
            p_arr_live = prices["Close"].values
            stock_rets_live = np.diff(p_arr_live[-61:]) / p_arr_live[-61:-1]
            fx_rets_live = np.diff(_jpy_closes[-min(61, len(_jpy_closes)):]) / _jpy_closes[-min(61, len(_jpy_closes)):-1]
            _ml = min(len(stock_rets_live), len(fx_rets_live))
            if _ml >= 20:
                _sr = stock_rets_live[:_ml]; _fr = fx_rets_live[:_ml]
                _vfx = np.var(_fr)
                if _vfx > 0:
                    _fx_beta_live = float(np.cov(_sr, _fr)[0, 1] / _vfx)

        fundamentals = {
            "per":                 per_live,
            "pbr":                 pbr_live,
            "roe":                 fd_raw.get("ROE"),
            "days_to_earnings":    None,
            "days_since_div_ex":   pit.get("days_since_div_ex"),
            "month":               today.month,
            "div_yield":           div_yield_live,
            "eps_growth":          pit.get("eps_growth"),
            "dps_growth":          pit.get("dps_growth"),
            # マクロ特徴量
            "vix":                 _macro.get("vix"),
            "us5":                 _macro.get("us5"),
            "us20":                _macro.get("us20"),
            # 新規IB特徴量
            "fx_beta":             _fx_beta_live,
            "jpy5":                _jpy.get("jpy5"),
            "eps_surprise":        pit.get("eps_surprise"),
            "bps_growth":          pit.get("bps_growth"),
            "piotroski":           pit.get("piotroski"),
            "payout":              pit.get("payout"),
            "accruals":            pit.get("accruals"),
        }

        feat = extract_features(
            prices["Close"].values,
            prices["Volume"].tolist() if "Volume" in prices.columns else None,
            nk_rets,
            fundamentals=fundamentals,
        )
        if feat is None:
            time.sleep(0.1)
            return None
        time.sleep(0.2)
        return (code, prices, feat)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with lock:
                    raw_data.append(result)

    print(f"有効銘柄: {len(raw_data)} 件")

    # フェーズ2: クロスセクショナルランク特徴量を付加（同日内での相対順位）
    if not raw_data:
        print("ERROR: 有効銘柄なし"); return
    feats_matrix = np.array([d[2] for d in raw_data], dtype=float)
    # 推論時: 全銘柄を同一日として扱い、セクター内相対モメンタムも計算
    from lib.utils import get_sector_cached as _gsc
    _sectors_for_batch = [_gsc(str(d[0])) for d in raw_data]
    feats_aug = add_cs_rank_features(feats_matrix, sectors=_sectors_for_batch)

    # フェーズ3: モデルスコア計算
    results = []
    for idx, (code, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1])
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) if drop_model else None
        alpha_rise_prob = float(alpha_rise_model.predict_proba([feat_aug])[0][1]) if alpha_rise_model else None
        alpha_drop_prob = float(alpha_drop_model.predict_proba([feat_aug])[0][1]) if alpha_drop_model else None
        close = float(prices["Close"].iloc[-1])
        rise_pct = round(rise_prob * 100, 1)
        drop_pct = round(drop_prob * 100, 1) if drop_prob is not None else None
        # 4モデルアンサンブル: net = (rise-drop) + (alpha_rise-alpha_drop)
        abs_net = (rise_pct - drop_pct) if drop_pct is not None else rise_pct
        alpha_net = 0.0
        if alpha_rise_prob is not None and alpha_drop_prob is not None:
            alpha_net = (alpha_rise_prob - alpha_drop_prob) * 100
        net = round(abs_net + alpha_net, 1)

        # ボラティリティ（feat[7] = vol20, 年率換算%）
        vol = round(feat[7], 1)
        if vol < 20:
            vol_label = "🟢低"
        elif vol < 40:
            vol_label = "🟡中"
        elif vol < 60:
            vol_label = "🟠高"
        else:
            vol_label = "🔴超高"

        # ネットスコア判定
        if net >= 15:
            judgment = "🟢強気買い"
        elif net >= 5:
            judgment = "🔵やや強気"
        elif net >= -5:
            judgment = "🟡中立    "
        else:
            judgment = "🟠弱気    "

        volumes = prices["Volume"].tolist() if "Volume" in prices.columns else []
        p_arr = prices["Close"].values
        ret_504 = float((p_arr[-1]-p_arr[-505])/p_arr[-505]) if len(p_arr) >= 505 else None
        p504 = p_arr[-504:] if len(p_arr) >= 504 else p_arr
        t504 = np.arange(len(p504), dtype=float)
        _coef504 = np.polyfit(t504, p504, 1)
        _pred504 = np.polyval(_coef504, t504)
        _ss_res504 = float(np.sum((p504 - _pred504)**2))
        _ss_tot504 = float(np.sum((p504 - p504.mean())**2))
        r2_504 = 1.0 - _ss_res504 / _ss_tot504 if _ss_tot504 > 0 else 0.0
        buy_ok = passes_buy_filter(feat, close, volumes)
        _fm = fund_map.get(code) or {}
        valid_vols = [v for v in volumes[-20:] if v is not None and not np.isnan(v)]
        _turnover_m = float(np.mean(valid_vols) * close / 1e6) if len(valid_vols) >= 10 else None
        recommend = recommend_from_scores(
            net, drop_pct, allow_buy=buy_ok, vol=vol,
            piotroski=_fm.get("piotroski"),
            pos52=float(feat[9]),
            bps_growth=_fm.get("bps_growth"),
            eps_surprise=_fm.get("eps_surprise"),
            ret90=float(feat[3]),
            turnover_m=_turnover_m,
        )

        # βフィルター: 日経強気時に低β銘柄の💎買いを降格
        if recommend == "💎 買い" and _nk225_bull and _nk225_closes is not None:
            p_arr_beta = prices["Close"].values
            beta = _calc_nk225_beta(p_arr_beta, _nk225_closes)
            if beta is not None and beta < BETA_MIN_BULL:
                recommend = "⏳ 方向感なし"

        # 日経比相対リターン
        p = prices["Close"].values
        s5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) >= 6  else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        s60 = (p[-1] - p[-61]) / p[-61] * 100 if len(p) >= 61 else 0
        rel5  = round(s5  - nk5,  2) if nk5  is not None else None
        rel20 = round(s20 - nk20, 2) if nk20 is not None else None
        rel60 = round(s60 - nk60, 2) if nk60 is not None else None
        rels = [r for r in [rel5, rel20, rel60] if r is not None]
        rs_score = round(sum(rels) / len(rels), 2) if rels else None

        cs_vol20_rank = round(float(feat_aug[32]) * 100, 0)  # ボラティリティのCS相対ランク(0-100%)

        row = {
            "銘柄コード": code,
            "銘柄名": names.get(code, ""),
            "直近株価(円)": round(close, 1),
            "上昇確率(%)": rise_pct,
            "下落確率(%)": drop_pct if drop_pct is not None else "-",
            "ネット(%)": net,
            "判定": judgment,
            "ボラ(%)": vol,
            "ボラ水準": vol_label,
            "ボラランク(%)": cs_vol20_rank,
            "推奨": recommend,
            "日経比5日(%)": rel5 if rel5 is not None else "-",
            "日経比20日(%)": rel20 if rel20 is not None else "-",
            "日経比60日(%)": rel60 if rel60 is not None else "-",
            "相対強度": rs_score if rs_score is not None else "-",
            "PER": (fund_map.get(code) or {}).get("PER"),
            "PBR": (fund_map.get(code) or {}).get("PBR"),
            "ROE(%)": (fund_map.get(code) or {}).get("ROE"),
            "piotroski":    (fund_map.get(code) or {}).get("piotroski"),
            "bps_growth":   (fund_map.get(code) or {}).get("bps_growth"),
            "eps_surprise": (fund_map.get(code) or {}).get("eps_surprise"),
            "pos52":        round(float(feat[9]), 3),
        }
        results.append(row)

    # ランキング（ネットスコア順）
    result_df = pd.DataFrame(results).sort_values("ネット(%)", ascending=False).reset_index(drop=True)
    result_df.index += 1
    result_df.insert(0, "順位", result_df.index)

    # PER/PBR/ROE は全銘柄 fund_map（pit eps/bps 由来）で既に設定済み

    # 表示（動的銘柄数: レジームに応じて 3/5/10）
    print(f"\n{'='*90}")
    regime_label_disp = {'bull': '📈強気', 'bear': '📉弱気', 'uncertain': '🔶中立'}.get(regime, regime)
    print(f"上位{dynamic_top_n}銘柄ランキング [{regime_label_disp}レジーム]（ネットスコア順）")
    if is_bear:
        print(f"⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")
    print(f"{'='*90}")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<16}  {'株価':>8}  {'ネット':>7}  {'判定':<12}  "
          f"{'PER':>6}  {'PBR':>5}  {'感情':>5}  {'Gトレ':>5}  推奨")
    print("-" * 140)
    for _, row in result_df.head(dynamic_top_n).iterrows():
        per_val = row.get("PER"); pbr_val = row.get("PBR")
        per_str = f"{per_val:>5.1f}x" if per_val is not None else "   N/A"
        pbr_str = f"{pbr_val:>4.2f}x" if pbr_val is not None else "  N/A"

        # NLP感情
        sent = row.get("感情スコア", 0.0) or 0.0
        sent_emoji = "😊" if sent > 0.3 else ("😞" if sent < -0.3 else "😐")
        sent_str = f"{sent_emoji}{sent:+.1f}"

        # Googleトレンド
        gtr = row.get("Gトレンド", 0.0) or 0.0
        gtr_str = f"{'↑' if gtr > 0.2 else ('↓' if gtr < -0.1 else '→')}{gtr:+.1f}"

        print(
            f"{int(row['順位']):>4}  {row['銘柄コード']:>6}  "
            f"{str(row['銘柄名']):<16}  "
            f"{row['直近株価(円)']:>8,.0f}円  "
            f"{row['ネット(%)']:>+6.1f}%  "
            f"{row['判定']:<12}  "
            f"{per_str}  {pbr_str}  "
            f"{sent_str:>5}  "
            f"{gtr_str:>5}  "
            f"{row['推奨']}"
        )

    # フェーズ4b: オルタナティブデータ取得（上位20銘柄）
    ALT_TOP = min(20, len(result_df))
    print(f"\nオルタナティブデータ取得中（上位{ALT_TOP}銘柄）...")
    print("  対象: Googleトレンド")
    alt_results = {}
    alt_errors = 0
    try:
        from lib.alt_data import get_alt_signals

        def _fetch_alt(code, name):
            try:
                return str(code), get_alt_signals(str(code), str(name))
            except Exception:
                return str(code), {}

        with ThreadPoolExecutor(max_workers=5) as _exc:
            _futures = {
                _exc.submit(_fetch_alt, row["銘柄コード"], row["銘柄名"]): row["銘柄コード"]
                for _, row in result_df.head(ALT_TOP).iterrows()
            }
            for _f in as_completed(_futures):
                try:
                    _code, _data = _f.result()
                    alt_results[_code] = _data
                except Exception:
                    alt_errors += 1

        def _safe_get(code, key, default=None):
            return alt_results.get(str(code), {}).get(key, default)

        result_df["Gトレンド"]      = result_df["銘柄コード"].astype(str).map(lambda x: _safe_get(x, "trend_score", 0.0))

        print(f"  取得完了: {len(alt_results)}件 / エラー: {alt_errors}件")
    except Exception as _ae:
        print(f"  オルタナティブデータ取得エラー: {_ae}")
        result_df["Gトレンド"] = 0.0

    # フェーズ4d: 上位銘柄の決算テキスト感情分析（Claude Haiku NLP）
    NLP_TOP = min(20, len(result_df))
    print(f"\n決算テキスト感情分析中（上位{NLP_TOP}銘柄、Claude Haiku）...")
    try:
        from lib.nlp_sentiment import get_earnings_sentiment
        sentiment_scores = {}
        for _, row in result_df.head(NLP_TOP).iterrows():
            c = str(row["銘柄コード"])
            sentiment_scores[c] = get_earnings_sentiment(c)
        result_df["感情スコア"] = result_df["銘柄コード"].astype(str).map(
            lambda x: sentiment_scores.get(x, 0.0)
        )
        # 強い悲観（< -0.5）の場合は 💎 買い → 方向感なし に降格
        for idx, row in result_df.iterrows():
            if row.get("推奨") == "💎 買い" and row.get("感情スコア", 0.0) <= -0.5:
                result_df.at[idx, "推奨"] = "⏳ 方向感なし"
                print(f"  ⚠️ {row['銘柄名']}({row['銘柄コード']}): 感情スコア{row['感情スコア']:.2f} → S買い降格")
        pos_count = (result_df.head(NLP_TOP)["感情スコア"] > 0.2).sum()
        neg_count = (result_df.head(NLP_TOP)["感情スコア"] < -0.2).sum()
        print(f"  楽観的: {pos_count}銘柄 / 悲観的: {neg_count}銘柄 / 中立: {NLP_TOP-pos_count-neg_count}銘柄")
    except Exception as _e:
        print(f"  感情分析スキップ（APIキー未設定 or エラー: {_e}）")
        result_df["感情スコア"] = 0.0

    # フェーズ5: 株主優待権利落ち日チェック（権利落ち日21日前以内は除外）
    buy_mask = result_df["推奨"] == "💎 買い"
    buy_codes = result_df.loc[buy_mask, "銘柄コード"].astype(str).tolist()
    if buy_codes:
        print(f"\n株主優待権利落ちチェック中（S買い {len(buy_codes)}銘柄）...")
        today = datetime.now().date()
        for code in buy_codes:
            days = _days_to_yutai_record(code, today)
            if days is not None and 0 <= days <= YUTAI_SKIP_DAYS:
                idx = result_df[result_df["銘柄コード"].astype(str) == code].index
                result_df.loc[idx, "推奨"] = "⏳ 方向感なし"
                name = result_df.loc[idx, "銘柄名"].values[0]
                print(f"  ⚠️ {name}({code}): 優待権利落ち{days}日前 → S買いを方向感なしに降格")

    # フェーズ7: 米国セクターETF前日リターンフィルター（リードラグ効果）
    # 強相関セクター(XLK/XLF/XLI/XLB/XLV/XLY)のETFが前日マイナスならS買い→方向感なし に降格
    buy_mask = result_df["推奨"] == "💎 買い"
    buy_codes = result_df.loc[buy_mask, "銘柄コード"].astype(str).tolist()
    if buy_codes:
        print(f"\n米国ETFリードラグフィルター中（S買い {len(buy_codes)}銘柄）...")
        _load_sector_cache()
        etf_rets = fetch_us_sector_etf_returns()
        if etf_rets:
            ret_str = " ".join(f"{k}:{v:+.1f}%" for k, v in sorted(etf_rets.items()))
            print(f"  前営業日ETFリターン: {ret_str}")
            degraded = []
            for code in buy_codes:
                etf = get_sector_etf(code)
                if etf not in STRONG_EFFECT_ETFS:
                    continue
                ret = etf_rets.get(etf)
                if ret is not None and ret < 0:
                    idx = result_df[result_df["銘柄コード"].astype(str) == code].index
                    name = result_df.loc[idx, "銘柄名"].values[0]
                    result_df.loc[idx, "推奨"] = "⏳ 方向感なし"
                    degraded.append(f"{name}({code})[{etf}:{ret:+.1f}%] S買い→方向感なし")
            _save_sector_cache()
            if degraded:
                print(f"  ⚠️ ETF前日マイナスのため降格: {', '.join(degraded)}")
            else:
                print(f"  ✅ 全S買い銘柄のETFは前日プラス（フィルター通過）")
        else:
            print(f"  ETFデータ取得失敗: フィルタースキップ")

    # フェーズ8: 相場リスク管制官 — マクロからリスクオン/オフを判定し、
    #            リスクオフ地合いではS買いを全件見送り（自動防御）
    from lib.risk_regime import assess as _assess_risk, summary_line as _risk_summary
    risk_verdict = _assess_risk(
        nk20=nk20,
        vix=_live_macro.get("vix"),
        jpy5=_live_jpy.get("jpy5"),
        us20=_live_macro.get("us20"),
        us5=_live_macro.get("us5"),
    )
    print(f"\n🛡️ 相場リスク管制官: {_risk_summary(risk_verdict)}")
    if risk_verdict["suppress_buy"]:
        risk_buy = result_df[result_df["推奨"] == "💎 買い"]["銘柄コード"].astype(str).tolist()
        for code in risk_buy:
            idx = result_df[result_df["銘柄コード"].astype(str) == code].index
            result_df.loc[idx, "推奨"] = "⏳ 方向感なし"
        if risk_buy:
            print(f"  🔴 リスクオフ地合い → S買い{len(risk_buy)}件を全て見送り（方向感なしに降格）")
    # 当日のリスク判定を保存（メール・Web・活動ログが参照）
    try:
        import json as _json
        from datetime import datetime as _dt
        _risk_out = {"date": _dt.now().strftime("%Y-%m-%d"), **risk_verdict}
        with open(os.path.join(BASE_DIR, "data", "risk_regime.json"), "w", encoding="utf-8") as _f:
            _json.dump(_risk_out, _f, ensure_ascii=False)
    except Exception as _e:
        print(f"  リスク判定の保存失敗（無視）: {_e}")

    # CSV保存
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(os.path.join(BASE_DIR, "data", "rankings"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "data", "rankings", f"ranking_{date_str}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")

    # DB保存
    from lib.db import save_daily_ranking
    db_date_str = datetime.now().strftime("%Y-%m-%d")
    db_rows = [
        {
            "code": str(row["銘柄コード"]),
            "name": row["銘柄名"],
            "close": row["直近株価(円)"],
            "rise_prob": row["上昇確率(%)"],
            "drop_prob": row["下落確率(%)"] if row["下落確率(%)"] != "-" else None,
            "net": row["ネット(%)"],
            "vol": row["ボラ(%)"],
            "recommend": row["推奨"],
            "rel20": row["日経比20日(%)"] if row["日経比20日(%)"] != "-" else None,
            "per":          row.get("PER"),
            "pbr":          row.get("PBR"),
            "piotroski":    row.get("piotroski"),
            "bps_growth":   row.get("bps_growth"),
            "eps_surprise": row.get("eps_surprise"),
            "pos52":        row.get("pos52"),
        }
        for _, row in result_df.iterrows()
    ]
    save_daily_ranking(db_date_str, db_rows)
    print(f"DB保存: {len(db_rows)}件 → stock_alert.db")

    # QA: 出力データの不変条件チェック（alert-only。違反でも処理は止めない）
    try:
        from lib.data_sanity import run_gate
        run_gate(db_rows, source="rank_stocks", alert=True)
    except Exception as _e:
        print(f"[rank_stocks] QAチェックでエラー（無視して継続）: {_e}")

    print("完了")


if __name__ == "__main__":
    main()

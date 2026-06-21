"""
utils.py - 共通関数モジュール
rank_stocks.py / backtest.py で共有
"""
import re
import os
import functools
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
SEQ_DAYS = 60
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")


def get_prices(code, days=400):
    ticker = f"{code}.T"
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        adjcloses  = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        raw_closes = result[0].get("indicators", {}).get("quote",    [{}])[0].get("close",    [])
        volumes    = result[0].get("indicators", {}).get("quote",    [{}])[0].get("volume",   [])
        if not timestamps or not adjcloses:
            return None
        # adjcloseがNoneの行はraw closeで補完（市場終了直後の計算ラグ対策）
        closes = [a if a is not None else r for a, r in zip(adjcloses, raw_closes)]
        df = pd.DataFrame(
            {"Close": closes, "Volume": volumes},
            index=pd.to_datetime(timestamps, unit="s", utc=True)
        )
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def get_price_at_date(code, target_date):
    """target_date前後の最も近い終値を返す（±14日範囲で探索）"""
    start_ts = int((target_date - timedelta(days=14)).timestamp())
    end_ts   = int((target_date + timedelta(days=14)).timestamp())
    ticker = f"{code}.T"
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        target_ts = target_date.timestamp()
        best_price, best_diff = None, float("inf")
        for ts, c in zip(timestamps, closes):
            if c is None:
                continue
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff, best_price = diff, c
        return best_price
    except Exception:
        return None


def get_market_index_df(ticker_encoded, days=2200):
    """Yahoo Finance から市場指数の日次終値を date-indexed DataFrame で返す。
    ticker_encoded: URLエンコード済みティッカー（例: %5EN225, %5EVIX, %5EGSPC）
    """
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_encoded}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        ts     = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Tokyo")
        df  = pd.DataFrame({"Close": closes}, index=idx).dropna()
        df.index = df.index.date
        return df
    except Exception:
        return None


def get_market_index_df_cached(cache_key, ticker_encoded, days=2200):
    """DBキャッシュ優先で市場指数を取得。差分のみ Yahoo から取得して保存。

    cache_key: DBのキー名 ('VIX' | 'SP500' | 'USDJPY')
    ticker_encoded: Yahoo Finance URLエンコード済みティッカー

    動作:
      1. DBの最終保存日を確認
      2. 今日以降のデータが欠けていれば差分のみ Yahoo 取得 → DB保存
      3. DBから指定 days 分を返す
      4. DB/Yahoo両方失敗時は None
    """
    from lib.db import get_market_index_latest_date, save_market_index_data, load_market_index_data
    from datetime import date as _date, timedelta as _td

    today = _date.today()
    latest_str = get_market_index_latest_date(cache_key)

    need_fetch = True
    if latest_str is not None:
        latest_date = _date.fromisoformat(latest_str)
        # 平日なら翌日、週末なら月曜を「次の営業日」と見なす
        # 簡易判定: 最終保存日が3日以内なら差分フェッチ済みとみなす
        days_since = (today - latest_date).days
        if days_since == 0:
            # 今日のデータまであり → DBから返す
            return load_market_index_data(cache_key, days=days)
        elif days_since <= 5:
            # 数日分だけ差分取得（週末 / 祝日対応）
            df_new = get_market_index_df(ticker_encoded, days=days_since + 5)
            if df_new is not None:
                df_new_diff = df_new[df_new.index > latest_date]
                save_market_index_data(cache_key, df_new_diff)
            need_fetch = False
        else:
            # 1週間以上古い → 差分取得
            fetch_days = min(days_since + 10, days)
            df_new = get_market_index_df(ticker_encoded, days=fetch_days)
            if df_new is not None:
                df_new_diff = df_new[df_new.index > latest_date]
                save_market_index_data(cache_key, df_new_diff)
            need_fetch = False

    if need_fetch:
        # DB未登録 → 全期間取得して保存
        df_full = get_market_index_df(ticker_encoded, days=days)
        if df_full is not None:
            save_market_index_data(cache_key, df_full)

    return load_market_index_data(cache_key, days=days)


@functools.lru_cache(maxsize=1)
def get_nikkei_returns():
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
           f"?interval=1d&period1={int((datetime.now()-timedelta(days=400)).timestamp())}"
           f"&period2={int(datetime.now().timestamp())}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None, None, None
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        closes = [c for c in closes if c is not None]
        if len(closes) < 61:
            return None, None, None
        p = closes
        r5  = round((p[-1]-p[-6]) /p[-6] *100, 2) if len(p)>=6  else 0
        r20 = round((p[-1]-p[-21])/p[-21]*100, 2) if len(p)>=21 else 0
        r60 = round((p[-1]-p[-61])/p[-61]*100, 2) if len(p)>=61 else 0
        return r5, r20, r60
    except Exception:
        return None, None, None


def classify_market_regime(nk_prices):
    """日経225の価格系列から相場レジームを判定する。

    Returns:
        'bull'      : 強気（スクリーナーなし推奨）
        'bear'      : 弱気（スクリーナーあり or シグナル停止）
        'uncertain' : 中間（スクリーナーあり推奨）

    判定基準:
        bull    : 日経 > SMA63 AND 20日リターン > -2%
        bear    : 日経 < SMA63 OR  20日リターン < -3%
        uncertain: それ以外
    """
    arr = [p for p in nk_prices if p is not None]
    if len(arr) < 63:
        return 'uncertain'
    arr = arr[-200:]  # 最新200日分のみ使用
    current = arr[-1]
    sma63   = sum(arr[-63:]) / 63
    sma200  = sum(arr[-200:]) / 200 if len(arr) >= 200 else sma63
    nk20    = (arr[-1] - arr[-21]) / arr[-21] * 100 if len(arr) >= 21 else 0
    nk60    = (arr[-1] - arr[-61]) / arr[-61] * 100 if len(arr) >= 61 else 0

    above_sma63  = current > sma63
    above_sma200 = current > sma200

    # 強気: SMA63/200の両方を上回り、かつ短期急落なし
    if above_sma63 and above_sma200 and nk20 > -2.0:
        return 'bull'
    # 弱気: SMA63を下回る or 短期急落 or 20日-5%超
    if not above_sma63 or nk20 < -3.0:
        return 'bear'
    # それ以外は中間
    return 'uncertain'


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas,  0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


RANK_FEAT_INDICES = [0, 1, 2, 6, 7, 9]  # ret5, ret20, ret60, rsi, vol20, pos52

_JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
_SECTOR_CACHE = None  # プロセス内キャッシュ {code: sector}

def _load_jpx_sector_map():
    """JPX Excelから {code: 33業種区分} を一括取得してDBキャッシュに保存"""
    from lib.db import get_all_sectors, save_all_sectors
    import io
    cache = get_all_sectors()
    if cache:
        return cache
    try:
        resp = requests.get(_JPX_URL, headers=HEADERS, timeout=30)
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()
        code_col = next((c for c in df.columns if "コード" in c), None)
        sec_col  = next((c for c in df.columns if "33業種区分" in c), None)
        if code_col and sec_col:
            for _, row in df.iterrows():
                code = str(row[code_col]).strip()
                sec  = str(row[sec_col]).strip()
                if code and sec and sec != "nan":
                    cache[code] = sec
            try:
                save_all_sectors(cache)
            except Exception:
                pass
    except (requests.RequestException, ValueError, OSError) as e:
        print(f"[WARN] JPX業種マップ取得失敗: {e}")
    return cache


def get_sector_cached(code):
    """業種を取得（プロセス内キャッシュ→CSVキャッシュ→JPX一括取得）"""
    global _SECTOR_CACHE
    if _SECTOR_CACHE is None:
        _SECTOR_CACHE = _load_jpx_sector_map()
    return _SECTOR_CACHE.get(str(code), "不明")


def add_cs_rank_features(X, dates=None, sectors=None):
    """Add 6 standard CS + 1 sector-relative CS = 7 total CS features.

    Training mode (dates provided): rank within each date group.
    Inference mode (dates=None): rank within the current batch (single day).
    sectors: optional sequence of sector labels (same length as X).
             If provided, adds sector-relative ret60 rank as 7th CS feature.
             If None, sector rank defaults to 0.5 for all rows.
    Returns augmented matrix shape (n, n_features + 7).
    """
    X = np.array(X, dtype=float)
    n = len(X)
    rank_matrix = np.full((n, len(RANK_FEAT_INDICES)), 0.5, dtype=float)
    RET60_IDX = 2  # ret60 is at index 2 in the feature vector

    if dates is not None:
        dates_arr = np.array(dates)
        _, inverse = np.unique(dates_arr, return_inverse=True)
        for d_idx in range(inverse.max() + 1):
            group = np.where(inverse == d_idx)[0]
            cnt = len(group)
            if cnt < 2:
                continue
            for j, fi in enumerate(RANK_FEAT_INDICES):
                vals = X[group, fi]
                order = np.argsort(np.argsort(vals))
                rank_matrix[group, j] = order / (cnt - 1)
    else:
        if n >= 2:
            for j, fi in enumerate(RANK_FEAT_INDICES):
                vals = X[:, fi]
                order = np.argsort(np.argsort(vals))
                rank_matrix[:, j] = order / (n - 1)

    # Sector-relative ret60 rank（セクター内での相対モメンタム）
    sector_rank = np.full(n, 0.5, dtype=float)
    if sectors is not None:
        sectors_arr = np.array(sectors)
        if dates is not None:
            dates_arr2 = np.array(dates)
            _, inv2 = np.unique(dates_arr2, return_inverse=True)
            for d_idx in range(inv2.max() + 1):
                date_group = np.where(inv2 == d_idx)[0]
                secs_in_group = sectors_arr[date_group]
                for sec in np.unique(secs_in_group):
                    sec_mask = secs_in_group == sec
                    sec_idx = date_group[sec_mask]
                    cnt = len(sec_idx)
                    if cnt < 2:
                        continue
                    vals = X[sec_idx, RET60_IDX]
                    order = np.argsort(np.argsort(vals))
                    sector_rank[sec_idx] = order / (cnt - 1)
        else:
            for sec in np.unique(sectors_arr):
                sec_idx = np.where(sectors_arr == sec)[0]
                cnt = len(sec_idx)
                if cnt < 2:
                    continue
                vals = X[sec_idx, RET60_IDX]
                order = np.argsort(np.argsort(vals))
                sector_rank[sec_idx] = order / (cnt - 1)

    return np.hstack([X, rank_matrix, sector_rank.reshape(-1, 1)])


def compute_seq_features(seq):
    """60日リターン系列 → 7次元要約統計量"""
    s = np.array(seq, dtype=float)
    if len(s) < 2:
        return [0.0] * 7
    mean_s = s.mean(); std_s = s.std()
    if std_s > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            ac = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    else:
        ac = 0.0
    if np.isnan(ac): ac = 0.0
    skew = float(((s - mean_s) ** 3).mean() / (std_s ** 3 + 1e-10))
    max_r = float(s.max())
    min_r = float(s.min())
    pos_ratio = float((s > 0).mean())
    t = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(t, s, 1)[0])
    mid = len(s) // 2
    recent_vs_early = float(s[mid:].mean() - s[:mid].mean())
    return [ac, skew, max_r, min_r, pos_ratio, slope, recent_vs_early]


def _days_to_nearest_event(from_date, months, day=25):
    """from_date から最も近い未来の(year, month, day)までの日数を返す。"""
    import calendar as _cal
    from datetime import date as _date
    best = 9999
    for m in months:
        for yr_add in [0, 1]:
            yr = from_date.year + yr_add
            last = _cal.monthrange(yr, m)[1]
            d = _date(yr, m, min(day, last))
            delta = (d - from_date).days
            if 0 <= delta < best:
                best = delta
    return best if best < 9999 else None


def extract_features(p, v=None, nk_rets=None, fundamentals=None):
    """54次元特徴量: テクニカル10 + トレンド反転5 + 出来高3 + 日経マクロ3 + 60日系列要約7 + 日経相対アルファ4 + ファンダメンタル11 + マクロ拡張4 + 新規IB8 + EDINET1
    fundamentals dict keys (all optional):
      per, pbr, roe, days_to_earnings, days_since_div_ex,
      month（カレンダー月 1-12），div_yield（配当利回り%）
      eps_growth（EPS前年比），dps_growth（DPS前年比）
      vix（VIX指数水準），us5（SP500 5日リターン%），us20（SP500 20日リターン%）
      fx_beta（USD/JPY 60日ベータ），jpy5（USD/JPY 5日リターン%）
      eps_surprise（EPS実績-線形予測乖離），bps_growth（BPS前年比）
      piotroski（簡易Fスコア 0-1），payout（配当性向 DPS/EPS），accruals（BSアクルーアル）
      edinet_holding（直近90日の大量保有報告 保有割合%）
    """
    if len(p) < 91 or p[-1] == 0:
        return None
    c = p[-1]

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
    seq_feat = compute_seq_features(seq_raw)

    # 日経相対アルファ4特徴量（nk5/nk20/nk60 は % 単位なので 0.01 倍して fraction に揃える）
    rel5  = ret5  - nk5  * 0.01
    rel20 = ret20 - nk20 * 0.01
    rel60 = ret60 - nk60 * 0.01
    alpha_momentum = rel5 - rel20 / 4  # アルファ加速度

    # ファンダメンタル特徴量
    import math as _math
    fd    = fundamentals or {}
    _per  = fd.get('per');   _pbr = fd.get('pbr');   _roe = fd.get('roe')
    _dte  = fd.get('days_to_earnings')
    _ddiv = fd.get('days_since_div_ex')
    _dyld = fd.get('div_yield')
    _mon  = fd.get('month')
    _epsg = fd.get('eps_growth')
    _dpsg = fd.get('dps_growth')
    _vix  = fd.get('vix')
    _us5  = fd.get('us5')
    _us20 = fd.get('us20')
    # 新規IB特徴量
    _fxb   = fd.get('fx_beta')
    _jpy5  = fd.get('jpy5')
    _epss  = fd.get('eps_surprise')
    _bpsg  = fd.get('bps_growth')
    _pio   = fd.get('piotroski')
    _pyout = fd.get('payout')
    _accr  = fd.get('accruals')
    _ehold = fd.get('edinet_holding')

    per_feat      = float(np.clip(_per  / 20.0 - 1.0, -1.0, 3.0)) if _per  is not None else 0.0
    pbr_feat      = float(np.clip(_pbr  /  1.5 - 1.0, -1.0, 4.0)) if _pbr  is not None else 0.0
    roe_feat      = float(np.clip(_roe  / 15.0,        -0.5, 2.0)) if _roe  is not None else 0.0
    earn_feat     = float(np.clip(_dte  / 90.0,         0.0, 1.0)) if _dte  is not None else 0.5
    div_ex_feat   = float(np.clip(_ddiv / 60.0,         0.0, 1.0)) if _ddiv is not None else 0.5
    sin_month     = _math.sin(2 * _math.pi * _mon / 12) if _mon is not None else 0.0
    cos_month     = _math.cos(2 * _math.pi * _mon / 12) if _mon is not None else 1.0
    div_yield_f   = float(np.clip(_dyld / 10.0,          0.0, 1.0)) if _dyld is not None else 0.0
    eps_growth_f  = float(np.clip(_epsg,                 -1.0, 3.0)) if _epsg is not None else 0.0
    dps_growth_f  = float(np.clip(_dpsg,                 -1.0, 2.0)) if _dpsg is not None else 0.0
    # マクロ拡張（VIX・SP500）
    vix_feat  = float(np.clip((_vix / 25.0) - 1.0, -1.0, 2.0)) if _vix  is not None else 0.0
    us5_f     = float(np.clip(_us5  / 10.0,         -1.0, 1.0)) if _us5  is not None else 0.0
    us20_f    = float(np.clip(_us20 / 20.0,         -1.0, 1.0)) if _us20 is not None else 0.0

    # ── Amihud非流動性（Amihud 2002）: |日次リターン|/出来高金額 の20日平均 ──────
    amihud_f = 2.0  # default = medium illiquidity
    if v is not None and len(v) >= 21:
        p_arr20 = p[-21:]
        v_arr20 = np.array([x if x is not None else np.nan for x in v[-20:]], dtype=float)
        abs_rets = np.abs(np.diff(p_arr20) / p_arr20[:-1])
        vols_yen = v_arr20 * p[-1] / 1e6   # 百万円単位
        valid = ~np.isnan(vols_yen) & (vols_yen > 0)
        if valid.sum() >= 10:
            illiq = float(np.mean(abs_rets[valid] / vols_yen[valid]))
            amihud_f = float(np.clip((np.log10(illiq + 1e-10) + 8.0) / 5.0, 0.0, 2.0))

    # ── 新規IB8特徴量 ──────────────────────────────────────────────────────────
    # USD/JPY 60日ベータ（為替感応度）: 2 = 輸出株, -1 = 内需株
    fx_beta_f     = float(np.clip(_fxb  / 2.0,   -1.0, 1.5)) if _fxb   is not None else 0.0
    # USD/JPY 5日リターン（円安/円高方向）: ±5%を±1に正規化
    jpy5_f        = float(np.clip(_jpy5 / 5.0,   -1.0, 1.0)) if _jpy5  is not None else 0.0
    # EPS surprise（実績 vs トレンド外挿の乖離率）
    eps_surprise_f = float(np.clip(_epss,         -1.5, 2.0)) if _epss  is not None else 0.0
    # BPS前年比成長率（簿価蓄積 — 資本効率の代理変数）
    bps_growth_f  = float(np.clip(_bpsg,          -0.5, 1.5)) if _bpsg  is not None else 0.0
    # Piotroski F-score簡易版（0=低品質, 1=高品質、0.5をデフォルト）
    piotroski_f   = float(np.clip(_pio,            0.0, 1.0)) if _pio   is not None else 0.5
    # 配当性向（DPS/EPS）: 高=成熟/還元型, 低=成長型
    payout_f      = float(np.clip(_pyout,          0.0, 1.5)) if _pyout is not None else 0.5
    # アクルーアル: J-Quants Sloan正確版 (NP-CFO)/TA×5 or BPSプロキシ（フォールバック）
    accruals_f    = float(np.clip(_accr,          -0.3, 0.5)) if _accr  is not None else 0.0
    # EDINET大量保有報告: 直近90日以内の保有割合（0=なし, 0-1正規化）
    edinet_hold_f = float(np.clip(_ehold / 50.0,   0.0, 1.0)) if _ehold is not None else 0.0

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + seq_feat + [rel5, rel20, rel60, alpha_momentum,
            per_feat, pbr_feat, roe_feat, earn_feat, div_ex_feat,
            sin_month, cos_month, div_yield_f, eps_growth_f,
            dps_growth_f, vix_feat, us5_f, us20_f,
            amihud_f, fx_beta_f, jpy5_f,
            eps_surprise_f, bps_growth_f, piotroski_f, payout_f, accruals_f,
            edinet_hold_f]

    if any(np.isnan(feat[:10])) or any(np.isinf(feat[:10])):
        return None
    return feat


class IsotonicCalibrated:
    """joblib保存互換: XGBoost + IsotonicRegressionキャリブレーションラッパー"""
    def __init__(self, model, iso):
        self.model = model
        self.iso   = iso

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])


_KABUTAN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en;q=0.9",
}

def get_fundamentals(code):
    """PER/PBR/ROE を kabutan.jp からスクレイピング"""
    result = {"PER": None, "PBR": None, "ROE": None}
    try:
        # PER / PBR: メインページ
        resp = requests.get(f"https://kabutan.jp/stock/?code={code}",
                            headers=_KABUTAN_HEADERS, timeout=8)
        if resp.status_code == 200:
            text = resp.text.replace("\n", "").replace("\t", "")
            idx = text.find('data-help="PER"')
            if idx != -1:
                vals = re.findall(r'<td>([\d.-]+)<span', text[idx:idx+600])
                if len(vals) >= 1:
                    result["PER"] = float(vals[0])
                if len(vals) >= 2:
                    result["PBR"] = float(vals[1])
        # ROE: finance ページ（最新年度の値、列順: 売上/利益/ROE/ROA/...）
        resp2 = requests.get(f"https://kabutan.jp/stock/finance/?code={code}",
                             headers=_KABUTAN_HEADERS, timeout=8)
        if resp2.status_code == 200:
            text2 = resp2.text.replace(" ", "").replace("\n", "").replace("\t", "")
            idx2 = text2.find('ROE">')
            if idx2 != -1:
                tbody_idx = text2.find("<tbody>", idx2)
                rows = re.findall(r'<tr><thscope="row".*?</tr>', text2[tbody_idx:tbody_idx+1200])
                if rows:
                    vals = re.findall(r'<td[^>]*>([\d,.-]+)</td>', rows[0])
                    if len(vals) >= 3:
                        try:
                            result["ROE"] = float(vals[2])
                        except ValueError:
                            pass
    except (requests.RequestException, ValueError, IndexError):
        pass
    return result


# ── 推奨ラベル（rank_stocks 共通） ───────────────────────────

def recommend_from_net(net, allow_buy=True):
    if net < -10:
        return "🔴 下降シグナル"
    if net < -5:
        return "⚠️ 弱気シグナル"
    return "⏳ 方向感なし"


def recommend_from_scores(net, drop_prob=None, allow_buy=True, vol=None,
                          piotroski=None, pos52=None, bps_growth=None, eps_surprise=None,
                          ret90=None, turnover_m=None):
    """💎 買い: drop_prob<2% AND net>=16 AND Piotroski>=6/9 AND pos52<0.80
               AND vol<=20% AND ret90>-25% AND turnover>=50M AND 業績改善（EPS>2% or BPS成長+ or データなし）"""
    has_fundamentals = (eps_surprise is not None or bps_growth is not None)
    biz_ok = (not has_fundamentals
              or (eps_surprise is not None and eps_surprise > 2.0)
              or (bps_growth is not None and bps_growth > 0))
    if (allow_buy
            and drop_prob is not None and drop_prob < 2.0
            and net >= 16.0
            and piotroski is not None and piotroski >= 0.67
            and pos52 is not None and pos52 < 0.80
            and (vol is None or vol <= 20.0)
            and (ret90 is None or ret90 > -0.25)
            and (turnover_m is None or turnover_m >= 50.0)
            and biz_ok):
        return "💎 買い"
    return recommend_from_net(net, allow_buy=allow_buy)

"""
utils.py - 共通関数モジュール
rank_stocks.py / alert_email.py / backtest.py で共有
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
SEQ_DAYS = 60


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
        closes  = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        volumes = result[0].get("indicators", {}).get("quote",    [{}])[0].get("volume",   [])
        if not timestamps or not closes:
            return None
        df = pd.DataFrame(
            {"Close": closes, "Volume": volumes},
            index=pd.to_datetime(timestamps, unit="s", utc=True)
        )
        return df.dropna()
    except Exception:
        return None


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


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas,  0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


def compute_seq_features(seq):
    """60日リターン系列 → 7次元要約統計量"""
    s = np.array(seq, dtype=float)
    if len(s) < 2:
        return [0.0] * 7
    mean_s = s.mean(); std_s = s.std()
    ac = float(np.corrcoef(s[:-1], s[1:])[0, 1]) if std_s > 0 else 0.0
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


def extract_features(p, v=None, nk_rets=None):
    """28次元特徴量: テクニカル10 + トレンド反転5 + 出来高3 + 日経マクロ3 + 60日系列要約7"""
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

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + seq_feat

    if any(np.isnan(feat[:10])) or any(np.isinf(feat[:10])):
        return None
    return feat

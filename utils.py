"""
utils.py - 共通関数モジュール
rank_stocks.py / alert_email.py / backtest.py で共有
"""
import re
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


def get_macro_df(symbol, days=2200):
    """マクロ指標の履歴を Date インデックスのDataFrameで返す。symbolの例: USDJPY=X, %5ETNX, %5EVIX"""
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval=1d&period1={start_ts}&period2={end_ts}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        ts     = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        idx    = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Tokyo")
        df     = pd.DataFrame({"Close": closes}, index=idx).dropna()
        df.index = df.index.date
        return df
    except Exception:
        return None


def get_macro_snapshot():
    """現在のマクロ指標を返す: (usdjpy_20d_return, ust10y_level/10, vix_level/100)"""
    usdjpy_df = get_macro_df("USDJPY=X", days=60)
    tnx_df    = get_macro_df("%5ETNX",   days=60)
    vix_df    = get_macro_df("%5EVIX",   days=60)
    usdjpy_20d = 0.0
    if usdjpy_df is not None and len(usdjpy_df) >= 21:
        p = usdjpy_df["Close"].values
        usdjpy_20d = (p[-1] - p[-21]) / p[-21]
    ust10y = 0.0
    if tnx_df is not None and len(tnx_df) >= 1:
        ust10y = float(tnx_df["Close"].values[-1]) / 10.0  # 例: 4.5% → 0.45
    vix = 0.0
    if vix_df is not None and len(vix_df) >= 1:
        vix = float(vix_df["Close"].values[-1]) / 100.0    # 例: 20 → 0.20
    return (round(usdjpy_20d, 4), round(ust10y, 4), round(vix, 4))


def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas,  0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    if losses == 0:
        return 100.0
    return 100 - 100 / (1 + gains / losses)


RANK_FEAT_INDICES        = [0, 1, 2, 6, 7, 9]  # ret5, ret20, ret60, rsi, vol20, pos52
SECTOR_RANK_FEAT_INDICES = [2, 7, 9]            # ret60, vol20, pos52（業種内相対）

_JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
_SECTOR_CACHE = None  # プロセス内キャッシュ {code: sector}

def _load_jpx_sector_map():
    """JPX Excelから {code: 33業種区分} を一括取得してCSVキャッシュに保存"""
    import os, io
    cache_path = os.path.expanduser("~/stock-alert/sector_cache.csv")
    cache = {}
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            cache = dict(zip(df["code"], df["sector"]))
        except Exception:
            pass
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
                pd.DataFrame(list(cache.items()), columns=["code", "sector"]).to_csv(cache_path, index=False)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] JPX業種マップ取得失敗: {e}")
    return cache


def get_sector_cached(code):
    """業種を取得（プロセス内キャッシュ→CSVキャッシュ→JPX一括取得）"""
    global _SECTOR_CACHE
    if _SECTOR_CACHE is None:
        _SECTOR_CACHE = _load_jpx_sector_map()
    return _SECTOR_CACHE.get(str(code), "不明")


def add_sector_rank_features(X, sectors, dates=None):
    """業種内パーセンタイルランク特徴量を3次元追加。

    Training mode (dates provided): (date, sector) グループ内でランク化。
    Inference mode (dates=None): sector グループ内でランク化（同日想定）。
    Returns augmented matrix shape (n, n_features + 3).
    """
    X = np.array(X, dtype=float)
    n = len(X)
    sectors = np.array([s if s else "不明" for s in sectors])
    rank_matrix = np.full((n, len(SECTOR_RANK_FEAT_INDICES)), 0.5, dtype=float)

    if dates is not None:
        dates = np.array(dates)
        # (date, sector) のペアでグループ化
        keys = np.array([f"{d}|{s}" for d, s in zip(dates, sectors)])
        unique_keys = np.unique(keys)
        for k in unique_keys:
            group = np.where(keys == k)[0]
            cnt = len(group)
            if cnt < 3:  # 業種内3銘柄未満はランク化しない
                continue
            for j, fi in enumerate(SECTOR_RANK_FEAT_INDICES):
                vals = X[group, fi]
                order = np.argsort(np.argsort(vals))
                rank_matrix[group, j] = order / (cnt - 1)
    else:
        unique_secs = np.unique(sectors)
        for s in unique_secs:
            group = np.where(sectors == s)[0]
            cnt = len(group)
            if cnt < 3:
                continue
            for j, fi in enumerate(SECTOR_RANK_FEAT_INDICES):
                vals = X[group, fi]
                order = np.argsort(np.argsort(vals))
                rank_matrix[group, j] = order / (cnt - 1)

    return np.hstack([X, rank_matrix])


def add_cs_rank_features(X, dates=None):
    """Add 6 cross-sectional percentile rank features to feature matrix X.

    Training mode (dates provided): rank within each date group.
    Inference mode (dates=None): rank within the current batch (single day).
    Returns augmented matrix shape (n, n_features + 6).
    """
    X = np.array(X, dtype=float)
    n = len(X)
    rank_matrix = np.full((n, len(RANK_FEAT_INDICES)), 0.5, dtype=float)

    if dates is not None:
        dates = np.array(dates)
        _, inverse = np.unique(dates, return_inverse=True)
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

    return np.hstack([X, rank_matrix])


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


def extract_features(p, v=None, nk_rets=None, macro_vals=None):
    """31次元特徴量: テクニカル10 + トレンド反転5 + 出来高3 + 日経マクロ3 + 60日系列要約7 + 海外マクロ3"""
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

    usdjpy_20d = macro_vals[0] if macro_vals is not None else 0.0
    ust10y     = macro_vals[1] if macro_vals is not None else 0.0
    vix_lvl    = macro_vals[2] if macro_vals is not None else 0.0

    seq_raw = (np.clip(np.diff(p[-(SEQ_DAYS+1):]) / p[-(SEQ_DAYS+1):-1], -0.2, 0.2)
               if len(p) >= SEQ_DAYS + 1 else np.zeros(SEQ_DAYS))
    seq_feat = compute_seq_features(seq_raw)

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + seq_feat + [usdjpy_20d, ust10y, vix_lvl]

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


class EnsembleCalibrated:
    """XGBoost + LightGBM アンサンブル + Isotonic Calibration"""
    def __init__(self, models, iso):
        self.models = models  # [xgb_model, lgb_model]
        self.iso    = iso

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        cal   = self.iso.predict(probs)
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
    except Exception:
        pass
    return result

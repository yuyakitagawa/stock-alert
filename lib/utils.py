"""
utils.py - 共通関数モジュール
rank_stocks.py / alert_email.py / backtest.py で共有
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
    """38次元特徴量: テクニカル10 + トレンド反転5 + 出来高3 + 日経マクロ3 + 60日系列要約7 + 日経相対アルファ4 + ファンダメンタル6
    fundamentals: dict with keys per/pbr/roe/days_to_earnings/days_to_dividend/days_to_yutai (all optional)
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

    # ファンダメンタル6特徴量
    # PER/PBR/ROE: バリュエーション・収益性
    # days_to_earnings: 決算前ドリフト
    # days_since_div_ex / days_since_yutai_ex: 権利落ち後の戻り買いゾーン
    fd  = fundamentals or {}
    _per = fd.get('per');   _pbr = fd.get('pbr');   _roe = fd.get('roe')
    _dte  = fd.get('days_to_earnings')
    _ddiv = fd.get('days_since_div_ex')
    _dyut = fd.get('days_since_yutai_ex')
    per_feat      = float(np.clip(_per  / 20.0 - 1.0, -1.0, 3.0)) if _per  is not None else 0.0
    pbr_feat      = float(np.clip(_pbr  /  1.5 - 1.0, -1.0, 4.0)) if _pbr  is not None else 0.0
    roe_feat      = float(np.clip(_roe  / 15.0,        -0.5, 2.0)) if _roe  is not None else 0.0
    earn_feat     = float(np.clip(_dte  / 90.0,         0.0, 1.0)) if _dte  is not None else 0.5
    div_ex_feat   = float(np.clip(_ddiv / 60.0,         0.0, 1.0)) if _ddiv is not None else 0.5
    yutai_ex_feat = float(np.clip(_dyut / 60.0,         0.0, 1.0)) if _dyut is not None else 0.5

    feat = [ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52,
            drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir,
            vr520, vr2060, vsurge, nk5, nk20, nk60] + seq_feat + [rel5, rel20, rel60, alpha_momentum,
            per_feat, pbr_feat, roe_feat, earn_feat, div_ex_feat, yutai_ex_feat]

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


# ── 推奨ラベル（rank_stocks / alert_email 共通） ───────────────────────────

def recommend_from_net(net, allow_buy=True):
    if net < -10:
        return "🔴 下降シグナル"
    if net < -5:
        return "⚠️ 弱気シグナル"
    return "⏳ 方向感なし"


def recommend_from_scores(net, drop_prob=None, allow_buy=True, vol=None):
    """S買い: 17<=net<=24 かつ drop_prob<4% かつ vol<=25%。上位3件まで（rank_stocks.pyフェーズ6）"""
    if allow_buy and drop_prob is not None and drop_prob < 4.0 and 17.0 <= net <= 24.0:
        if vol is None or vol <= 25.0:
            return "🥇 S買い"
    return recommend_from_net(net, allow_buy=allow_buy)

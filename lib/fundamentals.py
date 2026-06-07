"""
fundamentals.py
fundamentals_annual テーブルから point-in-time（先読みバイアスなし）の
ファンダメンタルを再構成する共有ロジック。学習(rf_train_v3)とバックテスト(backtest)で共用。

返すのは「その日に既知だった生の値」: eps / bps / roe / 各種確定日までの日数。
PER=price/eps, PBR=price/bps の計算と正規化は呼び出し側（価格を持つ側）で行う。
"""
from datetime import datetime, timedelta
from lib.utils import _days_to_nearest_event

_FUND_HIST = None    # {code: [ {fy_end, announce_date, eps, roe, bps}, ... ] (announce_date昇順)}
_YUTAI_MONTH = None   # {code: record_month or None}


def load_fundamentals_cache():
    """DBから年度別ファンダと優待月を一括ロード（プロセス内キャッシュ）。"""
    global _FUND_HIST, _YUTAI_MONTH
    if _FUND_HIST is not None:
        return
    try:
        from lib.db import load_all_fundamentals_annual, _conn
        _FUND_HIST = load_all_fundamentals_annual()
        _YUTAI_MONTH = {}
        with _conn() as con:
            for r in con.execute("SELECT code, has_yutai, record_month FROM yutai_cache"):
                _YUTAI_MONTH[str(r["code"])] = r["record_month"] if r["has_yutai"] else None
    except Exception:
        _FUND_HIST = {}
        _YUTAI_MONTH = {}


def get_pit_fundamentals(code, target_date):
    """target_date 時点で既知のファンダ生値を返す。データ皆無なら None。
    返り値: {eps, bps, roe, days_to_earnings, days_to_dividend, days_to_yutai}
    """
    load_fundamentals_cache()
    code = str(code)
    recs = _FUND_HIST.get(code, [])
    tgt_iso = target_date.isoformat()
    known = [r for r in recs if r["announce_date"] and r["announce_date"] <= tgt_iso]

    eps = bps = roe = None
    days_earn = None
    if known:
        latest = known[-1]  # announce_date 昇順 → 末尾が最新
        eps, bps, roe = latest.get("eps"), latest.get("bps"), latest.get("roe")
        last_ann = datetime.fromisoformat(known[-1]["announce_date"]).date()
        est_next = last_ann + timedelta(days=91)
        while est_next < target_date:
            est_next += timedelta(days=91)
        days_earn = (est_next - target_date).days

    ym = (_YUTAI_MONTH or {}).get(code)
    div_months = [ym] if ym else [3, 9]
    days_div = _days_to_nearest_event(target_date, div_months, day=28)
    days_yutai = _days_to_nearest_event(target_date, [ym], day=28) if ym else None

    if eps is None and bps is None and roe is None and not known and ym is None:
        return None
    return {
        "eps": eps, "bps": bps, "roe": roe,
        "days_to_earnings": days_earn,
        "days_to_dividend": days_div,
        "days_to_yutai": days_yutai,
    }


def pit_fundamental_features(code, target_date, price):
    """point-in-timeファンダを6特徴量(正規化済み)に変換。
    extract_features() のファンダ部と同一の正規化。データ無しは中立値。
    返り値: [per_feat, pbr_feat, roe_feat, earn_feat, div_feat, yutai_feat]
    """
    import numpy as np
    pf = pbf = rf = 0.0
    ef = df = yf = 0.5
    fd = get_pit_fundamentals(code, target_date)
    if fd is not None:
        eps, bps, roe = fd.get("eps"), fd.get("bps"), fd.get("roe")
        if eps is not None and eps > 0 and price > 0:
            pf = float(np.clip((price / eps) / 20.0 - 1.0, -1.0, 3.0))
        if bps is not None and bps > 0 and price > 0:
            pbf = float(np.clip((price / bps) / 1.5 - 1.0, -1.0, 4.0))
        if roe is not None:
            rf = float(np.clip(roe / 15.0, -0.5, 2.0))
        if fd.get("days_to_earnings") is not None:
            ef = float(np.clip(fd["days_to_earnings"] / 90.0, 0.0, 1.0))
        if fd.get("days_to_dividend") is not None:
            df = float(np.clip(fd["days_to_dividend"] / 60.0, 0.0, 1.0))
        if fd.get("days_to_yutai") is not None:
            yf = float(np.clip(fd["days_to_yutai"] / 60.0, 0.0, 1.0))
    return [pf, pbf, rf, ef, df, yf]

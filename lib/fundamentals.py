"""
fundamentals.py
fundamentals_annual テーブルから point-in-time（先読みバイアスなし）の
ファンダメンタルを再構成する共有ロジック。学習(rf_train_v3)とバックテスト(backtest)で共用。

特徴量（6次元）:
  PER, PBR, ROE         : バリュエーション・収益性
  days_to_earnings      : 次回決算まで日数（決算前ドリフト）
  days_since_div_ex     : 前回配当権利落ち日からの経過日数（戻り買いゾーン）
  days_since_yutai_ex   : 前回優待権利落ち日からの経過日数（同上）
"""
import calendar
from datetime import datetime, timedelta, date as _date_type
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


def _days_since_last_ex(target_date, record_months):
    """record_months の各月の直近過去の権利落ち日（月末-2営業日近似）からの経過日数。
    権利落ち直後（0日）〜完全回復（≥60日）をカバー。None は情報なし。"""
    if not record_months:
        return None
    best = None
    for m in record_months:
        for yr_adj in [0, -1]:
            yr = target_date.year + yr_adj
            try:
                last_day = calendar.monthrange(yr, m)[1]
                # 権利確定日: 月末2営業日前（簡易近似）
                record_dt = _date_type(yr, m, last_day) - timedelta(days=2)
                # 権利落ち日 = 権利確定日の翌営業日（≈翌日）
                ex_dt = record_dt + timedelta(days=1)
                delta = (target_date - ex_dt).days
                if 0 <= delta:  # 過去の権利落ち日のみ
                    if best is None or delta < best:
                        best = delta
            except ValueError:
                pass
    return best


def get_pit_fundamentals(code, target_date):
    """target_date 時点で既知のファンダ生値を返す。データ皆無なら None。
    返り値: {eps, bps, roe, days_to_earnings, days_to_dividend, days_to_yutai}
    """
    load_fundamentals_cache()
    code = str(code)
    recs = _FUND_HIST.get(code, [])
    tgt_iso = target_date.isoformat()
    known = [r for r in recs if r["announce_date"] and r["announce_date"] <= tgt_iso]

    eps = bps = roe = dps = None
    days_earn = None
    if known:
        latest = known[-1]  # announce_date 昇順 → 末尾が最新
        eps, bps, roe, dps = latest.get("eps"), latest.get("bps"), latest.get("roe"), latest.get("dps")
        last_ann = datetime.fromisoformat(known[-1]["announce_date"]).date()
        est_next = last_ann + timedelta(days=91)
        while est_next < target_date:
            est_next += timedelta(days=91)
        days_earn = (est_next - target_date).days

    # 配当・優待の権利落ち後経過日数（yutai_cacheの確定月から計算）
    ym = (_YUTAI_MONTH or {}).get(code)
    div_months = [ym] if ym else [3, 9]   # 優待月or典型的な配当月
    days_since_div  = _days_since_last_ex(target_date, div_months)
    days_since_yutai = _days_since_last_ex(target_date, [ym]) if ym else None

    # EPS成長率・ROEトレンド・DPS成長率（直近2期比較 — PIT準拠）
    eps_growth = None
    roe_trend  = None
    dps_growth = None
    if len(known) >= 2:
        prev = known[-2]
        if eps is not None and prev.get("eps") is not None and prev["eps"] != 0:
            eps_growth = (eps - prev["eps"]) / abs(prev["eps"])
        if roe is not None and prev.get("roe") is not None:
            roe_trend = roe - prev["roe"]   # ROEの前年差（%ポイント）
        if dps is not None and prev.get("dps") is not None and prev["dps"] != 0:
            dps_growth = (dps - prev["dps"]) / abs(prev["dps"])  # 配当成長率（増配シグナル）

    if eps is None and bps is None and roe is None and not known and ym is None:
        return None
    return {
        "eps": eps, "bps": bps, "roe": roe, "dps": dps,
        "days_to_earnings":    days_earn,
        "days_since_div_ex":   days_since_div,
        "days_since_yutai_ex": days_since_yutai,
        "eps_growth":          eps_growth,
        "roe_trend":           roe_trend,
        "dps_growth":          dps_growth,
    }


def pit_fundamental_features(code, target_date, price):
    """point-in-timeファンダを9特徴量(正規化済み)に変換。
    extract_features() のファンダ部と同一の正規化。データ無しは中立値。

    返り値: [per_feat, pbr_feat, roe_feat, earn_feat,
             div_ex_feat, yutai_ex_feat,
             sin_month, cos_month, div_yield_feat,
             eps_growth_feat, roe_trend_feat, dps_growth_feat]  ← 12次元

    div_ex_feat / yutai_ex_feat:
      0.0 = 権利落ち直後（戻り買いゾーン）  1.0 = 60日後（効果消滅）
    div_yield_feat:
      配当利回り% / 10（3%=0.3, 5%=0.5）。高利回り株を識別。
    sin_month, cos_month: 月の季節性を循環エンコード。
    eps_growth_feat: EPS前年比（-1〜3に正規化）。業績モメンタム。
    roe_trend_feat:  ROE前年差（%pt、-10〜+10に正規化）。TSE改革対応。
    """
    import numpy as np, math
    pf = pbf = rf = 0.0
    ef = div_ef = yutai_ef = 0.5
    div_yield_f = eps_gf = roe_tf = dps_gf = 0.0

    fd = get_pit_fundamentals(code, target_date)
    if fd is not None:
        eps, bps, roe, dps = fd.get("eps"), fd.get("bps"), fd.get("roe"), fd.get("dps")
        if eps is not None and eps > 0 and price > 0:
            pf = float(np.clip((price / eps) / 20.0 - 1.0, -1.0, 3.0))
        if bps is not None and bps > 0 and price > 0:
            pbf = float(np.clip((price / bps) / 1.5 - 1.0, -1.0, 4.0))
        if roe is not None:
            rf = float(np.clip(roe / 15.0, -0.5, 2.0))
        if fd.get("days_to_earnings") is not None:
            ef = float(np.clip(fd["days_to_earnings"] / 90.0, 0.0, 1.0))
        if fd.get("days_since_div_ex") is not None:
            div_ef = float(np.clip(fd["days_since_div_ex"] / 60.0, 0.0, 1.0))
        if fd.get("days_since_yutai_ex") is not None:
            yutai_ef = float(np.clip(fd["days_since_yutai_ex"] / 60.0, 0.0, 1.0))
        if dps is not None and dps > 0 and price > 0:
            div_yield_f = float(np.clip((dps / price * 100) / 10.0, 0.0, 1.0))
        # EPS成長率: (-100%〜+300%)を(-1〜3)にスケール
        if fd.get("eps_growth") is not None:
            eps_gf = float(np.clip(fd["eps_growth"], -1.0, 3.0))
        # ROEトレンド: ±10%ptを-1〜+1に正規化
        if fd.get("roe_trend") is not None:
            roe_tf = float(np.clip(fd["roe_trend"] / 10.0, -1.0, 1.0))
        # DPS成長率: -100%〜+200%を-1〜+2に正規化（増配シグナル — TSE資本改革文脈）
        if fd.get("dps_growth") is not None:
            dps_gf = float(np.clip(fd["dps_growth"], -1.0, 2.0))

    # 季節性: 月を循環エンコード（1月と12月が隣接するように）
    m = target_date.month
    sin_m = math.sin(2 * math.pi * m / 12)
    cos_m = math.cos(2 * math.pi * m / 12)

    return [pf, pbf, rf, ef, div_ef, yutai_ef, sin_m, cos_m, div_yield_f, eps_gf, roe_tf, dps_gf]

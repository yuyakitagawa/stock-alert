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
        from lib.db import load_all_fundamentals_annual, get_all_yutai
        _FUND_HIST = load_all_fundamentals_annual()
        _YUTAI_MONTH = {}
        for r in get_all_yutai():
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


def _jq_split_safe_bps(code, target_date):
    """J-Quants(jquants_fin_summary)の直近開示BPS(>0)を返す。
    開示ごとに分割後株数で再表示されるため、分割調整済み株価と整合する
    （fundamentals_annual=株探値は分割調整漏れで表示PBRが過小化することがある）。
    表示PER/PBR専用。Noneなら呼び出し側でfundamentals_annual値に委ねる。"""
    try:
        from lib.db import get_jquants_fin_history
        rows = get_jquants_fin_history(str(code), target_date.isoformat(), n=6)
    except Exception:
        return None
    for r in rows:
        b = r.get("bps")
        if b is not None and b > 0:
            return b
    return None


def get_pit_valuation(code, target_date):
    """表示用バリュエーション: target_date時点で既知の eps/bps を
    「それぞれ最新の非NULL値」で返す。
    翌期予想行は eps はあるが bps=None のことが多く、get_pit_fundamentals が
    予想行を最新採用すると bps=None になり PBR が欠損するため、
    eps と bps を独立に直近非NULLから拾う（PER/PBR表示専用、特徴量には不使用）。
    BPSは分割調整漏れを防ぐためJ-Quants値を優先し、無ければ株探値にフォールバック。
    返り値: {"eps": float|None, "bps": float|None}
    """
    load_fundamentals_cache()
    code = str(code)
    recs = _FUND_HIST.get(code, [])
    tgt_iso = target_date.isoformat()
    known = [r for r in recs if r["announce_date"] and r["announce_date"] <= tgt_iso]
    eps = next((r["eps"] for r in reversed(known) if r.get("eps") is not None), None)
    bps = next((r["bps"] for r in reversed(known) if r.get("bps") is not None), None)
    # 分割調整済み株価と整合するJ-Quants BPSを優先（表示PBRの過小化を防止）
    bps = _jq_split_safe_bps(code, target_date) or bps
    return {"eps": eps, "bps": bps}


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
    bps_growth = None
    if len(known) >= 2:
        prev = known[-2]
        if eps is not None and prev.get("eps") is not None and prev["eps"] != 0:
            eps_growth = (eps - prev["eps"]) / abs(prev["eps"])
        if roe is not None and prev.get("roe") is not None:
            roe_trend = roe - prev["roe"]   # ROEの前年差（%ポイント）
        if dps is not None and prev.get("dps") is not None and prev["dps"] != 0:
            dps_growth = (dps - prev["dps"]) / abs(prev["dps"])  # 配当成長率（増配シグナル）
        if bps is not None and prev.get("bps") is not None and prev["bps"] > 0:
            bps_growth = (bps - prev["bps"]) / prev["bps"]  # BPS前年比成長率

    # ── 投資銀行手法8特徴量 (Group A: 既存データから計算) ────────────────────
    # 1. EPS surprise: 線形トレンド外挿との乖離（Bernard & Thomas 1989）
    eps_surprise = None
    if len(known) >= 3:
        eps_t2 = known[-3].get("eps")
        eps_t1 = known[-2].get("eps")
        eps_t0 = eps
        if all(e is not None for e in [eps_t2, eps_t1, eps_t0]):
            trend = eps_t1 - eps_t2
            expected = eps_t1 + trend
            denom = abs(expected) if abs(expected) > 1.0 else (abs(eps_t1) if abs(eps_t1) > 1.0 else None)
            if denom:
                eps_surprise = (eps_t0 - expected) / denom

    # 2. Piotroski F-score 簡易版（Piotroski 2000、6シグナル、0-1スケール）
    pio_score = 0.0
    pio_max   = 0.0
    if roe is not None:
        pio_max += 2
        if roe > 0:                                         pio_score += 1
        if roe_trend is not None and roe_trend > 0:        pio_score += 1
    if eps is not None and eps_growth is not None:
        pio_max += 1
        if eps_growth > 0:                                  pio_score += 1
    if dps is not None and dps_growth is not None:
        pio_max += 2
        if dps_growth >= 0:                                 pio_score += 1  # 非減配
        if eps_growth is not None and eps_growth >= dps_growth: pio_score += 1  # 持続可能
    if bps_growth is not None:
        pio_max += 1
        if bps_growth > 0:                                  pio_score += 1  # 簿価増加
    piotroski = float(pio_score / pio_max) if pio_max > 0 else None

    # 3. Payout ratio（DPS / EPS — 資本還元度）
    payout = None
    if dps is not None and eps is not None and abs(eps) > 0.1:
        payout = max(0.0, min(dps / eps, 2.0))

    # 4. Balance sheet accruals proxy（Sloan 1996 — 会計発生主義）
    # 実際のBPS成長率 vs 内部留保で期待されるBPS成長率の差
    accruals = None
    if bps_growth is not None and roe is not None and len(known) >= 2:
        bps_prev_val = known[-2].get("bps")
        if bps_prev_val is not None and bps_prev_val > 0:
            dps_yield_on_bps = (dps / bps_prev_val) if dps else 0.0
            expected_retention = roe / 100 - dps_yield_on_bps
            accruals = bps_growth - expected_retention  # +なら過剰な簿価膨張

    # ── J-Quants 補完: Sloanアクルーアル（正確版）────────────────────────────
    # CFOデータが利用可能なら BPS プロキシを正確版で上書き
    try:
        from lib.db import get_jquants_fin_history_fy
        jq_fy = get_jquants_fin_history_fy(str(code), target_date.isoformat(), n=3)
        if len(jq_fy) >= 1:
            lq = jq_fy[0]
            np_v, cfo_v, ta_v = lq.get("np"), lq.get("cfo"), lq.get("ta")
            if np_v is not None and cfo_v is not None and ta_v and abs(ta_v) > 0:
                sloan_accruals = (np_v - cfo_v) / ta_v   # 正確なSloan(1996)
                accruals = sloan_accruals * 5.0           # BPSプロキシと同スケールに正規化
    except Exception:
        pass  # J-Quants未取得 → BPSプロキシのまま使用

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
        # 新規8特徴量（投資銀行手法）
        "eps_surprise":        eps_surprise,
        "bps_growth":          bps_growth,
        "piotroski":           piotroski,
        "payout":              payout,
        "accruals":            accruals,      # Sloan正確版 or BPSプロキシ
    }


def pit_fundamental_features(code, target_date, price):
    """point-in-timeファンダをファンダメンタル部の正規化済み辞書として返す。
    extract_features()に渡すfundamentals dictを生成するためのヘルパー。
    backtest.py が extract_features() を直接呼び出す際に使用。

    返り値: fundamentals dict（extract_features()のfd引数と互換）
    """
    import math
    fd = get_pit_fundamentals(code, target_date)
    m = target_date.month
    result = {"month": m}
    if fd is not None:
        eps = fd.get("eps")
        bps = fd.get("bps")
        dps = fd.get("dps")
        result["per"]             = (price / eps) if eps and eps > 0 and price > 0 else None
        # 注意: ここの pbr は60次元特徴量の一部。bps は fundamentals_annual(株探)由来で
        # 分割調整漏れの可能性があるが、学習済みモデルと特徴量分布の整合を保つため
        # ここでは敢えて変更しない（CLAUDE.md §0「特徴量は変更時要申告」）。
        # 分割調整済みBPSへの移行は金曜再学習時に申告のうえ一括で行うこと。
        result["pbr"]             = (price / bps) if bps and bps > 0 and price > 0 else None
        result["roe"]             = fd.get("roe")
        result["days_to_earnings"]  = fd.get("days_to_earnings")
        result["days_since_div_ex"] = fd.get("days_since_div_ex")
        result["div_yield"]       = (dps / price * 100) if dps and dps > 0 and price > 0 else None
        result["eps_growth"]      = fd.get("eps_growth")
        result["dps_growth"]      = fd.get("dps_growth")
        result["eps_surprise"]    = fd.get("eps_surprise")
        result["bps_growth"]      = fd.get("bps_growth")
        result["piotroski"]       = fd.get("piotroski")
        result["payout"]          = fd.get("payout")
        result["accruals"]        = fd.get("accruals")
    return result


# ── 進捗率（会社予想に対する累計実績の進み具合）─────────────────────────────
# GARP: 「未来の業績成長を割安なうちに」捉えるための先行指標。
# 四半期の累計実績NP ÷ 通期会社予想FNP を、その四半期の期待ペースと比較する。
# 期待を上回るペース（progress_vs_pace > 1）= 通期上方修正の先回りシグナル。
_QTR_EXPECTED_PACE = {"1Q": 0.25, "2Q": 0.50, "3Q": 0.75}


def get_progress_rate(code, as_of_date):
    """as_of_date 時点で既知の最新四半期開示から進捗率を point-in-time で返す。

    返り値: {
      "progress_ratio":   float | None,  # 累計実績NP / 通期予想FNP
      "progress_vs_pace": float | None,  # progress_ratio / 四半期期待ペース（>1=上振れ）
      "doc_type":         str   | None,  # 1Q/2Q/3Q（FYは中間進捗の意味を持たないためNone扱い）
      "disc_date":        str   | None,
    }
    データ不足時は各値 None。
    """
    from lib.db import get_jquants_fin_history
    none_result = {"progress_ratio": None, "progress_vs_pace": None,
                   "doc_type": None, "disc_date": None}
    as_of_iso = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
    rows = get_jquants_fin_history(str(code), as_of_iso, n=1)
    if not rows:
        return none_result
    r = rows[0]
    doc_type = r.get("doc_type")
    np_v = r.get("np")
    fnp_v = r.get("fnp")
    # 中間四半期（1Q/2Q/3Q）かつ 予想NPが正のときのみ進捗率が意味を持つ
    if doc_type not in _QTR_EXPECTED_PACE or np_v is None or fnp_v is None or fnp_v <= 0:
        return none_result
    progress_ratio = np_v / fnp_v
    pace = _QTR_EXPECTED_PACE[doc_type]
    return {
        "progress_ratio":   progress_ratio,
        "progress_vs_pace": progress_ratio / pace,
        "doc_type":         doc_type,
        "disc_date":        r.get("disc_date"),
    }

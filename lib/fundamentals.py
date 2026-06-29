"""
fundamentals.py
point-in-time（先読みバイアスなし）のファンダメンタルを再構成する共有ロジック。
学習(rf_train_v3)とバックテスト(backtest)で共用。

特徴量（6次元）:
  PER, PBR, ROE         : バリュエーション・収益性
  days_to_earnings      : 次回決算まで日数（決算前ドリフト）
  days_since_div_ex     : 前回配当権利落ち日からの経過日数（戻り買いゾーン）
  days_since_yutai_ex   : 前回優待権利落ち日からの経過日数（同上）
"""
import os
import calendar
from datetime import datetime, timedelta, date as _date_type
from lib.utils import _days_to_nearest_event

_JQUANTS_LOCAL = None

def _load_jquants_local():
    global _JQUANTS_LOCAL
    if _JQUANTS_LOCAL is not None:
        return _JQUANTS_LOCAL
    import pickle
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_jquants_fin.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            _JQUANTS_LOCAL = pickle.load(f)
        return _JQUANTS_LOCAL
    _JQUANTS_LOCAL = {}
    return _JQUANTS_LOCAL

def _local_jq_history(code, as_of_date_str, n=4):
    data = _load_jquants_local()
    rows = data.get(str(code), [])
    return sorted([r for r in rows if r.get("disc_date", "") <= as_of_date_str],
                  key=lambda x: x.get("disc_date", ""), reverse=True)[:n]

def _local_jq_history_fy(code, as_of_date_str, n=3):
    data = _load_jquants_local()
    rows = data.get(str(code), [])
    return sorted([r for r in rows if r.get("disc_date", "") <= as_of_date_str and r.get("doc_type", "").startswith("FY")],
                  key=lambda x: x.get("disc_date", ""), reverse=True)[:n]

_YUTAI_MONTH = None   # {code: record_month or None}


def load_fundamentals_cache():
    """DBから優待月を一括ロード（プロセス内キャッシュ）。"""
    global _YUTAI_MONTH
    if _YUTAI_MONTH is not None:
        return
    try:
        from lib.db import get_all_yutai
        _YUTAI_MONTH = {}
        for r in get_all_yutai():
            _YUTAI_MONTH[str(r["code"])] = r.get("yutai_month") or r.get("record_month")
    except Exception:
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
    開示ごとに分割後株数で再表示されるため、分割調整済み株価と整合する。
    表示PER/PBR専用。"""
    try:
        from lib.db import get_jquants_fin_history
        rows = get_jquants_fin_history(str(code), target_date.isoformat(), n=6)
        if not rows:
            rows = _local_jq_history(str(code), target_date.isoformat(), n=6)
    except Exception:
        rows = _local_jq_history(str(code), target_date.isoformat(), n=6)
    for r in rows:
        b = r.get("bps")
        if b is not None and b > 0:
            return b
    return None


def get_pit_valuation(code, target_date):
    """表示用バリュエーション: target_date時点で既知の eps/bps を返す。
    J-Quants(jquants_fin_summary)から取得。PER/PBR表示専用、特徴量には不使用。
    返り値: {"eps": float|None, "bps": float|None}
    """
    code = str(code)
    bps = _jq_split_safe_bps(code, target_date)
    eps = None
    try:
        from lib.db import get_jquants_fin_history
        rows = get_jquants_fin_history(str(code), target_date.isoformat(), n=6)
        for r in rows:
            e = r.get("eps")
            if e is not None:
                eps = e
                break
    except Exception:
        pass
    return {"eps": eps, "bps": bps}


def get_pit_fundamentals(code, target_date):
    """target_date 時点で既知のファンダ生値を返す。データ皆無なら None。
    返り値: {eps, bps, roe, days_to_earnings, days_to_dividend, days_to_yutai, ...}
    """
    load_fundamentals_cache()
    code = str(code)

    # 配当・優待の権利落ち後経過日数（優待確定月から計算）
    ym = (_YUTAI_MONTH or {}).get(code)
    div_months = [ym] if ym else [3, 9]
    days_since_div  = _days_since_last_ex(target_date, div_months)
    days_since_yutai = _days_since_last_ex(target_date, [ym]) if ym else None

    eps = bps = roe = dps = None
    eps_growth = bps_growth = eps_surprise = piotroski_score = payout = accruals = dps_growth = None
    cfo_margin = leverage = op_margin_improve = equity_ratio = sales_growth = forecast_revision = None
    has_jq = False

    try:
        from lib.db import get_jquants_fin_history, get_jquants_fin_history_fy
        td_iso = target_date.isoformat()

        rows = get_jquants_fin_history(code, td_iso, n=4)
        if not rows:
            rows = _local_jq_history(code, td_iso, n=4)
        if rows:
            has_jq = True
            latest = rows[0]
            eps = latest.get("eps")
            bps = _jq_split_safe_bps(code, target_date)
            dps = latest.get("div_ann")
            pr = latest.get("payout_ratio")
            if pr is not None:
                payout = pr / 100.0 if pr > 1.5 else pr

            np_v = latest.get("np")
            eq = latest.get("equity")
            if np_v is not None and eq and eq > 0:
                roe = (np_v / eq) * 100

            # EPS surprise: actual NP vs forecast NP
            fnp = latest.get("fnp")
            if np_v is not None and fnp is not None and fnp != 0:
                eps_surprise = (np_v - fnp) / abs(fnp)

        jq_fy = get_jquants_fin_history_fy(code, td_iso, n=3)
        if not jq_fy:
            jq_fy = _local_jq_history_fy(code, td_iso, n=3)
        if len(jq_fy) >= 2:
            has_jq = True
            curr, prev = jq_fy[0], jq_fy[1]

            curr_eps, prev_eps = curr.get("eps"), prev.get("eps")
            if curr_eps is not None and prev_eps is not None and prev_eps != 0:
                eps_growth = (curr_eps - prev_eps) / abs(prev_eps)

            curr_bps, prev_bps = curr.get("bps"), prev.get("bps")
            if curr_bps is not None and prev_bps is not None and prev_bps > 0:
                bps_growth = (curr_bps - prev_bps) / prev_bps

            curr_div, prev_div = curr.get("div_ann"), prev.get("div_ann")
            if curr_div is not None and prev_div is not None and prev_div > 0:
                dps_growth = (curr_div - prev_div) / prev_div

            # Piotroski F-Score (7 computable items, normalized to 0-1)
            score = 0
            items = 0
            if curr.get("np") is not None and curr.get("ta") and curr["ta"] > 0:
                items += 1
                if curr["np"] / curr["ta"] > 0: score += 1
            if curr.get("cfo") is not None:
                items += 1
                if curr["cfo"] > 0: score += 1
            if (curr.get("np") and curr.get("ta") and curr["ta"] > 0 and
                prev.get("np") and prev.get("ta") and prev["ta"] > 0):
                items += 1
                if curr["np"]/curr["ta"] > prev["np"]/prev["ta"]: score += 1
            if curr.get("cfo") is not None and curr.get("np") is not None:
                items += 1
                if curr["cfo"] > curr["np"]: score += 1
            if (curr.get("op") and curr.get("sales") and curr["sales"] > 0 and
                prev.get("op") and prev.get("sales") and prev["sales"] > 0):
                items += 1
                if curr["op"]/curr["sales"] > prev["op"]/prev["sales"]: score += 1
            if (curr.get("sales") and curr.get("ta") and curr["ta"] > 0 and
                prev.get("sales") and prev.get("ta") and prev["ta"] > 0):
                items += 1
                if curr["sales"]/curr["ta"] > prev["sales"]/prev["ta"]: score += 1
            if items >= 3:
                piotroski_score = score / items

        # 営業CFマージン (cfo/sales): キャッシュ創出力
        if jq_fy:
            lq = jq_fy[0]
            _cfo = lq.get("cfo"); _sales = lq.get("sales")
            if _cfo is not None and _sales and _sales > 0:
                cfo_margin = _cfo / _sales

        # 有利子負債比率 ((ta-equity)/equity): 財務レバレッジ
        if jq_fy:
            lq = jq_fy[0]
            _ta = lq.get("ta"); _eq = lq.get("equity")
            if _ta is not None and _eq and _eq > 0:
                leverage = (_ta - _eq) / _eq

        # 自己資本比率 (equity/ta)
        if jq_fy:
            lq = jq_fy[0]
            _ta = lq.get("ta"); _eq = lq.get("equity")
            if _ta and _ta > 0 and _eq is not None:
                equity_ratio = _eq / _ta

        # 売上成長率
        if len(jq_fy) >= 2:
            c_sal, p_sal = curr.get("sales"), prev.get("sales")
            if c_sal is not None and p_sal and p_sal > 0:
                sales_growth = (c_sal - p_sal) / abs(p_sal)

        # 営業利益率改善 (op/sales YoY差分)
        if len(jq_fy) >= 2:
            c_op, c_sal = curr.get("op"), curr.get("sales")
            p_op, p_sal = prev.get("op"), prev.get("sales")
            if (c_op is not None and c_sal and c_sal > 0 and
                p_op is not None and p_sal and p_sal > 0):
                op_margin_improve = (c_op / c_sal) - (p_op / p_sal)

        # 会社予想修正率: 最新予想NP vs 前回予想NP
        if len(rows) >= 2:
            r0_fnp = rows[0].get("fnp")
            r1_fnp = rows[1].get("fnp")
            if r0_fnp is not None and r1_fnp is not None and r1_fnp != 0:
                forecast_revision = (r0_fnp - r1_fnp) / abs(r1_fnp)

        # Sloan accruals
        if jq_fy:
            lq = jq_fy[0]
            np_v, cfo_v, ta_v = lq.get("np"), lq.get("cfo"), lq.get("ta")
            if np_v is not None and cfo_v is not None and ta_v and abs(ta_v) > 0:
                accruals = ((np_v - cfo_v) / ta_v) * 5.0
    except Exception:
        pass

    if not has_jq and ym is None:
        return None
    return {
        "eps": eps, "bps": bps, "roe": roe, "dps": dps,
        "days_to_earnings":    None,
        "days_since_div_ex":   days_since_div,
        "days_since_yutai_ex": days_since_yutai,
        "eps_growth":          eps_growth,
        "roe_trend":           None,
        "dps_growth":          dps_growth,
        "eps_surprise":        eps_surprise,
        "bps_growth":          bps_growth,
        "piotroski":           piotroski_score,
        "payout":              payout,
        "accruals":            accruals,
        "cfo_margin":          cfo_margin,
        "leverage":            leverage,
        "op_margin_improve":   op_margin_improve,
        "equity_ratio":        equity_ratio,
        "sales_growth":        sales_growth,
        "forecast_revision":   forecast_revision,
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
        result["cfo_margin"]      = fd.get("cfo_margin")
        result["leverage"]        = fd.get("leverage")
        result["op_margin_improve"] = fd.get("op_margin_improve")
        result["equity_ratio"]    = fd.get("equity_ratio")
        result["sales_growth"]    = fd.get("sales_growth")
        result["forecast_revision"] = fd.get("forecast_revision")
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

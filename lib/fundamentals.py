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
import calendar
import math
from datetime import datetime, timedelta, date as _date_type

_YUTAI_MONTH = None   # {code: record_month or None}


def _filter_asof(rows, as_of_iso, n, fy_only=False):
    """rows（disc_date降順の全履歴）から、指定日時点で既知の直近n件を返す。
    get_jquants_fin_history()/get_jquants_fin_history_fy()のSQLクエリと
    同じ結果になるよう、point-in-timeフィルタをメモリ上で再現する。"""
    out = []
    for r in rows:
        d = r.get("disc_date")
        if d is None or d > as_of_iso:
            continue
        if fy_only and not str(r.get("doc_type") or "").startswith("FY"):
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


# ── 株式分割の検出 ─────────────────────────────────────────────────
# jquants_fin_summary の1株当たり値（eps/bps/div_ann）は「開示時点の株数」ベース。
# 分割をまたぐと前期比が 1/分割比 に潰れ（378A: bps_growth −33% ← 実際は+33%）、
# 分割後の株価と分割前の開示を組み合わせた PER/PBR が 1/分割比 に過小化する。
_SPLIT_CANDS = [float(n) for n in range(2, 11)] + [1.0 / n for n in range(2, 11)]
_FIN_SPLIT_TOL = 0.10    # 開示間の株数比が分割比の±10%以内なら分割とみなす
_PRICE_SPLIT_TOL = 0.08  # 終値の前日比（web/dip_buy_alert.py と同じ基準）


def _near_split(x, tol):
    if x is None or x <= 0:
        return None
    for c in _SPLIT_CANDS:
        if abs(x / c - 1) <= tol:
            return c
    return None


def _implied_shares(r):
    """equity / bps = 1株当たり値の分母になっている株数（開示時点のベース）。"""
    eq, b = r.get("equity"), r.get("bps")
    if eq is None or b is None or eq <= 0 or b <= 0:
        return None
    return eq / b


def disclosure_split_ratio(curr, prev):
    """prev→curr の開示間に起きた株式分割の比率（1株→n株 の n。無ければ1.0）。

    判定は equity/bps（1株当たり値の分母の株数）の比 R を使い、増資・自社株買いと
    分割を切り分けるため、次の順で分割比に近いものを採る:
      1. (equity/bps/sh_out) の比: 期末後〜開示前の分割（期末株数は分割前のまま、
         BPSだけ分割後で再表示。378A: 1.997）
      2. sh_out の比: 期中の分割
      3. R そのもの
    誤検出ガード: 増資・自社株買い込みでも R/分割比 が 0.6〜1.5 に収まり、かつ補正後の
    BPS前期比が 0.5〜2.0倍 か、補正前より1倍に近いこと（sh_outの単位違い・大型増資を弾く。
    2026-09-27時点のFY開示で 416件を分割と判定、2134・9327・9602 等48件を除外）。
    """
    ic, ip = _implied_shares(curr), _implied_shares(prev)
    if ic is None or ip is None:
        return 1.0
    R = ic / ip
    sc, sp = curr.get("sh_out"), prev.get("sh_out")
    qr = sr = None
    if sc and sp and sc > 0 and sp > 0:
        qr = (ic / sc) / (ip / sp)
        sr = sc / sp
    s = _near_split(qr, _FIN_SPLIT_TOL) or _near_split(sr, _FIN_SPLIT_TOL) \
        or _near_split(R, _FIN_SPLIT_TOL)
    if s is None or not (0.6 <= R / s <= 1.5):
        return 1.0
    raw = curr["bps"] / prev["bps"]
    adj = raw * s
    if not (0.5 <= adj <= 2.0) and abs(math.log(adj)) >= abs(math.log(raw)):
        return 1.0
    return s


def price_split_jumps(dates, closes):
    """終値系列の分割段差 [(日付ISO, 分割比n), ...]。
    yahoo_price_cache は過去行を上書きしないため、分割調整前の行と調整後の行の境目で
    終値が 1/n に飛ぶ（378A: 2026-06-19 2138→1025.5）。段差より後の終値は分割後ベース。"""
    out = []
    if dates is None or closes is None:
        return out
    for i in range(1, len(closes)):
        p0, p1 = closes[i - 1], closes[i]
        if not p0 or not p1 or p0 <= 0 or p1 <= 0:
            continue
        r = p0 / p1
        if abs(r - 1) < 0.4:
            continue
        s = _near_split(r, _PRICE_SPLIT_TOL)
        if s is not None:
            out.append((str(dates[i])[:10], s))
    return out


def split_factor_to_price(src, known_rows, target_iso, jumps):
    """src開示の1株当たり値を、target日の株価と同じ株数ベースに直す倍率 k
    （eps/bps/dps を k で割る）。jumps=None（株価情報なし）なら補正しない。

    - src開示後〜target日の終値の段差: その日以降の株価は分割後ベース
    - src以降の開示で検出した分割のうち、終値に段差が無いもの: 株価が遡って
      調整済み（yfinance auto_adjust で取り直した区間）なので、それ以前の株価も分割後ベース
    known_rows: 手元にある開示（未来分を含んでよい。学習時は全履歴）。
    """
    if jumps is None or src is None:
        return 1.0
    src_d = str(src.get("disc_date"))
    k = 1.0
    for d, s in jumps:
        if src_d < d <= target_iso:
            k *= s
    later = sorted((r for r in known_rows or [] if str(r.get("disc_date")) > src_d
                    and _implied_shares(r) is not None),
                   key=lambda r: str(r["disc_date"]))
    prev = src if _implied_shares(src) is not None else None
    for c in later:
        if prev is not None:
            s = disclosure_split_ratio(c, prev)
            if s != 1.0:
                lo, hi = str(prev["disc_date"]), str(c["disc_date"])
                matched = any(lo < d <= hi and abs(sj / s - 1) <= 0.15 for d, sj in jumps)
                if not matched:
                    k *= s
        prev = c
    return k


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


def _jq_split_safe_bps_row(code, target_date, rows=None):
    """J-Quants(jquants_fin_summary)の直近開示でBPS(>0)を持つ行を返す。
    rows（銘柄の全履歴、disc_date降順）が渡された場合はDBに問い合わせず
    メモリ上でフィルタする（多数のas_of_dateを扱う学習ループ用）。"""
    try:
        if rows is not None:
            asof_rows = _filter_asof(rows, target_date.isoformat(), n=6)
        else:
            from lib.db import get_jquants_fin_history
            asof_rows = get_jquants_fin_history(str(code), target_date.isoformat(), n=6)
    except Exception:
        return None
    for r in asof_rows:
        b = r.get("bps")
        if b is not None and b > 0:
            return r
    return None


def _jq_split_safe_bps(code, target_date, rows=None, jumps=None):
    """直近開示BPS(>0)を、target日の株価と同じ株数ベースに直して返す。
    jumps: price_split_jumps() の結果（省略時は開示時点ベースのまま）。"""
    r = _jq_split_safe_bps_row(code, target_date, rows=rows)
    if r is None:
        return None
    return r["bps"] / split_factor_to_price(r, rows, target_date.isoformat(), jumps)


def get_pit_valuation(code, target_date, rows=None, jumps=None):
    """表示用バリュエーション: target_date時点で既知の eps/bps を返す。
    J-Quants(jquants_fin_summary)から取得。PER/PBR表示専用、特徴量には不使用。
    rows: _jq_split_safe_bps/get_pit_fundamentals と同じ（省略時はDB問い合わせ）。
    jumps: price_split_jumps() の結果。渡すと株式分割をまたいでも株価と同じ株数ベースにそろえる。
    返り値: {"eps": float|None, "bps": float|None}
    """
    code = str(code)
    bps = _jq_split_safe_bps(code, target_date, rows=rows, jumps=jumps)
    eps = None
    try:
        if rows is not None:
            asof_rows = _filter_asof(rows, target_date.isoformat(), n=6)
        else:
            from lib.db import get_jquants_fin_history
            asof_rows = get_jquants_fin_history(str(code), target_date.isoformat(), n=6)
        for r in asof_rows:
            e = r.get("eps")
            if e is not None:
                eps = e / split_factor_to_price(r, rows, target_date.isoformat(), jumps)
                break
    except Exception:
        pass
    return {"eps": eps, "bps": bps}


def get_pit_fundamentals(code, target_date, rows=None, jumps=None):
    """target_date 時点で既知のファンダ生値を返す。データ皆無なら None。
    rows: 銘柄の全履歴（disc_date降順）を渡すとDBに問い合わせずメモリ上で
    point-in-timeフィルタする（rf_train_v3.pyのように同一銘柄を多数の
    target_dateで呼ぶ場合の高速化用）。省略時は従来通りDB問い合わせ。
    jumps: price_split_jumps() の結果。渡すと eps/bps/dps を target日の株価と
    同じ株数ベースにそろえる（株式分割対策。省略時は開示時点ベース）。
    前期比（eps/bps/dps_growth）は jumps に関係なく開示間の分割を補正する。
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
    cfo_margin = leverage = op_margin_improve = None
    has_jq = False

    try:
        td_iso = target_date.isoformat()

        if rows is not None:
            n4_rows = _filter_asof(rows, td_iso, n=4)
        else:
            from lib.db import get_jquants_fin_history
            n4_rows = get_jquants_fin_history(code, td_iso, n=4)
        if n4_rows:
            has_jq = True
            latest = n4_rows[0]
            k = split_factor_to_price(latest, rows, td_iso, jumps)
            eps = latest.get("eps")
            eps = eps / k if eps is not None else None
            bps = _jq_split_safe_bps(code, target_date, rows=rows, jumps=jumps)
            dps = latest.get("div_ann")
            dps = dps / k if dps is not None else None
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

        if rows is not None:
            jq_fy = _filter_asof(rows, td_iso, n=3, fy_only=True)
        else:
            from lib.db import get_jquants_fin_history_fy
            jq_fy = get_jquants_fin_history_fy(code, td_iso, n=3)
        if len(jq_fy) >= 2:
            has_jq = True
            curr, prev = jq_fy[0], jq_fy[1]
            # 前期の1株当たり値を今期の株数ベースへ（分割が無ければ1.0）
            sp = disclosure_split_ratio(curr, prev)

            curr_eps, prev_eps = curr.get("eps"), prev.get("eps")
            prev_eps = prev_eps / sp if prev_eps is not None else None
            if curr_eps is not None and prev_eps is not None and prev_eps != 0:
                eps_growth = (curr_eps - prev_eps) / abs(prev_eps)

            curr_bps, prev_bps = curr.get("bps"), prev.get("bps")
            prev_bps = prev_bps / sp if prev_bps is not None else None
            if curr_bps is not None and prev_bps is not None and prev_bps > 0:
                bps_growth = (curr_bps - prev_bps) / prev_bps

            curr_div, prev_div = curr.get("div_ann"), prev.get("div_ann")
            prev_div = prev_div / sp if prev_div is not None else None
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

        # 営業利益率改善 (op/sales YoY差分)
        if len(jq_fy) >= 2:
            c_op, c_sal = curr.get("op"), curr.get("sales")
            p_op, p_sal = prev.get("op"), prev.get("sales")
            if (c_op is not None and c_sal and c_sal > 0 and
                p_op is not None and p_sal and p_sal > 0):
                op_margin_improve = (c_op / c_sal) - (p_op / p_sal)

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
    }


def pit_fundamental_features(code, target_date, price, rows=None, jumps=None):
    """point-in-timeファンダをファンダメンタル部の正規化済み辞書として返す。
    extract_features()に渡すfundamentals dictを生成するためのヘルパー。
    backtest.py が extract_features() を直接呼び出す際に使用。
    rows, jumps: get_pit_fundamentals()と同じ（省略時はDB問い合わせ・分割補正なし）。

    返り値: fundamentals dict（extract_features()のfd引数と互換）
    """
    fd = get_pit_fundamentals(code, target_date, rows=rows, jumps=jumps)
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
    return result



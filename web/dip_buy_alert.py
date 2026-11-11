#!/usr/bin/env python3
"""
web/dip_buy_alert.py — 市場急落日の押し目買い候補をLINEへ通知する

条件（2026-09-19 オーナー確定）:
  1. 日経平均の当日騰落が -2% 以下
  2. その銘柄の当日騰落が -4% 以下（下限なし）
  3. 株価300円以上・直近20日の平均売買代金1億円以上
  4. PBR < 1.5（株式分割は補正する。期末〜有報提出の間の分割も含む）
  5. 会社予想の純利益 > 前期実績の純利益（前期は黒字）
     予想は jquants_fin_summary → company_info の短信XBRL予想（tanshin_fc_*）→ 株探の会社予想（fc_*）の順
  6. 従業員数 1000人以上（Supabase company_info.employees。tools/fetch_employees.py が更新）
  各銘柄には有報の「事業の内容」から作った一文を添える（lib/company_profile.py）
  会社予想が未取得で 5 を判定できない銘柄のうち、ほかの条件をすべて満たすものは
  「参考」としてβ順に上位5つまで末尾に出す（判定には入れない。2026-10-18 オーナー確定）。
  並びはβ（過去250日・日経平均に対する感応度）の高い順。

なぜこの条件か（検証は docs/dip_buy_strategy.md）:
  日経-2%以下×-4%以下の銘柄は、10年（2015-2026）で63日後平均+10.7%・勝率64%。
  PBR<1.5 と予想増益で大負け（-15%以下）が減り、従業員1000人未満は成績が悪い。
  βが高いほど「市場に連れて下げただけ」の銘柄になり戻りが大きい（11年中9年で有効）。

  同じ銘柄の再通知は抑えない。買うかどうかは人間が判断する（オーナー方針）。

使い方:
  python web/dip_buy_alert.py             # 条件を満たせばLINEへ送る
  python web/dip_buy_alert.py --dry-run   # 送らずに本文を表示
  python web/dip_buy_alert.py --date 2026-09-02 --dry-run   # 過去日で確認
"""
import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

NIKKEI_DROP = -0.02
STOCK_DROP = -0.04
MIN_PRICE = 300
MIN_TURNOVER_YEN = 1e8
MAX_PBR = 1.5
MIN_EMPLOYEES = 1000
BETA_DAYS = 250
BETA_MIN_DAYS = 150
# 本決算が400日より前＝最新の本決算が未取得。判定は1年前の数字になるので本文で印を付ける
STALE_FY_DAYS = 400
# 分割の検出: 前日比が1/nまたはn（n=2..10）の±8%以内かつ±40%以上。
# 分割日にも株価自体が動くため、±3%では古河電工(0.104=1:10)・住友電工(0.268=1:4)を取りこぼした
SPLIT_TOL = 0.08
# LINEの1通の上限（lib.notify.MAX_CHARS と同じ）。ここに収まる分だけ並べる
MAX_CHARS = 4_000
# 予想未取得で判定できない銘柄の参考表示の件数。2026-04-25〜08-19の短信予想が欠けているための措置で、
# 予想が出そろえば（3月決算は2Q短信の11月中旬）該当がほぼ無くなり、表示も出なくなる
REF_MAX = 5
# 前の取引日の件数が無いとき、当日の終値がそろっている銘柄がこれ未満なら株価更新の失敗とみなす（平常時は約3,750）
MIN_PRICED_CODES = 1000
# 前の取引日に対してこの割合未満しか終値が無ければ欠けとみなす（平常の日次の増減は数銘柄）
PRICED_RATIO = 0.9

# 判定できなかった日を後の便で判定し直す範囲（判定日より前の平日の数）。日本時間0時を越えて始まった便は
# 当日の日足が取れないが、翌日の便では前日分がそろう（11/6分は11/6の便で1,073銘柄、11/9の便で3,746銘柄）
CATCHUP_WEEKDAYS = 5

JST = timezone(timedelta(hours=9))
# 東証の大引け（2024-11-05から15:30）。これより前は当日の終値がまだ無い
TSE_CLOSE_JST = (15, 30)


class PricesMissing(RuntimeError):
    """判定日の終値（個別株・日経平均）が欠けている。0時JSTを越えた便では想定内なので警告しない（judge の quiet）。
    DB障害・コードの不具合など他の例外は quiet でもその日のうちに警告する。"""


# ── 純粋関数（テスト対象） ─────────────────────────────────────────────

def session_date(now_jst: datetime) -> date:
    """判定する取引日。日次ジョブは遅れて日本時間の0時を越えることがある（2026-11-04分は
    11/5 00:55に実行され、11/5の終値が無いとして判定されなかった）ので、
    大引け前なら前の平日、土日なら金曜を返す。祝日は日経平均の終値が無いことで判別する。"""
    d = now_jst.date()
    if (now_jst.hour, now_jst.minute) < TSE_CLOSE_JST:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def prices_missing(n_priced: int, n_prev: int | None = None) -> bool:
    """判定日の終値が欠けているか。0件は休場日。前の取引日の件数があればその90%未満で欠けとみなす
    （2026-11-06は1,073銘柄＝前日3,746の29%で、旧基準の1,000以上を越えて取り直されなかった）。
    前の取引日の件数が無いときは MIN_PRICED_CODES 未満で欠けとみなす。"""
    if n_priced == 0:
        return False
    if n_prev:
        return n_priced < n_prev * PRICED_RATIO
    return n_priced < MIN_PRICED_CODES


def index_missing(n_priced: int, nk_last: date | None, asof: date) -> bool:
    """取引日（個別株の終値が1件でもある日）なのに日経平均の asof の終値が無いか。
    2026-11-06分は個別株1073銘柄そろい Step 0r を素通りし、日経平均だけ欠けて判定できなかった。"""
    return n_priced > 0 and (nk_last is None or nk_last < asof)


def missing_index_reason(asof: date, nk_last: date, n_priced: int) -> str | None:
    """日経平均の asof の終値が無いとき、休場日なら None、取得漏れなら警告文を返す。
    休場日は個別株の終値も1件も無い。1件でもあれば取引日なので日経平均の取得漏れ
    （2026-11-04は株価23銘柄だけ・日経平均なしで、黙って「通知対象なし」になっていた）。"""
    if n_priced == 0:
        return None
    return (f"日経平均の {asof} の終値が未取得です（最終 {nk_last}）。"
            f"個別株は{n_priced}銘柄に終値があり休場日ではありません（株価更新の失敗の疑い）")


def past_session_day(asof: date, now_jst: datetime) -> bool:
    """判定日の翌日（日本時間）以降に動いた便か。Yahoo は日本時間0時を越えると当日の日足を返さず、
    翌日の夕方までに埋まる（11/4・11/5・11/9・11/11分。翌日の便では3,746銘柄そろった）。
    このときの欠けは想定内なので、その日の便では警告せず翌日以降の判定し直しに任せる。"""
    return now_jst.date() > asof


def catchup_dates(asof: date, n: int = CATCHUP_WEEKDAYS) -> list[date]:
    """asof より前の平日 n 日を古い順に返す（祝日は find_candidates が休場日として飛ばす）。"""
    out, d = [], asof
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out[::-1]


def recovered_message(asof: date, nk_ret: float | None) -> str:
    """「判定できませんでした」と警告した日を後から判定し、通知対象が無かったときの本文。"""
    week = "月火水木金土日"[asof.weekday()]
    if nk_ret is None:
        result = "日経平均・個別株とも終値が無く休場日でした"
    elif nk_ret > NIKKEI_DROP:
        result = f"日経 {nk_ret * 100:+.1f}%（-2%に届かず）で通知対象なし"
    else:
        result = f"日経 {nk_ret * 100:+.1f}%。条件を満たす銘柄なし"
    return f"✅ 押し目買い候補\n{asof.month}/{asof.day}（{week}）分を判定し直しました: {result}"


def daily_return(closes) -> float | None:
    if closes is None or len(closes) < 2 or not closes[-2]:
        return None
    return float(closes[-1] / closes[-2] - 1)


def beta(stock_rets, index_rets, min_days: int = BETA_MIN_DAYS) -> float | None:
    """同じ日付で揃えた日次騰落率の配列から β = cov / var を返す。"""
    s = np.asarray(stock_rets, dtype=float)
    m = np.asarray(index_rets, dtype=float)
    ok = np.isfinite(s) & np.isfinite(m)
    s, m = s[ok], m[ok]
    if len(s) < min_days:
        return None
    var = m.var()
    if var <= 0:
        return None
    return float(((s - s.mean()) * (m - m.mean())).mean() / var)


def split_factor(dates, closes, after, until) -> float:
    """after < 日付 <= until の間に起きた株式分割の比率の積（分割が無ければ1.0）。

    yahoo_price_cache は過去行を上書きしないため、分割の前後で終値が飛ぶ。
    開示時点の1株当たり値（BPS等）にこの比率を掛けると当日の株数ベースにそろう。
    """
    cand = [1 / n for n in range(2, 11)] + [float(n) for n in range(2, 11)]
    k = 1.0
    for i in range(1, len(closes)):
        d = dates[i]
        if not (after < d <= until) or not closes[i - 1]:
            continue
        r = closes[i] / closes[i - 1]
        if abs(r - 1) >= 0.4 and any(abs(r / c - 1) <= SPLIT_TOL for c in cand):
            k *= r
    return k


def pick_fundamentals(rows: list[dict], asof: date) -> dict | None:
    """asof より前（前日まで）に開示された決算から、判定に使う値を取り出す。

    当日引け後の開示は、終値で買う時点では見えないので使わない。
    """
    past = [r for r in rows if r.get("disc_date") and date.fromisoformat(str(r["disc_date"])[:10]) < asof]
    if not past:
        return None
    past.sort(key=lambda r: str(r["disc_date"]))
    fy = [r for r in past if r.get("doc_type") == "FY" and r.get("bps") and r.get("np") is not None]
    if not fy:
        return None
    f = fy[-1]
    # 予想は「直近の本決算より後の年度」のものだけ使う。同じ年度の予想と実績を比べると
    # 増益判定の意味が無くなる（2026-04-25以降はEDINET由来で予想が入らず、放置すると起きる）
    fc = [r for r in past if r.get("fnp") is not None and str(r.get("fy_end") or "") > str(f.get("fy_end") or "")]
    fy_end = str(f.get("fy_end") or "")[:10]
    return {
        "fy_disc": date.fromisoformat(str(f["disc_date"])[:10]),
        "fy_end": date.fromisoformat(fy_end) if len(fy_end) == 10 else None,
        "prev_bps": float(fy[-2]["bps"]) if len(fy) >= 2 else None,
        "np_prev": float(fy[-2]["np"]) if len(fy) >= 2 else None,
        # 直近の本決算と同じ年度の会社予想（実績が予想を上回ったかの参考表示用）
        "fnp_ly": next((float(r["fnp"]) for r in reversed(past)
                        if r.get("fnp") is not None and str(r.get("fy_end") or "") == str(f.get("fy_end") or "")), None),
        "bps": float(f["bps"]),
        "np": float(f["np"]),
        "fnp": float(fc[-1]["fnp"]) if fc else None,
    }


def bps_split_factor(dates, closes, fund: dict, asof: date) -> float:
    """本決算のBPSを asof の株価と同じ株数ベースにそろえる比率（PBR = 終値 / (BPS × 比率)）。

    開示後の分割に加え、期末〜開示の間の分割も見る。有報のBPSは期末の株数ベースのことが多く、
    開示より前に株価だけ分割後になる（古河電工: 2026-03期、6/3に1:10分割、有報6/24のBPS 5,928円は
    分割前ベースで、PBR 0.57 に見えていた。実際は約5.7）。
    ただし前期BPSから分割比ぶん下がっていれば、有報が分割後ベースに直してあるので掛けない。
    """
    k = split_factor(dates, closes, fund["fy_disc"], asof)
    if fund.get("fy_end") is None:
        return k
    gap = split_factor(dates, closes, fund["fy_end"], fund["fy_disc"])
    prev = fund.get("prev_bps")
    restated = prev is not None and prev > 0 and fund["bps"] / prev < gap * 1.5
    return k * gap if gap != 1.0 and not restated else k


def company_forecast(row: dict | None, fund: dict, asof: date) -> float | None:
    """company_info の会社予想の純利益（円）。使えなければ None。

    jquants_fin_summary は 2026-04-25〜08-19 の短信予想が欠けている（TDnetのXBRLは約1か月で消え、
    遡って取れない）。company_info で補う。先に最新の決算短信のXBRLから読んだ予想（tanshin_fc_*）、
    無ければ株探の決算ページの「予」行（fc_*）。どちらも jquants 側と同じく、asof より前（前日まで）に
    出ていて、直近の本決算より後の年度の予想に限る。company_info は百万円単位なので円に直す。
    """
    fy_end = fund.get("fy_end")
    if not row or fy_end is None:
        return None
    for np_key, end_key, ann_key in (("tanshin_fc_np", "tanshin_fc_fy_end", "tanshin_disclosed_at"),
                                     ("fc_np", "fc_period", "fc_announced")):
        v, end, ann = row.get(np_key), row.get(end_key), row.get(ann_key)
        if v is None or not end or not ann or date.fromisoformat(str(ann)[:10]) >= asof:
            continue
        if str(end).replace(".", "-")[:7] <= fy_end.isoformat()[:7]:
            continue
        return float(v) * 1e6
    return None


def passes_fundamentals(close: float, fund: dict, k_fy: float) -> tuple[bool, float | None, float | None]:
    """(合格か, PBR, 予想増益率)。PBRは分割補正後、予想増益は前期黒字が前提。"""
    pbr = close / (fund["bps"] * k_fy) if fund["bps"] > 0 else None
    growth = fund["fnp"] / fund["np"] - 1 if fund["np"] > 0 and fund["fnp"] is not None else None
    ok = pbr is not None and pbr < MAX_PBR and growth is not None and growth > 0
    return ok, pbr, growth


def ref_note(p: dict) -> str:
    """参考表示の補足。前期の実績増益率と、前期の会社予想を上回ったか（どちらも成績は未検証）。"""
    out = []
    if p.get("np_prev") and p["np_prev"] > 0 and p.get("np") is not None:
        out.append(f"前期{(p['np'] / p['np_prev'] - 1) * 100:+.0f}%")
    if p.get("fnp_ly") and p.get("np") is not None:
        out.append("予想超" if p["np"] > p["fnp_ly"] else "予想未達")
    return "・".join(out)


def build_message(asof: date, nk_ret: float, picks: list[dict], no_forecast: int = 0,
                  refs: list[dict] | None = None, late: bool = False) -> str:
    """1銘柄2行（銘柄名＋コード／数字）で全件並べる。

    LINEの1通は5,000文字（lib.notify は4,000で切る）。急落が深い日は候補が数百になり
    全部は入らないので、入る分だけ並べて残りは件数で示す（途中で切れて末尾の条件文が
    消えるのを防ぐ）。
    """
    week = "月火水木金土日"[asof.weekday()]
    head = [f"📉 押し目買い候補 {asof.month}/{asof.day}（{week}）",
            f"日経 {nk_ret * 100:+.1f}%／該当{len(picks)}銘柄（β順）", ""]
    if late:
        head.insert(0, "⏰ 株価の取得が遅れたため後から判定しました")
    tail = []
    if refs:
        tail += ["", "📎 参考: 会社予想が未取得だが他の条件はクリア（β順・判定外）"]
        for p in refs[:REF_MAX]:
            note = ref_note(p)
            tail += [f"・{p['name']} {p['code']}",
                     f"  {p['ret'] * 100:+.1f}% ｜ β{p['beta']:.2f} ｜ PBR{p['pbr']:.2f}" + (f" ｜ {note}" if note else "")]
    if no_forecast:
        tail.append(f"※会社予想が未取得で増益を判定できず除外: {no_forecast}銘柄（PBR<1.5は満たす）")
    tail += ["", "条件: 日経-2%以下×当日-4%以下×PBR<1.5×予想増益×従業員1000人以上",
             "検証は当日終値買い・63日保有。引け後の通知なので約定は翌日以降"
             + ("（今回は通知が1日以上遅れている）" if late else "")]

    body, shown = [], 0
    budget = MAX_CHARS - len("\n".join(head + tail)) - 20  # 20は「…ほかN銘柄」の行ぶん
    for i, p in enumerate(picks, 1):
        stale = " ⚠決算古" if p.get("stale") else ""
        pair = [f"{i} {p['name']} {p['code']}",
                f"  {p['ret'] * 100:+.1f}% ｜ β{p['beta']:.2f} ｜ PBR{p['pbr']:.2f} "
                f"｜ 増益{p['growth'] * 100:+.0f}%{stale}"]
        if p.get("business"):
            pair.append(f"  └ {p['business']}")
        add = len("\n".join(pair)) + 1
        if budget - add < 0:
            break
        budget -= add
        body += pair
        shown = i
    if shown < len(picks):
        body.append(f"…ほか{len(picks) - shown}銘柄")
    return "\n".join(head + body + tail)


# ── データ取得 ─────────────────────────────────────────────────────────

def _load_names() -> dict:
    try:
        with open(os.path.join(BASE_DIR, "data", "code_name_map.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _fin_rows(codes: list[str]) -> dict[str, list[dict]]:
    import lib.supabase_client as sb
    out: dict[str, list[dict]] = {}
    for i in range(0, len(codes), 40):
        q = ",".join(f'"{c}"' for c in codes[i:i + 40])
        rows = sb.select("jquants_fin_summary",
                         f"code=in.({q})&select=code,disc_date,doc_type,fy_end,np,bps,fnp&order=disc_date.asc")
        for r in rows:
            out.setdefault(str(r["code"]), []).append(r)
    return out


def _company_forecasts(codes: list[str]) -> dict[str, dict]:
    import lib.supabase_client as sb
    out: dict[str, dict] = {}
    for i in range(0, len(codes), 100):
        q = ",".join(f'"{c}"' for c in codes[i:i + 100])
        for r in sb.select("company_info", f"code=in.({q})&or=(fc_np.not.is.null,tanshin_fc_np.not.is.null)"
                                 "&select=code,fc_period,fc_np,fc_announced,tanshin_fc_fy_end,tanshin_fc_np,tanshin_disclosed_at"):
            out[str(r["code"])] = r
    return out


def nikkei_last_date(asof: date) -> date | None:
    """asof 以前で日経平均の終値がある最終日。"""
    from lib.utils import get_market_index_df_cached
    nk = get_market_index_df_cached("N225", "%5EN225", 30)
    if nk is None:
        return None
    nk = nk[nk.index <= asof]
    return nk.index[-1] if len(nk) else None


def count_priced(asof: date) -> int:
    """asof の終値がある銘柄数（price_store のミラー）。"""
    from lib import price_store
    price_store.enable()
    return sum(1 for c in price_store.codes()
               if price_store.frame(c, start=asof.isoformat(), end=asof.isoformat()) is not None)


def prev_count_priced(asof: date, max_back: int = 7) -> int | None:
    """asof より前で、終値が1件でもある直近の日の銘柄数（祝日・土日を飛ばす）。"""
    d = asof
    for _ in range(max_back):
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        n = count_priced(d)
        if n:
            return n
    return None


def find_candidates(asof: date) -> tuple[float | None, list[dict], int, list[dict]]:
    from lib import price_store
    from lib.utils import get_market_index_df_cached

    nk = get_market_index_df_cached("N225", "%5EN225", 500)
    if nk is None or len(nk) < BETA_MIN_DAYS:
        raise RuntimeError("日経平均の株価が取得できません")
    nk = nk[nk.index <= asof]
    price_store.enable()
    if nk.index[-1] != asof:
        n_priced = count_priced(asof)
        reason = missing_index_reason(asof, nk.index[-1], n_priced)
        if reason:
            raise PricesMissing(reason)
        print(f"[dip] {asof} は休場日（日経平均・個別株とも終値なし）")
        return None, [], 0, []
    nk_ret = daily_return(nk["Close"].values)
    print(f"[dip] 日経平均 {asof}: {nk_ret * 100:+.2f}%")
    if nk_ret is None or nk_ret > NIKKEI_DROP:
        return nk_ret, [], 0, []

    nk_r = nk["Close"].pct_change()
    stage = []
    n_today = 0
    for code in price_store.codes():
        df = price_store.frame(code, end=asof.isoformat(), days=(date.today() - asof).days + 420)
        if df is None or len(df) < 22 or df.index[-1] != asof:
            continue
        n_today += 1
        closes = df["Close"].values
        r = daily_return(closes)
        if r is None or r > STOCK_DROP or closes[-1] < MIN_PRICE:
            continue
        if split_factor(df.index[-2:], closes[-2:], df.index[-2], asof) != 1.0:
            continue  # 当日が分割日（下げに見えるだけ）
        if float(np.mean(closes[-20:] * df["Volume"].values[-20:])) < MIN_TURNOVER_YEN:
            continue
        sr = df["Close"].pct_change()
        sr = sr.where(sr.abs() < 0.4)  # 分割日の見かけの騰落をβに入れない
        idx = sr.index[:-1][-BETA_DAYS:]  # 当日を含めない
        b = beta(sr.reindex(idx).values, nk_r.reindex(idx).values)
        if b is None:
            continue
        stage.append({"code": code, "ret": r, "close": float(closes[-1]), "beta": b,
                      "dates": list(df.index), "closes": closes})
    print(f"[dip] 当日の終値がある銘柄: {n_today} / 株価条件を満たす銘柄: {len(stage)}")
    n_all, n_prev = count_priced(asof), prev_count_priced(asof)  # --check-prices と同じ数え方
    if n_all == 0 or prices_missing(n_all, n_prev):
        # Step 0（株価更新）が落ちた日に「該当なし」で黙って終わらないようにする
        raise PricesMissing(f"{asof} の終値がある銘柄が{n_all}件しかありません"
                           f"（前の取引日 {n_prev}件。株価更新の失敗の疑い）")

    fins = _fin_rows([s["code"] for s in stage])
    kfc = _company_forecasts([s["code"] for s in stage])
    stage2, nofc = [], []
    no_forecast = n_ci = 0
    for s in stage:
        fund = pick_fundamentals(fins.get(s["code"], []), asof)
        if not fund:
            continue
        if fund["fnp"] is None:
            fund["fnp"] = company_forecast(kfc.get(s["code"]), fund, asof)
            n_ci += fund["fnp"] is not None
        k = bps_split_factor(s["dates"], s["closes"], fund, asof)
        if fund["fnp"] is None:
            # 会社予想が無い＝増益を判定できない。PBRは満たしていたかだけ数えて本文に出す
            pbr = s["close"] / (fund["bps"] * k)
            if pbr < MAX_PBR:
                no_forecast += 1
                s.update(pbr=pbr, np=fund["np"], np_prev=fund["np_prev"], fnp_ly=fund["fnp_ly"])
                nofc.append(s)
            continue
        ok, pbr, growth = passes_fundamentals(s["close"], fund, k)
        if ok:
            s.update(pbr=pbr, growth=growth, stale=(asof - fund["fy_disc"]).days > STALE_FY_DAYS)
            stage2.append(s)
    print(f"[dip] PBR<1.5・予想増益: {len(stage2)}（予想をcompany_infoで補った: {n_ci} / "
          f"PBR<1.5だが会社予想が無く判定できず: {no_forecast}）")

    from lib import company_profile, employees
    emp_map = employees.get([s["code"] for s in stage2 + nofc])  # company_info.employees、無い銘柄だけYahooで補う
    emps = [emp_map.get(s["code"]) for s in stage2]
    names = _load_names()
    profiles = company_profile.load([s["code"] for s in stage2])  # 有報の「事業の内容」（company_info.business）
    picks = []
    for s, e in zip(stage2, emps):
        if e is not None and e >= MIN_EMPLOYEES:
            prof = profiles.get(s["code"]) or {}
            picks.append({k: s[k] for k in ("code", "ret", "beta", "pbr", "growth", "stale")}
                         | {"name": names.get(s["code"], ""), "employees": e,
                            "business": company_profile.headline(prof.get("business") or "")})
    unknown = sum(e is None for e in emps)
    print(f"[dip] 従業員1000人以上: {len(picks)}（従業員数を取得できず除外: {unknown}）")
    picks.sort(key=lambda p: p["beta"], reverse=True)
    refs = sorted(({k: s[k] for k in ("code", "ret", "beta", "pbr", "np", "np_prev", "fnp_ly")}
                   | {"name": names.get(s["code"], "")}
                   for s in nofc if (emp_map.get(s["code"]) or 0) >= MIN_EMPLOYEES),
                  key=lambda p: p["beta"], reverse=True)
    print(f"[dip] 参考（予想未取得・他の条件クリア）: {len(refs)}")
    return nk_ret, picks, no_forecast, refs


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import notify

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="判定日（既定: 直近の取引日。JST15:30より前なら前の平日）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-prices", action="store_true",
                    help="判定日の終値（個別株・日経平均）がそろっていれば0、欠けていれば1で終わる（daily_alert の再取得判定用）")
    a = ap.parse_args()
    asof = date.fromisoformat(a.date) if a.date else session_date(datetime.now(JST))
    if a.check_prices:
        n, n_prev = count_priced(asof), prev_count_priced(asof)
        nk_last = nikkei_last_date(asof)  # Yahooから差分を取り直してDBに保存する
        print(f"[dip] {asof} の終値がある銘柄: {n}（前の取引日 {n_prev}） / 日経平均の最終日: {nk_last}")
        return 1 if prices_missing(n, n_prev) or index_missing(n, nk_last, asof) else 0

    if not a.date:
        # 先に、過去の平日で判定できていない日を判定し直す（送信済みの日は飛ばす）
        days = catchup_dates(asof)
        for d in days:
            # 判定し直しでそろわないときの警告は直前の平日（前夜の便が0時越えで黙った日）だけ。
            # それより前の日は前の便で警告済み。日経平均の取得自体が落ちた便で5日分の警告が並ばないようにする
            judge(d, notify, a.dry_run, late=True, warn_late=(d == days[-1]))
    quiet = not a.date and past_session_day(asof, datetime.now(JST))
    return judge(asof, notify, a.dry_run, quiet=quiet)


def judge(asof: date, notify, dry_run: bool, late: bool = False, quiet: bool = False,
          warn_late: bool = True) -> int:
    """asof を判定して送る。late=True は後の便での判定し直し（送信済みなら何もしない）。
    quiet=True は日本時間0時を越えた便（当日の終値が無いのは想定内）: 終値の欠け（PricesMissing）なら警告しない。
    後の便の判定し直し（late）でもそろわなければ、そこで初めて警告する（一度だけ）。"""
    if late and (notify.was_sent(f"dip_buy_{asof.isoformat()}")
                 or notify.was_sent(f"dip_buy_recovered_{asof.isoformat()}")):
        return 0
    try:
        nk_ret, picks, no_forecast, refs = find_candidates(asof)
    except Exception as e:
        print(f"[dip] ⚠ {asof} を判定できませんでした: {e}")
        if quiet and isinstance(e, PricesMissing):
            print(f"[dip] 日本時間0時を越えた便のため警告しない（次の便で {asof} を判定し直す）")
            return 0
        if late and (not warn_late or not isinstance(e, PricesMissing)
                     or notify.was_sent(f"dip_buy_error_{asof.isoformat()}")):
            return 1  # 警告は直前の平日の終値欠けだけ・一度だけ（障害は当日分の警告で知らせる）。次の便でまた判定し直す
        if not dry_run:
            notify.warn("押し目買い候補", "判定できませんでした",
                        detail=(f"{e}。翌日以降の便でもそろいません（株価更新の失敗の疑い）" if late else
                                f"{e}。後の便（{CATCHUP_WEEKDAYS}平日以内）で株価がそろえば自動で判定し直して送ります"),
                        dedupe_key=f"dip_buy_error_{asof.isoformat()}")
        return 1
    if not picks and not no_forecast:
        print(f"[dip] {asof}: 通知対象なし")
        # 「判定できませんでした」と送った日は、判定し直した結果も知らせて区切りを付ける
        if late and not dry_run and notify.was_sent(f"dip_buy_error_{asof.isoformat()}"):
            notify.push_once(f"dip_buy_recovered_{asof.isoformat()}", recovered_message(asof, nk_ret),
                             window_hours=math.inf)
        return 0
    # 候補0でも「予想が無くて判定できなかった」銘柄があれば知らせる（データ欠落で黙らない）
    text = build_message(asof, nk_ret, picks, no_forecast, refs, late=late)
    print(text)
    if dry_run:
        return 0
    # watchdog の再実行で同じ日に二重送信しない。判定し直しは一度送った日に二度と送らない
    notify.push_once(f"dip_buy_{asof.isoformat()}", text, **({"window_hours": math.inf} if late else {}))
    return 0

if __name__ == "__main__":
    sys.exit(main())

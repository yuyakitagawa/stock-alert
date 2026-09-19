#!/usr/bin/env python3
"""
web/dip_buy_alert.py — 市場急落日の押し目買い候補をLINEへ通知する

条件（2026-09-19 オーナー確定）:
  1. 日経平均の当日騰落が -2% 以下
  2. その銘柄の当日騰落が -4% 以下（下限なし）
  3. 株価300円以上・直近20日の平均売買代金1億円以上
  4. PBR < 1.5（株式分割は補正する）
  5. 会社予想の純利益 > 前期実績の純利益（前期は黒字）
  6. 従業員数 1000人以上（Supabase company_employees。tools/fetch_employees.py が更新）
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
MAX_LINES = 40
# 当日の終値がそろっている銘柄がこれ未満なら株価更新の失敗とみなす（平常時は約3,800）
MIN_PRICED_CODES = 1000

JST = timezone(timedelta(hours=9))


# ── 純粋関数（テスト対象） ─────────────────────────────────────────────

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
    return {
        "fy_disc": date.fromisoformat(str(f["disc_date"])[:10]),
        "bps": float(f["bps"]),
        "np": float(f["np"]),
        "fnp": float(fc[-1]["fnp"]) if fc else None,
    }


def passes_fundamentals(close: float, fund: dict, k_fy: float) -> tuple[bool, float | None, float | None]:
    """(合格か, PBR, 予想増益率)。PBRは分割補正後、予想増益は前期黒字が前提。"""
    pbr = close / (fund["bps"] * k_fy) if fund["bps"] > 0 else None
    growth = fund["fnp"] / fund["np"] - 1 if fund["np"] > 0 and fund["fnp"] is not None else None
    ok = pbr is not None and pbr < MAX_PBR and growth is not None and growth > 0
    return ok, pbr, growth


def build_message(asof: date, nk_ret: float, picks: list[dict], no_forecast: int = 0) -> str:
    lines = [
        f"📉 押し目買い候補 {asof:%Y-%m-%d}",
        f"日経平均 {nk_ret * 100:+.1f}% ／ 該当 {len(picks)}銘柄（β順）",
        "",
    ]
    for i, p in enumerate(picks[:MAX_LINES], 1):
        stale = " ⚠決算古" if p.get("stale") else ""
        lines.append(
            f"{i}. {p['code']} {p['name']} {p['ret'] * 100:+.1f}% "
            f"β{p['beta']:.2f} PBR{p['pbr']:.2f} 増益{p['growth'] * 100:+.0f}%{stale}"
        )
    if len(picks) > MAX_LINES:
        lines.append(f"…ほか{len(picks) - MAX_LINES}銘柄")
    if no_forecast:
        lines.append(f"※会社予想が未取得で増益を判定できず除外: {no_forecast}銘柄（PBR<1.5は満たす）")
    lines += [
        "",
        "条件: 日経-2%以下×当日-4%以下×PBR<1.5×予想増益×従業員1000人以上",
        "検証は当日終値買い・63日保有（10年で平均+10%前後）。この通知は引け後なので約定は翌日以降",
    ]
    return "\n".join(lines)


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


def find_candidates(asof: date) -> tuple[float | None, list[dict], int]:
    from lib import price_store
    from lib.utils import get_market_index_df_cached

    nk = get_market_index_df_cached("N225", "%5EN225", 500)
    if nk is None or len(nk) < BETA_MIN_DAYS:
        raise RuntimeError("日経平均の株価が取得できません")
    nk = nk[nk.index <= asof]
    if nk.index[-1] != asof:
        print(f"[dip] 日経平均の {asof} の終値が無い（休場日か未取得）: 最終 {nk.index[-1]}")
        return None, [], 0
    nk_ret = daily_return(nk["Close"].values)
    print(f"[dip] 日経平均 {asof}: {nk_ret * 100:+.2f}%")
    if nk_ret is None or nk_ret > NIKKEI_DROP:
        return nk_ret, [], 0

    nk_r = nk["Close"].pct_change()
    price_store.enable()
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
    if n_today < MIN_PRICED_CODES:
        # Step 0（株価更新）が落ちた日に「該当なし」で黙って終わらないようにする
        raise RuntimeError(f"{asof} の終値がある銘柄が{n_today}件しかありません（株価更新の失敗の疑い）")

    fins = _fin_rows([s["code"] for s in stage])
    stage2 = []
    no_forecast = 0
    for s in stage:
        fund = pick_fundamentals(fins.get(s["code"], []), asof)
        if not fund:
            continue
        k = split_factor(s["dates"], s["closes"], fund["fy_disc"], asof)
        if fund["fnp"] is None:
            # 会社予想が無い＝増益を判定できない。PBRは満たしていたかだけ数えて本文に出す
            if s["close"] / (fund["bps"] * k) < MAX_PBR:
                no_forecast += 1
            continue
        ok, pbr, growth = passes_fundamentals(s["close"], fund, k)
        if ok:
            s.update(pbr=pbr, growth=growth, stale=(asof - fund["fy_disc"]).days > STALE_FY_DAYS)
            stage2.append(s)
    print(f"[dip] PBR<1.5・予想増益: {len(stage2)}（PBR<1.5だが会社予想が無く判定できず: {no_forecast}）")

    from lib import employees
    emp_map = employees.get([s["code"] for s in stage2])  # company_employees、無い銘柄だけYahooで補う
    emps = [emp_map.get(s["code"]) for s in stage2]
    names = _load_names()
    picks = []
    for s, e in zip(stage2, emps):
        if e is not None and e >= MIN_EMPLOYEES:
            picks.append({k: s[k] for k in ("code", "ret", "beta", "pbr", "growth", "stale")}
                         | {"name": names.get(s["code"], ""), "employees": e})
    unknown = sum(e is None for e in emps)
    print(f"[dip] 従業員1000人以上: {len(picks)}（従業員数を取得できず除外: {unknown}）")
    picks.sort(key=lambda p: p["beta"], reverse=True)
    return nk_ret, picks, no_forecast


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import notify

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="判定日（既定: JSTの今日）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    asof = date.fromisoformat(a.date) if a.date else datetime.now(JST).date()

    try:
        nk_ret, picks, no_forecast = find_candidates(asof)
    except Exception as e:
        print(f"[dip] ⚠ 判定できませんでした: {e}")
        if not a.dry_run:
            notify.warn("押し目買い候補", "判定できませんでした", detail=str(e),
                        dedupe_key=f"dip_buy_error_{asof.isoformat()}")
        return 1
    if not picks and not no_forecast:
        print("[dip] 通知対象なし")
        return 0
    # 候補0でも「予想が無くて判定できなかった」銘柄があれば知らせる（データ欠落で黙らない）
    text = build_message(asof, nk_ret, picks, no_forecast)
    print(text)
    if a.dry_run:
        return 0
    # watchdog の再実行で同じ日に二重送信しない
    notify.push_once(f"dip_buy_{asof.isoformat()}", text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

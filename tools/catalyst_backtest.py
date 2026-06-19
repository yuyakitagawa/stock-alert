#!/usr/bin/env python3
"""
tools/catalyst_backtest.py
カタリスト候補スクリーン（PBR<1×低ROE×自己資本比率×利益の質A/B）の
ヒストリカル・バックテスト。point-in-time（開示日 disc_date ≤ 基準日）で再構成する。

目的: 「A/Bの利益の質フィルター（化粧/斜陽除外）が、実リターンの平均・勝率・大勝率を
改善するか」を数値で確認する（CLAUDE.md §5 のマージ規律に沿った可否判断）。

データ源（本番DBキャッシュ前提）:
  - price_cache:        株価（基準日close・保有後close）
  - jquants_fin_summary: bps/equity/ta/op/sales/np（disc_date ≤ 基準日のみ＝先読み無し）

ファンダは J-Quants 由来（kabutanクラウドブロック対策）。ROE は直近FYの np/equity で算出。

Usage:
  python3 tools/catalyst_backtest.py --start 2024-06-01 --end 2026-03-01 --hold 90 --top 5 --step 21
"""
import sys, os, argparse, sqlite3
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.db import DB_PATH
from lib.earnings_quality import assess_earnings_quality

BIG_WIN = 15.0  # 大勝率の閾値(%)


def close_asof(con, code, d):
    r = con.execute("SELECT close FROM price_cache WHERE code=? AND date<=? "
                    "ORDER BY date DESC LIMIT 1", (code, d)).fetchone()
    return r[0] if r and r[0] else None


def close_on_or_after(con, code, d):
    r = con.execute("SELECT close FROM price_cache WHERE code=? AND date>=? "
                    "ORDER BY date ASC LIMIT 1", (code, d)).fetchone()
    return r[0] if r and r[0] else None


def turnover_asof(con, code, d, days=20):
    rows = con.execute("SELECT close, volume FROM price_cache WHERE code=? AND date<=? "
                       "ORDER BY date DESC LIMIT ?", (code, d, days)).fetchall()
    vals = [c * v for c, v in rows if c and v]
    if len(vals) < max(5, days // 2):
        return None
    return (sum(vals) / len(vals)) / 1e6


def bps_asof(con, code, d):
    r = con.execute("SELECT bps FROM jquants_fin_summary WHERE code=? AND disc_date<=? "
                    "AND bps IS NOT NULL AND bps>0 ORDER BY disc_date DESC LIMIT 1",
                    (code, d)).fetchone()
    return r[0] if r else None


def equity_ratio_asof(con, code, d):
    r = con.execute("SELECT equity, ta FROM jquants_fin_summary WHERE code=? AND disc_date<=? "
                    "AND equity IS NOT NULL AND ta IS NOT NULL AND ta>0 "
                    "ORDER BY disc_date DESC LIMIT 1", (code, d)).fetchone()
    return (r[0] / r[1]) if r and r[1] else None


def roe_asof(con, code, d):
    r = con.execute("SELECT np, equity FROM jquants_fin_summary WHERE code=? AND disc_date<=? "
                    "AND doc_type='FY' AND np IS NOT NULL AND equity IS NOT NULL AND equity>0 "
                    "ORDER BY disc_date DESC LIMIT 1", (code, d)).fetchone()
    return (r[0] / r[1] * 100.0) if r and r[1] else None


def fy_rows_asof(con, code, d):
    """disc_date ≤ d の FY 実績行を assess_earnings_quality 形式で返す（fy_end昇順）＋予想行。"""
    rows = con.execute("SELECT fy_end, sales, op, np FROM jquants_fin_summary "
                       "WHERE code=? AND disc_date<=? AND doc_type='FY' AND op IS NOT NULL "
                       "ORDER BY fy_end ASC", (code, d)).fetchall()
    out = [{"fy_end": r[0], "is_forecast": False, "revenue": r[1],
            "op_profit": r[2], "net_income": r[3]} for r in rows]
    fc = con.execute("SELECT fop, fsales, fnp FROM jquants_fin_summary "
                     "WHERE code=? AND disc_date<=? AND fop IS NOT NULL "
                     "ORDER BY disc_date DESC LIMIT 1", (code, d)).fetchone()
    if fc and fc[0] is not None:
        out.append({"fy_end": "fc", "is_forecast": True, "revenue": fc[1],
                    "op_profit": fc[0], "net_income": fc[2]})
    return out


def select_candidates(con, codes, d, args, use_quality):
    """基準日 d 時点のカタリスト候補（上位 args.top）を返す。"""
    cands = []
    for code in codes:
        close = close_asof(con, code, d)
        bps = bps_asof(con, code, d)
        roe = roe_asof(con, code, d)
        if close is None or bps is None or bps <= 0 or roe is None:
            continue
        pbr = close / bps
        if pbr >= args.pbr_max or roe >= args.roe_max:
            continue
        eq = equity_ratio_asof(con, code, d)
        if eq is None or eq < args.equity_min:
            continue
        turn = turnover_asof(con, code, d)
        if turn is None or turn < args.min_turnover:
            continue
        score = (1.0 - pbr) * eq
        if use_quality:
            q = assess_earnings_quality(fy_rows_asof(con, code, d))
            if q["exclude"]:
                continue
            score *= (1.0 + q["bonus"])
        cands.append((score, code, close))
    cands.sort(key=lambda x: -x[0])
    return cands[:args.top]


def run(con, codes, dates, args, use_quality):
    rets = []
    for d in dates:
        target = (date.fromisoformat(d) + timedelta(days=args.hold)).isoformat()
        for score, code, entry in select_candidates(con, codes, d, args, use_quality):
            exitp = close_on_or_after(con, code, target)
            if exitp:
                rets.append((exitp / entry - 1.0) * 100.0)
    return rets


def summarize(rets, hold):
    if not rets:
        return "  該当なし（候補0）"
    n = len(rets)
    avg = sum(rets) / n
    win = sum(1 for r in rets if r > 0) / n * 100
    big = sum(1 for r in rets if r >= BIG_WIN) / n * 100
    # 近似CAGR（平均保有リターンを年率換算）
    cagr = ((1 + avg / 100) ** (365.0 / hold) - 1) * 100
    return (f"  件数{n} / 平均{avg:+.2f}% / 勝率{win:.0f}% / 大勝率(≥{BIG_WIN:.0f}%){big:.0f}% "
            f"/ 近似CAGR{cagr:+.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--end", default="2026-03-01")
    p.add_argument("--hold", type=int, default=90, help="保有日数")
    p.add_argument("--step", type=int, default=21, help="リバランス間隔（日）")
    p.add_argument("--top", type=int, default=5, help="保有銘柄数")
    p.add_argument("--pbr-max", type=float, default=1.0)
    p.add_argument("--roe-max", type=float, default=8.0)
    p.add_argument("--equity-min", type=float, default=0.5)
    p.add_argument("--min-turnover", type=float, default=500.0)
    args = p.parse_args()

    con = sqlite3.connect(DB_PATH)
    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM price_cache").fetchall()]

    d0, d1 = date.fromisoformat(args.start), date.fromisoformat(args.end)
    dates = []
    d = d0
    while d <= d1:
        dates.append(d.isoformat())
        d += timedelta(days=args.step)

    print(f"カタリストBT: {args.start}〜{args.end} / {len(dates)}リバランス点 / 保有{args.hold}日 / 上位{args.top}")
    print(f"ゲート: PBR<{args.pbr_max} / ROE<{args.roe_max}% / 自己資本>{args.equity_min*100:.0f}% / 代金≥{args.min_turnover:.0f}M\n")

    rets_off = run(con, codes, dates, args, use_quality=False)
    rets_on = run(con, codes, dates, args, use_quality=True)
    con.close()

    print("A/Bなし（ゲートのみ）:")
    print(summarize(rets_off, args.hold))
    print("A/Bあり（化粧/斜陽除外＋成長加点）:")
    print(summarize(rets_on, args.hold))


if __name__ == "__main__":
    main()

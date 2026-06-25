"""
sell_signal_backtest.py
売りシグナル戦略の比較バックテスト

テスト戦略:
  A: 現行 (方向感なし + 弱気 + 下降)
  B: 弱気 + 下降 のみ (方向感なしを無視)
  C: 下降 のみ
  D: 日経クラッシュ中は方向感なしを抑制 (Nikkei 5日リターン < -N%)
  E: 日経クラッシュ中は方向感なし+弱気を抑制

Usage:
  python3 tools/sell_signal_backtest.py
  python3 tools/sell_signal_backtest.py --start 2026-01-01 --end 2026-05-20
  python3 tools/sell_signal_backtest.py --crash-threshold -5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2026-01-01")
parser.add_argument("--end",   default="2026-05-20")
parser.add_argument("--crash-threshold", type=float, default=-5.0,
                    help="日経5日リターンがこの値(%)を下回ったらクラッシュ判定 (default: -5.0)")
args = parser.parse_args()

SBUY_URL   = os.getenv("SUPABASE_URL")
SBUY_KEY   = os.getenv("SUPABASE_SERVICE_KEY")
HEADERS    = {"apikey": SBUY_KEY, "Authorization": f"Bearer {SBUY_KEY}"}
CRASH_THRESH = args.crash_threshold

# ── データ取得 ─────────────────────────────────────────────────────────────

def fetch_all(params):
    rows, offset = [], 0
    while True:
        p = {**params, "limit": 1000, "offset": offset}
        r = requests.get(f"{SBUY_URL}/rest/v1/gen_rankings", params=p, headers=HEADERS)
        batch = r.json()
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows

print("Supabaseからデータ取得中...")
buy_rows  = fetch_all({"recommend": "eq.S買い", "order": "date.asc",
                        "select": "code,name,close,date",
                        "date": f"gte.{args.start}", "date2": f"lte.{args.end}"})
# Supabase では同一パラメータキーが使えないので範囲フィルターを別途
buy_rows  = [r for r in fetch_all({"recommend": "eq.S買い", "order": "date.asc",
                                    "select": "code,name,close,date"})
             if args.start <= r["date"] <= args.end]

all_sell  = fetch_all({"recommend": "in.(方向感なし,弱気シグナル,下降シグナル)",
                        "order": "date.asc", "select": "code,close,date,recommend"})
latest    = fetch_all({"date": f"eq.{args.end}", "select": "code,close"})
latest_map = {r["code"]: r["close"] for r in latest}

print(f"  S買い: {len(buy_rows)}行  売りシグナル全: {len(all_sell)}行")

# ── 日経225 5日リターン計算 ────────────────────────────────────────────────

print("日経225データ取得中...")
from lib.db import load_market_index_data
_nk_df = load_market_index_data("N225", days=2200)
if _nk_df is not None and len(_nk_df) > 0:
    nk_close = _nk_df["Close"].squeeze()
    nk_close.index = pd.Index([d.strftime("%Y-%m-%d") for d in _nk_df.index])
else:
    nk_close = pd.Series(dtype=float)
nk_5d_ret = {}  # date -> 5日前比リターン(%)
dates_sorted = sorted(nk_close.index)
for i, d in enumerate(dates_sorted):
    if i >= 5:
        prev = nk_close.iloc[i - 5]
        cur  = nk_close.iloc[i]
        nk_5d_ret[d] = (cur - prev) / prev * 100
print(f"  日経225: {len(nk_5d_ret)}日分")

def is_crash_day(date_str):
    return nk_5d_ret.get(date_str, 0) < CRASH_THRESH

# ── シミュレーションコア ───────────────────────────────────────────────────

def run_sim(label, sell_set_func):
    """
    sell_set_func(row) -> bool: この行が売りシグナルとして有効かどうか
    """
    first_buy = {}
    for row in buy_rows:
        if row["code"] not in first_buy:
            first_buy[row["code"]] = row

    first_sell = {}
    for row in all_sell:
        code = row["code"]
        buy  = first_buy.get(code)
        if not buy or code in first_sell:
            continue
        if row["date"] <= buy["date"]:
            continue
        if sell_set_func(row):
            first_sell[code] = row

    pnls, months = [], {}
    for code, buy in first_buy.items():
        sell = first_sell.get(code)
        if sell:
            pct = (sell["close"] - buy["close"]) / buy["close"] * 100
        else:
            cur = latest_map.get(code, buy["close"])
            pct = (cur - buy["close"]) / buy["close"] * 100
        pnls.append(pct)
        # 月別集計
        m = buy["date"][:7]
        months.setdefault(m, []).append(pct)

    n    = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    avg  = sum(pnls) / n if n else 0
    mg   = max(pnls) if pnls else 0
    ml   = min(pnls) if pnls else 0
    sold = len(first_sell)
    held = n - sold

    month_str = "  ".join(
        f"{m}[{len(v)}件 勝率{sum(1 for x in v if x>0)/len(v)*100:.0f}% 平均{sum(v)/len(v):+.1f}%]"
        for m, v in sorted(months.items())
    )
    return {
        "label": label, "n": n, "sold": sold, "held": held,
        "wins": wins, "avg": avg, "mg": mg, "ml": ml,
        "month_str": month_str,
    }

# ── 戦略定義 ─────────────────────────────────────────────────────────────

SELL_ALL    = {"方向感なし", "弱気シグナル", "下降シグナル"}
SELL_WEAK   = {"弱気シグナル", "下降シグナル"}
SELL_DOWN   = {"下降シグナル"}
SELL_WEAK_D = {"弱気シグナル", "下降シグナル"}

def strat_A(row):  # 現行
    return row["recommend"] in SELL_ALL

def strat_B(row):  # 弱気+下降のみ
    return row["recommend"] in SELL_WEAK

def strat_C(row):  # 下降のみ
    return row["recommend"] in SELL_DOWN

def strat_D(row):  # クラッシュ中は方向感なし抑制
    rec = row["recommend"]
    if rec == "方向感なし" and is_crash_day(row["date"]):
        return False
    return rec in SELL_ALL

def strat_E(row):  # クラッシュ中は方向感なし+弱気抑制
    rec = row["recommend"]
    if rec in ("方向感なし", "弱気シグナル") and is_crash_day(row["date"]):
        return False
    return rec in SELL_ALL

strategies = [
    ("A: 現行（方向感なし+弱気+下降）",  strat_A),
    ("B: 弱気+下降 のみ",                strat_B),
    ("C: 下降 のみ",                     strat_C),
    (f"D: クラッシュ中({CRASH_THRESH}%)は方向感なし抑制",  strat_D),
    (f"E: クラッシュ中({CRASH_THRESH}%)は方向感なし+弱気抑制", strat_E),
]

# ── 結果出力 ─────────────────────────────────────────────────────────────

results = [run_sim(label, fn) for label, fn in strategies]

print(f"\n{'='*72}")
print(f"売りシグナル戦略バックテスト  {args.start} 〜 {args.end}  (クラッシュ閾値: {CRASH_THRESH}%)")
print(f"{'='*72}")
print(f"{'戦略':<40}  {'勝率':>5}  {'平均リタ':>8}  {'最大利':>8}  {'最大損':>8}  保有/決済")
print(f"{'─'*72}")
for r in results:
    print(f"{r['label']:<40}  {r['wins']/r['n']*100:>4.1f}%  {r['avg']:>+8.2f}%  {r['mg']:>+8.2f}%  {r['ml']:>+8.2f}%  {r['held']}/{r['sold']}")

print(f"\n{'─'*72}")
print("月別内訳:")
for r in results:
    print(f"  [{r['label'][:25]}]  {r['month_str']}")

# クラッシュ日カレンダー
print(f"\n日経5日リターン < {CRASH_THRESH}% だった日:")
crash_days = sorted(d for d, v in nk_5d_ret.items() if v < CRASH_THRESH)
crash_in_range = [d for d in crash_days if args.start <= d <= args.end]
for d in crash_in_range:
    print(f"  {d}  Nikkei 5d: {nk_5d_ret[d]:+.1f}%")

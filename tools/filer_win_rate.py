#!/usr/bin/env python3
"""
tools/filer_win_rate.py
投資家（EDINET提出者）別の「乗っかり勝率」バックテスト。

「この投資家が大量保有報告書で買い増しを開示したとき、63営業日(3ヶ月)後まで
保有していたら儲かったか」を投資家別に集計する。tools/catalyst_backtest.pyと
同じ指標セット（平均リターン・勝率・大勝率、BIG_WIN=15%）を踏襲する。

対象は「買い」開示のみ（tools/scan_large_holdings.pyのis_noise_match()で
自己申告・訂正報告書・過半数超・売り方向を除外した残り）。開示からまだ
63営業日経っていないイベントは結果未確定として除外する。

Usage:
  python3 tools/filer_win_rate.py                  # 全投資家（開示件数の多い順）
  python3 tools/filer_win_rate.py --min-n 3         # 3件以上開示している投資家のみ
  python3 tools/filer_win_rate.py --hold 63
  python3 tools/filer_win_rate.py --out data/filer_win_rate.csv
"""
import sys, os, argparse, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb
from lib.db import get_price_raw
from tools.scan_large_holdings import is_noise_match

BIG_WIN = 15.0  # 大勝率の閾値(%)。catalyst_backtest.pyと同じ定義。


def fetch_buy_events() -> list[dict]:
    """edinet_large_holdings全件から「買い」開示のみを抽出する。"""
    rows = sb.select(
        "edinet_large_holdings",
        "select=filer_name,issuer_code,issuer_name,disc_date,holding_ratio,"
        "holding_ratio_prior,doc_description&order=disc_date.asc",
    )
    events = []
    for r in rows:
        if not r.get("issuer_code") or not r.get("filer_name"):
            continue
        reason = is_noise_match(
            r["filer_name"], r.get("issuer_name") or "",
            r.get("doc_description") or "",
            r.get("holding_ratio"), r.get("holding_ratio_prior"),
        )
        if reason:
            continue
        events.append(r)
    return events


def fetch_filer_categories() -> dict:
    """filer_name → 投資家分類(13分類)。"""
    rows = sb.select("edinet_filer_classification", "select=filer_name,category")
    return {r["filer_name"]: r["category"] for r in rows}


def compute_outcome(prices: list, disc_date: str, hold_days: int) -> "float | None":
    """開示日以降の最初のcloseをエントリー、hold_days営業日後のcloseをエグジットに
    リターン(%)を返す。エグジットがまだ無ければNone（結果未確定）。"""
    entry_idx = None
    for i, (d, close, _vol) in enumerate(prices):
        if d >= disc_date and close:
            entry_idx = i
            break
    if entry_idx is None:
        return None
    exit_idx = entry_idx + hold_days
    if exit_idx >= len(prices):
        return None
    entry_close = prices[entry_idx][1]
    exit_close = prices[exit_idx][1]
    if not entry_close or not exit_close:
        return None
    return (exit_close / entry_close - 1.0) * 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hold", type=int, default=63, help="保有期間（営業日）")
    p.add_argument("--min-n", type=int, default=1, help="最低サンプル件数（未満は表示しない）")
    p.add_argument("--out", type=str, default=None, help="CSV出力先（省略時は標準出力のみ）")
    p.add_argument("--no-persist", action="store_true",
                   help="Supabase filer_win_rateへの保存をスキップ（確認用）")
    args = p.parse_args()

    print("EDINET大量保有報告書を取得中...")
    events = fetch_buy_events()
    print(f"  買い開示: {len(events)}件（自己申告・訂正・過半数超・売りを除外済み）")

    categories = fetch_filer_categories()

    print("銘柄別の株価キャッシュを取得中...")
    codes = sorted({e["issuer_code"] for e in events})
    price_by_code = {}
    for code in codes:
        raw = get_price_raw(code)
        price_by_code[code] = raw or []

    outcomes = []  # (filer_name, category, return_pct)
    immature = 0
    for e in events:
        prices = price_by_code.get(e["issuer_code"])
        if not prices:
            continue
        ret = compute_outcome(prices, e["disc_date"], args.hold)
        if ret is None:
            immature += 1
            continue
        category = categories.get(e["filer_name"], "その他")
        outcomes.append((e["filer_name"], category, ret))

    print(f"  結果確定: {len(outcomes)}件 / 結果未確定(まだ{args.hold}営業日経っていない): {immature}件\n")

    if not outcomes:
        print("結果確定イベントが0件のため集計できません。")
        return

    # 投資家別集計
    filer_stats = {}
    for filer, category, ret in outcomes:
        s = filer_stats.setdefault(filer, {"category": category, "n": 0, "wins": 0,
                                            "big_wins": 0, "returns": []})
        s["n"] += 1
        s["wins"] += 1 if ret > 0 else 0
        s["big_wins"] += 1 if ret >= BIG_WIN else 0
        s["returns"].append(ret)

    rows = []
    for filer, s in filer_stats.items():
        if s["n"] < args.min_n:
            continue
        n = s["n"]
        win_rate = s["wins"] / n
        avg_return = sum(s["returns"]) / n
        big_win_rate = s["big_wins"] / n
        rows.append({
            "filer_name": filer,
            "category": s["category"],
            "n": n,
            "win_rate": round(win_rate * 100, 1),
            "avg_return": round(avg_return, 2),
            "big_win_rate": round(big_win_rate * 100, 1),
            "hold_days": args.hold,
        })

    rows.sort(key=lambda r: (-r["win_rate"], -r["n"]))

    print(f"{'投資家':40s} {'分類':14s} {'n':>4s} {'勝率':>7s} {'平均リターン':>10s} {'大勝率':>7s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['filer_name'][:40]:40s} {r['category']:14s} {r['n']:>4d} "
              f"{r['win_rate']:>6.1f}% "
              f"{r['avg_return']:>+9.2f}% {r['big_win_rate']:>6.1f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n→ {args.out} に保存しました（{len(rows)}件）")

    if not args.no_persist:
        sb.upsert("filer_win_rate", rows, on_conflict="filer_name")
        print(f"→ Supabase filer_win_rate に保存しました（{len(rows)}件）")


if __name__ == "__main__":
    main()

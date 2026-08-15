#!/usr/bin/env python3
"""
tools/filer_win_rate.py
投資家（EDINET提出者）別の「乗っかりリターン」バックテスト。

「この投資家が大量保有報告書で買い増しを開示したとき、その回の推定取得金額
（保有比率の変化幅×発行済株式数×株価）をそのまま63営業日(3ヶ月)後まで
保有していたら、損益はいくらだったか」を投資家別に集計する（億円単位）。
1株だけ買ったと仮定する方式では実際の投資規模（数億〜数百億円）と乖離するため、
web/publish_blog_articles.pyのestimate_deal_amount_oku()と同じ考え方で実額を概算する。

発行済株式数はyfinance（レート制限あり）ではなく、point-in-time
（開示日as_ofで取得済みの直近開示分、先読み無し）のjquants_fin_summary.sh_outを使う
（EDINET決算XBRL由来でSupabaseに既にキャッシュ済みのため、大量イベントの一括処理でも
API呼び出しが発生しない）。

対象は「買い」開示のみ（tools/scan_large_holdings.pyのis_noise_match()で
自己申告・訂正報告書・過半数超・売り方向を除外した残り）。開示からまだ
63営業日経っていないイベント、発行済株式数が取得できないイベントは集計から除外する。

Usage:
  python3 tools/filer_win_rate.py                  # 全投資家（トータルリターンの多い順）
  python3 tools/filer_win_rate.py --min-n 3         # 3件以上開示している投資家のみ
  python3 tools/filer_win_rate.py --hold 63
  python3 tools/filer_win_rate.py --out data/filer_win_rate.csv
"""
import sys, os, argparse, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb
from lib.db import get_price_raw
from tools.scan_large_holdings import is_noise_match


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


def fetch_shares_outstanding_history() -> dict:
    """issuer_code → [(disc_date, sh_out), ...]（disc_date昇順）。
    point-in-timeでas_of以前の最新値を引けるようにする（先読み防止）。"""
    rows = sb.select(
        "jquants_fin_summary",
        "select=code,disc_date,sh_out&sh_out=not.is.null&order=disc_date.asc",
    )
    history: dict = {}
    for r in rows:
        history.setdefault(r["code"], []).append((r["disc_date"], r["sh_out"]))
    return history


def shares_outstanding_asof(history: list, as_of: str) -> "float | None":
    """as_of以前で最新のsh_out（先読み無し）。"""
    value = None
    for d, sh in history:
        if d > as_of:
            break
        value = sh
    return value


def compute_outcome(prices: list, disc_date: str, hold_days: int) -> "tuple[float, float] | None":
    """開示日以降の最初のcloseをエントリー、hold_days営業日後のcloseをエグジットに
    (entry_close, exit_close)を返す。エグジットがまだ無ければNone（結果未確定）。"""
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
    return entry_close, exit_close


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

    print("銘柄別の株価キャッシュ・発行済株式数を取得中...")
    codes = sorted({e["issuer_code"] for e in events})
    price_by_code = {}
    for code in codes:
        raw = get_price_raw(code)
        price_by_code[code] = raw or []
    sh_out_history = fetch_shares_outstanding_history()

    outcomes = []  # (filer_name, category, deal_amount_yen, return_pct)
    immature = 0
    no_shares = 0
    for e in events:
        prices = price_by_code.get(e["issuer_code"])
        if not prices:
            continue
        outcome = compute_outcome(prices, e["disc_date"], args.hold)
        if outcome is None:
            immature += 1
            continue
        entry_close, exit_close = outcome

        ratio_prior = e.get("holding_ratio_prior")
        ratio = e.get("holding_ratio")
        if ratio is None:
            continue
        ratio_change = (ratio - ratio_prior) if ratio_prior is not None else ratio
        if ratio_change <= 0:
            continue

        sh_out = shares_outstanding_asof(sh_out_history.get(e["issuer_code"], []), e["disc_date"])
        if not sh_out:
            no_shares += 1
            continue

        deal_amount_yen = sh_out * entry_close * (ratio_change / 100)
        return_pct = exit_close / entry_close - 1.0
        category = categories.get(e["filer_name"], "その他")
        outcomes.append((e["filer_name"], category, deal_amount_yen, deal_amount_yen * return_pct))

    print(f"  結果確定: {len(outcomes)}件 / 結果未確定(まだ{args.hold}営業日経っていない): {immature}件"
          f" / 発行済株式数が取得できず除外: {no_shares}件\n")

    if not outcomes:
        print("結果確定イベントが0件のため集計できません。")
        return

    # 投資家別集計（推定取得金額(円)ベースの損益）
    filer_stats = {}
    for filer, category, deal_amount_yen, return_yen in outcomes:
        s = filer_stats.setdefault(
            filer, {"category": category, "n": 0, "deal_amount": 0.0, "returns": []}
        )
        s["n"] += 1
        s["deal_amount"] += deal_amount_yen
        s["returns"].append(return_yen)

    OKU = 1e8
    rows = []
    for filer, s in filer_stats.items():
        if s["n"] < args.min_n:
            continue
        n = s["n"]
        total_return_oku = sum(s["returns"]) / OKU
        rows.append({
            "filer_name": filer,
            "category": s["category"],
            "n": n,
            "total_return_oku": round(total_return_oku, 2),
            "hold_days": args.hold,
        })

    rows.sort(key=lambda r: -r["total_return_oku"])

    print(f"{'投資家':40s} {'分類':14s} {'n':>4s} {'トータル(億円)':>14s}")
    print("-" * 78)
    for r in rows:
        print(f"{r['filer_name'][:40]:40s} {r['category']:14s} {r['n']:>4d} "
              f"{r['total_return_oku']:>+13.2f}")

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

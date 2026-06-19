#!/usr/bin/env python3
"""
tools/screen_catalyst_candidates.py
カタリスト候補スクリーン（GARP・割安成長株投資の補助ツール）

4つのカタリスト（親子上場解消/TOB・アクティビスト介入・業界再編寡占化・不採算撤退/構造改革）
が「起きやすい構造的特徴」を持つ割安株を、既存の無料データから洗い出す。

確定した「疑惑」ではなく「起きやすい候補」のスクリーン。真の先回り（誰が大量保有したか）は
EDINET 大量保有報告書が必要（別途登録）。

抽出条件（デフォルト）:
  - PBR < 1.0       … 簿価割れ＝買収・改革で是正余地（TOB/親子解消の妙味）
  - ROE < 8%        … 資本効率が低い＝改革・アクティビストの標的
  - 自己資本比率 > 50% … 資産・現金が厚い＝TOB原資/株主還元余地
  + 独占率ウォッチリスト該当 … ③業界寡占の追い風（フラグ表示）

Usage:
  python3 tools/screen_catalyst_candidates.py
  python3 tools/screen_catalyst_candidates.py --pbr-max 0.8 --roe-max 6 --equity-min 0.6 --top 50
"""
import sys, os, csv, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.supabase_client as sb


def load_name_map():
    """code→name。J-Quants全銘柄マップ(data/code_name_map.json)優先、無ければランキングCSV。"""
    import json
    name_map = {}
    if os.path.exists("data/code_name_map.json"):
        name_map = json.load(open("data/code_name_map.json", encoding="utf-8"))
    files = sorted(glob.glob("data/rankings/ranking_*.csv"), reverse=True)
    if files:
        with open(files[0], encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                c = str(row.get("銘柄コード", "")).strip()
                if c and c not in name_map:
                    name_map[c] = row.get("銘柄名", "")
    return name_map


def load_monopoly_set():
    """独占率ウォッチリストの code→note。"""
    mono = {}
    path = "data/pricing_power_watchlist.csv"
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                c = str(row.get("code", "")).strip()
                if c:
                    mono[c] = f"{row.get('product','')}({row.get('domestic_share','')})"
    return mono


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pbr-max", type=float, default=1.0)
    p.add_argument("--roe-max", type=float, default=8.0)
    p.add_argument("--equity-min", type=float, default=0.5, help="自己資本比率の下限（0-1）")
    p.add_argument("--min-turnover", type=float, default=100.0, help="平均売買代金の下限（百万円）。流動性フィルタ")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--out", type=str, default="data/catalyst_candidates.csv")
    args = p.parse_args()

    name_map = load_name_map()
    mono = load_monopoly_set()

    print("universe: Supabase(RPC)でスクリーニング中...")
    rows = sb.rpc("screen_catalyst_candidates", {
        "pbr_max": args.pbr_max,
        "roe_max": args.roe_max,
        "equity_min": args.equity_min,
        "min_turnover": args.min_turnover,
    }) or []

    cands = []
    for r in rows:
        code = r["code"]
        cands.append({
            "code": code, "name": name_map.get(code, ""),
            "pbr": round(r["pbr"], 2), "roe": round(r["roe"], 1),
            "equity_ratio": round(r["equity_ratio"] * 100, 1),
            "turnover_m": round(r["turnover_m"], 0),
            "monopoly": mono.get(code, ""),
            "score": round(r["score"], 3),
        })

    cands.sort(key=lambda x: -x["score"])

    print(f"\n該当: {len(cands)}銘柄（PBR<{args.pbr_max} / ROE<{args.roe_max}% / 自己資本比率>{args.equity_min*100:.0f}% / 売買代金≥{args.min_turnover:.0f}百万円）\n")
    print(f"{'コード':<6}{'銘柄名':<22}{'PBR':>6}{'ROE%':>7}{'自己資本%':>9}{'代金(百万)':>10}  寡占/メモ")
    print("-" * 88)
    for c in cands[:args.top]:
        nm = (c["name"] or "")[:20]
        flag = f"  🏰{c['monopoly']}" if c["monopoly"] else ""
        print(f"{c['code']:<6}{nm:<22}{c['pbr']:>6}{c['roe']:>7}{c['equity_ratio']:>9}{c['turnover_m']:>10.0f}{flag}")

    # CSV保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "pbr", "roe", "equity_ratio", "turnover_m", "monopoly", "score"])
        w.writeheader()
        w.writerows(cands)
    print(f"\n全{len(cands)}件を保存: {args.out}")


if __name__ == "__main__":
    main()

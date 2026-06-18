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

利益の質フィルター（--quality、デフォルトON）:
  低ROEの"理由"を問い、化粧決算と斜陽事業を機械的に除外する。
  - A 利益の質: 営業赤字 / 純利益>営業益×1.5（一過性益の水増し疑い）を除外
  - B 本業方向: 直近実績の営業利益が前年比減益を除外
  - 加減点: 売上3期CAGR・営業利益率トレンド・会社予想方向で score を調整
  除外された銘柄は理由付きで data/catalyst_excluded.csv に出力（人手レビュー用）。

Usage:
  python3 tools/screen_catalyst_candidates.py
  python3 tools/screen_catalyst_candidates.py --pbr-max 0.8 --roe-max 6 --equity-min 0.6 --top 50
  python3 tools/screen_catalyst_candidates.py --no-quality   # 品質フィルター無効
"""
import sys, os, csv, argparse, sqlite3, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.db import DB_PATH
from lib.kabutan_earnings import fetch_kabutan_earnings
from lib.earnings_quality import assess_earnings_quality


def latest_close(con, code):
    r = con.execute(
        "SELECT close FROM price_cache WHERE code=? ORDER BY date DESC LIMIT 1", (code,)
    ).fetchone()
    return r[0] if r and r[0] else None


def latest_bps_roe(con, code):
    """fundamentals_annual から直近の非NULL bps と roe を独立に取得。"""
    rows = con.execute(
        "SELECT roe, bps FROM fundamentals_annual WHERE code=? ORDER BY fy_end DESC LIMIT 6",
        (code,)
    ).fetchall()
    roe = next((r[0] for r in rows if r[0] is not None), None)
    bps = next((r[1] for r in rows if r[1] is not None), None)
    return bps, roe


def latest_bps_split_safe(con, code):
    """分割調整済み株価(price_cache)と整合するBPSを返す。
    J-Quants(jquants_fin_summary)のBPSは開示ごとに分割後株数で再表示されるため、
    株式分割を実施した銘柄でも分割調整漏れが起きない。直近開示のbps(>0)を採用。
    J-Quants未取得の銘柄は None（呼び出し側でfundamentals_annual値にフォールバック）。"""
    r = con.execute(
        "SELECT bps FROM jquants_fin_summary "
        "WHERE code=? AND bps IS NOT NULL AND bps>0 "
        "ORDER BY disc_date DESC LIMIT 1", (code,)
    ).fetchone()
    return r[0] if r else None


def latest_equity_ratio(con, code):
    """jquants_fin_summary から直近の equity/ta（自己資本比率）。"""
    r = con.execute(
        "SELECT equity, ta FROM jquants_fin_summary "
        "WHERE code=? AND equity IS NOT NULL AND ta IS NOT NULL AND ta>0 "
        "ORDER BY disc_date DESC LIMIT 1", (code,)
    ).fetchone()
    if r and r[1]:
        return r[0] / r[1]
    return None


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


def avg_turnover_m(con, code, days=20):
    """直近days日の平均売買代金（百万円）。流動性フィルタ用。"""
    rows = con.execute(
        "SELECT close, volume FROM price_cache WHERE code=? ORDER BY date DESC LIMIT ?",
        (code, days)
    ).fetchall()
    vals = [c * v for c, v in rows if c and v]
    if len(vals) < max(5, days // 2):
        return None
    return (sum(vals) / len(vals)) / 1e6


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
    p.add_argument("--no-quality", action="store_true",
                   help="利益の質フィルター(A/B)を無効化（化粧/斜陽の除外を行わない）")
    p.add_argument("--excluded-out", type=str, default="data/catalyst_excluded.csv",
                   help="品質フィルターで除外した銘柄の出力先（人手レビュー用）")
    args = p.parse_args()

    con = sqlite3.connect(DB_PATH)
    name_map = load_name_map()
    mono = load_monopoly_set()

    codes = [r[0] for r in con.execute("SELECT DISTINCT code FROM price_cache").fetchall()]
    print(f"universe: {len(codes)}銘柄をスクリーニング中...")

    cands = []
    excluded = []
    excl_counts = {"営業赤字": 0, "化粧決算": 0, "本業減益": 0}
    for code in codes:
        close = latest_close(con, code)
        bps_kab, roe = latest_bps_roe(con, code)
        # PBRは分割調整済み株価と整合するJ-Quants BPSを優先（分割調整漏れ防止）。
        # J-Quants未取得の銘柄のみ fundamentals_annual(株探) 値にフォールバック。
        bps = latest_bps_split_safe(con, code) or bps_kab
        if close is None or bps is None or bps <= 0 or roe is None:
            continue
        pbr = close / bps
        if pbr >= args.pbr_max or roe >= args.roe_max:
            continue
        eq_ratio = latest_equity_ratio(con, code)
        if eq_ratio is None or eq_ratio < args.equity_min:
            continue
        # 流動性フィルタ（実際に売買できる水準に絞る）
        turn = avg_turnover_m(con, code)
        if turn is None or turn < args.min_turnover:
            continue
        # 候補スコア: 割安(PBR低)×財務余力(自己資本比率高)。低いほど買収/改革妙味が大きい
        score = (1.0 - pbr) * eq_ratio

        row = {
            "code": code, "name": name_map.get(code, ""),
            "pbr": round(pbr, 2), "roe": round(roe, 1),
            "equity_ratio": round(eq_ratio * 100, 1),
            "turnover_m": round(turn, 0),
            "monopoly": mono.get(code, ""),
            "score": round(score, 3),
        }

        # 利益の質フィルター（A: 化粧除外 / B: 斜陽除外）＋ 成長加減点
        if not args.no_quality:
            q = assess_earnings_quality(fetch_kabutan_earnings(code))
            row.update({
                "op_yoy": q["op_yoy"], "op_margin": q["op_margin"],
                "rev_cagr3": q["rev_cagr3"], "fc_dir": q["fc_dir"],
                "growth_bonus": q["bonus"],
            })
            if q["exclude"]:
                excl_counts[q["exclude"]] = excl_counts.get(q["exclude"], 0) + 1
                excluded.append({**row, "reason": q["exclude"],
                                 "note": "; ".join(q["notes"])})
                continue
            # 成長方向で base score を調整（gateを通った候補の並べ替え）
            row["score"] = round(score * (1.0 + q["bonus"]), 3)

        cands.append(row)

    cands.sort(key=lambda x: -x["score"])
    con.close()

    q_on = not args.no_quality
    print(f"\n該当: {len(cands)}銘柄（PBR<{args.pbr_max} / ROE<{args.roe_max}% / 自己資本比率>{args.equity_min*100:.0f}% / 売買代金≥{args.min_turnover:.0f}百万円"
          f"{' / 利益の質A・B' if q_on else ''}）")
    if q_on:
        print(f"品質フィルター除外: 営業赤字{excl_counts['営業赤字']} / 化粧決算{excl_counts['化粧決算']} / 本業減益{excl_counts['本業減益']}（計{len(excluded)}件）")
    if q_on:
        print(f"\n{'コード':<6}{'銘柄名':<20}{'PBR':>6}{'ROE%':>6}{'自己資本%':>8}{'営業益YoY%':>10}{'売上CAGR%':>9}{'予想':>8}  寡占/メモ")
    else:
        print(f"\n{'コード':<6}{'銘柄名':<22}{'PBR':>6}{'ROE%':>7}{'自己資本%':>9}{'代金(百万)':>10}  寡占/メモ")
    print("-" * 96)
    for c in cands[:args.top]:
        nm = (c["name"] or "")[:18]
        flag = f"  🏰{c['monopoly']}" if c["monopoly"] else ""
        if q_on:
            yoy = f"{c.get('op_yoy')}" if c.get("op_yoy") is not None else "-"
            cagr = f"{c.get('rev_cagr3')}" if c.get("rev_cagr3") is not None else "-"
            print(f"{c['code']:<6}{nm:<20}{c['pbr']:>6}{c['roe']:>6}{c['equity_ratio']:>8}{yoy:>10}{cagr:>9}{c.get('fc_dir','-'):>8}{flag}")
        else:
            print(f"{c['code']:<6}{nm:<20}{c['pbr']:>6}{c['roe']:>7}{c['equity_ratio']:>9}{c['turnover_m']:>10.0f}{flag}")

    # 除外された地雷を理由付きでログ表示（化粧/斜陽の妥当性を人手レビューしやすく）
    if q_on and excluded:
        print(f"\n── 品質フィルター除外 {len(excluded)}件（地雷レビュー用）──")
        print(f"{'コード':<6}{'銘柄名':<20}{'理由':<8}  詳細")
        print("-" * 80)
        for e in sorted(excluded, key=lambda x: x["reason"]):
            print(f"{e['code']:<6}{(e['name'] or '')[:18]:<20}{e['reason']:<8}  {e['note']}")

    # CSV保存
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base_fields = ["code", "name", "pbr", "roe", "equity_ratio", "turnover_m", "monopoly", "score"]
    q_fields = ["op_yoy", "op_margin", "rev_cagr3", "fc_dir", "growth_bonus"] if q_on else []
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + q_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    print(f"\n全{len(cands)}件を保存: {args.out}")

    # 除外銘柄を理由付きで保存（人手レビュー用）
    if q_on and excluded:
        os.makedirs(os.path.dirname(args.excluded_out), exist_ok=True)
        with open(args.excluded_out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=base_fields + ["reason", "note"], extrasaction="ignore")
            w.writeheader()
            w.writerows(excluded)
        print(f"除外{len(excluded)}件を保存: {args.excluded_out}")


if __name__ == "__main__":
    main()

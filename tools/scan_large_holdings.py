#!/usr/bin/env python3
"""
tools/scan_large_holdings.py
EDINET 大量保有スキャナー（GARP・イベント駆動の先回り検出）

毎日 EDINET から大量保有報告書（5%ルール）をスキャンして edinet_holdings に蓄積し、
カタリスト候補（screen_catalyst_candidates.py の出力）と突合する。

  「構造的に改革・買収が起きやすい候補」× 「実際に誰かが 5%超を買い集めた事実」
  = 本物の先回り候補（構造的候補 × 実際の買い集め）

過去の歴史的検証はできない（開示は提出後に蓄積するフォワード方式）。日次で回し続けて
イベントを溜め、構造的候補にヒットしたものを通知する。

Usage:
  python3 tools/scan_large_holdings.py                 # 直近7日スキャン＋突合
  python3 tools/scan_large_holdings.py --days 30       # 遡る日数
  python3 tools/scan_large_holdings.py --candidates data/catalyst_candidates.csv
"""
import sys, os, csv, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.edinet import scan_large_holdings, verify_api, _api_key
from lib.db import get_edinet_holdings_recent


def load_candidate_codes(path: str) -> dict:
    """カタリスト候補CSV（code列）を code→行dict で読み込む。無ければ空dict。"""
    cands = {}
    if not path or not os.path.exists(path):
        return cands
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            c = str(row.get("code", "")).strip()
            if c:
                cands[c] = row
    return cands


def load_name_map() -> dict:
    path = "data/code_name_map.json"
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7, help="EDINETを遡る日数（蓄積用）")
    p.add_argument("--match-days", type=int, default=30, help="突合で見る蓄積イベントの範囲")
    p.add_argument("--candidates", type=str, default="data/catalyst_candidates.csv")
    p.add_argument("--out", type=str, default="data/edinet_holding_matches.csv")
    p.add_argument("--no-fetch", action="store_true", help="スキャンせずDB蓄積分だけ突合")
    p.add_argument("--verify", action="store_true", help="APIキーの有効性のみ確認して終了")
    args = p.parse_args()

    if args.verify:
        v = verify_api()
        print(f"EDINET APIキー検証: date={v['date']} status={v['status']} "
              f"総件数={v['total']} 大量保有={v['large']} → {v['reason']}")
        if v["ok"]:
            print("✅ PASS: キーは有効。大量保有報告書を取得できます。")
            sys.exit(0)
        else:
            print(f"❌ FAIL: {v['reason']}")
            sys.exit(1)

    if not _api_key():
        print("⚠️ EDINET_API_KEY が未設定です（.env または環境変数）。スキャンをスキップします。")
        if args.no_fetch:
            pass
        else:
            return

    if not args.no_fetch and _api_key():
        print(f"EDINET 大量保有報告書をスキャン中（直近{args.days}日）...")
        recs = scan_large_holdings(days_back=args.days, persist=True)
        n350 = sum(1 for r in recs if r["doc_type_code"] == "350")
        n360 = sum(1 for r in recs if r["doc_type_code"] == "360")
        print(f"  取得: 大量保有報告書{n350}件 / 変更報告書{n360}件（計{len(recs)}件）")

    cands = load_candidate_codes(args.candidates)
    name_map = load_name_map()
    print(f"カタリスト候補: {len(cands)}銘柄（{args.candidates}）")

    # DB蓄積分から突合
    holdings = get_edinet_holdings_recent(
        days=args.match_days,
        codes=list(cands.keys()) if cands else None,
    )

    if not cands:
        print("\n候補CSVが無いため全大量保有イベントを表示します（突合なし）:")

    matches = []
    for h in holdings:
        code = h.get("sec_code")
        if cands and code not in cands:
            continue
        cand = cands.get(code, {})
        matches.append({
            "code": code,
            "name": name_map.get(code, cand.get("name", "")),
            "filer_name": h.get("filer_name", ""),
            "doc_type": "大量保有" if h.get("doc_type_code") == "350" else "変更",
            "disc_date": h.get("disc_date", ""),
            "pbr": cand.get("pbr", ""),
            "roe": cand.get("roe", ""),
            "equity_ratio": cand.get("equity_ratio", ""),
            "doc_description": (h.get("doc_description") or "").strip(),
        })

    title = "構造的候補 × 実際の買い集め（本物の先回り候補）" if cands else "大量保有イベント（直近）"
    print(f"\n🎯 {title}: {len(matches)}件\n")
    print(f"{'コード':<6}{'銘柄名':<20}{'種別':<8}{'開示日':<12}{'提出者':<24}  概要")
    print("-" * 100)
    for m in matches[:60]:
        nm = (m["name"] or "")[:18]
        filer = (m["filer_name"] or "")[:22]
        print(f"{m['code'] or '----':<6}{nm:<20}{m['doc_type']:<8}{m['disc_date']:<12}{filer:<24}  {m['doc_description'][:40]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "code", "name", "filer_name", "doc_type", "disc_date",
            "pbr", "roe", "equity_ratio", "doc_description"])
        w.writeheader()
        w.writerows(matches)
    print(f"\n全{len(matches)}件を保存: {args.out}")


if __name__ == "__main__":
    main()

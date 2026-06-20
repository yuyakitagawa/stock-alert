#!/usr/bin/env python3
"""
tools/scan_large_holdings.py
EDINET 大量保有スキャナー（GARP・イベント駆動の先回り検出）

毎日 EDINET から大量保有報告書（5%ルール）をスキャンして edinet_large_holdings に蓄積し、
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
from lib.db import get_edinet_large_holdings_recent


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


def _normalize_name(s: str) -> str:
    """社名比較用に正規化（法人格・記号・空白を除去）。"""
    if not s:
        return ""
    for tok in ["株式会社", "(株)", "（株）", "ホールディングス", "HD", " ", "　", "・"]:
        s = s.replace(tok, "")
    return s.strip()


# 「買い集め」ではない＝先回りシグナルとして不要な開示の語
_SELL_KEYWORDS = ["譲渡", "売却", "売出", "処分"]


def is_noise_match(filer_name: str, issuer_name: str, doc_description: str) -> "str | None":
    """突合ヒットがノイズなら理由を返す（先回りシグナルでないもの）。問題なければ None。

    - self_filing: 提出者≒対象企業（自己申告。第三者の買い集めではない）
    - sell:        概要が譲渡/売却/処分（買いではない）
    """
    if any(k in (doc_description or "") for k in _SELL_KEYWORDS):
        return "sell"
    f = _normalize_name(filer_name)
    i = _normalize_name(issuer_name)
    if f and i and (f in i or i in f):
        return "self_filing"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7, help="EDINETを遡る日数（蓄積用）")
    p.add_argument("--start", type=str, default=None,
                   help="バックフィル開始日 YYYY-MM-DD（指定時はこの日から当日まで全走査・--daysは無視）")
    p.add_argument("--match-days", type=int, default=30, help="突合で見る蓄積イベントの範囲")
    p.add_argument("--candidates", type=str, default="data/catalyst_candidates.csv")
    p.add_argument("--out", type=str, default="data/edinet_holding_matches.csv")
    p.add_argument("--no-fetch", action="store_true", help="スキャンせずDB蓄積分だけ突合")
    p.add_argument("--verify", action="store_true", help="APIキーの有効性のみ確認して終了")
    p.add_argument("--no-exclude", action="store_true",
                   help="ノイズ除外（自己申告・譲渡/売却）を無効化して全ヒットを表示")
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
        if args.start:
            print(f"EDINET 大量保有報告書をバックフィル中（{args.start} 〜 当日・土日スキップ）...")
            recs = scan_large_holdings(start_date=args.start, persist=True, sleep_sec=0.2)
        else:
            print(f"EDINET 大量保有報告書をスキャン中（直近{args.days}日）...")
            recs = scan_large_holdings(days_back=args.days, persist=True)
        n350 = sum(1 for r in recs if r["doc_type_code"] == "350")
        n360 = sum(1 for r in recs if r["doc_type_code"] == "360")
        print(f"  取得: 大量保有報告書{n350}件 / 変更報告書{n360}件（計{len(recs)}件）")

    cands = load_candidate_codes(args.candidates)
    name_map = load_name_map()
    print(f"カタリスト候補: {len(cands)}銘柄（{args.candidates}）")

    # DB蓄積分から突合
    holdings = get_edinet_large_holdings_recent(
        days=args.match_days,
        codes=list(cands.keys()) if cands else None,
    )

    if not cands:
        print("\n候補CSVが無いため全大量保有イベントを表示します（突合なし）:")

    matches = []
    excluded = {"self_filing": 0, "sell": 0}
    for h in holdings:
        code = h.get("sec_code")
        if cands and code not in cands:
            continue
        cand = cands.get(code, {})
        issuer_name = name_map.get(code, cand.get("name", ""))
        filer = h.get("filer_name", "")
        desc = (h.get("doc_description") or "").strip()
        # 自己申告・売り/譲渡の報告を除外（第三者の買い集めだけ残す）
        if not args.no_exclude:
            reason = is_noise_match(filer, issuer_name, desc)
            if reason:
                excluded[reason] += 1
                continue
        ratio = h.get("holding_ratio")
        matches.append({
            "code": code,
            "name": issuer_name,
            "filer_name": filer,
            "doc_type": "大量保有" if h.get("doc_type_code") == "350" else "変更",
            "disc_date": h.get("disc_date", ""),
            "holding_ratio": ratio,
            "pbr": cand.get("pbr", ""),
            "roe": cand.get("roe", ""),
            "equity_ratio": cand.get("equity_ratio", ""),
            "doc_description": desc,
        })

    title = "構造的候補 × 実際の買い集め（本物の先回り候補）" if cands else "大量保有イベント（直近）"
    if not args.no_exclude and (excluded["self_filing"] or excluded["sell"]):
        print(f"\n（ノイズ除外: 自己申告{excluded['self_filing']}件 / 譲渡・売却{excluded['sell']}件）")
    print(f"\n🎯 {title}: {len(matches)}件\n")
    print(f"{'コード':<6}{'銘柄名':<20}{'種別':<8}{'開示日':<12}{'保有%':>6}  {'提出者':<24}  概要")
    print("-" * 110)
    for m in matches[:60]:
        nm = (m["name"] or "")[:18]
        filer = (m["filer_name"] or "")[:22]
        ratio_str = f"{m['holding_ratio']:.1f}" if m.get("holding_ratio") is not None else "  -"
        print(f"{m['code'] or '----':<6}{nm:<20}{m['doc_type']:<8}{m['disc_date']:<12}{ratio_str:>6}  {filer:<24}  {m['doc_description'][:40]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "code", "name", "filer_name", "doc_type", "disc_date",
            "holding_ratio", "pbr", "roe", "equity_ratio", "doc_description"])
        w.writeheader()
        w.writerows(matches)
    print(f"\n全{len(matches)}件を保存: {args.out}")


if __name__ == "__main__":
    main()

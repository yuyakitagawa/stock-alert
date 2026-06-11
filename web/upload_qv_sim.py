"""
web/upload_qv_sim.py — QVバックテスト結果を Supabase web_qv_sim テーブルへアップロード

使い方:
  python3 web/upload_qv_sim.py --csv simulations/backtests/strategy_v2_qv_2026-01-01_2026-06-12.csv
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import requests
from dotenv import load_dotenv

load_dotenv()

SB_URL = os.getenv("SUPABASE_URL", "").strip()
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_ANON_KEY", "")).strip()

def upsert(rows: list[dict]) -> None:
    if not rows:
        return
    resp = requests.post(
        f"{SB_URL}/rest/v1/web_qv_sim",
        headers={
            "apikey": SB_KEY,
            "Authorization": f"Bearer {SB_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        json=rows,
        timeout=30,
    )
    if not resp.ok:
        print(f"upsert failed: {resp.status_code} {resp.text[:300]}")
    else:
        print(f"upsert {len(rows)} 件 完了")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ret = float(r["return"]) if r.get("return") else None
            rows.append({
                "code":         r["code"],
                "name":         r["name"],
                "entry_date":   r["entry"],
                "exit_date":    r["exit"] if r.get("exit") else None,
                "entry_price":  float(r["entry_px"]) if r.get("entry_px") else None,
                "exit_price":   float(r["exit_px"])  if r.get("exit_px")  else None,
                "return_pct":   ret,
                "reason":       r.get("reason"),
                "held_days":    int(r["held_days"])   if r.get("held_days") else None,
                "status":       "active" if r.get("reason") == "期末" else "closed",
            })

    print(f"{len(rows)} 件読み込み")
    upsert(rows)


if __name__ == "__main__":
    main()

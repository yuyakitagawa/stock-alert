"""
web/update_gem_sim.py — 💎買いシグナルの実績シミュレーションを gen_qv_sim へ同期

日次CI (daily_alert.yml) の export_to_web.py 後に実行する。
gen_rankings の recommend='💎 買い' を追跡し、最大90日保有のシミュレーションを構築。

使い方:
  python3 web/update_gem_sim.py          # 日次更新（新規💎エントリー＋既存ポジション更新）
  python3 web/update_gem_sim.py --reset  # 全件リセット＆再構築
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

SB_URL = os.getenv("SUPABASE_URL", "").strip()
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_ANON_KEY", "")).strip()
HOLD_LIMIT = 90
DECISION_INTERVAL = 14  # 💎チェックは2週間おき（頻繁すぎるエントリーを防ぐ）
MAX_POSITIONS = 10


def sb_headers():
    return {
        "apikey": SB_KEY,
        "Authorization": f"Bearer {SB_KEY}",
        "Content-Type": "application/json",
    }


def sb_get(table, query=""):
    url = f"{SB_URL}/rest/v1/{table}?{query}"
    r = requests.get(url, headers=sb_headers(), timeout=30)
    return r.json() if r.ok else []


def sb_upsert(table, rows):
    if not rows:
        return
    h = sb_headers()
    h["Prefer"] = "resolution=merge-duplicates"
    for i in range(0, len(rows), 200):
        batch = rows[i:i+200]
        r = requests.post(f"{SB_URL}/rest/v1/{table}", headers=h, json=batch, timeout=30)
        if r.ok:
            print(f"  upsert {len(batch)}件 完了")
        else:
            print(f"  upsert 失敗: {r.status_code} {r.text[:200]}")


def sb_delete(table, query):
    url = f"{SB_URL}/rest/v1/{table}?{query}"
    r = requests.delete(url, headers=sb_headers(), timeout=30)
    return r.ok


def get_all_trading_dates():
    rows = sb_get("gen_rankings", "select=date&order=date.asc&limit=1")
    if not rows:
        return []
    first = rows[0]["date"]
    rows2 = sb_get("gen_rankings", "select=date&order=date.desc&limit=1")
    last = rows2[0]["date"] if rows2 else first
    dates = sb_get("gen_rankings", f"select=date&order=date.asc&limit=500")
    return sorted(set(r["date"] for r in dates))


def get_gems_on_date(d):
    rows = sb_get("gen_rankings",
                  f"date=eq.{d}&recommend=eq.💎 買い&order=net.desc&limit=50"
                  "&select=code,name,close,net,drop_prob,vol")
    return rows


def get_price_on_date(code, d):
    rows = sb_get("gen_rankings",
                  f"date=eq.{d}&code=eq.{code}&select=close,drop_prob,recommend")
    if rows and rows[0].get("close"):
        return rows[0]
    return None


def build_simulation():
    """gen_rankings の全日付から💎シグナルを追跡してシミュレーションを構築。"""
    print("gen_rankings の全日付を取得中...")
    all_dates = get_all_trading_dates()
    if not all_dates:
        print("日付データなし"); return []

    print(f"  {len(all_dates)}営業日 ({all_dates[0]} ~ {all_dates[-1]})")

    positions = {}   # code -> {entry_date, entry_price, name}
    trades = []
    last_entry_date = None

    for di, d in enumerate(all_dates):
        # 既存ポジションの更新（90日チェック、drop急騰チェック）
        for code in list(positions.keys()):
            pos = positions[code]
            entry_idx = all_dates.index(pos["entry_date"])
            held_days = di - entry_idx

            sell = None
            row = get_price_on_date(code, d)
            current_price = row["close"] if row else None
            current_drop = row.get("drop_prob") if row else None

            if held_days >= HOLD_LIMIT:
                sell = "時間切れ"
            elif current_drop is not None and current_drop >= 10.0:
                sell = "drop急騰"

            if sell and current_price:
                ret = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append({
                    "code": code,
                    "name": pos["name"],
                    "entry_date": pos["entry_date"],
                    "exit_date": d,
                    "entry_price": round(pos["entry_price"], 1),
                    "exit_price": round(current_price, 1),
                    "return_pct": round(ret, 2),
                    "reason": sell,
                    "held_days": held_days,
                    "status": "closed",
                })
                del positions[code]

        # 新規💎エントリー（前回エントリーからDECISION_INTERVAL日以上経過）
        if last_entry_date is not None:
            last_idx = all_dates.index(last_entry_date)
            if di - last_idx < DECISION_INTERVAL:
                continue

        open_slots = MAX_POSITIONS - len(positions)
        if open_slots <= 0:
            continue

        gems = get_gems_on_date(d)
        if not gems:
            continue

        added = 0
        for g in gems:
            if added >= open_slots:
                break
            code = g["code"]
            if code in positions:
                continue
            positions[code] = {
                "entry_date": d,
                "entry_price": g["close"],
                "name": g["name"],
            }
            added += 1

        if added > 0:
            last_entry_date = d
            print(f"  {d}: +{added}件エントリー（計{len(positions)}ポジション）")

    # 期末: 残ポジションを active として記録
    last_d = all_dates[-1]
    for code, pos in positions.items():
        entry_idx = all_dates.index(pos["entry_date"])
        held_days = len(all_dates) - 1 - entry_idx
        row = get_price_on_date(code, last_d)
        current_price = row["close"] if row else pos["entry_price"]
        ret = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
        trades.append({
            "code": code,
            "name": pos["name"],
            "entry_date": pos["entry_date"],
            "exit_date": last_d,
            "entry_price": round(pos["entry_price"], 1),
            "exit_price": round(current_price, 1),
            "return_pct": round(ret, 2),
            "reason": "期末",
            "held_days": held_days,
            "status": "active",
        })

    return trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="全件削除して再構築")
    args = p.parse_args()

    if not SB_URL or not SB_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定"); return

    if args.reset:
        print("gen_qv_sim を全件削除...")
        sb_delete("gen_qv_sim", "id=gt.0")

    trades = build_simulation()
    trades = [t for t in trades if t["entry_date"] >= "2026-01-01"]
    if not trades:
        print("トレードなし"); return

    import numpy as np
    rets = [t["return_pct"] for t in trades]
    wins = sum(1 for r in rets if r > 0)
    print(f"\n{'='*50}")
    print(f"トレード: {len(trades)}件")
    print(f"勝率: {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)")
    print(f"平均リターン: {np.mean(rets):+.2f}%")
    print(f"累計: {sum(rets):+.1f}%")
    print(f"{'='*50}")

    print("\nSupabase へアップロード...")
    sb_upsert("gen_qv_sim", trades)
    print("完了")


if __name__ == "__main__":
    main()

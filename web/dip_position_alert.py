#!/usr/bin/env python3
"""
web/dip_position_alert.py — 押し目買いで買った銘柄の「保有63営業日」をLINEで知らせる

押し目買い候補（web/dip_buy_alert.py）の検証は「当日終値で買い、63営業日保有」
（docs/dip_buy_strategy.md）。実際に買った銘柄を data/dip_positions.json に書いておくと、
買付日から63営業日たった日に、その時点の損益を添えて1回だけ通知する（notify.push_once、
同じ銘柄・買付日では二度送らない）。売るかどうかは人間が決める。

営業日は日経平均の終値がある日で数える（祝日表を持たない）。ジョブが落ちた日を取りこぼさないよう、
63営業日「以上」で送る（送信済みなら push_once が抑える）。

使い方:
  python web/dip_position_alert.py              # 満了した銘柄があれば送る
  python web/dip_position_alert.py --dry-run    # 送らずに本文と残り営業日を表示
  python web/dip_position_alert.py --date 2027-01-20 --dry-run
"""
import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

BASE_DIR = os.getenv("STOCK_ALERT_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

HOLD_DAYS = 63
POSITIONS_PATH = os.path.join(BASE_DIR, "data", "dip_positions.json")
JST = timezone(timedelta(hours=9))


def days_held(trading_dates, buy_date: date, asof: date) -> int:
    """買付日の翌営業日から asof までの営業日数（買付日当日は0日目）。"""
    return sum(1 for d in trading_dates if buy_date < d <= asof)


def build_message(asof: date, rows: list[dict]) -> str:
    """rows: position + {"close": 現在値 or None, "held": 営業日数}"""
    lines = [f"⏰ 押し目買い 保有{HOLD_DAYS}営業日 {asof.month}/{asof.day}",
             "検証の前提（63営業日保有）に達した銘柄です。売るかは判断してください", ""]
    total_cost = total_val = 0.0
    for p in rows:
        head = f"・{p['name']} {p['code']}（{p['buy_date'][5:].replace('-', '/')}買付 {p['qty']}株）"
        if p.get("close") is None:
            lines += [head, f"  取得{p['price']:,.0f} → 現在値が取れません"]
            continue
        cost, val = p["price"] * p["qty"], p["close"] * p["qty"]
        total_cost += cost
        total_val += val
        lines += [head, f"  取得{p['price']:,.0f} → {p['close']:,.0f}（{(p['close'] / p['price'] - 1) * 100:+.1f}%・"
                        f"{val - cost:+,.0f}円）"]
    if total_cost:
        lines += ["", f"合計 {total_val - total_cost:+,.0f}円（{(total_val / total_cost - 1) * 100:+.1f}%）"]
    return "\n".join(lines)


def load_positions(path: str = POSITIONS_PATH) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _close(code: str, asof: date) -> float | None:
    from lib import price_store
    df = price_store.frame(code, end=asof.isoformat(), days=15)
    return None if df is None or df.empty else float(df["Close"].values[-1])


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    from lib import notify, price_store
    from lib.utils import get_market_index_df_cached

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", help="判定日（YYYY-MM-DD、既定は今日JST）")
    a = ap.parse_args()
    asof = date.fromisoformat(a.date) if a.date else datetime.now(JST).date()

    positions = load_positions()
    if not positions:
        print("[position] data/dip_positions.json に保有が無い")
        return 0
    nk = get_market_index_df_cached("N225", "%5EN225", 200)
    if nk is None or nk.empty:
        print("[position] ⚠ 日経平均の日付が取れないため判定できません")
        return 1
    dates = [d for d in nk.index if d <= asof]

    price_store.enable()
    due: dict[str, list[dict]] = {}
    for p in positions:
        held = days_held(dates, date.fromisoformat(p["buy_date"]), asof)
        print(f"[position] {p['code']} {p['name']} {p['buy_date']}買付: {held}/{HOLD_DAYS}営業日")
        if held >= HOLD_DAYS:
            due.setdefault(p["buy_date"], []).append(p | {"held": held, "close": _close(p["code"], asof)})
    for buy_date, rows in due.items():
        text = build_message(asof, rows)
        print(text)
        if not a.dry_run:
            codes = "_".join(sorted(r["code"] for r in rows))
            notify.push_once(f"dip_exit_{buy_date}_{codes}", text, window_hours=math.inf)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""押し目買いの保有満了通知（web/dip_position_alert.py）のユニットテスト。

実行: python3 tests/test_dip_position_alert.py  /  pytest tests/test_dip_position_alert.py
"""
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.dip_position_alert import HOLD_DAYS, build_message, days_held, load_positions


def test_days_held_counts_trading_days_after_buy_date():
    dates = [date(2026, 10, 16) + timedelta(days=i) for i in range(10)]
    dates = [d for d in dates if d.weekday() < 5]          # 10/16(金)〜10/23(金)の営業日
    assert days_held(dates, date(2026, 10, 16), date(2026, 10, 16)) == 0   # 買付日当日は0
    assert days_held(dates, date(2026, 10, 16), date(2026, 10, 23)) == 5


def test_build_message_shows_pnl_and_total():
    rows = [{"code": "6118", "name": "アイダ", "buy_date": "2026-10-16", "qty": 100, "price": 1168, "close": 1300.0},
            {"code": "7966", "name": "リンテック", "buy_date": "2026-10-16", "qty": 1, "price": 5730, "close": None}]
    text = build_message(date(2027, 1, 20), rows)
    assert f"保有{HOLD_DAYS}営業日 1/20" in text
    assert "・アイダ 6118（10/16買付 100株）" in text
    assert "取得1,168 → 1,300（+11.3%・+13,200円）" in text
    assert "現在値が取れません" in text
    assert "合計 +13,200円（+11.3%）" in text


def test_positions_file_is_valid():
    ps = load_positions()
    assert ps, "data/dip_positions.json が空"
    for p in ps:
        assert {"code", "name", "buy_date", "qty", "price"} <= set(p)
        date.fromisoformat(p["buy_date"])


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

"""押し目買い候補のLINE通知（web/dip_buy_alert.py）のユニットテスト。

実行: python3 tests/test_dip_buy_alert.py  /  pytest tests/test_dip_buy_alert.py
"""
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.dip_buy_alert import (beta, build_message, daily_return, passes_fundamentals,
                               pick_fundamentals, split_factor, MAX_LINES)


def test_daily_return():
    assert abs(daily_return(np.array([100.0, 95.0])) - (-0.05)) < 1e-12
    assert daily_return(np.array([100.0])) is None


def test_beta_recovers_slope():
    """銘柄 = 1.5 × 日経 + ノイズ なら β≒1.5。"""
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 250)
    s = 1.5 * m + rng.normal(0, 0.002, 250)
    assert abs(beta(s, m) - 1.5) < 0.05


def test_beta_needs_enough_days():
    assert beta(np.zeros(100), np.ones(100)) is None


def test_split_factor_detects_split_after_disclosure():
    """開示後の1:10分割（分割日に株価も-4%動いた）を拾う。±3%では取りこぼしていた。"""
    d = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
    closes = np.array([50000.0, 50910.0, 5310.0])  # 古河電工 2026-06-03 の実例（比率0.104）
    assert abs(split_factor(d, closes, date(2025, 5, 13), date(2026, 7, 28)) - 0.1043) < 0.001


def test_split_factor_ignores_split_before_disclosure_and_real_drops():
    d = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 3, 2), date(2026, 3, 3)]
    closes = np.array([1000.0, 500.0, 500.0, 300.0])  # 1/6の分割は開示前、3/3の-40%は分割比に近くない
    assert split_factor(d, closes, date(2026, 2, 1), date(2026, 3, 31)) == 1.0


def test_pick_fundamentals_uses_only_disclosures_before_the_day():
    rows = [
        {"disc_date": "2025-05-13", "doc_type": "FY", "fy_end": "2025-03-31", "bps": 1000, "np": 100, "fnp": None},
        {"disc_date": "2025-11-10", "doc_type": "2Q", "fy_end": "2026-03-31", "bps": None, "np": 40, "fnp": 120},
        {"disc_date": "2026-09-02", "doc_type": "1Q", "fy_end": "2027-03-31", "bps": None, "np": 10, "fnp": 50},  # 当日開示は見えない
    ]
    f = pick_fundamentals(rows, date(2026, 9, 2))
    assert f["bps"] == 1000 and f["np"] == 100 and f["fnp"] == 120
    assert f["fy_disc"] == date(2025, 5, 13)


def test_pick_fundamentals_none_without_fy():
    assert pick_fundamentals([{"disc_date": "2025-11-10", "doc_type": "2Q", "fy_end": "2026-03-31", "np": 40, "fnp": 120}],
                             date(2026, 1, 1)) is None


def test_pick_fundamentals_ignores_forecast_of_same_year_as_latest_actual():
    """FY2026/3の実績が入った後に、FY2026/3の予想（古い）と比べて増益判定しない。

    2026-04-25以降の決算はEDINET由来で会社予想が入らないため、放置すると必ず起きる。"""
    rows = [
        {"disc_date": "2025-11-10", "doc_type": "2Q", "fy_end": "2026-03-31", "bps": None, "np": 40, "fnp": 120},
        {"disc_date": "2026-06-27", "doc_type": "FY", "fy_end": "2026-03-31", "bps": 1100, "np": 130, "fnp": None},
    ]
    f = pick_fundamentals(rows, date(2026, 9, 2))
    assert f["np"] == 130 and f["fnp"] is None
    assert passes_fundamentals(500.0, f, 1.0)[0] is False


def test_passes_fundamentals_split_adjusted_pbr():
    """分割補正なしだとPBR0.62で通ってしまう銘柄を、補正後のPBR≒6で落とす。"""
    fund = {"bps": 4845.0, "np": 100.0, "fnp": 110.0}
    ok, pbr, _ = passes_fundamentals(3008.0, fund, 1.0)
    assert ok and pbr < 1.5
    ok, pbr, _ = passes_fundamentals(3008.0, fund, 0.1043)
    assert not ok and pbr > 5


def test_passes_fundamentals_requires_growth_from_profit():
    assert passes_fundamentals(500.0, {"bps": 1000.0, "np": 100.0, "fnp": 100.0}, 1.0)[0] is False  # 増益でない
    assert passes_fundamentals(500.0, {"bps": 1000.0, "np": -50.0, "fnp": 10.0}, 1.0)[0] is False   # 前期赤字
    assert passes_fundamentals(500.0, {"bps": 1000.0, "np": 100.0, "fnp": 120.0}, 1.0)[0] is True


def test_build_message_truncates_and_marks_stale():
    picks = [{"code": str(1000 + i), "name": "銘柄", "ret": -0.05, "beta": 1.2, "pbr": 1.0,
              "growth": 0.1, "stale": i == 0} for i in range(MAX_LINES + 3)]
    text = build_message(date(2026, 9, 2), -0.029, picks)
    assert "日経平均 -2.9%" in text and f"該当 {MAX_LINES + 3}銘柄" in text
    assert "⚠決算古" in text.splitlines()[3]
    assert "…ほか3銘柄" in text
    assert len(text) < 4000


def test_build_message_reports_missing_forecasts():
    text = build_message(date(2026, 9, 2), -0.029, [], no_forecast=12)
    assert "該当 0銘柄" in text and "判定できず除外: 12銘柄" in text


def test_employees_stale_codes_orders_missing_first_then_oldest():
    from lib.employees import stale_codes
    rows = {"1000": {"fetched_date": "2026-09-18"},   # 新しい→対象外
            "2000": {"fetched_date": "2026-07-01"},
            "3000": {"fetched_date": "2026-06-01"}}
    got = stale_codes(["1000", "2000", "3000", "4000"], rows, 30, date(2026, 9, 19))
    assert got == ["4000", "3000", "2000"]


def test_employees_get_reads_table_and_fills_only_missing():
    import lib.employees as emp
    saved = {}
    orig = (emp.load, emp.fetch_many, emp.save)
    emp.load = lambda codes=None: {"1000": {"employees": 5000, "fetched_date": "2026-09-01"},
                                   "2000": {"employees": None, "fetched_date": "2026-09-01"}}
    emp.fetch_many = lambda codes, workers=8: {c: 1234 for c in codes}
    emp.save = lambda values, today=None: saved.update(values) or True
    try:
        got = emp.get(["1000", "2000", "3000"])
    finally:
        emp.load, emp.fetch_many, emp.save = orig
    assert got == {"1000": 5000, "2000": None, "3000": 1234}  # 値なし(None)の行は取り直さない
    assert saved == {"3000": 1234}


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

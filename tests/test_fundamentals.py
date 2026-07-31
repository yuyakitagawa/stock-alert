"""point-in-timeファンダ（lib/fundamentals）のユニットテスト。
_filter_asof / get_pit_fundamentals(rows=...) が、DBに都度問い合わせる従来経路と
同じpoint-in-time結果（未来の開示を先読みしない）を返すことを確認する。

実行: python3 tests/test_fundamentals.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.fundamentals import _filter_asof, get_pit_fundamentals

# disc_date降順（get_jquants_fin_history_all()と同じ並び）
_ROWS = [
    {"disc_date": "2026-06-01", "doc_type": "FY", "eps": 120, "bps": 800,
     "np": 500, "ta": 5000, "equity": 2000, "cfo": 300, "op": 400,
     "sales": 3000, "div_ann": 20, "payout_ratio": 20, "fnp": 480},
    {"disc_date": "2026-03-01", "doc_type": "FY", "eps": 100, "bps": 700,
     "np": 400, "ta": 4500, "equity": 1800, "cfo": 250, "op": 350,
     "sales": 2800, "div_ann": 18, "payout_ratio": 18, "fnp": 420},
    {"disc_date": "2025-03-01", "doc_type": "FY", "eps": 80, "bps": 600,
     "np": 300, "ta": 4000, "equity": 1600, "cfo": 200, "op": 300,
     "sales": 2500, "div_ann": 15, "payout_ratio": 15, "fnp": 310},
]


def test_filter_asof_excludes_future_disclosures():
    """as_of日より後のdisc_dateは含めない（先読みバイアス防止）。"""
    out = _filter_asof(_ROWS, "2026-02-01", n=10)
    assert [r["disc_date"] for r in out] == ["2025-03-01"]


def test_filter_asof_respects_limit_n():
    out = _filter_asof(_ROWS, "2026-12-31", n=2)
    assert [r["disc_date"] for r in out] == ["2026-06-01", "2026-03-01"]


def test_filter_asof_fy_only():
    rows = _ROWS + [{"disc_date": "2026-05-01", "doc_type": "1Q", "eps": 30}]
    out = _filter_asof(rows, "2026-12-31", n=10, fy_only=True)
    assert all(r["doc_type"] == "FY" for r in out)
    assert len(out) == 3


def test_pit_fundamentals_no_lookahead():
    """2026-03-01開示より前の時点では、その開示のepsが見えてはいけない。"""
    fd = get_pit_fundamentals("1234", date(2026, 2, 1), rows=_ROWS)
    assert fd is not None
    assert fd["eps"] == 80  # 2025-03-01時点の開示のみ既知


def test_pit_fundamentals_sees_latest_after_disclosure():
    fd = get_pit_fundamentals("1234", date(2026, 7, 1), rows=_ROWS)
    assert fd is not None
    assert fd["eps"] == 120
    # eps_growth = (120-100)/100
    assert abs(fd["eps_growth"] - 0.2) < 1e-9


def test_pit_fundamentals_none_without_any_disclosure_or_yutai():
    fd = get_pit_fundamentals("9999", date(2020, 1, 1), rows=[])
    assert fd is None


if __name__ == "__main__":
    test_filter_asof_excludes_future_disclosures()
    test_filter_asof_respects_limit_n()
    test_filter_asof_fy_only()
    test_pit_fundamentals_no_lookahead()
    test_pit_fundamentals_sees_latest_after_disclosure()
    test_pit_fundamentals_none_without_any_disclosure_or_yutai()
    print("OK: test_fundamentals (6 tests)")

"""point-in-timeファンダ（lib/fundamentals）のユニットテスト。
_filter_asof / get_pit_fundamentals(rows=...) が、DBに都度問い合わせる従来経路と
同じpoint-in-time結果（未来の開示を先読みしない）を返すことを確認する。

実行: python3 tests/test_fundamentals.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.fundamentals import (_filter_asof, get_pit_fundamentals, get_pit_valuation,
                              disclosure_split_ratio, price_split_jumps)

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


# 378A（ヒット）: 2026-07-01 に1:2分割。FY2025は分割前、FY2026は分割後ベース（期末株数は分割前のまま）
_HIT = [
    {"disc_date": "2026-09-24", "doc_type": "FY", "eps": 70.42, "bps": 405.04,
     "equity": 5781169000, "np": 984167000, "sh_out": 7136600, "div_ann": 20},
    {"disc_date": "2025-08-14", "doc_type": "FY", "eps": 162.79, "bps": 609.08,
     "equity": 3391000000, "np": 905000000, "sh_out": 5560000, "div_ann": 17.5},
]


def test_split_ratio_detects_post_period_split_with_issuance():
    """増資(+28%)と分割(×2)が重なっても分割比2を取り出す（Rは2.56で素直に丸めると3になる）。"""
    assert disclosure_split_ratio(_HIT[0], _HIT[1]) == 2.0


def test_split_ratio_ignores_normal_growth():
    assert disclosure_split_ratio(_ROWS[0], _ROWS[1]) == 1.0


def test_split_ratio_rejects_split_that_worsens_bps_continuity():
    """株数比が2倍近くでも、BPSが落ちていない（大型増資）なら分割とみなさない。"""
    curr = {"bps": 140, "equity": 2800, "sh_out": 20}
    prev = {"bps": 100, "equity": 1000, "sh_out": 10.9}
    assert disclosure_split_ratio(curr, prev) == 1.0


def test_bps_growth_is_split_adjusted():
    fd = get_pit_fundamentals("378A", date(2026, 9, 25), rows=_HIT)
    assert abs(fd["bps_growth"] - (405.04 / (609.08 / 2) - 1)) < 1e-9  # +33%
    assert abs(fd["eps_growth"] - (70.42 / (162.79 / 2) - 1)) < 1e-9


def test_price_split_jumps():
    dates = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-22"]
    assert price_split_jumps(dates, [2100, 2138, 1025.5, 1030]) == [("2026-06-19", 2.0)]
    # 通常の値動き・-30%の急落は段差にしない
    assert price_split_jumps(dates, [1000, 700, 690, 1000]) == []


def test_valuation_uses_price_jump_after_disclosure():
    """分割前の開示×分割後の株価（7/1〜9/23の状態）: 段差の後は1株当たり値を1/2にする。"""
    known = _HIT[1:]  # 9/24開示前（ライブ運用と同じく未来の開示なし）
    jumps = [("2026-06-19", 2.0)]
    v = get_pit_valuation("378A", date(2026, 8, 1), rows=known, jumps=jumps)
    assert abs(v["bps"] - 609.08 / 2) < 1e-9 and abs(v["eps"] - 162.79 / 2) < 1e-9
    # 段差より前は分割前ベースのまま
    v = get_pit_valuation("378A", date(2026, 6, 1), rows=known, jumps=jumps)
    assert v["bps"] == 609.08
    # 分割後の開示が出たら補正しない
    v = get_pit_valuation("378A", date(2026, 9, 25), rows=_HIT, jumps=jumps)
    assert v["bps"] == 405.04


def test_valuation_uses_later_disclosure_when_prices_are_back_adjusted():
    """株価が遡って分割調整済み（段差なし）の学習データでは、後続開示の分割比で割り戻す。"""
    fd = get_pit_fundamentals("378A", date(2026, 3, 1), rows=_HIT, jumps=[])
    assert abs(fd["bps"] - 609.08 / 2) < 1e-9 and abs(fd["dps"] - 17.5 / 2) < 1e-9
    # 株価情報を渡さない呼び出し（従来経路）は補正しない
    fd = get_pit_fundamentals("378A", date(2026, 3, 1), rows=_HIT)
    assert fd["bps"] == 609.08


if __name__ == "__main__":
    test_filter_asof_excludes_future_disclosures()
    test_filter_asof_respects_limit_n()
    test_filter_asof_fy_only()
    test_pit_fundamentals_no_lookahead()
    test_pit_fundamentals_sees_latest_after_disclosure()
    test_pit_fundamentals_none_without_any_disclosure_or_yutai()
    test_split_ratio_detects_post_period_split_with_issuance()
    test_split_ratio_ignores_normal_growth()
    test_split_ratio_rejects_split_that_worsens_bps_continuity()
    test_bps_growth_is_split_adjusted()
    test_price_split_jumps()
    test_valuation_uses_price_jump_after_disclosure()
    test_valuation_uses_later_disclosure_when_prices_are_back_adjusted()
    print("OK: test_fundamentals (13 tests)")

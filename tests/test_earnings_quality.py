"""利益の質フィルター（lib/earnings_quality）のユニットテスト。

化粧決算・営業赤字・本業減益のゲート除外と、健全候補への成長加減点を検証する。
実行: python3 tests/test_earnings_quality.py  /  pytest tests/test_earnings_quality.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.earnings_quality import (
    assess_earnings_quality, revenue_cagr3, op_margin_declining_3,
)


def _row(fy, rev, op, net, fc=False):
    return {"fy_end": fy, "is_forecast": fc, "revenue": rev,
            "op_profit": op, "net_income": net}


def test_cosmetic_earnings_excluded():
    """純利益が営業益×1.5を超える＝一過性益の水増し → 化粧決算で除外。"""
    rows = [_row("2023", 1000, 100, 90), _row("2024", 1000, 80, 300)]
    r = assess_earnings_quality(rows)
    assert r["exclude"] == "化粧決算"


def test_operating_deficit_excluded():
    """営業赤字は最優先で除外。"""
    rows = [_row("2023", 1000, 50, 40), _row("2024", 1000, -30, -20)]
    assert assess_earnings_quality(rows)["exclude"] == "営業赤字"


def test_operating_decline_excluded():
    """黒字でも前年比減益＝本業縮小で除外。YoYも算出される。"""
    rows = [_row("2023", 1000, 120, 110), _row("2024", 950, 100, 95)]
    r = assess_earnings_quality(rows)
    assert r["exclude"] == "本業減益"
    assert r["op_yoy"] == -16.7


def test_healthy_growth_kept_with_bonus():
    """増収増益・売上CAGRプラス・増収増益予想 → 通過し bonus が正。"""
    rows = [_row("2022", 800, 60, 55), _row("2023", 900, 80, 70),
            _row("2024", 1000, 110, 100), _row("2025", 1200, 140, 120, fc=True)]
    r = assess_earnings_quality(rows)
    assert r["exclude"] is None
    assert r["bonus"] > 0
    assert r["fc_dir"] == "増収増益"


def test_no_data_kept():
    """実績データが無い銘柄は判定不能＝除外しない（keep）。"""
    r = assess_earnings_quality([])
    assert r["data"] is False
    assert r["exclude"] is None


def test_cosmetic_does_not_fire_on_normal_ratio():
    """純利益が営業益×1.5以内なら化粧扱いしない（増益のケース）。"""
    rows = [_row("2023", 1000, 100, 70), _row("2024", 1000, 120, 90)]
    assert assess_earnings_quality(rows)["exclude"] is None


def test_revenue_cagr3():
    """3期（2区間）の売上CAGRを正しく計算する。"""
    acts = [{"revenue": 1000}, {"revenue": 1100}, {"revenue": 1210}]
    assert abs(revenue_cagr3(acts) - 0.10) < 1e-6


def test_op_margin_declining_3():
    """営業利益率が3期連続低下していれば True。"""
    declining = [_row("2022", 1000, 200, 150), _row("2023", 1000, 150, 120),
                 _row("2024", 1000, 100, 80)]
    assert op_margin_declining_3(declining) is True
    flat_up = [_row("2022", 1000, 100, 80), _row("2023", 1000, 150, 120),
               _row("2024", 1000, 200, 150)]
    assert op_margin_declining_3(flat_up) is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\n{len(fns)} passed")

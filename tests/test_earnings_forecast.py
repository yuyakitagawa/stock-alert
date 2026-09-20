"""決算短信XBRLからの会社予想抽出（lib/earnings_forecast.py）のユニットテスト。

実行: python3 tests/test_earnings_forecast.py  /  pytest tests/test_earnings_forecast.py
"""
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.earnings_forecast import parse_forecast


def _zip(body: str, path: str = "XBRLData/Summary/tse-scedjpsy-ixbrl.htm") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(path, body)
    return buf.getvalue()


def _fact(name, ctx, value, scale="6", sign=""):
    s = f' sign="{sign}"' if sign else ""
    return (f'<ix:nonFraction contextRef="{ctx}" name="tse-ed-t:{name}" scale="{scale}"{s}>'
            f"{value}</ix:nonFraction>")


def _fy_end(ctx, value):
    return f'<ix:nonNumeric contextRef="{ctx}" name="tse-ed-t:FiscalYearEnd">{value}</ix:nonNumeric>'


def test_parses_current_year_forecast():
    body = (_fy_end("CurrentYearInstant", "2026-12-31")
            + _fact("NetSales", "CurrentYearDuration_ConsolidatedMember_ForecastMember", "8,555")
            + _fact("OperatingIncome", "CurrentYearDuration_ConsolidatedMember_ForecastMember", "753")
            + _fact("ProfitAttributableToOwnersOfParent",
                    "CurrentYearDuration_ConsolidatedMember_ForecastMember", "391"))
    assert parse_forecast(_zip(body)) == {"fy_end": "2026-12-31", "fnp": 391e6, "fop": 753e6, "fsales": 8555e6}


def test_annual_report_uses_next_year_forecast():
    """通期短信は来期予想が載る。NextYearInstant が無ければ今期末の1年後を使う。"""
    body = (_fy_end("CurrentYearInstant", "2026-06-30")
            + _fact("ProfitAttributableToOwnersOfParent",
                    "CurrentYearDuration_ConsolidatedMember_ForecastMember", "100")
            + _fact("ProfitAttributableToOwnersOfParent",
                    "NextYearDuration_ConsolidatedMember_ForecastMember", "194"))
    got = parse_forecast(_zip(body))
    assert got["fy_end"] == "2027-06-30" and got["fnp"] == 194e6


def test_prefers_consolidated_over_non_consolidated():
    body = (_fy_end("CurrentYearInstant", "2027-03-31")
            + _fact("NetIncome", "CurrentYearDuration_NonConsolidatedMember_ForecastMember", "50")
            + _fact("NetIncome", "CurrentYearDuration_ConsolidatedMember_ForecastMember", "80"))
    assert parse_forecast(_zip(body))["fnp"] == 80e6


def test_negative_forecast_and_ignores_range_members():
    """赤字予想の sign="-" を拾い、レンジ予想の上限/下限は使わない。"""
    body = (_fy_end("CurrentYearInstant", "2027-03-31")
            + _fact("ProfitAttributableToOwnersOfParent",
                    "CurrentYearDuration_ConsolidatedMember_LowerMember_ForecastMember", "10")
            + _fact("ProfitAttributableToOwnersOfParent",
                    "CurrentYearDuration_ConsolidatedMember_ForecastMember", "413", sign="-"))
    assert parse_forecast(_zip(body))["fnp"] == -413e6


def test_none_when_no_forecast_or_broken_zip():
    body = _fy_end("CurrentYearInstant", "2027-03-31") + _fact(
        "ProfitAttributableToOwnersOfParent", "CurrentYearDuration_ConsolidatedMember", "100")
    assert parse_forecast(_zip(body)) is None          # 実績しかない短信（予想未定）
    assert parse_forecast(b"not a zip") is None
    assert parse_forecast(_zip("x", path="XBRLData/Attachment/other.htm")) is None


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

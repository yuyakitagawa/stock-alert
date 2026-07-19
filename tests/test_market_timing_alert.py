"""大口保有動向セクション（web/market_timing_alert.build_large_holdings_section）のユニットテスト。

実行: python3 tests/test_market_timing_alert.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.market_timing_alert import build_large_holdings_section


def test_empty_holdings_returns_empty_string():
    assert build_large_holdings_section([]) == ""


def test_formats_entries_with_name_and_ratio():
    holdings = [{
        "issuer_code": "8058", "name": "三菱商事", "filer_name": "○○ファンド",
        "doc_type_code": "350", "holding_ratio": 5.2, "disc_date": "2026-07-17",
    }]
    msg = build_large_holdings_section(holdings)
    assert "三菱商事(8058)" in msg
    assert "○○ファンド" in msg
    assert "5.2%" in msg
    assert "大量保有" in msg


def test_change_report_labelled_correctly():
    holdings = [{
        "issuer_code": "7203", "name": "トヨタ自動車", "filer_name": "△△投信",
        "doc_type_code": "360", "holding_ratio": 8.0, "disc_date": "2026-07-17",
    }]
    msg = build_large_holdings_section(holdings)
    assert "変更" in msg


def test_missing_name_falls_back_to_code():
    holdings = [{
        "issuer_code": "9999", "name": "", "filer_name": "誰か",
        "doc_type_code": "350", "holding_ratio": None, "disc_date": "2026-07-17",
    }]
    msg = build_large_holdings_section(holdings)
    assert "9999" in msg
    assert "-" in msg  # holding_ratio欠損時のプレースホルダ


def test_truncates_over_limit_with_count():
    holdings = [
        {"issuer_code": str(1000 + i), "name": f"銘柄{i}", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 5.0, "disc_date": "2026-07-17"}
        for i in range(8)
    ]
    msg = build_large_holdings_section(holdings, limit=5)
    assert "銘柄0" in msg
    assert "銘柄4" in msg
    assert "銘柄5" not in msg
    assert "他3件" in msg


if __name__ == "__main__":
    test_empty_holdings_returns_empty_string()
    test_formats_entries_with_name_and_ratio()
    test_change_report_labelled_correctly()
    test_missing_name_falls_back_to_code()
    test_truncates_over_limit_with_count()
    print("OK: test_market_timing_alert (5 tests)")

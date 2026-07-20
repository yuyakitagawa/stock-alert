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
        "doc_description": "大量保有報告書",
    }]
    msg = build_large_holdings_section(holdings)
    assert "三菱商事(8058)" in msg
    assert "○○ファンド" in msg
    assert "5.2%" in msg
    assert "大量保有" in msg
    assert "📈買い" in msg


def test_sell_disclosure_labelled_as_sell():
    """譲渡/売却は除外せず、方向性(📉売り)を表示して見せる。"""
    holdings = [{
        "issuer_code": "4813", "name": "ＡＣＣＥＳＳ", "filer_name": "清原達郎",
        "doc_type_code": "350", "holding_ratio": 13.3, "disc_date": "2026-07-17",
        "doc_description": "変更報告書（短期大量譲渡）",
    }]
    msg = build_large_holdings_section(holdings)
    assert "📉売り" in msg
    assert "📈買い" not in msg


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


def test_watchlist_code_prioritized_over_higher_ratio():
    """ウォッチ銘柄は保有比率が小さくても最優先で表示される。"""
    holdings = [
        {"issuer_code": "9999", "name": "非ウォッチ大口", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 40.0, "disc_date": "2026-07-17"},
        {"issuer_code": "8058", "name": "三菱商事", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 5.1, "disc_date": "2026-07-17"},
    ]
    msg = build_large_holdings_section(holdings, watch_codes={"8058"}, limit=1)
    assert "三菱商事" in msg
    assert "⭐" in msg
    assert "非ウォッチ大口" not in msg


def test_non_watchlist_sorted_by_ratio_magnitude():
    """ウォッチ対象が無ければ保有比率(動きの大きさ)が大きい順。"""
    holdings = [
        {"issuer_code": "1111", "name": "小口", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 5.0, "disc_date": "2026-07-17"},
        {"issuer_code": "2222", "name": "大口", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 40.0, "disc_date": "2026-07-17"},
    ]
    msg = build_large_holdings_section(holdings, limit=1)
    assert "大口" in msg
    assert "小口" not in msg


def test_individual_filer_deprioritized_below_institution():
    """個人名の提出者は、保有比率が大きくても法人/ファンドより後ろに回る。"""
    holdings = [
        {"issuer_code": "1111", "name": "個人大量保有", "filer_name": "清原　達郎",
         "doc_type_code": "350", "holding_ratio": 40.0, "disc_date": "2026-07-17"},
        {"issuer_code": "2222", "name": "ファンド保有", "filer_name": "○○アセットマネジメント株式会社",
         "doc_type_code": "350", "holding_ratio": 5.0, "disc_date": "2026-07-17"},
    ]
    msg = build_large_holdings_section(holdings, limit=1)
    assert "ファンド保有" in msg
    assert "個人大量保有" not in msg


def test_ratio_change_shown_when_same_filer_has_multiple_disclosures():
    """同一提出者の開示が期間内に複数あれば「最古%→最新%」で変化を見せる。"""
    holdings = [
        {"issuer_code": "4813", "name": "ＡＣＣＥＳＳ", "filer_name": "清原　達郎",
         "doc_type_code": "350", "holding_ratio": 27.45, "disc_date": "2026-07-01",
         "doc_description": "変更報告書"},
        {"issuer_code": "4813", "name": "ＡＣＣＥＳＳ", "filer_name": "清原　達郎",
         "doc_type_code": "350", "holding_ratio": 13.30, "disc_date": "2026-07-17",
         "doc_description": "変更報告書（短期大量譲渡）"},
    ]
    msg = build_large_holdings_section(holdings, limit=1)
    assert "27.4%→13.3%" in msg


def test_ratio_unchanged_shows_single_value_not_range():
    """提出者は複数回開示されていても比率が同じなら範囲表示にしない。"""
    holdings = [
        {"issuer_code": "8058", "name": "三菱商事", "filer_name": "○○ファンド",
         "doc_type_code": "350", "holding_ratio": 5.2, "disc_date": "2026-07-10",
         "doc_description": "大量保有報告書"},
        {"issuer_code": "8058", "name": "三菱商事", "filer_name": "○○ファンド",
         "doc_type_code": "360", "holding_ratio": 5.2, "disc_date": "2026-07-17",
         "doc_description": "訂正報告書"},
    ]
    msg = build_large_holdings_section(holdings, limit=1)
    assert "5.2%" in msg
    assert "→" not in msg


if __name__ == "__main__":
    test_empty_holdings_returns_empty_string()
    test_formats_entries_with_name_and_ratio()
    test_sell_disclosure_labelled_as_sell()
    test_change_report_labelled_correctly()
    test_missing_name_falls_back_to_code()
    test_truncates_over_limit_with_count()
    test_watchlist_code_prioritized_over_higher_ratio()
    test_non_watchlist_sorted_by_ratio_magnitude()
    test_individual_filer_deprioritized_below_institution()
    test_ratio_change_shown_when_same_filer_has_multiple_disclosures()
    test_ratio_unchanged_shows_single_value_not_range()
    print("OK: test_market_timing_alert (11 tests)")

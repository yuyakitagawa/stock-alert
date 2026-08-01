"""大口保有動向セクション（web/market_timing_alert.build_large_holdings_section）のユニットテスト。

実行: python3 tests/test_market_timing_alert.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.market_timing_alert import build_large_holdings_section, build_watchlist_section, BLOG_SITE_URL


def test_empty_holdings_returns_empty_string():
    assert build_large_holdings_section([]) == ""


def test_includes_blog_link_when_holdings_present():
    """大口保有の話がある場合はmicroCMSブログへのリンクも案内する。"""
    holdings = [{
        "issuer_code": "8058", "name": "三菱商事", "filer_name": "○○ファンド",
        "doc_type_code": "350", "holding_ratio": 5.2, "disc_date": "2026-07-17",
        "doc_description": "大量保有報告書",
    }]
    msg = build_large_holdings_section(holdings)
    assert BLOG_SITE_URL in msg


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


def test_ratio_decrease_labelled_as_sell_even_without_keyword():
    """概要が「変更報告書」とだけ書かれ売買方向のキーワードが無くても、
    holding_ratio_prior(直前保有割合)との比較で保有比率が減っていれば📉売りと表示する
    （実例: 東芝がキオクシアHD株を一部売却し16.10%→15.10%になった開示が
    概要のキーワードのみに頼る旧ロジックでは📈買いに誤表示されていたバグ）。"""
    holdings = [{
        "issuer_code": "285A", "name": "キオクシアホールディングス", "filer_name": "株式会社東芝",
        "doc_type_code": "360", "holding_ratio": 15.10, "holding_ratio_prior": 16.10,
        "disc_date": "2026-07-23", "doc_description": "変更報告書",
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


def test_fresh_disclosure_outranks_stale_watchlist_hit():
    """開示日が新しいものが最優先。古い日付のウォッチ銘柄ヒットが新しい開示を
    何日も押しのけ続ける「毎日同じ古い日付が出る」バグの再発防止。"""
    holdings = [
        {"issuer_code": "8058", "name": "三菱商事", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 20.0, "disc_date": "2026-07-21"},
        {"issuer_code": "9999", "name": "非ウォッチ新着", "filer_name": "F",
         "doc_type_code": "350", "holding_ratio": 5.0, "disc_date": "2026-07-23"},
    ]
    msg = build_large_holdings_section(holdings, watch_codes={"8058"}, limit=1)
    assert "非ウォッチ新着" in msg
    assert "三菱商事" not in msg


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


def test_watchlist_shows_buy_mark_below_threshold():
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 5.0, "close": 3000.0, "recommend": "💎 買い"}}
    msg = build_watchlist_section(watchlist, ranking_map)
    assert "🔔買い時！" in msg


def test_watchlist_suppresses_buy_mark_when_ranking_says_sell():
    """dp閾値だけ見れば買い時でも、ランキング本体（品質フィルター込み）が
    売り検討と判定済みなら、同一銘柄で矛盾した案内をしない。"""
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 5.0, "close": 3000.0, "recommend": "🔴 売り検討"}}
    msg = build_watchlist_section(watchlist, ranking_map)
    assert "🔔買い時！" not in msg
    assert "⚠️売り検討" in msg


def test_watchlist_shows_sell_mark_above_sell_threshold():
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 25.0, "close": 3000.0, "recommend": "⏳ 方向感なし"}}
    msg = build_watchlist_section(watchlist, ranking_map)
    assert "⚠️売り検討" in msg


def test_watchlist_warns_on_system_sell_even_within_default_sell_threshold_gap():
    """dp_sell_thresholdの既定値20%は、ランキング本体の売り検討基準（drop_prob>=10%等）より緩い。
    dpが8〜20%の間（買い時でも既定の売り閾値到達でもない）でも、ランキングが既に
    「🔴 売り検討」と判定していれば、既定値のままの利用者にも必ず警告する。"""
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 12.0, "close": 3000.0, "recommend": "🔴 売り検討"}}
    msg = build_watchlist_section(watchlist, ranking_map)
    assert "⚠️売り検討" in msg
    assert "他" not in msg  # 要約行に押し込められていないこと(個別表示されていること)


def test_watchlist_summarizes_unchanged_stocks_instead_of_listing_each():
    """閾値未達で変化のない銘柄を毎日個別表示すると同文の繰り返しで読み飛ばされる
    （通知疲れ）ため、件数だけの要約行にまとめる。"""
    watchlist = [
        {"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0},
        {"code": "9984", "name": "ソフトバンクG", "dp_threshold": 8.0, "dp_sell_threshold": 20.0},
    ]
    ranking_map = {
        "7203": {"drop_prob": 12.0, "close": 3000.0, "recommend": "⏳ 方向感なし"},
        "9984": {"drop_prob": 13.0, "close": 8000.0, "recommend": "⏳ 方向感なし"},
    }
    msg = build_watchlist_section(watchlist, ranking_map)
    assert "他2銘柄は閾値未到達で変化なし" in msg
    assert "トヨタ" in msg  # 要約行の中に銘柄名は含める
    assert "下落確率12.0%" not in msg  # 個別行としては出さない


def test_watchlist_shows_previous_day_change_for_actionable_stocks():
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 5.0, "close": 3000.0, "recommend": "💎 買い"}}
    msg = build_watchlist_section(watchlist, ranking_map, prev_dp_map={"7203": 6.5})
    assert "前日比-1.5pt" in msg


def test_watchlist_omits_change_when_no_previous_data():
    watchlist = [{"code": "7203", "name": "トヨタ", "dp_threshold": 8.0, "dp_sell_threshold": 20.0}]
    ranking_map = {"7203": {"drop_prob": 5.0, "close": 3000.0, "recommend": "💎 買い"}}
    msg = build_watchlist_section(watchlist, ranking_map, prev_dp_map={})
    assert "前日比" not in msg


if __name__ == "__main__":
    test_empty_holdings_returns_empty_string()
    test_formats_entries_with_name_and_ratio()
    test_includes_blog_link_when_holdings_present()
    test_sell_disclosure_labelled_as_sell()
    test_ratio_decrease_labelled_as_sell_even_without_keyword()
    test_change_report_labelled_correctly()
    test_missing_name_falls_back_to_code()
    test_truncates_over_limit_with_count()
    test_watchlist_code_prioritized_over_higher_ratio()
    test_non_watchlist_sorted_by_ratio_magnitude()
    test_individual_filer_deprioritized_below_institution()
    test_ratio_change_shown_when_same_filer_has_multiple_disclosures()
    test_fresh_disclosure_outranks_stale_watchlist_hit()
    test_ratio_unchanged_shows_single_value_not_range()
    test_watchlist_shows_buy_mark_below_threshold()
    test_watchlist_suppresses_buy_mark_when_ranking_says_sell()
    test_watchlist_shows_sell_mark_above_sell_threshold()
    test_watchlist_warns_on_system_sell_even_within_default_sell_threshold_gap()
    test_watchlist_summarizes_unchanged_stocks_instead_of_listing_each()
    test_watchlist_shows_previous_day_change_for_actionable_stocks()
    test_watchlist_omits_change_when_no_previous_data()
    print("OK: test_market_timing_alert (21 tests)")

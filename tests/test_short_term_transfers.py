"""短期大量譲渡（譲渡の相手方・単価）パーサのユニットテスト。

実行: python3 tests/test_short_term_transfers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.edinet import parse_short_term_transfers, summarize_disposals
from web.publish_blog_articles import deal_amount_label, format_transfer_facts

_TAG = ("jplvh_cor:DetailsOfAcquisitionsAndDisposalsOfStocksEtcIssuedByIssuerOfSaidStocksEtc"
        "DuringLast60DaysIfCategorizedAsShortTermLargeVolumeTransferTextBlock")


def _xbrl(inner_rows: str) -> str:
    """XBRL本文中のテキストブロック（中身はエスケープされたHTML）を組み立てる。"""
    header = (
        "&lt;tr&gt;&lt;td&gt;年月日&lt;/td&gt;&lt;td&gt;株券等の種類&lt;/td&gt;&lt;td&gt;数量&lt;/td&gt;"
        "&lt;td&gt;割合&lt;/td&gt;&lt;td&gt;市場内外取引の別&lt;/td&gt;&lt;td&gt;取得又は処分の別&lt;/td&gt;"
        "&lt;td&gt;譲渡の相手方&lt;/td&gt;&lt;td&gt;単価&lt;/td&gt;&lt;/tr&gt;"
    )
    return (f'<{_TAG} contextRef="x">&lt;table&gt;{header}{inner_rows}&lt;/table&gt;</{_TAG}>')


# 日立製作所→日立建機（2026-08-25, S100YXY1）の実データ形式
HITACHI = _xbrl(
    "&lt;tr&gt;&lt;td class=\"COL_L\"&gt;2026年8月19日&lt;/td&gt;&lt;td&gt;普通株式&lt;/td&gt;"
    "&lt;td&gt;21,462,310&lt;/td&gt;&lt;td&gt;9.98&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;SMBC日興証券株式会社&lt;/td&gt;&lt;td&gt;5,227&lt;/td&gt;&lt;/tr&gt;"
)

# 空セルが自己終了タグ、日付が和暦、単価が「4,730円」表記、取得と処分が混在（C&I HD形式）
MIXED = _xbrl(
    "&lt;tr&gt;&lt;td&gt;&lt;span&gt;令和8年7月15日&lt;/span&gt;&lt;/td&gt;&lt;td&gt;株券&lt;/td&gt;"
    "&lt;td&gt;453,800&lt;/td&gt;&lt;td&gt;5.09&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;取得&lt;/td&gt;"
    "&lt;td class=\"COL_L\"/&gt;&lt;td&gt;4,730円&lt;/td&gt;&lt;/tr&gt;"
    "&lt;tr&gt;&lt;td&gt;令和8年7月16日&lt;/td&gt;&lt;td&gt;株券&lt;/td&gt;"
    "&lt;td&gt;10,000&lt;/td&gt;&lt;td&gt;0.11&lt;/td&gt;&lt;td&gt;市場内&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;市場内取引のため不明&lt;/td&gt;&lt;td class=\"COL_R\"/&gt;&lt;/tr&gt;"
)

# 単元未満株の端数処分が主取引と同じ表に並ぶ（Umios→林兼産業形式）
WITH_ODD_LOT = _xbrl(
    "&lt;tr&gt;&lt;td&gt;2026年8月17日&lt;/td&gt;&lt;td&gt;普通株式&lt;/td&gt;&lt;td&gt;46&lt;/td&gt;"
    "&lt;td&gt;0.00&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;単元未満株式の売却のため不明&lt;/td&gt;&lt;td&gt;887&lt;/td&gt;&lt;/tr&gt;"
    "&lt;tr&gt;&lt;td&gt;2026年8月18日&lt;/td&gt;&lt;td&gt;普通株式&lt;/td&gt;&lt;td&gt;225,200&lt;/td&gt;"
    "&lt;td&gt;2.53&lt;/td&gt;&lt;td&gt;市場内&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;大東通商株式会社&lt;/td&gt;&lt;td&gt;887&lt;/td&gt;&lt;/tr&gt;"
)


def test_parses_counterparty_and_unit_price():
    rows = parse_short_term_transfers(HITACHI)
    assert len(rows) == 1
    r = rows[0]
    assert r["date"] == "2026-08-19"
    assert r["shares"] == 21462310
    assert r["counterparty"] == "SMBC日興証券株式会社"
    assert r["unit_price"] == 5227.0
    assert r["action"] == "処分"
    assert r["venue"] == "市場外"


def test_exact_amount_replaces_market_price_estimate():
    """5,227円×21,462,310株＝1,121.8億円。開示日終値からの概算1,274.9億円ではなく実額を出す。"""
    s = summarize_disposals(parse_short_term_transfers(HITACHI), 9.98)
    assert s["amount_oku"] == 1121.8
    assert s["counterparties"] == ["SMBC日興証券株式会社"]
    assert s["unit_price"] == 5227.0


def test_exact_amount_rejected_when_table_does_not_explain_ratio_change():
    """表の処分だけで今回の比率変化を説明できないときは実額を採らない（概算へフォールバック）。"""
    assert summarize_disposals(parse_short_term_transfers(HITACHI), 25.0)["amount_oku"] is None


def test_era_dates_self_closing_cells_and_mixed_actions():
    rows = parse_short_term_transfers(MIXED)
    assert len(rows) == 2
    assert rows[0]["date"] == "2026-07-15"  # 令和8年 = 2026年
    assert rows[0]["action"] == "取得"
    assert rows[0]["counterparty"] is None  # 空セル（自己終了タグ）でも列がずれない
    assert rows[0]["unit_price"] == 4730.0  # 「4,730円」から数値を取る
    assert rows[1]["counterparty"] == "市場内取引のため不明"
    # 取得と処分が混在する60日間の売買記録からは実額を復元できない
    s = summarize_disposals(rows, 5.09)
    assert s["amount_oku"] is None
    assert s["counterparties"] == []  # 「不明」は相手方として扱わない


def test_representative_price_comes_from_largest_row():
    """端数処分（46株）ではなく主取引（225,200株）の単価・日付を代表値にする。"""
    s = summarize_disposals(parse_short_term_transfers(WITH_ODD_LOT), 2.53)
    assert s["date"] == "2026-08-18"
    assert s["venue"] == "市場内"
    assert s["counterparties"] == ["大東通商株式会社"]


# 新株予約権証券の譲渡（シェアレコ→日本文化数寄財団, 2026-06-25）。単価1円は株価ではない
WARRANT = _xbrl(
    "&lt;tr&gt;&lt;td&gt;2026年2月5日&lt;/td&gt;&lt;td&gt;新株予約権証券&lt;/td&gt;&lt;td&gt;1,915,500&lt;/td&gt;"
    "&lt;td&gt;22.73&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;一般財団法人日本文化数寄財団&lt;/td&gt;&lt;td&gt;1&lt;/td&gt;&lt;/tr&gt;"
)

# 連続譲渡の2枚目。表には前回開示済みの行（8/3 18.85%）も並ぶ（三井金属→ナカボーテック）
SERIAL = _xbrl(
    "&lt;tr&gt;&lt;td&gt;2026年8月3日&lt;/td&gt;&lt;td&gt;株券（普通株式）&lt;/td&gt;&lt;td&gt;490,500&lt;/td&gt;"
    "&lt;td&gt;18.85&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;ヒューリック株式会社&lt;/td&gt;&lt;td&gt;5,438&lt;/td&gt;&lt;/tr&gt;"
    "&lt;tr&gt;&lt;td&gt;2026年8月4日&lt;/td&gt;&lt;td&gt;株券（普通株式）&lt;/td&gt;&lt;td&gt;190,000&lt;/td&gt;"
    "&lt;td&gt;7.30&lt;/td&gt;&lt;td&gt;市場外&lt;/td&gt;&lt;td&gt;処分&lt;/td&gt;"
    "&lt;td&gt;株式会社ナカボーテック&lt;/td&gt;&lt;td&gt;5,930&lt;/td&gt;&lt;/tr&gt;"
)


def test_warrant_transfer_is_not_treated_as_share_price():
    """新株予約権の単価（1円）を株式の売却金額として出さない。相手方だけは見せる。"""
    s = summarize_disposals(parse_short_term_transfers(WARRANT), 23.0)
    assert s["amount_oku"] is None
    assert s["counterparties"] == ["一般財団法人日本文化数寄財団"]


def test_serial_disposal_picks_the_row_matching_this_filing():
    """2枚目の表には前回ぶんの行も並ぶ。今回の変化幅と一致する行だけを今回の譲渡として使う。"""
    rows = parse_short_term_transfers(SERIAL)
    s = summarize_disposals(rows, 7.30)
    assert s["amount_oku"] == 11.3  # 190,000株×5,930円
    assert s["counterparties"] == ["株式会社ナカボーテック"]
    # 1枚目（変化幅18.85pt）は合計が一致しないので、その行だけを使う
    first = summarize_disposals(rows[:1], 18.85)
    assert first["amount_oku"] == 26.7
    assert first["counterparties"] == ["ヒューリック株式会社"]


def test_no_table_returns_empty():
    assert parse_short_term_transfers("<xbrl>変更報告書</xbrl>") == []
    assert parse_short_term_transfers(None) == []
    assert summarize_disposals([], 1.0)["amount_oku"] is None


def test_amount_label_drops_estimate_prefix_only_for_exact():
    assert deal_amount_label(True, True) == "売却金額"
    assert deal_amount_label(True, False) == "推定売却金額"
    assert deal_amount_label(False, False) == "推定取得金額"


def test_transfer_facts_line_for_prompt():
    s = summarize_disposals(parse_short_term_transfers(HITACHI), 9.98)
    line = format_transfer_facts(s)
    assert "SMBC日興証券株式会社" in line
    assert "5,227円" in line
    assert "21,462,310株" in line
    assert format_transfer_facts({}) == ""


if __name__ == "__main__":
    test_parses_counterparty_and_unit_price()
    test_exact_amount_replaces_market_price_estimate()
    test_exact_amount_rejected_when_table_does_not_explain_ratio_change()
    test_era_dates_self_closing_cells_and_mixed_actions()
    test_representative_price_comes_from_largest_row()
    test_warrant_transfer_is_not_treated_as_share_price()
    test_serial_disposal_picks_the_row_matching_this_filing()
    test_no_table_returns_empty()
    test_amount_label_drops_estimate_prefix_only_for_exact()
    test_transfer_facts_line_for_prompt()
    print("OK: test_short_term_transfers (10 tests)")

"""web/x_disclosure_facts.py のユニットテスト。SupabaseもX APIも呼ばない。

実行: python3 tests/test_x_disclosure_facts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import x_disclosure_facts as m


def _row(**over):
    row = {
        "doc_id": "D1", "filer_name": "株式会社テスト", "issuer_code": "1234",
        "issuer_name": "テスト工業株式会社", "disc_date": "2026-08-28",
        "holding_ratio": 8.5, "holding_ratio_prior": None,
        "funding_total": 1_400_000_000, "funding_own": 0, "funding_borrowings": 1_400_000_000,
        "obligation_date": None, "doc_description": "大量保有報告書",
    }
    row.update(over)
    return row


def test_individuals_are_excluded_even_when_the_master_has_no_row():
    """分類マスターに載っていない個人名がそのまま投稿に載るのを防ぐ。
    私人を名指しで問題視する投稿になるため、判定は保守側に倒す。"""
    assert m.looks_like_individual("蕪木　登") is True
    assert m.looks_like_individual("大村　禎史") is True
    assert m.looks_like_individual("株式会社ナノバンク") is False
    assert m.looks_like_individual("公益財団法人双葉電子記念財団") is False
    assert m.looks_like_individual("ＵＬＴＩＭＡＴＥ　ＣＬＡＳＳＩＣ　ＩＮＶＥＳＴＭＥＮＴ　ＬＬＣ") is False
    assert m.looks_like_individual("ＭＴＭ　Ｃａｐｉｔａｌ株式会社") is False


def test_fully_borrowed_needs_an_explicit_zero_for_own_funds():
    """自己資金の欄が空の開示は「全額借入」と判定しない（未取得と0を区別できないため）。"""
    assert m.is_fully_borrowed(_row()) is True
    assert m.is_fully_borrowed(_row(funding_own=None)) is False
    assert m.is_fully_borrowed(_row(funding_own=5_000_000)) is False
    assert m.is_fully_borrowed(_row(funding_borrowings=0)) is False


def test_late_days_only_counts_beyond_the_threshold():
    """しきい値以下の提出と、報告義務発生日が無い開示はNone。"""
    assert m.late_days(_row(obligation_date="2026-05-01")) == 119
    assert m.late_days(_row(obligation_date="2026-08-20")) is None   # 8日は遅延としない
    assert m.late_days(_row(obligation_date=None)) is None


def test_pick_uses_only_new_filings_for_the_borrowed_slot():
    """変更報告書の取得資金は保有分全体の資金であって、その回の買い増し分ではない。
    「0.11ポイント買い増し」と「取得資金14億円」を並べると誤読させるので新規のみ使う。"""
    amendment = _row(doc_description="変更報告書", holding_ratio=3.0, holding_ratio_prior=2.89)
    kind, row = m.pick([amendment], set(), set())
    assert kind is None and row is None
    kind, row = m.pick([_row()], set(), set())
    assert kind == "borrowed"


def test_pick_skips_recently_posted_stocks_and_individuals():
    """同じ銘柄が短期間に続けて出るのを避ける。個人名義も対象外。"""
    assert m.pick([_row()], set(), {"1234"}) == (None, None)
    assert m.pick([_row(filer_name="蕪木　登")], set(), set()) == (None, None)
    assert m.pick([_row()], {"株式会社テスト"}, set()) == (None, None)


def test_build_text_never_claims_a_yearly_rarity_count():
    """本表(XBRL)の解析済みは今年の開示の約2%しかなく、「今年N件」は全体の件数にならない。
    実測で全額借入を「今年4件」と書きかけたが、直近30日だけで5件あった。"""
    text = m.build_text("borrowed", _row())
    assert "今年" not in text and "件" not in text
    assert "取得資金14億円は全額が借入" in text
    assert "テスト工業(1234)を8.5%取得" in text
    assert "http" not in text


def test_build_text_never_claims_a_direction_for_late_filings():
    """遅延の枠には変更報告書も混じる。比率が下がった届出を「取得」「買い増し」と
    書くと誤報になるため、現在の保有比率だけを言う。"""
    down = _row(doc_description="変更報告書", holding_ratio=3.0, holding_ratio_prior=8.0,
                obligation_date="2026-05-01")
    text = m.build_text("late", down)
    assert "買い増し" not in text and "取得" not in text
    assert "テスト工業(1234)の3.0%を持つ" in text


def test_build_text_returns_none_when_the_delay_cannot_be_measured():
    """報告義務発生日が無い開示で「None日たってから提出」と書かないこと。"""
    assert m.build_text("late", _row(obligation_date=None)) is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

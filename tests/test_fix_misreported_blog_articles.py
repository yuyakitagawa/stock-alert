"""公開済み記事の数字を是正するツール（tools/fix_misreported_blog_articles.py）の
対象判定のユニットテスト。株価の再概算とmicroCMSはモックする。

実行: python3 tests/test_fix_misreported_blog_articles.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.fix_misreported_blog_articles as m

NEW_REPORT = {
    "doc_id": "S100AAAA", "issuer_code": "4369", "disc_date": "2026-08-28",
    "filer_name": "野村證券株式会社", "doc_type_code": "350",
    "doc_description": "大量保有報告書", "holding_ratio": 10.41, "holding_ratio_prior": None,
}
CHANGE_REPORT = {
    "doc_id": "S100BBBB", "issuer_code": "9450", "disc_date": "2026-08-28",
    "filer_name": "光通信株式会社", "doc_type_code": "350",
    "doc_description": "変更報告書", "holding_ratio": 7.30, "holding_ratio_prior": 6.30,
}


def _article(**over) -> dict:
    art = {"id": "a1", "stockCode": "4369", "dealDate": "2026-08-28T00:00:00.000Z",
           "filerName": "野村證券株式会社", "title": "野村證券がトリケミカル研究所の0.46%を新規保有",
           "ratioChangePct": 0.46, "dealAmount": 1.2}
    art.update(over)
    return art


def setup_function() -> None:
    m.HISTORY.clear()


# ------------------------------------------------ 共同保有の合算で比率そのものが変わったケース

def test_new_report_is_corrected_when_the_stored_ratio_changed():
    """共同保有の合算対応で holding_ratio が「筆頭保有者の1枠」から合算に変わった。
    新規の大量保有報告書は前回0%として組み直し、記事の変化幅とのズレで拾う。"""
    m.HISTORY.clear()
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=27.0):
        fix = m.corrected_values(_article(), NEW_REPORT)
    assert fix is not None
    assert fix["prior_ratio"] == 0.0
    assert fix["signed_change"] == 10.41
    assert fix["is_sell"] is False
    assert fix["deal_amount"] == 27.0


def test_new_report_is_left_alone_when_the_ratio_still_matches():
    """比率が変わっていない記事まで書き換えない（大半はこちら）。"""
    m.HISTORY.clear()
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=27.0):
        assert m.corrected_values(_article(ratioChangePct=10.41), NEW_REPORT) is None


def test_change_report_without_prior_or_history_is_still_skipped():
    """変更報告書で前回比率が取れないものを0%扱いすると全量が動いたことになる。
    実態とかけ離れるので従来どおり是正しない。"""
    m.HISTORY.clear()
    row = {**CHANGE_REPORT, "holding_ratio_prior": None}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=1.0):
        assert m.corrected_values(_article(stockCode="9450", ratioChangePct=7.30), row) is None


def test_change_report_uses_history_when_prior_is_missing():
    """履歴から前回比率を補える変更報告書は従来どおり是正する。"""
    m.HISTORY[("9450", "光通信株式会社")] = [("2026-06-01", 6.30)]
    row = {**CHANGE_REPORT, "holding_ratio_prior": None}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        fix = m.corrected_values(
            _article(stockCode="9450", filerName="光通信株式会社",
                     title="光通信がファイバーゲートの7.30%を新規保有", ratioChangePct=7.30),
            row)
    assert fix is not None
    assert fix["prior_ratio"] == 6.30
    assert fix["signed_change"] == 1.00


# ------------------------------------------------ 既存の判定を壊していないこと

def test_change_report_field_mismatch_is_corrected():
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        fix = m.corrected_values(
            _article(stockCode="9450", filerName="光通信株式会社", ratioChangePct=7.30), CHANGE_REPORT)
    assert fix is not None
    assert fix["signed_change"] == 1.00


def test_change_report_matching_article_is_skipped():
    """変化幅もタイトルも合っている記事は触らない（タイトルに「新規保有」が残っていると
    前回比率>0との矛盾で別条件に引っかかるため、正しいタイトルで確認する）。"""
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        assert m.corrected_values(
            _article(stockCode="9450", filerName="光通信株式会社", ratioChangePct=1.00,
                     title="光通信がファイバーゲートの保有比率を7.30%に引き上げ"),
            CHANGE_REPORT) is None


def test_sell_disclosure_keeps_the_negative_sign():
    row = {**CHANGE_REPORT, "holding_ratio": 5.30, "holding_ratio_prior": 6.30}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        fix = m.corrected_values(
            _article(stockCode="9450", filerName="光通信株式会社", ratioChangePct=7.30), row)
    assert fix is not None
    assert fix["is_sell"] is True
    assert fix["signed_change"] == -1.00


def test_correction_report_has_zero_deal_amount():
    row = {**CHANGE_REPORT, "doc_description": "訂正報告書", "doc_type_code": "360"}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        fix = m.corrected_values(
            _article(stockCode="9450", filerName="光通信株式会社", ratioChangePct=7.30), row)
    assert fix is not None
    assert fix["is_correction"] is True
    assert fix["deal_amount"] == 0.0


def test_missing_holding_ratio_is_skipped():
    row = {**NEW_REPORT, "holding_ratio": None}
    assert m.corrected_values(_article(), row) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        m.HISTORY.clear()
        t()
        print(f"  ok {t.__name__}")
    print(f"✅ {len(tests)} tests passed")

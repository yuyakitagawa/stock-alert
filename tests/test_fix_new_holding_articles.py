#!/usr/bin/env python3
"""公開済み「新規保有」誤記事の訂正ツール（tools/fix_new_holding_articles.py）のユニットテスト。
ネットワーク（microCMS・Supabase・yfinance）は全てモックする。"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.fix_new_holding_articles as m


ARTICLE = {
    "id": "abc123",
    "title": "テスト製紙（3878）、テストファンドが7.6%を新規保有｜大量保有報告書",
    "stockCode": "3878",
    "stockName": "テスト製紙",
    "dealDate": "2026-08-18T00:00:00.000Z",
    "ratioChangePct": 7.6,
    "dealAmount": 120.0,
}


def test_find_disclosure_returns_row_for_single_filer():
    rows = [{"filer_name": "テストファンド", "doc_type_code": "350",
             "doc_description": "変更報告書", "holding_ratio": 7.6, "holding_ratio_prior": 7.1}]
    with mock.patch.object(m.sb, "select", return_value=rows):
        assert m.find_disclosure("3878", "2026-08-18")["filer_name"] == "テストファンド"


def test_find_disclosure_none_when_multiple_filers():
    """同じ日に複数の提出者がぶら下がる銘柄は、どの行が記事の元か特定できないので触らない。"""
    rows = [
        {"filer_name": "A", "doc_type_code": "350", "doc_description": "変更報告書",
         "holding_ratio": 7.6, "holding_ratio_prior": 7.1},
        {"filer_name": "B", "doc_type_code": "350", "doc_description": "変更報告書",
         "holding_ratio": 5.2, "holding_ratio_prior": 5.0},
    ]
    with mock.patch.object(m.sb, "select", return_value=rows):
        assert m.find_disclosure("3878", "2026-08-18") is None


def test_corrected_payload_rebuilds_title_amount_and_change():
    """前回比率が取れる開示は、変化幅・推定金額・タイトルを実態に合わせて作り直す。"""
    row = {"filer_name": "テストファンド", "doc_type_code": "350",
           "doc_description": "変更報告書", "holding_ratio": 7.6, "holding_ratio_prior": 7.1}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=8.0):
        payload = m.corrected_payload(ARTICLE, row)
    assert payload["ratioChangePct"] == 0.5     # 7.6 - 7.1（過大な7.6ではない）
    assert payload["dealAmount"] == 8.0
    assert "新規保有" not in payload["title"]
    assert "保有比率7.6%に引き上げ" in payload["title"]


def test_corrected_payload_signs_change_negative_for_sell():
    row = {"filer_name": "テストファンド", "doc_type_code": "350",
           "doc_description": "変更報告書", "holding_ratio": 5.1, "holding_ratio_prior": 8.0}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=3.0):
        payload = m.corrected_payload(ARTICLE, row)
    assert payload["ratioChangePct"] == -2.9
    assert "引き下げ" in payload["title"]


def test_corrected_payload_none_without_prior_ratio():
    """前回比率が無い開示は、誤った数字を別の誤った数字に置き換えないようNoneを返す。"""
    row = {"filer_name": "ＦＭＲ　ＬＬＣ", "doc_type_code": "350",
           "doc_description": "変更報告書（特例対象株券等）",
           "holding_ratio": 7.6, "holding_ratio_prior": None}
    assert m.corrected_payload(ARTICLE, row) is None


def test_corrected_payload_none_when_amount_cannot_be_estimated():
    row = {"filer_name": "テストファンド", "doc_type_code": "350",
           "doc_description": "変更報告書", "holding_ratio": 7.6, "holding_ratio_prior": 7.1}
    with mock.patch.object(m, "estimate_deal_amount_oku", return_value=None):
        assert m.corrected_payload(ARTICLE, row) is None


if __name__ == "__main__":
    test_find_disclosure_returns_row_for_single_filer()
    test_find_disclosure_none_when_multiple_filers()
    test_corrected_payload_rebuilds_title_amount_and_change()
    test_corrected_payload_signs_change_negative_for_sell()
    test_corrected_payload_none_without_prior_ratio()
    test_corrected_payload_none_when_amount_cannot_be_estimated()
    print("全テスト成功 (6件)")

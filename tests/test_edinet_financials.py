"""lib/edinet_financials.py のXBRL抽出（連結優先・タグ完全一致）と書類種別の選別。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.edinet_financials import (  # noqa: E402
    _XBRL_TAGS, _detect_doc_type, _extract_float, extract_financial_docs,
)


def _fact(tag, ctx, val):
    return f'<jpcrp_cor:{tag} contextRef="{ctx}" unitRef="JPY" decimals="-6">{val}</jpcrp_cor:{tag}>'


class TestExtractFloat(unittest.TestCase):
    def test_ifrs_prefers_consolidated_over_non_consolidated(self):
        # IFRS会社の有報: 日本基準の単体売上が先に出てくる（4183三井化学の実例）
        xbrl = "".join([
            _fact("NetSalesSummaryOfBusinessResults", "CurrentYearDuration_NonConsolidatedMember", "749791000000"),
            _fact("RevenueIFRSSummaryOfBusinessResults", "CurrentYearDuration", "1668754000000"),
        ])
        self.assertEqual(_extract_float(xbrl, _XBRL_TAGS["sales"]), 1668754000000.0)

    def test_ifrs_profit_uses_owners_of_parent(self):
        xbrl = "".join([
            _fact("ProfitLossIFRSSummaryOfBusinessResults", "CurrentYearDuration", "46910000000"),
            _fact("ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults", "CurrentYearDuration", "34378000000"),
        ])
        self.assertEqual(_extract_float(xbrl, _XBRL_TAGS["np"]), 34378000000.0)

    def test_non_consolidated_only_company_falls_back(self):
        xbrl = _fact("NetSalesSummaryOfBusinessResults", "CurrentYearDuration_NonConsolidatedMember", "5000000000")
        self.assertEqual(_extract_float(xbrl, _XBRL_TAGS["sales"]), 5000000000.0)

    def test_tag_match_is_exact(self):
        # 部分一致だと OperatingIncome が NonOperatingIncome に当たっていた
        xbrl = "".join([
            _fact("NonOperatingIncome", "CurrentYearDuration", "894000000"),
            _fact("OperatingIncome", "CurrentYearDuration", "27937000000"),
        ])
        self.assertEqual(_extract_float(xbrl, _XBRL_TAGS["op"]), 27937000000.0)

    def test_prior_year_is_skipped(self):
        xbrl = "".join([
            _fact("RevenueIFRSSummaryOfBusinessResults", "Prior1YearDuration", "1"),
            _fact("RevenueIFRSSummaryOfBusinessResults", "CurrentYearDuration", "2"),
        ])
        self.assertEqual(_extract_float(xbrl, _XBRL_TAGS["sales"]), 2.0)


class TestDocTypes(unittest.TestCase):
    def test_only_annual_and_half_year_reports(self):
        results = [{"docID": f"S{c}", "docTypeCode": c, "secCode": "41830", "filerName": "x"}
                   for c in ("120", "130", "140", "160", "170", "350")]
        got = sorted(d["doc_type_code"] for d in extract_financial_docs(results, "2026-06-26"))
        self.assertEqual(got, ["120", "160"])

    def test_detect_doc_type(self):
        self.assertEqual(_detect_doc_type("120", ""), "FY")
        self.assertEqual(_detect_doc_type("160", ""), "2Q")


if __name__ == "__main__":
    unittest.main(verbosity=2)

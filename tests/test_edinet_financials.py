"""lib/edinet_financials.py のXBRL抽出（連結優先・タグ完全一致）と書類種別の選別。"""
import os
import sys
import unittest
from datetime import date
from unittest import mock

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


class TestScanSkipsSaved(unittest.TestCase):
    """保存済みの (code, disc_date) はXBRLを取りに行かず、日ごとに保存する。"""

    def _run(self, existing, force=False):
        import lib.edinet_financials as ef
        docs = [{"docID": "S1", "docTypeCode": "120", "secCode": "58010", "filerName": "a"},
                {"docID": "S2", "docTypeCode": "120", "secCode": "58020", "filerName": "b"}]
        parse = mock.Mock(side_effect=lambda doc_id, dtc, ds: {"code": {"S1": "5801", "S2": "5802"}[doc_id],
                                                               "disc_date": ds, "doc_type": "FY", "np": 1.0})
        upsert = mock.Mock()
        keys = mock.Mock(return_value=existing)

        class _D(date):
            @classmethod
            def today(cls):
                return date(2026, 6, 30)  # 火曜

        with mock.patch.object(ef, "fetch_documents_list", return_value=docs), \
             mock.patch.object(ef, "parse_financial_xbrl", parse), \
             mock.patch.object(ef, "date", _D), \
             mock.patch("lib.db.get_jquants_fin_keys", keys), \
             mock.patch("lib.db.bulk_upsert_jquants_fin_summary", upsert):
            out = ef.scan_financial_reports(start_date="2026-06-29", sleep_sec=0, force=force)
        return out, parse, upsert, keys

    def test_skips_saved_docs(self):
        out, parse, upsert, keys = self._run({("5801", "2026-06-29"), ("5801", "2026-06-30")})
        keys.assert_called_once_with("2026-06-29", "2026-06-30")
        self.assertEqual([c.args[0] for c in parse.call_args_list], ["S2", "S2"])
        self.assertEqual(len(out), 2)

    def test_saves_per_day(self):
        _, _, upsert, _ = self._run(set())
        self.assertEqual(upsert.call_count, 2)
        self.assertEqual({r["disc_date"] for r in upsert.call_args_list[0].args[0]}, {"2026-06-29"})

    def test_force_refetches_saved_docs(self):
        _, parse, _, keys = self._run({("5801", "2026-06-29"), ("5802", "2026-06-29")}, force=True)
        keys.assert_not_called()
        self.assertEqual(parse.call_count, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

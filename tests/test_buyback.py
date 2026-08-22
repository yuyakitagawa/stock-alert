"""自社株買い開示の分類・取得枠抽出（lib/buyback.py）のユニットテスト。ネットワーク・LLMは使わない。

実行: python3 tests/test_buyback.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.buyback as m

TOSTNET_TEXT = """
当社は、2026年８月21日開催の取締役会において、自己株式取得に係る事項及びその具体的取得方法を決議いたしました。
２．取得の方法
本日（2026年８月21日）の終値 5,810円で、2026年８月24日午前８時45分の東京証券取引所の自己株式立会外買付取引（ToSTNeT-3）において買付けの委託を行う。
３．取得の内容
（２）取得する株式の総数  180,000株（上限）
               （自己株式を除く発行済株式の総数に対する割合1.53％）
（３）株式の取得価額の総額 1,045,800,000円（上限）
５．自己株式の消却の内容
（２）消却する株式の数    上記３．により取得した自己株式の全数
（ご参考）発行済株式の総数（自己株式を除く）   11,765,560株
"""

MARKET_TEXT = """
当社は、2026 年８月 18 日付の会社法第 370 条に基づく取締役会の決議に代わる書面決議により決定しました。
（２）取得し得る株式の総数（上限） 6,420,000 株
（３）株式の取得価額の総額（上限） 340 億円
（４）取得期間 2026 年８月 19 日から 2026 年 11 月 30 日まで
（５）取得の方法 取引一任 方式による東京証券取引所における市場買付
"""

AMEND_TEXT = """
（2026 年２月９日取締役会決議） （2026 年８月 18 日取締役会決議）
株式の総数 310,000 株（上限） 460,000 株（上限）
（発行済株式総数 （自己株式を除く） に対する割合 5.86％） （発行済株式総数 （自己株式を除く） に対する割合 9.12％）
価額の総額 700,000,000 円（上限） 1,100,000,000 円（上限）
（４）取得期間 2026 年 2 月 10 日～2027 年 2 月９日
（５）取得方法 福岡証券取引所における市場買付
"""


def test_classify_title():
    assert m.classify_buyback_title("自己株式取得に係る事項の決定に関するお知らせ") == "decision"
    assert m.classify_buyback_title("自己株式の取得及び自己株式立会外買付取引（ToSTNeT-3）による自己株式の買付けに関するお知らせ") == "decision"
    assert m.classify_buyback_title("自己株式の取得状況に関するお知らせ") == "progress"
    # 進捗語が混ざれば決定語（決定/消却等）があっても進捗
    assert m.classify_buyback_title("自己株式の取得結果及び取得終了並びに自己株式の消却に関するお知らせ") == "progress"
    assert m.classify_buyback_title("コミットメント型自己株式取得（FCSR）における事後調整のお知らせ") == "progress"
    assert m.classify_buyback_title("自己株式の取得に関するご参考") == "other"


def test_direct_pdf_url_strips_redirector():
    assert m.direct_pdf_url("https://webapi.yanoshin.jp/rd.php?https://www.release.tdnet.info/inbs/1.pdf") == \
        "https://www.release.tdnet.info/inbs/1.pdf"
    assert m.direct_pdf_url("https://www.release.tdnet.info/inbs/1.pdf") == "https://www.release.tdnet.info/inbs/1.pdf"
    assert m.direct_pdf_url("") == ""


def test_regex_tostnet_single_day_with_cancel():
    r = m.extract_facts_regex(TOSTNET_TEXT, "自己株式の取得及び…並びに自己株式の消却に関するお知らせ", "2026-08-21T18:15:00+00:00")
    assert r["max_shares"] == 180000
    assert r["max_amount_yen"] == 1_045_800_000
    assert r["ratio_pct"] == 1.53
    assert r["period_from"] == r["period_to"] == "2026-08-24"
    assert r["board_date"] == "2026-08-21"
    assert r["method"].startswith("ToSTNeT-3")
    assert r["will_cancel"] is True


def test_regex_market_purchase_with_oku_unit_and_period():
    r = m.extract_facts_regex(MARKET_TEXT, "自己株式取得に係る事項の決定に関するお知らせ", "2026-08-19T00:00:00+00:00")
    assert r["max_shares"] == 6_420_000
    assert r["max_amount_yen"] == 340 * 100_000_000
    assert r["ratio_pct"] is None  # 本文に割合が無い開示は埋めない
    assert (r["period_from"], r["period_to"]) == ("2026-08-19", "2026-11-30")
    assert r["method"] == "取引一任方式による市場買付"
    assert r["will_cancel"] is False
    # 書面決議で「日付+取締役会」の並びが無い場合は開示日を決議日とみなす
    assert r["board_date"] == "2026-08-19"


def test_regex_amendment_takes_after_column():
    r = m.extract_facts_regex(AMEND_TEXT, "自己株式取得に係る事項の一部変更に関するお知らせ", "2026-08-18")
    assert r["max_shares"] == 460000
    assert r["max_amount_yen"] == 1_100_000_000
    assert r["ratio_pct"] == 9.12
    assert r["board_date"] == "2026-02-09"
    assert r["method"] == "市場買付"


def test_extract_buyback_facts_skips_llm_when_regex_succeeds():
    with mock.patch.object(m, "extract_facts_llm", side_effect=AssertionError("LLM should not be called")):
        r = m.extract_buyback_facts(TOSTNET_TEXT, "2108", "自己株式の取得…", "2026-08-21")
    assert r["max_amount_yen"] == 1_045_800_000


def test_extract_buyback_facts_falls_back_to_llm_and_keeps_regex_fields():
    llm = {k: None for k in m._FACT_KEYS}
    llm.update({"max_amount_yen": 500_000_000, "purpose": "資本効率の向上", "method": "LLM方式"})
    with mock.patch.object(m, "extract_facts_llm", return_value=llm):
        r = m.extract_buyback_facts("取得期間 2026年9月1日から2026年12月31日まで 市場買付", "0000", "自己株式取得に係る事項の決定", "2026-08-20")
    assert r["max_amount_yen"] == 500_000_000
    assert r["purpose"] == "資本効率の向上"
    assert r["method"] == "市場買付"           # 正規表現で取れた項目はLLMで上書きしない
    assert r["period_to"] == "2026-12-31"


def test_coerce_facts_rejects_invalid_values():
    r = m._coerce_facts({"max_shares": "-5", "max_amount_yen": "1,000", "ratio_pct": "150",
                         "board_date": "2026/8/1", "period_from": "不明", "will_cancel": None, "method": "  "})
    assert r["max_shares"] is None
    assert r["max_amount_yen"] == 1000
    assert r["ratio_pct"] is None
    assert r["board_date"] == "2026-08-01"
    assert r["period_from"] is None
    assert r["will_cancel"] is None
    assert r["method"] is None


def test_pending_decisions_filters_progress_and_done():
    rows = [
        {"code": "1111", "disclosed_at": "2026-08-20T10:00:00+00:00", "title": "自己株式取得に係る事項の決定に関するお知らせ", "doc_url": "u1"},
        {"code": "2222", "disclosed_at": "2026-08-20T10:00:00+00:00", "title": "自己株式の取得状況に関するお知らせ", "doc_url": "u2"},
        {"code": "3333", "disclosed_at": "2026-08-20T10:00:00+00:00", "title": "自己株式取得に係る事項の決定に関するお知らせ", "doc_url": "u3"},
        {"code": "4444", "disclosed_at": "2026-08-20T10:00:00+00:00", "title": "自己株式取得に係る事項の決定に関するお知らせ", "doc_url": None},
    ]
    done = [{"code": "3333", "disclosed_at": "2026-08-20T10:00:00+00:00", "title": "自己株式取得に係る事項の決定に関するお知らせ"}]
    with mock.patch.object(m.sb, "select", side_effect=[rows, done]):
        out = m.pending_decisions(3)
    assert [r["code"] for r in out] == ["1111"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

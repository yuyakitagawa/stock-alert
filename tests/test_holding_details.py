"""保有目的・取得資金パーサ（XBRL本表）のユニットテスト。

実行: python3 tests/test_holding_details.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.edinet import (average_acquisition_price, classify_purpose,
                        parse_holding_details)


def _member(edinet_code: str, n: int) -> str:
    return f"FilingDateInstant_jplvh010000-lvh_{edinet_code}-000FilerLargeVolumeHolder{n}Member"


def _tag(name: str, ctx: str, value: str) -> str:
    return f'<jplvh_cor:{name} contextRef="{ctx}">{value}</jplvh_cor:{name}>'


# 成成→東京コスモス電機（S100YZ5C形式）。単独提出・自己資金0の全額借入。
SINGLE = "".join([
    _tag("TotalNumberOfStocksEtcHeld", _member("E37851", 1), "1332000"),
    _tag("TotalNumberOfStocksEtcHeld", "FilingDateInstant", "1332000"),
    _tag("TotalNumberOfOutstandingStocksEtc", "FilingDateInstant", "6825860"),
    _tag("AmountOfOwnFund", _member("E37851", 1), "0"),
    _tag("TotalAmountOfBorrowings", _member("E37851", 1), "905892000"),
    _tag("TotalAmountOfFundingForAcquisition", _member("E37851", 1), "905892000"),
    _tag("DateWhenFilingRequirementAroseCoverPage", "FilingDateInstant", "2026-08-13"),
    _tag("PurposeOfHolding", _member("E37851", 1), "純投資及び状況に応じて重要提案行為などを行うこと。"),
    _tag("ActOfMakingImportantProposalEtc", _member("E37851", 1), "該当事項なし"),
])

# 蕪木→インターファクトリー（S100YZ24形式）。共同保有3名で目的の文言が全員違い、
# 取得資金は合算contextが無いので提出者分を足し上げる必要がある。
# 3人目には AmountOfOwnFund が無い＝タグが飛ぶので、貪欲マッチだと次の提出者の
# 値まで飲み込んでしまう（実データで踏んだ）。
JOINT = "".join([
    _tag("TotalNumberOfStocksEtcHeld", _member("E35963", 1), "1380000"),
    _tag("TotalNumberOfStocksEtcHeld", _member("E35963", 2), "200000"),
    _tag("TotalNumberOfStocksEtcHeld", _member("E35963", 3), "220000"),
    _tag("TotalNumberOfStocksEtcHeld", "FilingDateInstant", "1800000"),
    _tag("TotalNumberOfOutstandingStocksEtc", "FilingDateInstant", "4125700"),
    _tag("AmountOfOwnFund", _member("E35963", 1), "8000000"),
    _tag("TotalAmountOfFundingForAcquisition", _member("E35963", 1), "8000000"),
    _tag("AmountOfOwnFund", _member("E35963", 2), "4484000"),
    _tag("TotalAmountOfFundingForAcquisition", _member("E35963", 2), "4484000"),
    _tag("TotalAmountOfBorrowings", _member("E35963", 3), "99000000"),
    _tag("TotalAmountOfFundingForAcquisition", _member("E35963", 3), "99000000"),
    _tag("PurposeOfHolding", _member("E35963", 1), "発行会社の創業者かつ代表取締役であり、安定株主として保有しております。"),
    _tag("PurposeOfHolding", _member("E35963", 2), "発行会社の創業者の配偶者であり、安定株主として保有しております。"),
    _tag("PurposeOfHolding", _member("E35963", 3), "発行会社の代表取締役の資産管理会社であり、安定株主として保有しております。"),
])

# 全部売却した変更報告書（S100YYUR形式）。株数0・資金の記載なし。
SOLD_OUT = "".join([
    _tag("TotalNumberOfStocksEtcHeld", "FilingDateInstant", "0"),
    _tag("TotalNumberOfOutstandingStocksEtc", "FilingDateInstant", "1854000"),
    _tag("PurposeOfHolding", _member("E00001", 1), "純投資"),
])


def test_single_filer():
    d = parse_holding_details(SINGLE)
    assert d["shares_held"] == 1332000, d
    assert d["shares_outstanding"] == 6825860, d
    assert d["funding_total"] == 905892000, d
    assert d["funding_own"] == 0, d
    assert d["funding_borrowings"] == 905892000, d
    assert d["obligation_date"] == "2026-08-13", d
    assert d["important_proposal"] == "該当事項なし", d
    assert average_acquisition_price(d["funding_total"], d["shares_held"]) == 680.1


def test_joint_filers_sum_funding():
    d = parse_holding_details(JOINT)
    # 株数は合算contextを優先する（足し上げても同じ値になること）
    assert d["shares_held"] == 1800000, d
    # 取得資金は合算contextが無いので提出者3名分の合計
    assert d["funding_total"] == 8000000 + 4484000 + 99000000, d
    # 自己資金は1・2人目だけ、借入は3人目だけ。取りこぼしも飲み込みもしない
    assert d["funding_own"] == 12484000, d
    assert d["funding_borrowings"] == 99000000, d
    # 目的は提出者ごとに違うので全部残す
    assert d["purpose_of_holding"].count("\n") == 2, d
    assert d["purpose_of_holding"].startswith("発行会社の創業者かつ代表取締役"), d


def test_sold_out_has_no_unit_price():
    d = parse_holding_details(SOLD_OUT)
    assert d["shares_held"] == 0, d
    assert d["funding_total"] is None, d
    assert average_acquisition_price(d["funding_total"], d["shares_held"]) is None


def test_empty_xbrl():
    d = parse_holding_details(None)
    assert set(d) == {"purpose_of_holding", "important_proposal", "shares_held",
                      "shares_outstanding", "funding_total", "funding_own",
                      "funding_borrowings", "obligation_date"}
    assert all(v is None for v in d.values()), d


def test_classify_purpose():
    # 「純投資及び状況に応じて重要提案行為など」は純投資ではなく重要提案行為等に寄せる
    assert classify_purpose("純投資及び状況に応じて重要提案行為などを行うこと。") == "重要提案行為等"
    assert classify_purpose("企業価値の向上を目的として重要提案行為等を行うこと") == "重要提案行為等"
    assert classify_purpose("経営支援を行うことにより、発行者の企業価値を向上させるため。") == "経営参加"
    assert classify_purpose("政策投資 （長期安定株主としての政策保有）") == "政策保有"
    assert classify_purpose("安定株主として長期保有しております。") == "安定株主"
    assert classify_purpose("純投資") == "純投資"
    assert classify_purpose("証券業務に係る商品在庫として保有している。") == "純投資"
    assert classify_purpose(None) is None
    assert classify_purpose("なんとも判別できない記載") is None


def test_average_acquisition_price_guards():
    assert average_acquisition_price(None, 100) is None
    assert average_acquisition_price(1000, None) is None
    assert average_acquisition_price(1000, 0) is None
    assert average_acquisition_price(1000, 3) == 333.3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok {t.__name__}")
    print(f"✅ {len(tests)} tests passed")

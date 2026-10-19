"""有報からの事業情報抽出（lib/company_profile.py）のユニットテスト。

実行: python3 tests/test_company_profile.py  /  pytest tests/test_company_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.company_profile import headline, parse_profile, row_from_document

XBRL = """<xbrl>
<jpcrp_cor:DescriptionOfBusinessTextBlock contextRef="X">
 &lt;p&gt;３ 【事業の内容】 当社グループは、当社及び連結子会社11社により構成されております。
 なお、2025年７月１日付で子会社を統合しました。
 資源循環事業においては、工場から排出される金属スクラップの回収・販売を行っております。&lt;/p&gt;
</jpcrp_cor:DescriptionOfBusinessTextBlock>
<jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock contextRef="X">
 海外展開の加速と人材確保が課題であります。
</jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock>
<jpcrp_cor:NumberOfEmployees contextRef="X" unitRef="pure">1,226</jpcrp_cor:NumberOfEmployees>
<jpcrp_cor:AverageAgeYearsInformationAboutReportingCompanyInformationAboutEmployees contextRef="X">46.7</jpcrp_cor:AverageAgeYearsInformationAboutReportingCompanyInformationAboutEmployees>
<jpcrp_cor:AverageAnnualSalaryInformationAboutReportingCompanyInformationAboutEmployees contextRef="X">11,265,100</jpcrp_cor:AverageAnnualSalaryInformationAboutReportingCompanyInformationAboutEmployees>
</xbrl>"""


def test_parse_profile_pulls_text_and_numbers():
    p = parse_profile(XBRL)
    assert p["business"].startswith("当社グループは")       # 見出し「３ 【事業の内容】」は落とす
    assert "<p>" not in p["business"] and "&lt;" not in p["business"]
    assert p["issues"].startswith("海外展開")
    assert p["employees"] == 1226 and p["avg_age"] == 46.7 and p["avg_salary"] == 11265100


def test_headline_skips_composition_and_notes():
    """構成説明（〜により構成されております）と補足（なお〜）を飛ばし、事業の文を採る。"""
    assert headline(parse_profile(XBRL)["business"]).startswith("資源循環事業においては")


def test_headline_takes_business_part_of_composition_sentence():
    """構成説明と事業説明が1文にまとまっている有報（2026-10-15 リンテック・アイダの通知で構成説明が出た）。"""
    lintec = ("当社グループは、当社、子会社43社、関連会社３社およびその他の関係会社の子会社１社で構成され、"
              "「印刷材・産業工材関連」、「電子・光学関連」の各事業に関する製品の製造・加工・販売を主な内容とし、"
              "さらに事業展開をしております。 当社グループの事業における位置付けは次のとおりであります。")
    assert headline(lintec).startswith("「印刷材・産業工材関連」")
    aida = ("当社グループ(当社及び国内・海外子会社)は、プレス機械を主力とする鍛圧機械等の製造・販売並びにサービスを"
            "主な事業として、当社、連結子会社22社で構成されております。 当社グループの事業内容及び主要な連結子会社の"
            "位置づけを示すと、以下のとおりであります。")
    assert headline(aida).startswith("プレス機械を主力とする")
    # 事業説明の後ろに続く「位置づけは、次のとおり」は落とす
    jr = ("当社グループは、当社及び子会社２社で構成され、主として建設事業及び不動産事業を展開しており、"
          "当企業集団に係る位置づけは、次のとおりであります。")
    assert headline(jr) == "主として建設事業及び不動産事業を展開しており"


def test_headline_truncates():
    long = "当社は、" + "半導体製造装置の開発および販売を行っております" * 5 + "。"
    h = headline(long, limit=20)
    assert len(h) == 21 and h.endswith("…")


def test_headline_empty_and_fallback():
    assert headline("") == ""
    assert headline("当社グループは、当社及び子会社３社により構成されております。") != ""  # 候補が無ければ先頭文


def test_row_from_document_requires_code_and_business():
    doc = {"secCode": "58980", "docID": "S100XXXX", "periodEnd": "2026-03-31",
           "submitDateTime": "2026-09-18 10:00"}
    row = row_from_document(doc, XBRL)
    assert row["code"] == "5898" and row["doc_id"] == "S100XXXX" and row["disc_date"] == "2026-09-18"
    assert row_from_document({"secCode": ""}, XBRL) is None
    assert row_from_document(doc, "<xbrl></xbrl>") is None        # 事業の内容が無い書類は保存しない


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

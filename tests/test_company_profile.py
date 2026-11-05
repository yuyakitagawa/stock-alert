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
    # 構成説明しか無ければ空欄（会社の構成だけの文は出さない。2026-10-30 日本ゼオン・サンワテクノス）
    assert headline("当社グループは、当社及び子会社３社により構成されております。") == ""


def test_headline_handles_parens_tables_and_headings():
    """2026-10-30 の通知で崩れた形: 括弧内の「。」で文が切れる・表の行・（分野名）の見出し。"""
    thomson = ("当社の企業集団は、当社および子会社14社で構成され、針状ころ軸受および直動案内機器等"
               "(以下、「軸受等」といいます。)ならびに諸機械部品の製造・販売を単一の事業として運営しております。 "
               "従いまして、当社および連結子会社(以下、「当社グループ」といいます。)は、単一セグメントであります。")
    assert headline(thomson).startswith("針状ころ軸受および直動案内機器等ならびに")
    nichicon = ("当社グループ（当社および当社の関係会社）は、当社、子会社25社および関連会社３社により構成されており、"
                "その主な事業内容と、主要な会社の当社グループの事業に係る位置付けおよびセグメントとの関連は、次のとおりです。 "
                "（コンデンサ事業） 当事業では、主にアルミ電解コンデンサの製造販売等を行っています。 "
                "セグメント 主要な製品 主要な生産会社 主要な販売会社 コンデンサ事業 アルミ電解コンデンサ")
    assert headline(nichicon) == "当事業では、主にアルミ電解コンデンサの製造販売等を行っています"
    zeon = ("当社グループ（当社及び当社の関係会社）は、当社、子会社56社及び関連会社９社で構成されており、"
            "主な事業内容と事業を構成している当社及び関係会社の当該事業における位置付けは次のとおりであります。 "
            "事業区分 主要製品等 主要な会社 エラストマー素材事業 合成ゴムの製造販売。")
    assert headline(zeon) == ""
    sekisui = ("当社及び当社の関係会社(当社、国内子会社16社(2026年３月31日現在)により構成)においては、"
               "公共分野、民間分野の２部門に関係する事業を主として行っており、各事業における位置付け等は次のとおりである。")
    assert headline(sekisui, limit=80) == "当社及び当社の関係会社においては、公共分野、民間分野の２部門に関係する事業を主として行っており"


def test_row_from_document_requires_code_and_business():
    doc = {"secCode": "58980", "docID": "S100XXXX", "periodEnd": "2026-03-31",
           "submitDateTime": "2026-09-18 10:00"}
    row = row_from_document(doc, XBRL)
    assert row["code"] == "5898" and row["doc_id"] == "S100XXXX" and row["disc_date"] == "2026-09-18"
    assert row_from_document({"secCode": ""}, XBRL) is None
    assert row_from_document(doc, "<xbrl></xbrl>") is None        # 事業の内容が無い書類は保存しない


def test_save_and_load_use_company_info_columns():
    """事業情報は company_info に統合。統合前の名前で渡し、列名（profile_*）へ読み替えて保存する。"""
    from unittest import mock
    import lib.company_profile as cp
    with mock.patch.object(cp.sb, "upsert", return_value=True) as up:
        cp.save([{"code": "5801", "business": "電線", "issues": None, "employees": 100, "avg_age": 40.0,
                  "avg_salary": 8_000_000, "avg_service": 15.0, "fy_end": "2026-03-31",
                  "doc_id": "S100X", "disc_date": "2026-06-24"}])
    table, rows = up.call_args.args[:2]
    assert table == "company_info"
    r = rows[0]
    assert r["profile_employees"] == 100 and r["profile_fy_end"] == "2026-03-31"
    assert r["profile_doc_id"] == "S100X" and r["profile_disc_date"] == "2026-06-24" and r["profile_fetched_at"]
    assert "employees" not in r and "fy_end" not in r  # 株探・Yahooの列を上書きしない
    with mock.patch.object(cp.sb, "select", return_value=[{"code": "5801", "business": "電線"}]) as sel:
        assert cp.load(["5801"])["5801"]["business"] == "電線"
    q = sel.call_args.args[1]
    assert "profile_doc_id=not.is.null" in q and "employees:profile_employees" in q


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

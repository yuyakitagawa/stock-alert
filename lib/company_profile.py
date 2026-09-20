"""
lib/company_profile.py — 有価証券報告書（EDINET）から事業情報を抜き出す

押し目買いの通知（web/dip_buy_alert.py）に「どんな会社か」を1行添えるために使う。
出どころは有報のXBRLにある記述ブロックで、EDINET APIから無料で取れる。
Claude APIは使わない（要約せず、先頭の一文を機械的に切り出す）。

取るもの:
  事業の内容（DescriptionOfBusinessTextBlock）      → business（先頭800字）
  対処すべき課題（...IssuesToAddressEtcTextBlock）   → issues（先頭400字）
  従業員の状況                                       → employees / avg_age / avg_salary / avg_service

保存先は Supabase の `company_profile`（supabase/create_company_profile.sql）。
本文をそのまま持つとDBが膨らむ（有報1本で3MB前後あり、Free枠は500MB）ので、
先頭だけを残す。
"""
import html
import re
from datetime import date

import lib.supabase_client as sb

TABLE = "company_profile"
DOC_TYPE_ANNUAL = "120"          # 有価証券報告書
BUSINESS_CHARS = 800
ISSUES_CHARS = 400
HEADLINE_CHARS = 40

_BLOCKS = {
    "business": "DescriptionOfBusinessTextBlock",
    "issues": "BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock",
}
_NUMS = {
    "employees": "NumberOfEmployees",
    "avg_age": "AverageAgeYearsInformationAboutReportingCompanyInformationAboutEmployees",
    "avg_salary": "AverageAnnualSalaryInformationAboutReportingCompanyInformationAboutEmployees",
    "avg_service": "AverageLengthOfServiceYearsInformationAboutReportingCompanyInformationAboutEmployees",
}
# 「３ 【事業の内容】」のような見出しと、本文の頭に必ず付く構成会社の説明を落とす
_HEAD_RE = re.compile(r"^[０-９0-9]+\s*[【\[][^】\]]+[】\]]\s*")
# 本文の頭には「当社グループは、当社及び連結子会社11社（…）で構成されております」のような
# 会社の構成だけを述べる文が来ることが多い。何をしている会社かは、その次の文から始まる。
_SKIP_SENTENCE = re.compile(
    r"(構成され|持株会社制|純粋持株会社|により構成|から構成|以下の通りであります|記載のとおり|"
    r"記載しており|セグメントと同一|経理の状況|当連結会計年度より|位置付け|次のとおり|"
    r"^なお|^また|^これら|^その他|^以上|^[）\)）]|^（)")
# 「何をしている会社か」を述べている文の目印
_BUSINESS_WORD = re.compile(r"(事業|製造|販売|開発|提供|サービス|運営|手掛け|営んで)")


def _text_block(xbrl: str, tag: str) -> str:
    m = re.search(rf"<jpcrp[^:>]*:{tag}[^>]*>([\s\S]*?)</jpcrp[^:>]*:{tag}>", xbrl)
    if not m:
        return ""
    t = html.unescape(m.group(1))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[\s　]+", " ", t).strip()
    return _HEAD_RE.sub("", t).strip()


def _number(xbrl: str, tag: str) -> float | None:
    m = re.search(rf"<jpcrp[^:>]*:{tag}[^>]*>([\d,.]+)<", xbrl)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def headline(business: str, limit: int = HEADLINE_CHARS) -> str:
    """事業の内容から通知に載せる一文を作る。

    先頭の「当社グループは、当社及び連結子会社11社（…）で構成され」のような構成説明を飛ばし、
    最初に出てくる「何をしている会社か」の文を採る。
    """
    t = (business or "").strip()
    if not t:
        return ""
    cut = lambda s: (s[:limit].rstrip("、,") + "…") if len(s) > limit else s
    cands = []
    for raw in t.split("。"):
        s = re.sub(r"^[、,。\s]+", "", raw).strip()
        if len(s) < 10 or _SKIP_SENTENCE.search(s):
            continue
        if _BUSINESS_WORD.search(s):
            return cut(s)
        cands.append(s)
    return cut(cands[0]) if cands else cut(t.split("。")[0].strip())


def parse_profile(xbrl: str) -> dict:
    """有報のXBRL本文から保存する項目を取り出す。取れなかった項目は None/空。"""
    out = {k: (_text_block(xbrl, tag) or None) for k, tag in _BLOCKS.items()}
    if out.get("business"):
        out["business"] = out["business"][:BUSINESS_CHARS]
    if out.get("issues"):
        out["issues"] = out["issues"][:ISSUES_CHARS]
    for key, tag in _NUMS.items():
        v = _number(xbrl, tag)
        out[key] = int(v) if v is not None and key in ("employees", "avg_salary") else v
    return out


def save(rows: list[dict]) -> bool:
    return sb.upsert(TABLE, rows, on_conflict="code") if rows else True


def load(codes: list[str] | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    q = "select=code,business,employees,avg_age,avg_salary,disc_date"
    if codes is None:
        for r in sb.select(TABLE, q):
            out[str(r["code"])] = r
        return out
    for i in range(0, len(codes), 100):
        ids = ",".join(f'"{c}"' for c in codes[i:i + 100])
        for r in sb.select(TABLE, f"code=in.({ids})&{q}"):
            out[str(r["code"])] = r
    return out


def row_from_document(doc: dict, xbrl: str, today: date | None = None) -> dict | None:
    """EDINETの書類メタデータ＋XBRL本文から company_profile の1行を作る。"""
    code = (doc.get("secCode") or "")[:4]
    if not code or not xbrl:
        return None
    p = parse_profile(xbrl)
    if not p.get("business"):
        return None
    return {"code": code, **p, "fy_end": doc.get("periodEnd"), "doc_id": doc.get("docID"),
            "disc_date": doc.get("submitDateTime", "")[:10] or (today or date.today()).isoformat()}

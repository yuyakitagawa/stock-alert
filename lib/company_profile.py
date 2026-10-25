"""
lib/company_profile.py — 有価証券報告書（EDINET）から事業情報を抜き出す

押し目買いの通知（web/dip_buy_alert.py）に「どんな会社か」を1行添えるために使う。
出どころは有報のXBRLにある記述ブロックで、EDINET APIから無料で取れる。
Claude APIは使わない（要約せず、先頭の一文を機械的に切り出す）。

取るもの:
  事業の内容（DescriptionOfBusinessTextBlock）      → business（先頭800字）
  対処すべき課題（...IssuesToAddressEtcTextBlock）   → issues（先頭400字）
  従業員の状況                                       → employees / avg_age / avg_salary / avg_service

保存先は Supabase の `company_info`（2026-10-25 に旧 company_profile を統合）。
列名は company_info 側で衝突しないよう employees→profile_employees、fy_end/doc_id/disc_date/fetched_at に
profile_ を付けている。この関数群の呼び出し側は統合前の名前のまま扱う（save/load で読み替える）。
本文をそのまま持つとDBが膨らむ（有報1本で3MB前後あり、Free枠は500MB）ので、
先頭だけを残す。
"""
import html
import re
from datetime import date, datetime, timezone

import lib.supabase_client as sb

TABLE = "company_info"
# 統合前の名前 → company_info の列名（名前が同じ business/issues/avg_* は載せない）
_COLS = {"employees": "profile_employees", "fy_end": "profile_fy_end",
         "doc_id": "profile_doc_id", "disc_date": "profile_disc_date"}
_HAS = "profile_doc_id=not.is.null"  # 有報を取り込んだ行だけ（決算・従業員数だけの行を除く）
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
    r"記載しており|セグメントと同一|経理の状況|当連結会計年度より|位置付け|位置づけ|次のとおり|次の通り|"
    r"以下のとおり|"
    r"^なお|^また|^これら|^その他|^以上|^[）\)）]|^（)")
# 「何をしている会社か」を述べている文の目印
_BUSINESS_WORD = re.compile(r"(事業|製造|販売|開発|提供|サービス|運営|手掛け|営んで)")
# 構成説明と事業の説明が1文にまとまっている形。何をしている会社かの部分だけを取り出す
#   後ろ: 「…子会社43社…で構成され、各事業に関する製品の製造・加工・販売を主な内容とし…」（リンテック）
#   前:   「当社グループ(…)は、プレス機械…の製造・販売…を主な事業として、当社、連結子会社22社…で構成」（アイダ）
_COMPOSED_THEN = re.compile(r"構成(?:され|し)(?:ており|ています)?[、,]")
_MAIN_THEN_COMPOSED = re.compile(r"^(?:.{0,60}?は[、,])?(.+?)を主(?:な|たる)(?:事業|内容|業務)(?:として|とし)[、,].*構成")
_LIST_INTRO = re.compile(r"[、,][^、,]*(?:位置付け|位置づけ|次のとおり|次の通り|以下のとおり|以下の通り)")


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


def _business_part(s: str) -> str:
    """構成説明の文から事業の説明部分を切り出す。該当しなければ空文字。"""
    m = _COMPOSED_THEN.search(s)
    if m:
        part = s[m.end():]
        cut_at = _LIST_INTRO.search(part)  # 「…を展開しており、当企業集団に係る位置づけは、次のとおり」
        if cut_at:
            part = part[:cut_at.start()]
    else:
        m = _MAIN_THEN_COMPOSED.search(s)
        part = m.group(1) if m else ""
    part = part.strip()
    if len(part) < 10 or part.endswith("は") or not _BUSINESS_WORD.search(part):
        return ""
    return part


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
        if len(s) < 10:
            continue
        part = _business_part(s)
        if part:
            return cut(part)
        if _SKIP_SENTENCE.search(s):
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
    now = datetime.now(timezone.utc).isoformat()
    out = [{_COLS.get(k, k): v for k, v in r.items()} | {"profile_fetched_at": now} for r in rows]
    return sb.upsert(TABLE, out, on_conflict="code") if out else True


def load(codes: list[str] | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    q = f"{_HAS}&select=code,business,employees:profile_employees,avg_age,avg_salary,disc_date:profile_disc_date"
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
    """EDINETの書類メタデータ＋XBRL本文から1行を作る（統合前の列名。save() が読み替える）。"""
    code = (doc.get("secCode") or "")[:4]
    if not code or not xbrl:
        return None
    p = parse_profile(xbrl)
    if not p.get("business"):
        return None
    return {"code": code, **p, "fy_end": doc.get("periodEnd"), "doc_id": doc.get("docID"),
            "disc_date": doc.get("submitDateTime", "")[:10] or (today or date.today()).isoformat()}

"""
lib/earnings_forecast.py — 決算短信（TDnet）のXBRLから会社予想を取り込む

なぜ必要か:
  会社予想（純利益・営業利益・売上）は J-Quants Free から取っていたが、2026-04-24 分で
  提供が終わった。以降の決算は EDINET（有価証券報告書・半期報告書）由来で、有報には
  会社予想が載らない。そのため jquants_fin_summary の fnp は 2026-04-25 以降 0件になり、
  押し目買い通知（web/dip_buy_alert.py）の「予想増益」が判定できない銘柄が増えていた。
  予想が載っているのは決算短信（TDnet）で、XBRLが無料で取れる。

保存:
  jquants_fin_summary に doc_type='短信予想' の行として入れる（code, disc_date が主キー）。
  実績（FY/1Q/2Q/3Q）の行は触らない。`lib/fundamentals.py` の年次フィルタは doc_type が
  "FY" で始まる行だけを見るので、この行が混ざっても既存の処理は変わらない。

XBRLの形:
  決算短信サマリーはインラインXBRL（XBRLData/Summary/*-ixbrl.htm）。予想値は実績と同じ
  要素名で、contextRef に "ForecastMember" が入る。期は contextRef の先頭で分かる
  （CurrentYearDuration=今期予想／NextYearDuration=来期予想）。
"""
import io
import re
import zipfile
from datetime import date

import requests

HEADERS = {"User-Agent": "stock-alert/1.0 (+https://kujira-watch.com)"}
LIST_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list/{cond}.json"
DOC_TYPE = "短信予想"

# 純利益・営業利益・売上の要素名（日本基準とIFRSの両方。末尾一致で拾う）
_NP_TAGS = ("ProfitAttributableToOwnersOfParent", "NetIncome", "ProfitLossAttributableToOwnersOfParent")
_OP_TAGS = ("OperatingIncome", "OperatingProfit")
_SALES_TAGS = ("NetSales", "Sales", "Revenue", "OperatingRevenues")


def _attrs(tag: str) -> dict:
    return dict(re.findall(r'([\w:]+)="([^"]*)"', tag))


def _value(a: dict, text: str) -> float | None:
    """ix:nonFraction の値（scale・sign・カンマを解決）。"""
    t = re.sub(r"<[^>]+>", "", text).strip().replace(",", "").replace("△", "-")
    if not t or a.get("xsi:nil") == "true":
        return None
    try:
        v = float(t) * (10 ** int(a.get("scale", 0) or 0))
    except ValueError:
        return None
    return -v if a.get("sign") == "-" else v


def _name_matches(name: str, tags: tuple) -> bool:
    base = name.split(":")[-1]
    return any(base == t or base == t + "IFRS" for t in tags)


def parse_forecast(zip_bytes: bytes) -> dict | None:
    """決算短信のXBRL(zip)から会社予想を取り出す。予想が無い短信は None。

    連結を優先し、連結が無ければ単体。今期予想（CurrentYearDuration）と来期予想
    （NextYearDuration、通期の短信に載る）の両方があれば新しい年度の方を採る。
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        names = [n for n in zf.namelist() if n.endswith("-ixbrl.htm") and "/Summary/" in n]
        if not names:
            return None
        html = zf.read(names[0]).decode("utf-8", "ignore")
    except Exception:
        return None

    fy_end = {}
    for m in re.finditer(r"<ix:nonNumeric([^>]*)>([\s\S]{0,80}?)</ix:nonNumeric>", html):
        a = _attrs(m.group(1))
        if a.get("name", "").split(":")[-1] == "FiscalYearEnd":
            v = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                fy_end[a.get("contextRef", "")] = v

    found: dict[str, dict] = {}
    for m in re.finditer(r"<ix:nonFraction([^>]*)>([^<]*)</ix:nonFraction>", html):
        a = _attrs(m.group(1))
        ctx, name = a.get("contextRef", ""), a.get("name", "")
        if "ForecastMember" not in ctx or "LowerMember" in ctx or "UpperMember" in ctx:
            continue
        period = "next" if ctx.startswith("NextYear") else "current" if ctx.startswith("CurrentYear") else None
        if period is None:
            continue
        consolidated = "NonConsolidatedMember" not in ctx
        v = _value(a, m.group(2))
        if v is None:
            continue
        slot = found.setdefault(period, {"consolidated": False})
        if slot.get("consolidated") and not consolidated:
            continue  # 連結が既にあるなら単体で上書きしない
        if consolidated and not slot.get("consolidated"):
            slot.clear()
            slot["consolidated"] = True
        for key, tags in (("fnp", _NP_TAGS), ("fop", _OP_TAGS), ("fsales", _SALES_TAGS)):
            if _name_matches(name, tags) and key not in slot:
                slot[key] = v

    for period in ("next", "current"):  # 通期短信なら来期予想を優先
        slot = found.get(period)
        if not slot or "fnp" not in slot:
            continue
        end = fy_end.get("NextYearInstant" if period == "next" else "CurrentYearInstant")
        if end is None and period == "next" and fy_end.get("CurrentYearInstant"):
            cur = date.fromisoformat(fy_end["CurrentYearInstant"])
            end = cur.replace(year=cur.year + 1).isoformat()
        if not end:
            continue
        return {"fy_end": end, "fnp": slot.get("fnp"), "fop": slot.get("fop"), "fsales": slot.get("fsales")}
    return None


def list_summaries(day: date) -> list[dict]:
    """その日にTDnetへ出た決算短信（訂正を除く・XBRLあり）。"""
    try:
        r = requests.get(LIST_URL.format(cond=day.strftime("%Y%m%d")),
                         params={"limit": 2000}, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"[forecast] TDnet一覧 HTTP {r.status_code}: {day}")
            return []
        items = [i.get("Tdnet", {}) for i in r.json().get("items", [])]
    except Exception as e:
        print(f"[forecast] TDnet一覧の取得に失敗 ({day}): {e}")
        return []
    out = []
    for it in items:
        title = (it.get("title") or "").strip()
        code = (it.get("company_code") or "").strip()
        if "決算短信" not in title or "訂正" in title or not it.get("url_xbrl"):
            continue
        if len(code) == 5 and code.endswith("0"):
            code = code[:4]
        out.append({"code": code, "title": title, "url_xbrl": it["url_xbrl"],
                    "disclosed_at": (it.get("pubdate") or "")[:10]})
    return out


def fetch_forecast(item: dict) -> dict | None:
    """短信1件ぶんの行（jquants_fin_summary の形）。予想が無ければ None。"""
    try:
        r = requests.get(item["url_xbrl"], headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
    except Exception:
        return None
    f = parse_forecast(r.content)
    if not f:
        return None
    return {"code": item["code"], "disc_date": item["disclosed_at"], "doc_type": DOC_TYPE,
            "fy_end": f["fy_end"], "fnp": f["fnp"], "fop": f["fop"], "fsales": f["fsales"]}

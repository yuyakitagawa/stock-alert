"""
lib/edinet_financials.py
EDINET API v2 から有価証券報告書・四半期報告書のXBRLを取得し、
財務データ（BS/PL/CF）を抽出するモジュール。

jquants_fin_summary テーブルと同じスキーマで保存し、
J-Quants Free期限切れ後の代替データソースとして機能する。

docTypeCode:
  120 = 有価証券報告書（年次）
  130 = 四半期報告書
  140 = 半期報告書
"""
import re
from datetime import date, timedelta
from lib.edinet import _api_key, _fetch_xbrl_text, fetch_documents_list, _normalize_sec_code

_FIN_DOC_TYPES = {"120", "130", "140"}

# XBRL タクソノミ要素名 → 抽出ターゲット
# jpcrp_cor: (日本基準の連結・個別共通プレフィックス)
# jppfs_cor: (日本基準の財務諸表プレフィックス)
_XBRL_TAGS = {
    "sales": [
        "NetSalesSummaryOfBusinessResults",
        "NetSales",
        "RevenueIFRSSummaryOfBusinessResults",
        "RevenueIFRS",
        "OperatingRevenue1SummaryOfBusinessResults",
    ],
    "op": [
        "OperatingIncomeSummaryOfBusinessResults",
        "OperatingIncome",
        "OperatingProfitIFRSSummaryOfBusinessResults",
    ],
    "np": [
        "ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults",
        "ProfitLossAttributableToOwnersOfParent",
        "NetIncomeLossSummaryOfBusinessResults",
        "NetIncomeLoss",
        "ProfitLossIFRSSummaryOfBusinessResults",
    ],
    "ta": [
        "TotalAssetsSummaryOfBusinessResults",
        "TotalAssets",
        "TotalAssetsIFRSSummaryOfBusinessResults",
    ],
    "equity": [
        "NetAssetsSummaryOfBusinessResults",
        "NetAssets",
        "EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
        "TotalEquityIFRSSummaryOfBusinessResults",
    ],
    "eps": [
        "BasicEarningsLossPerShareSummaryOfBusinessResults",
        "BasicEarningsLossPerShare",
    ],
    "bps": [
        "NetAssetsPerShareSummaryOfBusinessResults",
        "NetAssetsPerShare",
        "BookValuePerShareOfEquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
    ],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults",
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults",
    ],
    "div_ann": [
        "AnnualDividendPerShareSummaryOfBusinessResults",
        "DividendPerShareAnnual",
        "DividendPaidPerShareSummaryOfBusinessResults",
    ],
    "sh_out": [
        "TotalNumberOfIssuedSharesSummaryOfBusinessResults",
        "TotalNumberOfIssuedShares",
    ],
    # 予想
    "fsales": [
        "NetSalesForecast",
        "RevenueIFRSForecast",
    ],
    "fop": [
        "OperatingIncomeForecast",
    ],
    "fnp": [
        "ProfitLossAttributableToOwnersOfParentForecast",
        "NetIncomeLossForecast",
    ],
}


def _extract_float(xbrl_text: str, tag_names: list[str]) -> float | None:
    """XBRL本文から指定タグ名のいずれかにマッチする数値を抽出する。
    contextRef に CurrentYear* や CurrentQuarter* を含むものを優先。
    """
    for tag in tag_names:
        pattern = rf'<[^>]*{tag}[^>]*contextRef="([^"]*)"[^>]*>\s*([+-]?[\d,]+\.?\d*)\s*<'
        matches = re.findall(pattern, xbrl_text)
        if not matches:
            pattern = rf'<[^>]*:{tag}[^>]*contextRef="([^"]*)"[^>]*>\s*([+-]?[\d,]+\.?\d*)\s*<'
            matches = re.findall(pattern, xbrl_text)
        if not matches:
            continue

        # CurrentYear/CurrentQuarter を優先（累計実績）
        best_val = None
        for ctx, val_str in matches:
            ctx_lower = ctx.lower()
            if "prior" in ctx_lower or "lastquarter" in ctx_lower:
                continue
            val = float(val_str.replace(",", ""))
            if "currentyear" in ctx_lower or "currentquarter" in ctx_lower:
                return val
            if best_val is None:
                best_val = val
        if best_val is not None:
            return best_val
    return None


def _extract_sec_code(xbrl_text: str) -> str | None:
    """XBRLからSecurityCodeDEIを抽出して4桁に正規化。"""
    m = re.search(r'<[^>]*SecurityCodeDEI[^>]*>\s*([0-9A-Za-z]{4,5})\s*<', xbrl_text)
    if m:
        return _normalize_sec_code(m.group(1))
    return None


def _extract_fiscal_year_end(xbrl_text: str) -> str | None:
    """XBRLから決算期末日を抽出。"""
    m = re.search(r'<[^>]*CurrentFiscalYearEndDateDEI[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*<', xbrl_text)
    if m:
        return m.group(1)
    # --MM-DD 形式
    m = re.search(r'<[^>]*CurrentFiscalYearEndDateDEI[^>]*>\s*--(\d{2}-\d{2})\s*<', xbrl_text)
    if m:
        return m.group(1)
    return None


def _detect_doc_type(doc_type_code: str, xbrl_text: str) -> str:
    """docTypeCode + XBRL内容からFY/1Q/2Q/3Qを判定。"""
    if doc_type_code == "120":
        return "FY"
    # 四半期: XBRL内のquarterを探す
    m = re.search(r'<[^>]*CurrentFiscalYearStartDateDEI[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*<', xbrl_text)
    m2 = re.search(r'<[^>]*CurrentPeriodEndDateDEI[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*<', xbrl_text)
    if m and m2:
        from datetime import date as _d
        try:
            start = _d.fromisoformat(m.group(1))
            end = _d.fromisoformat(m2.group(1))
            months = (end.year - start.year) * 12 + end.month - start.month
            if months <= 4:
                return "1Q"
            elif months <= 7:
                return "2Q"
            elif months <= 10:
                return "3Q"
            else:
                return "FY"
        except Exception:
            pass
    if doc_type_code == "140":
        return "2Q"
    return "1Q"


def parse_financial_xbrl(doc_id: str, doc_type_code: str, disc_date: str) -> dict | None:
    """EDINET XBRLから財務データを抽出し、jquants_fin_summary互換dictを返す。"""
    xbrl_text = _fetch_xbrl_text(doc_id)
    if not xbrl_text:
        return None

    code = _extract_sec_code(xbrl_text)
    if not code:
        return None

    doc_type = _detect_doc_type(doc_type_code, xbrl_text)
    fy_end = _extract_fiscal_year_end(xbrl_text)

    result = {
        "code": code,
        "disc_date": disc_date,
        "doc_type": doc_type,
        "fy_end": fy_end,
    }

    for field, tags in _XBRL_TAGS.items():
        result[field] = _extract_float(xbrl_text, tags)

    # 派生指標
    eps = result.get("eps")
    div = result.get("div_ann")
    if eps and eps > 0 and div and div > 0:
        result["payout_ratio"] = round(div / eps, 3)
    else:
        result["payout_ratio"] = None

    # tr_sh (自己株式) - XBRLから取れる場合は追加
    tr_sh = _extract_float(xbrl_text, [
        "NumberOfTreasurySharesSummaryOfBusinessResults",
        "TreasuryShare",
    ])
    result["tr_sh"] = tr_sh

    # 最低限のデータ品質チェック
    has_data = any(result.get(f) is not None for f in ["sales", "op", "np", "eps"])
    if not has_data:
        return None

    return result


def extract_financial_docs(results: list, disc_date: str) -> list[dict]:
    """documents.json の results から決算書類を抽出してメタデータを返す。"""
    docs = []
    for r in results:
        dtc = str(r.get("docTypeCode", ""))
        if dtc not in _FIN_DOC_TYPES:
            continue
        sec = r.get("secCode")
        if not sec:
            continue
        docs.append({
            "doc_id": r.get("docID"),
            "doc_type_code": dtc,
            "filer_name": r.get("filerName"),
            "sec_code": _normalize_sec_code(str(sec)),
            "disc_date": disc_date,
        })
    return [d for d in docs if d["doc_id"]]


def scan_financial_reports(days_back: int = 30, persist: bool = True,
                           start_date: str | None = None,
                           skip_weekends: bool = True,
                           sleep_sec: float = 1.0) -> list[dict]:
    """指定期間のEDINET決算書類をスキャンし、財務データを抽出してDBに保存。

    Args:
        days_back: 遡る日数
        persist: DBに保存するか
        start_date: 開始日(YYYY-MM-DD)。指定時はdays_back無視
        skip_weekends: 土日スキップ
        sleep_sec: XBRL取得間のスリープ(秒)

    Returns: 取得した財務データdictのリスト
    """
    import time
    from lib.db import bulk_upsert_jquants_fin_summary as upsert_jquants_fin_summary

    today = date.today()
    if start_date:
        d0 = date.fromisoformat(start_date)
        dates = [d0 + timedelta(days=i) for i in range((today - d0).days + 1)]
    else:
        dates = [today - timedelta(days=i) for i in range(days_back)]

    if skip_weekends:
        dates = [d for d in dates if d.weekday() < 5]

    all_records = []
    for d in dates:
        ds = d.isoformat()
        results = fetch_documents_list(ds)
        if not results:
            continue

        fin_docs = extract_financial_docs(results, ds)
        if not fin_docs:
            continue

        print(f"  {ds}: {len(fin_docs)}件の決算書類")

        for doc in fin_docs:
            parsed = parse_financial_xbrl(doc["doc_id"], doc["doc_type_code"], ds)
            if parsed:
                all_records.append(parsed)
                print(f"    ✅ {parsed['code']} {doc.get('filer_name','')} "
                      f"({parsed['doc_type']}) sales={parsed.get('sales')} np={parsed.get('np')}")
            else:
                print(f"    ⚠ 解析失敗: {doc['doc_id']} {doc.get('filer_name','')}")
            if sleep_sec:
                time.sleep(sleep_sec)

    if persist and all_records:
        upsert_jquants_fin_summary(all_records)
        print(f"\n  DB保存: {len(all_records)}件")

    return all_records

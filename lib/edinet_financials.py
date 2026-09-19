"""
lib/edinet_financials.py
EDINET API v2 から有価証券報告書・半期報告書のXBRLを取得し、
財務データ（BS/PL/CF）を抽出するモジュール。

jquants_fin_summary テーブルと同じスキーマで保存し、
J-Quants Free期限切れ後の代替データソースとして機能する。

docTypeCode（取得対象は120と160のみ）:
  120 = 有価証券報告書（年次）
  160 = 半期報告書
  130 = 訂正有価証券報告書 … 過去期の数値が当日の開示として保存され「最新決算」を上書きするため対象外
  140 = 四半期報告書 … 2024年4月に制度廃止（1Q/3Qは決算短信のみ＝EDINETに無い）
  2026-09-19まで 120/130/140 を取得しており、半期報告書が1件も入らず訂正報告書が混入していた。
"""
import re
from datetime import date, timedelta
from lib.edinet import _fetch_xbrl_text, fetch_documents_list, _normalize_sec_code

_FIN_DOC_TYPES = {"120", "160"}

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
        "OperatingProfitLossIFRS",
    ],
    "np": [
        "ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults",
        "ProfitLossAttributableToOwnersOfParent",
        "NetIncomeLossSummaryOfBusinessResults",
        "NetIncomeLoss",
        "ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
        "ProfitLossAttributableToOwnersOfParentIFRS",
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
        "BasicEarningsLossPerShareIFRSSummaryOfBusinessResults",
        "BasicEarningsLossPerShareIFRS",
    ],
    "bps": [
        "NetAssetsPerShareSummaryOfBusinessResults",
        "NetAssetsPerShare",
        "BookValuePerShareOfEquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults",
        # 名前に反してIFRSの「1株当たり親会社所有者帰属持分」＝BPS（4183で 持分÷株数 と一致を確認）
        "EquityToAssetRatioIFRSSummaryOfBusinessResults",
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
    """XBRL本文から指定タグ名のいずれかに一致する数値を抽出する。

    1周目は連結（contextRef に Member が付かない当期）だけを見る。IFRS会社の有報には
    日本基準の単体数値（*_NonConsolidatedMember）も載っており、先に単体を拾うと
    売上・EPS・BPSが単体の値になる（実例: 4183三井化学 FY2026 で売上7,498億円＝単体）。
    連結の無い会社は2周目で単体（NonConsolidatedMember）を使う。
    タグ名は完全一致（部分一致だと OperatingIncome が NonOperatingIncome にも当たる）。
    """
    for allow_non_consolidated in (False, True):
        for tag in tag_names:
            pattern = rf'<[\w-]+:{tag}\s[^>]*contextRef="([^"]*)"[^>]*>\s*([+-]?[\d,]+\.?\d*)\s*<'
            best_val = None
            for ctx, val_str in re.findall(pattern, xbrl_text):
                ctx_lower = ctx.lower()
                if "prior" in ctx_lower or "lastquarter" in ctx_lower:
                    continue
                if "member" in ctx_lower:
                    if not (allow_non_consolidated and ctx_lower.endswith("_nonconsolidatedmember")):
                        continue
                val = float(val_str.replace(",", ""))
                if ctx_lower.startswith(("currentyear", "currentquarter", "currentytd", "interim")):
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
    """docTypeCode から FY（有価証券報告書）/ 2Q（半期報告書）を返す。"""
    return "FY" if doc_type_code == "120" else "2Q"


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
                           end_date: str | None = None,
                           skip_weekends: bool = True,
                           sleep_sec: float = 1.0,
                           force: bool = False) -> list[dict]:
    """指定期間のEDINET決算書類をスキャンし、財務データを抽出してDBに保存。

    Args:
        days_back: 遡る日数
        persist: DBに保存するか
        start_date: 開始日(YYYY-MM-DD)。指定時はdays_back無視
        end_date: 終了日(YYYY-MM-DD)。省略時は今日。長期間のbackfillを分割するのに使う
        skip_weekends: 土日スキップ
        sleep_sec: XBRL取得間のスリープ(秒)
        force: 保存済み (code, disc_date) もXBRLを取り直して上書きする（抽出ロジック修正後の再取得用）

    保存済みの (code, disc_date) はXBRLを取りに行かない（一覧APIの secCode で判定）。
    これで遡る日数を広げても毎日の取得量は新着分だけで済み、日次ジョブが数日落ちても
    次に成功した日に取りこぼしを拾える。保存は日ごと（途中でジョブが止まっても済んだ日は残る）。

    Returns: 取得した財務データdictのリスト
    """
    import time
    from lib.db import bulk_upsert_jquants_fin_summary as upsert_jquants_fin_summary
    from lib.db import get_jquants_fin_keys

    last = date.fromisoformat(end_date) if end_date else date.today()
    if start_date:
        d0 = date.fromisoformat(start_date)
        dates = [d0 + timedelta(days=i) for i in range((last - d0).days + 1)]
    else:
        dates = [last - timedelta(days=i) for i in range(days_back)]

    if skip_weekends:
        dates = [d for d in dates if d.weekday() < 5]
    if not dates:
        return []

    existing = set() if force else get_jquants_fin_keys(min(dates).isoformat(), max(dates).isoformat())

    all_records = []
    for d in dates:
        ds = d.isoformat()
        results = fetch_documents_list(ds)
        if not results:
            continue

        fin_docs = extract_financial_docs(results, ds)
        todo = [doc for doc in fin_docs if (doc["sec_code"], ds) not in existing]
        if not todo:
            continue

        print(f"  {ds}: {len(fin_docs)}件の決算書類（未保存 {len(todo)}件）")

        day_records = []
        for doc in todo:
            parsed = parse_financial_xbrl(doc["doc_id"], doc["doc_type_code"], ds)
            if parsed:
                day_records.append(parsed)
                print(f"    ✅ {parsed['code']} {doc.get('filer_name','')} "
                      f"({parsed['doc_type']}) sales={parsed.get('sales')} np={parsed.get('np')}")
            else:
                print(f"    ⚠ 解析失敗: {doc['doc_id']} {doc.get('filer_name','')}")
            if sleep_sec:
                time.sleep(sleep_sec)

        if persist and day_records:
            upsert_jquants_fin_summary(day_records)
        all_records.extend(day_records)

    if persist and all_records:
        print(f"\n  DB保存: {len(all_records)}件")

    return all_records

"""
lib/tdnet.py
TDnet 適時開示情報を「やのしん WEB-API」(非公式・個人運営) 経由で取得するモジュール。

⚠️ 隔離設計（停止リスク対策）:
  - データ源が個人運営のため、いつ停止してもおかしくない。
  - 保存先は ext_tdnet_disclosures（ext_ プレフィックス）に隔離。
  - コア・パイプライン（rank/screener/train）はこのモジュールに一切依存しない。
  - 失敗時は常に [] を返す（例外を伝播させない）。

API: https://webapi.yanoshin.jp/webapi/tdnet/list/{cond}.json
  cond 例:
    - 証券コード4桁     → "7203"
    - 期間指定          → "recent" / "today" / "yesterday"
    - 日付範囲          → "20260601-20260627"
"""
import re
import time
import requests
from datetime import date, timedelta

_API_BASE = "https://webapi.yanoshin.jp/webapi/tdnet/list"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-alert/1.0)"}

# 適時開示タイトル → カタリスト分類
_CATEGORY_RULES = [
    ("業績上方修正", ["上方修正", "業績予想の修正", "通期連結業績予想の修正"]),
    ("業績下方修正", ["下方修正"]),
    ("増配",         ["増配", "配当予想の修正", "配当の修正"]),
    ("減配",         ["減配", "無配"]),
    ("自社株買い",   ["自己株式の取得", "自己株式取得", "自社株買い"]),
    ("株式分割",     ["株式分割"]),
    ("M&A・資本提携", ["株式取得", "子会社化", "合併", "資本提携", "業務提携", "TOB", "公開買付"]),
    ("決算",         ["決算short", "決算説明", "四半期報告", "決算short信"]),
    ("自己株式処分", ["自己株式の処分", "第三者割当"]),
    ("株主優待",     ["株主優待"]),
]


def _classify(title: str) -> str | None:
    """開示タイトルからカタリスト種別を判定。該当なしは None。"""
    for cat, keywords in _CATEGORY_RULES:
        for kw in keywords:
            if kw in title:
                return cat
    return None


def _normalize_code(raw: str) -> str | None:
    """やのしんの company_code（5桁 '72030' 等）を4桁に正規化。"""
    if not raw:
        return None
    s = str(raw).strip()
    if len(s) == 5 and s.endswith("0"):
        return s[:4]
    if len(s) == 4:
        return s
    return s[:4] if len(s) >= 4 else None


def fetch_disclosures(cond: str, limit: int = 100) -> list[dict]:
    """やのしん API から適時開示一覧を取得して整形して返す。

    Args:
        cond: 検索条件（証券コード / "recent" / "20260601-20260627" 等）
        limit: 最大取得件数

    Returns: 整形済み dict のリスト（失敗時は []）。
    """
    url = f"{_API_BASE}/{cond}.json"
    try:
        resp = requests.get(url, params={"limit": limit}, headers=_HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"[tdnet] HTTP {resp.status_code}: {cond}")
            return []
        data = resp.json()
    except Exception as e:
        print(f"[tdnet] 取得失敗 ({cond}): {e}")
        return []

    items = data.get("items", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        td = it.get("Tdnet", {}) if isinstance(it, dict) else {}
        code = _normalize_code(td.get("company_code", ""))
        title = (td.get("title") or "").strip()
        pubdate = (td.get("pubdate") or "").strip()
        if not code or not title or not pubdate:
            continue
        out.append({
            "code": code,
            "disclosed_at": pubdate,
            "title": title,
            "category": _classify(title),
            "doc_url": td.get("document_url") or None,
            "source": "yanoshin",
        })
    return out


def scan_disclosures(days_back: int = 3, codes: list[str] | None = None,
                     persist: bool = True, sleep_sec: float = 1.0,
                     only_categorized: bool = False) -> list[dict]:
    """直近 days_back 日分の適時開示をスキャンして ext_tdnet_disclosures に保存。

    Args:
        days_back:        遡る日数
        codes:            指定時はその銘柄のみ（ウォッチリスト限定取得用）
        persist:          DB保存するか
        sleep_sec:        リクエスト間スリープ
        only_categorized: True ならカタリスト分類できた開示のみ保存

    Returns: 取得した開示 dict のリスト。
    """
    records: list[dict] = []

    if codes:
        # 銘柄指定: 各コードで個別取得
        for code in codes:
            recs = fetch_disclosures(code, limit=50)
            records.extend(recs)
            if sleep_sec:
                time.sleep(sleep_sec)
    else:
        # 期間指定: 日付範囲でまとめて取得
        end = date.today()
        start = end - timedelta(days=days_back)
        cond = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
        records = fetch_disclosures(cond, limit=2000)

    if only_categorized:
        records = [r for r in records if r["category"]]

    if persist and records:
        _persist(records)

    return records


def _persist(records: list[dict]) -> None:
    """ext_tdnet_disclosures へ upsert（隔離テーブル）。"""
    try:
        import lib.supabase_client as sb
        sb.upsert("ext_tdnet_disclosures", records,
                  on_conflict="code,disclosed_at,title")
        print(f"[tdnet] DB保存: {len(records)}件 → ext_tdnet_disclosures")
    except Exception as e:
        print(f"[tdnet] DB保存失敗: {e}")


def get_recent_disclosures(code: str, n: int = 5) -> list[dict]:
    """指定銘柄の直近の適時開示を ext_tdnet_disclosures から取得（LINE Bot用）。
    DB未保存・取得失敗時は []。"""
    try:
        import lib.supabase_client as sb
        return sb.select(
            "ext_tdnet_disclosures",
            f"code=eq.{code}&order=disclosed_at.desc&select=disclosed_at,title,category,doc_url",
            limit=n,
        )
    except Exception:
        return []

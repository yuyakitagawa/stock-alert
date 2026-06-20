"""
lib/edinet.py
EDINET API v2 経由で大量保有報告書（5%ルール）を取得するモジュール。

目的（GARP・イベント駆動）:
  「構造的に買収・改革が起きやすい候補（カタリストスクリーン）」× 「実際に誰かが
  5%超を買い集めた事実（大量保有報告書）」を突合し、本物の先回り候補を洗い出す。

必要なもの:
  - 環境変数 EDINET_API_KEY（EDINET API v2 のサブスクリプションキー）
    ※ .env に置けば自動ロード。クラウドは GitHub Secrets に登録。

設計方針（既存 alt_data.py に倣う）:
  - DBキャッシュ前提（日次スキャンで edinet_large_holdings に蓄積）
  - 失敗時は常に [] を返す（例外は伝播しない）
  - documents.json のメタデータ + XBRL本文から保有割合を取得。

docTypeCode:
  350 = 大量保有報告書
  360 = 変更報告書（保有割合の増減・追加取得）
"""
import os
import re
import io
import zipfile
import requests
from datetime import date, timedelta

_API_BASE = "https://api.edinet-fsa.go.jp/api/v2"
_LARGE_HOLDING_TYPES = {"350", "360"}  # 大量保有報告書 / 変更報告書


def _api_key() -> str:
    """EDINET_API_KEY を返す。.env も探索。未設定なら空文字。"""
    key = os.environ.get("EDINET_API_KEY", "")
    if not key:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                line = line.strip()
                if line.startswith("EDINET_API_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip()
                    break
    return key


def _normalize_sec_code(sec_code: str) -> "str | None":
    """EDINET の secCode を 4桁の証券コードに正規化。
    形式: 4桁数字（7794）、5桁末尾0（77940）、英字混じり（268A, 268A0）
    """
    if not sec_code:
        return None
    s = str(sec_code).strip().upper()
    if len(s) == 5 and s.endswith("0"):
        return s[:4]
    if len(s) == 4:
        return s
    return s[:4] if len(s) >= 4 else None


def fetch_documents_list(target_date: "str | date") -> list:
    """指定日に EDINET へ提出された書類メタデータ一覧を返す。

    Returns: documents.json の results 配列（取得失敗時は []）。
    """
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    key = _api_key()
    if not key:
        return []
    try:
        resp = requests.get(
            f"{_API_BASE}/documents.json",
            params={"date": target_date, "type": 2, "Subscription-Key": key},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("results", []) or []
    except Exception:
        return []


def verify_api(target_date: "str | date | None" = None) -> dict:
    """APIキーの有効性を確認する。指定日（既定=直近の平日）の書類一覧を取得し、
    HTTPステータス・総件数・大量保有件数を返す。

    Returns: {'ok': bool, 'reason': str, 'status': int|None, 'total': int, 'large': int, 'date': str}
    """
    if not _api_key():
        return {"ok": False, "reason": "EDINET_API_KEY 未設定", "status": None,
                "total": 0, "large": 0, "date": ""}
    if target_date is None:
        d = date.today()
        while d.weekday() >= 5:  # 土日は遡る
            d -= timedelta(days=1)
        target_date = d
    if isinstance(target_date, date):
        target_date = target_date.isoformat()
    try:
        resp = requests.get(
            f"{_API_BASE}/documents.json",
            params={"date": target_date, "type": 2, "Subscription-Key": _api_key()},
            timeout=15,
        )
        status = resp.status_code
        if status != 200:
            reason = "キー無効/権限不足" if status in (401, 403) else f"HTTP {status}"
            return {"ok": False, "reason": reason, "status": status,
                    "total": 0, "large": 0, "date": target_date}
        results = resp.json().get("results", []) or []
        large = sum(1 for r in results
                    if str(r.get("docTypeCode", "")) in _LARGE_HOLDING_TYPES)
        return {"ok": True, "reason": "OK", "status": status,
                "total": len(results), "large": large, "date": target_date}
    except Exception as e:
        return {"ok": False, "reason": f"例外: {e}", "status": None,
                "total": 0, "large": 0, "date": target_date}


def _fetch_xbrl_text(doc_id: str) -> "str | None":
    """XBRL本文（ZIP内 PublicDoc/*.xbrl）をテキストで返す。取得失敗時は None。"""
    key = _api_key()
    if not key or not doc_id:
        return None
    try:
        resp = requests.get(
            f"{_API_BASE}/documents/{doc_id}",
            params={"type": 1, "Subscription-Key": key},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"    XBRL HTTP {resp.status_code}: {doc_id}")
            return None
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        xbrl_names = [n for n in zf.namelist()
                      if "PublicDoc" in n and n.endswith(".xbrl")]
        if not xbrl_names:
            print(f"    XBRL内に.xbrlファイルなし: {doc_id} files={zf.namelist()[:5]}")
            return None
        return zf.read(xbrl_names[0]).decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    XBRL例外: {doc_id} {e}")
        return None


def fetch_xbrl_details(doc_id: str) -> dict:
    """XBRL本文から対象銘柄コード(issuer_code)と保有割合(holding_ratio)を抽出する。

    Returns: {"issuer_code": str|None, "holding_ratio": float|None}
    """
    result = {"issuer_code": None, "holding_ratio": None}
    xbrl_text = _fetch_xbrl_text(doc_id)
    if not xbrl_text:
        print(f"    ⚠ XBRL取得失敗: {doc_id}")
        return result

    # 対象銘柄の証券コード
    # 優先1: SecurityCodeOfIssuer（発行者コード = 対象銘柄）
    # 優先2: SecurityCodeDEI（nil でないもの）
    # コード形式: 4桁数字 or 3桁数字+英字（268A等、2024年以降のIPO）
    sec_patterns = [
        r'<[^>]*SecurityCodeOfIssuer[^>]*>\s*([0-9A-Za-z]{4,5})\s*<',
        r'<[^>]*SecurityCodeDEI[^>]*(?<!nil="true")>\s*([0-9A-Za-z]{4,5})\s*<',
    ]
    for pat in sec_patterns:
        m = re.search(pat, xbrl_text)
        if m:
            result["issuer_code"] = _normalize_sec_code(m.group(1))
            break

    # 保有割合（提出後）
    # PerLastReport（前回割合）や Notes（注記）を除外し、現在の保有割合のみ取得
    # 値は小数（0.0778 = 7.78%）またはパーセント（7.78）の両方がありうる
    ratio_patterns = [
        r'<[^>]*HoldingRatioOfShareCertificatesEtc(?:DEI)?[\s>](?:(?!PerLastReport)[^>])*>\s*([0-9]*\.?[0-9]+)\s*<',
        r'<[^>]*:HoldingRatioOfShareCertificatesEtc[\s>][^>]*>\s*([0-9]*\.?[0-9]+)\s*<',
    ]
    for pat in ratio_patterns:
        m = re.search(pat, xbrl_text)
        if m:
            val = float(m.group(1))
            if val < 1.0:
                val *= 100
            result["holding_ratio"] = round(val, 2)
            break

    return result


def fetch_holding_ratio(doc_id: str) -> "float | None":
    """後方互換: XBRL本文から保有割合(%)のみ返す。"""
    return fetch_xbrl_details(doc_id).get("holding_ratio")


def extract_large_holdings(results: list, disc_date: str) -> list:
    """documents.json の results から大量保有報告書（350/360）のみ抽出して整形。

    Returns: list of dict（doc_id, filer_name, doc_type_code,
             doc_description, submit_date, disc_date）。
    """
    records = []
    for r in results:
        if str(r.get("docTypeCode", "")) not in _LARGE_HOLDING_TYPES:
            continue
        records.append({
            "doc_id": r.get("docID"),
            "filer_name": r.get("filerName"),
            "doc_type_code": str(r.get("docTypeCode", "")),
            "doc_description": r.get("docDescription"),
            "submit_date": r.get("submitDateTime"),
            "disc_date": disc_date,
            "holding_ratio": None,   # XBRL取得後に埋める
            "issuer_code": None,     # XBRL取得後に埋める
        })
    return [x for x in records if x["doc_id"]]


def scan_large_holdings(days_back: int = 7, persist: bool = True,
                        start_date: "str | None" = None,
                        skip_weekends: bool = True, sleep_sec: float = 0.0,
                        fetch_xbrl: bool = True) -> list:
    """直近 days_back 日分（または start_date 以降）の大量保有報告書をスキャンしてDB蓄積。

    Args:
        days_back:     何日前まで遡るか（当日含む）。start_date 指定時は無視。
        persist:       True なら edinet_large_holdings テーブルへ upsert。
        start_date:    'YYYY-MM-DD'。指定するとこの日から当日までを全て走査（バックフィル用）。
        skip_weekends: 土日はEDINET提出が無いのでスキップ（API呼び出し削減）。
        sleep_sec:     各日リクエスト間の待機（バックフィルでEDINETに優しく）。
        fetch_xbrl:    True なら XBRL本文から保有割合を取得。Falseならメタデータのみ。

    Returns: 取得した全レコード（dict のリスト）。
    """
    import time
    from lib.db import upsert_edinet_large_holdings

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
        recs = extract_large_holdings(results, disc_date=ds)
        if fetch_xbrl:
            for rec in recs:
                details = fetch_xbrl_details(rec["doc_id"])
                rec["holding_ratio"] = details["holding_ratio"]
                rec["issuer_code"] = details["issuer_code"]
                if not rec["issuer_code"]:
                    print(f"    ⚠ issuer_code取得失敗: {rec['doc_id']} filer={rec.get('filer_name')}")
                if sleep_sec:
                    time.sleep(sleep_sec)
            recs = [r for r in recs if r.get("issuer_code")]
        if recs and persist:
            upsert_edinet_large_holdings(recs)
        all_records.extend(recs)
        if sleep_sec:
            time.sleep(sleep_sec)
    return all_records


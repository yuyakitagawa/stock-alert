"""
Supabase へ当日のランキング・メタ・決算データをエクスポートする。
rank_stocks.py の後、alert_email.py の後に実行する（Step 5）。

依存: requests, python-dotenv
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[export_to_web] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定。スキップします。")
    sys.exit(0)


def _sb_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def _upsert(table: str, rows: list[dict]) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = requests.post(url, headers=_sb_headers(), json=batch, timeout=30)
        if not resp.ok:
            print(f"[export_to_web] {table} upsert failed: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"[export_to_web] {table}: {len(batch)} 行 upsert 完了")


_EMOJI_MAP = {
    "🥇 S買い":        "S買い",
    "⏳ 方向感なし":   "方向感なし",
    "🟡 高値警戒":     "方向感なし",
    "高値警戒":        "方向感なし",
    "—":               "—",
}


def _clean_recommend(value: str) -> str:
    return _EMOJI_MAP.get(value, value)


def export_rankings(today: str) -> list[dict]:
    import lib.supabase_client as sb
    rows = sb.select(
        "gen_rankings",
        f"date=eq.{today}&order=net.desc"
        "&select=date,code,name,close,rise_prob,drop_prob,net,vol,"
        "recommend,rel20,per,pbr,piotroski,bps_growth,eps_surprise,pos52"
    )

    records = []
    for i, r in enumerate(rows, 1):
        records.append({
            "date":       r["date"],
            "code":       r["code"],
            "rank":       i,
            "name":       r["name"],
            "close":      r["close"],
            "rise_prob":  r["rise_prob"],
            "drop_prob":  r["drop_prob"],
            "net":        r["net"],
            "vol":        r["vol"],
            "recommend":  _clean_recommend(r["recommend"]),
            "rel20":      r["rel20"],
            "per":          r["per"],
            "pbr":          r["pbr"],
            "piotroski":    r["piotroski"],
            "bps_growth":   r["bps_growth"],
            "eps_surprise": r["eps_surprise"],
            "pos52":        r["pos52"],
        })
    return records


def export_stock_meta(ranking_rows: list[dict]) -> None:
    from lib.db import get_all_sectors
    sector_map = get_all_sectors()

    meta_rows = []
    seen = set()
    for r in ranking_rows:
        code = r["code"]
        if code in seen:
            continue
        seen.add(code)
        meta_rows.append({
            "code":    code,
            "name":    r["name"],
            "sector":  sector_map.get(str(code)),
        })
    _upsert("jpx_stock_list", meta_rows)


def export_earnings(codes: list[str]) -> None:
    return


def export_market_compare(today: str) -> None:
    """日経 vs S&P500 相対強弱アドバイザーの当日判定（data/market_compare.json）をSupabaseへupsert。"""
    import json as _json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "market_compare.json")
    if not os.path.exists(path):
        print("[export_to_web] market_compare.json なし。市場比較エクスポートをスキップ。")
        return
    try:
        with open(path, encoding="utf-8") as f:
            v = _json.load(f)
    except Exception as e:
        print(f"[export_to_web] market_compare.json 読み込み失敗: {e}")
        return
    row = {
        "date":    today,
        "verdict": v.get("verdict"),
        "score":   v.get("score"),
        "label":   v.get("label"),
        "reasons": v.get("reasons", []),
        "nk5": v.get("nk5"), "nk20": v.get("nk20"), "nk60": v.get("nk60"),
        "us5": v.get("us5"), "us20": v.get("us20"), "us60": v.get("us60"),
    }
    _upsert("gen_market_compare?on_conflict=date", [row])


def main() -> None:
    today = date.today().isoformat()
    print(f"[export_to_web] {today} のデータをエクスポート開始")

    # 1. ランキング
    ranking_rows = export_rankings(today)
    if not ranking_rows:
        print(f"[export_to_web] {today} のランキングデータなし。終了します。")
        return

    # QA: web公開前にデータ整合性をチェック（alert-only。壊れたデータがweb/メールに
    #     出ても気づけるようパイプライン側と二重化。critical でも公開は止めない）
    try:
        from lib.data_sanity import run_gate
        run_gate(ranking_rows, source="export_to_web", alert=True)
    except Exception as _e:
        print(f"[export_to_web] QAチェックでエラー（無視して継続）: {_e}")

    _upsert("gen_rankings", ranking_rows)

    # 2. 企業メタ
    export_stock_meta(ranking_rows)

    # 3. 決算カレンダー
    codes = [r["code"] for r in ranking_rows]
    export_earnings(codes)

    # 4. 日経 vs S&P500 相対強弱アドバイザーの当日判定をエクスポート
    export_market_compare(today)

    print("[export_to_web] エクスポート完了")


if __name__ == "__main__":
    main()

"""
Supabase へ当日のランキング・メタ・決算データをエクスポートする。
rank_stocks.py の後、alert_email.py の後に実行する（Step 5）。

依存: requests, python-dotenv, anthropic
"""
import json
import os
import sqlite3
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
from lib.utils import recommend_from_scores

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_TOP_N = int(os.getenv("AI_TOP_N", "20"))

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[export_to_web] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定。スキップします。")
    sys.exit(0)

from lib.db import DB_PATH


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
    "🔴 下降シグナル": "下降シグナル",
    "⚠️ 弱気シグナル": "弱気シグナル",
    "🟡 高値警戒":     "方向感なし",
    "高値警戒":        "方向感なし",
    "🔻 売り検討":     "下降シグナル",
    "売り検討":        "下降シグナル",
}


def _clean_recommend(value: str) -> str:
    return _EMOJI_MAP.get(value, value)


def export_rankings(today: str) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT date, code, name, close, rise_prob, drop_prob, net, vol,
                  recommend, rel20, stop_loss, per, pbr
           FROM daily_ranking
           WHERE date = ?
           ORDER BY net DESC""",
        (today,),
    ).fetchall()
    con.close()

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
            "stop_loss":  r["stop_loss"],
            "per":        r["per"],
            "pbr":        r["pbr"],
        })
    return records


def export_stock_meta(ranking_rows: list[dict]) -> None:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    sector_map = {
        r["code"]: r["sector"]
        for r in con.execute("SELECT code, sector FROM sector_cache").fetchall()
    }
    con.close()

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
            "market":  None,
            "per":     r["per"],
            "pbr":     r["pbr"],
        })
    _upsert("web_stock_meta", meta_rows)


def export_earnings(codes: list[str]) -> None:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"""SELECT code, next_date FROM earnings_cache
            WHERE code IN ({",".join("?" * len(codes))})""",
        codes,
    ).fetchall()
    con.close()

    earnings_rows = [{"code": r["code"], "next_date": r["next_date"]} for r in rows]
    _upsert("web_earnings", earnings_rows)


def generate_ai_analyses(today: str, top_rows: list[dict]) -> None:
    if not ANTHROPIC_API_KEY:
        print("[export_to_web] ANTHROPIC_API_KEY 未設定。AI解析スキップ。")
        return

    try:
        import anthropic
    except ImportError:
        print("[export_to_web] anthropic パッケージ未インストール。AI解析スキップ。")
        return

    # 既存キャッシュ確認
    codes = [r["code"] for r in top_rows]
    check_url = f"{SUPABASE_URL}/rest/v1/ai_analyses?date=eq.{today}&code=in.({','.join(codes)})&select=code"
    resp = requests.get(check_url, headers=_sb_headers(), timeout=10)
    cached_codes = {r["code"] for r in resp.json()} if resp.ok else set()

    # メタ情報取得
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    sector_map = {
        r["code"]: r["sector"]
        for r in con.execute("SELECT code, sector FROM sector_cache").fetchall()
    }
    con.close()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    SYSTEM_PROMPT = """あなたは日本株の投資アナリストです。
提供されたデータをもとに、個人投資家向けに投資判断を説明してください。

以下のJSON形式で出力してください:
{
  "summary": "150字以内の日本語説明（なぜこのスコアなのか、主な根拠を具体的に）",
  "bull_points": ["強気材料1", "強気材料2", "強気材料3"],
  "bear_points": ["リスク1", "リスク2"]
}

ルール:
- summary は平易な日本語で、数値を具体的に引用すること
- bull_points / bear_points は各2〜3項目
- 投資を勧誘する表現は避け、あくまで分析として述べること
- JSON以外の文章は出力しないこと"""

    ai_rows = []
    for r in top_rows:
        code = r["code"]
        if code in cached_codes:
            print(f"[export_to_web] {code}: AI解析キャッシュ済みスキップ")
            continue

        net = r.get("net") or 0
        rel20 = r.get("rel20") or 0
        user_prompt = "\n".join([
            f"銘柄: {r.get('name') or code}（{code}）",
            f"セクター: {sector_map.get(str(code), '不明')}",
            f"ネットスコア: {'+' if net >= 0 else ''}{net:.1f}%",
            f"上昇確率: {r.get('rise_prob') or 0:.1f}%",
            f"下落確率: {r.get('drop_prob') or 0:.1f}%",
            f"推奨シグナル: {r.get('recommend') or '—'}",
            f"日経比20日リターン: {'+' if rel20 >= 0 else ''}{rel20:.1f}%",
            f"年率ボラティリティ: {r.get('vol') or 0:.1f}%",
            "PER: {}".format(f"{r.get('per'):.1f}x" if r.get('per') else '—'),
            "PBR: {}".format(f"{r.get('pbr'):.1f}x" if r.get('pbr') else '—'),
        ])

        try:
            msg = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = msg.content[0].text if msg.content else "{}"
            parsed = json.loads(raw)
        except Exception as e:
            print(f"[export_to_web] {code}: AI生成エラー: {e}")
            continue

        ai_rows.append({
            "code":          code,
            "date":          today,
            "summary":       parsed.get("summary"),
            "bull_points":   parsed.get("bull_points", []),
            "bear_points":   parsed.get("bear_points", []),
            "model_version": "opus-4-7",
        })
        print(f"[export_to_web] {code}: AI解析完了")

    if ai_rows:
        _upsert("ai_analyses", ai_rows)


def main() -> None:
    today = date.today().isoformat()
    print(f"[export_to_web] {today} のデータをエクスポート開始")

    # 1. ランキング
    ranking_rows = export_rankings(today)
    if not ranking_rows:
        print(f"[export_to_web] {today} のランキングデータなし。終了します。")
        return
    _upsert("web_rankings", ranking_rows)

    # 2. 企業メタ
    export_stock_meta(ranking_rows)

    # 3. 決算カレンダー
    codes = [r["code"] for r in ranking_rows]
    export_earnings(codes)

    # 4. AI解析（上位 N 銘柄）
    top_rows = [r for r in ranking_rows if r.get("recommend") == "S買い"][:AI_TOP_N]
    generate_ai_analyses(today, top_rows)

    # 5. Next.js ISRキャッシュを即時無効化
    site_url = os.getenv("SITE_URL", "")
    secret = os.getenv("INTERNAL_SEND_SECRET", "")
    if site_url:
        try:
            r = requests.get(f"{site_url}/api/revalidate?secret={secret}", timeout=15)
            print(f"[export_to_web] キャッシュ無効化: {r.status_code}")
        except Exception as e:
            print(f"[export_to_web] キャッシュ無効化失敗（無視）: {e}")

    print("[export_to_web] エクスポート完了")


if __name__ == "__main__":
    main()

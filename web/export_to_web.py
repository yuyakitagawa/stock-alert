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


def _sb_get(table: str, query: str = "") -> list[dict]:
    """Supabaseからテーブルを全件読み出す（QA読み戻し用）。
       SupabaseのRESTは1ページ最大1000行のため offset でページングする。失敗時は[]。"""
    base = f"{SUPABASE_URL}/rest/v1/{table}"
    page_size = 1000
    offset = 0
    out: list[dict] = []
    try:
        while True:
            sep = "&" if query else ""
            url = f"{base}?{query}{sep}limit={page_size}&offset={offset}"
            resp = requests.get(url, headers=_sb_headers(), timeout=30)
            if not resp.ok:
                break
            rows = resp.json()
            out.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        return out
    except Exception as e:
        print(f"[export_to_web] {table} 読み出し失敗: {e}")
        return out


def _description_targets() -> list[str]:
    """会社説明があるべき銘柄コード = 値上げ力ウォッチリスト + 保有株。
    これらはユーザーが必ず閲覧する銘柄なので説明欠損をQAが指摘する対象とする。"""
    import csv as _csv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codes: set[str] = set()
    # ウォッチリスト
    wl_path = os.path.join(root, "data", "pricing_power_watchlist.csv")
    try:
        with open(wl_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                c = str(row.get("code", "")).strip()
                if c:
                    codes.add(c.zfill(4))
    except Exception:
        pass
    # 保有株（スプシ→CSVフォールバック）
    try:
        from lib.sheets_helper import load_watch_list
        for c in load_watch_list().keys():
            c = str(c).strip()
            if c:
                codes.add(c.zfill(4))
    except Exception:
        pass
    return sorted(codes)


def qa_site_check(today: str, ranking_rows: list[dict], expected_ai: int) -> None:
    """全upsert完了後、Supabaseから読み戻してサイト全体の整合性を検証（alert-only）。"""
    try:
        from lib.data_sanity import run_site_gate
        # ライブ状態を読み戻して検証（upsert失敗・欠損も捕捉）
        live_rankings = _sb_get("web_rankings",
                                f"date=eq.{today}&select=date,code,rise_prob,drop_prob,net,recommend")
        meta = _sb_get("web_stock_meta", "select=code,sector&limit=5000")
        ai = _sb_get("ai_analyses", f"date=eq.{today}&select=code,summary,verdict,date")
        earnings = _sb_get("web_earnings", "select=code&limit=1")
        # 会社説明（詳細ページ「この会社について」）のカバレッジ検査用
        descriptions = _sb_get(
            "ai_analyses",
            "model_version=eq.company-desc-v1&select=code,summary&limit=5000")
        desc_targets = _description_targets()
        context = {
            "date":         today,
            "rankings":     live_rankings if live_rankings else ranking_rows,
            "stock_meta":   meta,
            "ai_analyses":  ai,
            "earnings":     earnings,
            "expected_ai":  expected_ai,
            "descriptions": descriptions,
            "desc_targets": desc_targets,
        }
        run_site_gate(context, source="export_to_web(site)", alert=True)
    except Exception as e:
        print(f"[export_to_web] サイトQAチェックでエラー（無視して継続）: {e}")


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

    try:
        from lib.kabutan_earnings import fetch_kabutan_earnings, format_earnings_for_prompt
        _kabutan_ok = True
    except ImportError:
        _kabutan_ok = False

    SYSTEM_PROMPT = """あなたは日本株の投資アナリストです。
提供されたデータ（AIモデルスコア＋実際の決算業績）をもとに、個人投資家向けに投資判断を説明してください。

以下のJSON形式で出力してください:
{
  "summary": "150字以内の日本語説明（なぜこのスコアなのか、業績トレンドを踏まえて具体的に）",
  "bull_points": ["強気材料1", "強気材料2", "強気材料3"],
  "bear_points": ["リスク1", "リスク2"],
  "verdict": "買い推奨" | "様子見" | "見送り"
}

ルール:
- summary は平易な日本語で、決算数値（売上・営業利益・EPS）を具体的に引用すること
- bull_points / bear_points は各2〜3項目
- verdict は業績トレンドとモデルスコアを総合的に判断すること
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

        # kabutan.jp から実際の決算業績を取得
        earnings_text = ""
        if _kabutan_ok:
            try:
                e_rows = fetch_kabutan_earnings(str(code))
                earnings_text = "\n" + format_earnings_for_prompt(e_rows)
            except Exception as _e:
                earnings_text = ""

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
        ]) + earnings_text

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
            "verdict":       parsed.get("verdict"),
            "model_version": "opus-4-7",
        })
        print(f"[export_to_web] {code}: AI解析完了 verdict={parsed.get('verdict','—')}")

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

    # QA: web公開前にデータ整合性をチェック（alert-only。壊れたデータがweb/メールに
    #     出ても気づけるようパイプライン側と二重化。critical でも公開は止めない）
    try:
        from lib.data_sanity import run_gate
        run_gate(ranking_rows, source="export_to_web", alert=True)
    except Exception as _e:
        print(f"[export_to_web] QAチェックでエラー（無視して継続）: {_e}")

    _upsert("web_rankings", ranking_rows)

    # 2. 企業メタ
    export_stock_meta(ranking_rows)

    # 3. 決算カレンダー
    codes = [r["code"] for r in ranking_rows]
    export_earnings(codes)

    # 4. AI解析（ネットスコア上位10銘柄）
    top_rows = ranking_rows[:10]
    generate_ai_analyses(today, top_rows)

    # 5. Top10シミュレーション更新
    update_top10_simulation(ranking_rows, today)

    # 6. QA: サイト全体の整合性チェック（全upsert後にライブ状態を読み戻して検証）
    qa_site_check(today, ranking_rows, expected_ai=len(top_rows))

    # 7. Next.js ISRキャッシュを即時無効化
    site_url = os.getenv("SITE_URL", "")
    secret = os.getenv("INTERNAL_SEND_SECRET", "")
    if site_url:
        try:
            r = requests.get(f"{site_url}/api/revalidate?secret={secret}", timeout=15)
            print(f"[export_to_web] キャッシュ無効化: {r.status_code}")
        except Exception as e:
            print(f"[export_to_web] キャッシュ無効化失敗（無視）: {e}")

    print("[export_to_web] エクスポート完了")


def _upsert_simulation(rows: list[dict]) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/web_simulation?on_conflict=period,round,code"
    headers = _sb_headers()
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = requests.post(url, headers=headers, json=batch, timeout=30)
        if not resp.ok:
            print(f"[export_simulation] upsert failed: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"[export_simulation] {len(batch)} 行 upsert 完了")


TOP10_SIM_SELL_THRESH = 5.0  # net < 5% で売却（信号消滅基準）


def update_top10_simulation(ranking_rows: list[dict], today: str) -> None:
    """
    Top10シミュレーション日次更新:
    1. アクティブポジションを今日のnetスコアでチェック → net<5%なら決済
    2. 今日のtop10に新規入りした銘柄を追加
    3. Supabase web_top10_sim テーブルへ upsert
    """
    from lib.db import (init_db, get_top10_sim_active,
                        upsert_top10_sim_entry, close_top10_sim_position)
    init_db()

    # 今日のnet辞書
    today_net   = {r["code"]: r.get("net", 0) or 0 for r in ranking_rows}
    today_price = {r["code"]: r.get("close")        for r in ranking_rows}
    today_name  = {r["code"]: r.get("name")         for r in ranking_rows}
    top10_codes = {r["code"] for r in ranking_rows[:10]}

    # ── 1. アクティブポジションの決済チェック ──────────────────────────────
    active = get_top10_sim_active()
    active_codes = {p["code"] for p in active}
    for pos in active:
        code = pos["code"]
        cur_net   = today_net.get(code, 0)
        cur_price = today_price.get(code)
        if cur_net < TOP10_SIM_SELL_THRESH and cur_price:
            close_top10_sim_position(
                pos["entry_date"], code, today, cur_net, cur_price
            )
            print(f"[top10_sim] 決済: {code} net={cur_net:.1f}% entry={pos['entry_date']}")

    # ── 2. 新規エントリー（top10に入ってかつ未保有） ─────────────────────
    for code in top10_codes:
        if code not in active_codes:
            net   = today_net.get(code, 0)
            price = today_price.get(code)
            name  = today_name.get(code, "")
            if price and net >= TOP10_SIM_SELL_THRESH:
                upsert_top10_sim_entry(today, code, name, net, price)
                print(f"[top10_sim] エントリー: {code} {name} net={net:.1f}%")

    # ── 3. Supabase へ upsert ──────────────────────────────────────────────
    from lib.db import get_top10_sim_active, get_top10_sim_recent_exits
    all_rows = get_top10_sim_active() + get_top10_sim_recent_exits(30)
    if all_rows:
        _upsert("web_top10_sim", all_rows)
        print(f"[top10_sim] {len(all_rows)}件 upsert完了")


def export_simulation_results() -> None:
    """simulations/backtests/rolling21d_*.csv を web_simulation テーブルへ一括 upsert"""
    import glob
    import csv
    import re

    pattern = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "simulations", "backtests", "rolling21d_*.csv"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        print("[export_simulation] CSVファイルが見つかりません:", pattern)
        return

    all_rows: list[dict] = []
    for fp in files:
        m = re.search(r"rolling21d_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.csv$", fp)
        if not m:
            continue
        period_start, period_end = m.group(1), m.group(2)
        period = f"{period_start}_{period_end}"

        with open(fp, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("selected", "0")).strip() != "1":
                    continue
                try:
                    all_rows.append({
                        "period": period,
                        "period_start": period_start,
                        "period_end": period_end,
                        "round": int(row["ラウンド"]),
                        "entry_date": row["entry"],
                        "exit_date": row["exit"],
                        "code": str(row["code"]),
                        "name": row.get("銘柄名", ""),
                        "net_score": float(row["net"]) if row.get("net") else None,
                        "return_pct": float(row["return"]) if row.get("return") else None,
                    })
                except (KeyError, ValueError):
                    pass

    print(f"[export_simulation] {len(all_rows)}件 upsert → web_simulation")
    # web_simulationは on_conflict=period,round,code で重複解決
    _upsert_simulation(all_rows)
    print("[export_simulation] 完了")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simulation":
        export_simulation_results()
    else:
        main()

"""
Supabase へ当日のランキング・メタ・決算データをエクスポートする。
rank_stocks.py の後、alert_email.py の後に実行する（Step 5）。

依存: requests, python-dotenv, anthropic
"""
import json
import os
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


def _description_targets(ranking_rows: list[dict] | None = None) -> list[str]:
    """会社説明があるべき銘柄コード。
    会社説明は全銘柄が対象なので、当日ランキングの全銘柄をターゲットにする
    （ウォッチリスト＋保有株も保険として含める）。QA がカバレッジ欠損を指摘する。"""
    import csv as _csv
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    codes: set[str] = set()
    # 当日ランキングの全銘柄（=サイトで閲覧されうる全銘柄）
    for r in (ranking_rows or []):
        c = str(r.get("code", "")).strip()
        if c:
            codes.add(c.zfill(4))
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
        live_rankings = _sb_get("gen_rankings",
                                f"date=eq.{today}&select=date,code,rise_prob,drop_prob,net,recommend")
        meta = _sb_get("jpx_stock_list", "select=code,sector&limit=5000")
        ai = _sb_get("gen_ai_analyses", f"date=eq.{today}&select=code,summary,verdict,date")
        earnings = []
        # 会社説明（詳細ページ「この会社について」）のカバレッジ検査用
        descriptions = _sb_get(
            "gen_ai_analyses",
            "model_version=eq.company-desc-v1&select=code,summary&limit=5000")
        desc_targets = _description_targets(live_rankings if live_rankings else ranking_rows)
        context = {
            "date":         today,
            "rankings":     live_rankings if live_rankings else ranking_rows,
            "stock_meta":   meta,
            "gen_ai_analyses":  ai,
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
    "🟡 高値警戒":     "方向感なし",
    "高値警戒":        "方向感なし",
    "—":               "—",
}


def _clean_recommend(value: str) -> str:
    return _EMOJI_MAP.get(value, value)


def _prob_band(p) -> str:
    """上昇/下落確率を段階ラベルに変換する。
    モデル確率はIsotonic較正で数十段の階段値になり、小数表示だと多数の銘柄が同値になって
    過度に精密な印象を与えるため、高/やや高/中/やや低/低の5段階で示す（Web/メール表示と統一）。"""
    if p is None:
        return "-"
    if p >= 30: return "高"
    if p >= 22: return "やや高"
    if p >= 14: return "中"
    if p >= 7:  return "やや低"
    return "低"


def export_rankings(today: str) -> list[dict]:
    import lib.supabase_client as sb
    rows = sb.select(
        "gen_rankings",
        f"date=eq.{today}&order=net.desc"
        "&select=date,code,name,close,rise_prob,drop_prob,net,vol,"
        "recommend,rel20,per,pbr,piotroski,bps_growth,eps_surprise,pos52,"
        "is_etf,etf_benchmark,etf_strategy,etf_expense_ratio,etf_return_1y,etf_return_3y"
    )

    records = []
    for i, r in enumerate(rows, 1):
        rec = {
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
            "is_etf":            r.get("is_etf", False),
            "etf_benchmark":     r.get("etf_benchmark"),
            "etf_strategy":      r.get("etf_strategy"),
            "etf_expense_ratio": r.get("etf_expense_ratio"),
            "etf_return_1y":     r.get("etf_return_1y"),
            "etf_return_3y":     r.get("etf_return_3y"),
        }
        records.append(rec)
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
    check_url = f"{SUPABASE_URL}/rest/v1/gen_ai_analyses?date=eq.{today}&code=in.({','.join(codes)})&select=code"
    resp = requests.get(check_url, headers=_sb_headers(), timeout=10)
    cached_codes = {r["code"] for r in resp.json()} if resp.ok else set()

    # メタ情報取得
    from lib.db import get_all_sectors
    sector_map = get_all_sectors()

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
            f"上昇度: {_prob_band(r.get('rise_prob'))}（AI予測の段階。高〜低の5段階）",
            f"下落度: {_prob_band(r.get('drop_prob'))}（AI予測の段階。高〜低の5段階）",
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
        _upsert("gen_ai_analyses", ai_rows)


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

    # 4. AI解析（ネットスコア上位10銘柄）
    top_rows = ranking_rows[:10]
    generate_ai_analyses(today, top_rows)

    # 5. QA: サイト全体の整合性チェック（全upsert後にライブ状態を読み戻して検証）
    qa_site_check(today, ranking_rows, expected_ai=len(top_rows))

    print("[export_to_web] エクスポート完了")


def _upsert_simulation(rows: list[dict]) -> None:
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/gen_simulation?on_conflict=period,round,code"
    headers = _sb_headers()
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = requests.post(url, headers=headers, json=batch, timeout=30)
        if not resp.ok:
            print(f"[export_simulation] upsert failed: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"[export_simulation] {len(batch)} 行 upsert 完了")




def export_simulation_results() -> None:
    """simulations/backtests/rolling21d_*.csv を gen_simulation テーブルへ一括 upsert"""
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

    print(f"[export_simulation] {len(all_rows)}件 upsert → gen_simulation")
    # gen_simulationは on_conflict=period,round,code で重複解決
    _upsert_simulation(all_rows)
    print("[export_simulation] 完了")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--simulation":
        export_simulation_results()
    else:
        main()

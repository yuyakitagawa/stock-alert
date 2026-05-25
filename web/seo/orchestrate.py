#!/usr/bin/env python3
"""SEO記事パイプライン
Usage:
  python3 web/seo/orchestrate.py daily   # S買い新規記事（毎日）
  python3 web/seo/orchestrate.py weekly  # SEOマネージャー週次レビュー（毎週月曜）
"""
import sys, os, json, sqlite3, requests
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY]):
    print("[seo] 環境変数未設定。スキップ。")
    sys.exit(0)

import anthropic
import web.seo.agents as agents
agents.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

from lib.db import DB_PATH
TODAY = date.today().isoformat()


# ── Supabase ヘルパー ──────────────────────────────────────────────────────

def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def _upsert(row: dict) -> None:
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/articles",
        headers=_sb_headers(), json=[row], timeout=30,
    )
    if resp.ok:
        print(f"    ✅ 公開: {row['slug']}")
    else:
        print(f"    ❌ 公開失敗: {resp.status_code} {resp.text[:200]}")


def _article_exists(slug: str) -> bool:
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/articles?slug=eq.{slug}&limit=1",
        headers=_sb_headers(), timeout=10,
    )
    return resp.ok and len(resp.json()) > 0


def _get_all_articles() -> list:
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/articles?order=signal_date.desc&limit=50",
        headers=_sb_headers(), timeout=10,
    )
    return resp.json() if resp.ok else []


# ── DB ────────────────────────────────────────────────────────────────────

def _get_buy_signals() -> list:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT code, name, net, rise_prob, drop_prob, vol
        FROM daily_ranking
        WHERE date = ? AND recommend LIKE '%S買い%'
        ORDER BY net DESC
    """, (TODAY,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"code": r[0], "name": r[1], "net": r[2], "rise_prob": r[3],
         "drop_prob": r[4], "vol": r[5], "sector": "", "signal_date": TODAY}
        for r in rows
    ]


# ── コアパイプライン ───────────────────────────────────────────────────────

def run_pipeline(stock: dict, slug: str = None, rewrite_count: int = 0) -> None:
    """KW選定 → リサーチ → 構成設計 → 執筆 → 品質チェック → 公開"""
    slug = slug or f"{stock['code']}-{TODAY}"
    name, code = stock['name'], stock['code']
    print(f"\n  [{name}（{code}）] slug={slug}")

    print("    1/6 KW選定...")
    kw = agents.kw_agent(stock)
    print(f"       主KW: {kw['primary_kw']}")

    print("    2/6 リサーチ...")
    research = agents.research_agent(kw, stock)

    print("    3/6 構成設計...")
    design = agents.design_agent(kw, research, stock)

    print("    4/6 執筆...")
    body = agents.writer_agent(design, stock, kw)

    print("    5/6 品質チェック...")
    qa = agents.qa_agent(body, kw)
    print(f"       スコア: {qa.get('score', '?')}/100  pass={qa.get('pass')}")

    if not qa.get("pass") and qa.get("feedback"):
        print("       → リライト（QAフィードバック反映）...")
        body = agents.writer_agent(design, stock, kw, qa_feedback=qa["feedback"])

    print("    6/6 公開...")
    _upsert({
        "slug":             slug,
        "code":             code,
        "name":             name,
        "title":            design.get("title", f"{name} S買いシグナル"),
        "body":             body,
        "signal_date":      stock.get("signal_date", TODAY),
        "net_score":        stock.get("net"),
        "target_keyword":   kw.get("primary_kw"),
        "meta_description": design.get("meta_description"),
        "seo_score":        qa.get("score", 70),
        "rewrite_count":    rewrite_count,
        "published_at":     f"{TODAY}T09:00:00+09:00",
    })


# ── 毎日モード ─────────────────────────────────────────────────────────────

def run_daily() -> None:
    print(f"=== SEO Daily {TODAY} ===")
    signals = _get_buy_signals()
    if not signals:
        print("S買いシグナルなし"); return
    print(f"{len(signals)}銘柄を処理")
    for stock in signals:
        slug = f"{stock['code']}-{TODAY}"
        if _article_exists(slug):
            print(f"  スキップ（既存）: {slug}"); continue
        run_pipeline(stock, slug=slug)


# ── 週次モード ─────────────────────────────────────────────────────────────

def run_weekly() -> None:
    print(f"=== SEO Weekly {TODAY} ===")
    articles = _get_all_articles()
    if not articles:
        print("記事なし"); return
    print(f"{len(articles)}記事を分析中...")

    print("\n[アナリティクス]")
    analytics = agents.analytics_agent(articles)
    print(f"  リライト候補: {analytics.get('priority_rewrites', [])}")

    print("\n[SEOマネージャー 週次レビュー]")
    plan = agents.seo_manager(articles, analytics)
    print(f"  今週の方針: {plan.get('strategy_note', '')}")
    print(f"  リライト指示: {len(plan.get('rewrites', []))}件")

    for task in plan.get("rewrites", []):
        slug    = task["slug"]
        article = next((a for a in articles if a["slug"] == slug), None)
        if not article:
            print(f"  スキップ（記事未発見）: {slug}"); continue

        print(f"\n  リライト: {slug}")
        print(f"  理由: {task.get('reason', '')} / 注力: {task.get('focus', '')}")

        stock = {
            "code":        article["code"],
            "name":        article["name"],
            "net":         article.get("net_score") or 0,
            "rise_prob":   0,
            "drop_prob":   0,
            "sector":      "",
            "signal_date": article["signal_date"],
        }
        run_pipeline(
            stock,
            slug=slug,
            rewrite_count=(article.get("rewrite_count") or 0) + 1,
        )

    print("\n=== 週次レビュー完了 ===")


# ── エントリーポイント ─────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "weekly":
        run_weekly()
    else:
        run_daily()

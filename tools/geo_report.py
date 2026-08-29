"""
tools/geo_report.py

GEO（生成AI検索での引用最適化）のPDCAに使うレポート（手動実行専用）。
「AIに読まれているか」と「AIから人が来ているか」を1画面で前期間と比較する。

なぜ専用ツールが要るか:
  - AIクローラー（GPTBot / OAI-SearchBot / ClaudeBot / PerplexityBot 等）はJSを実行しないため
    GA4に一切載らない。巡回の実態は `blog_crawler_log`（proxy.tsが記録）にしか無い。
  - 逆に「AIの回答を読んだ人がリンクを踏んで来た」訪問はGA4にしか無い
    （GA4はこの流入を medium=ai-assistant として分類する）。
  両方を並べないと「AIに読まれてはいるが人は来ていない」のか「そもそも読まれていない」のかが
  区別できず、打つ手が決まらない。

出すもの（すべて前期間との差分付き）:
  1. AIクローラー別の巡回数
  2. AI巡回のページ種別内訳（記事 / 銘柄 / 投資家 / 集計ページ …）
  3. ChatGPT-User・PerplexityBot が取りに来たページTOP … 回答生成のためのその場取得＝引用の代理指標
  4. AI経由の実訪問（GA4: セッション数・エンゲージメント率・着地ページ）
  5. AIクローラーが当たっている「存在しないURL」と「リダイレクト経由のURL」
     （実例: 廃止した/disclosuresに30日で58回当たり続けていた）

必要な設定:
  - SUPABASE_URL / SUPABASE_SERVICE_KEY（.env）… 1〜3・5に必要
  - GA4_PROPERTY_ID と gcp_key.json … 4に必要（無ければ4だけスキップする）

実行:
  python3 tools/geo_report.py              # 直近28日 vs その前の28日
  python3 tools/geo_report.py --days 7
  python3 tools/geo_report.py --limit 20
"""
import argparse
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote, unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import supabase_client as sb  # noqa: E402
from tools.ga4_clicks import access_token, delta, parse_rows, run_report  # noqa: E402

DEFAULT_DAYS = 28
DEFAULT_LIMIT = 15

# AIの学習・検索インデックス・回答生成のために巡回してくるクローラー。
# crawlers.ts の BOT_PATTERNS が付ける bot_name と一致させること。
AI_BOTS = [
    "GPTBot", "ChatGPT-User", "OAI-SearchBot",
    "ClaudeBot", "Claude-Web", "anthropic-ai",
    "PerplexityBot", "Amazonbot", "Applebot",
    "meta-externalagent", "meta-externalfetcher", "Bytespider",
]
# このUAは「AIが回答を作るためにその場で取りに来た」もの＝そのURLが回答に使われた可能性が高い。
# 学習・インデックス目的の一括クロール（GPTBot / OAI-SearchBot）とは意味が違うので分けて見る。
ON_DEMAND_BOTS = ["ChatGPT-User", "PerplexityBot"]

# GA4の sessionSource がこれらを含む流入をAI経由とみなす。medium=ai-assistant でも拾うが、
# GA4の分類が追いついていない新顔サービスをホスト名側でも拾えるようにしておく。
AI_REFERRAL_HOSTS = [
    "chatgpt.com", "openai.com", "perplexity.ai", "gemini.google.com",
    "bard.google.com", "claude.ai", "copilot.microsoft.com", "you.com", "felo.ai",
]
AI_REFERRAL_MEDIUM = "ai-assistant"

# 実在するページの形。ここに載っていないパスをAIクローラーが取りに来ていたら、
# 404を踏んでいる可能性が高い（クロール枠の無駄＋既にAI側に載った参照が切れている）。
EXACT_PATHS = {
    "/", "/about", "/contact", "/faq", "/privacy", "/terms",
    "/activists", "/buybacks", "/investors", "/monthly", "/stocks", "/trending", "/weekly",
    "/ranking/returns", "/ranking/activist",
    "/robots.txt", "/sitemap.xml", "/sitemap-index.xml", "/feed.xml", "/llms.txt",
    "/ads.txt", "/favicon.ico", "/manifest.webmanifest", "/logo",
}
PREFIX_PATTERNS = [
    re.compile(r"^/articles/[^/]+$"),
    # 証券コードは数字4桁だけでなく英字を含むもの（新規上場銘柄の603A等）もある。
    re.compile(r"^/stocks/[0-9A-Za-z]+$"),
    re.compile(r"^/investors/[^/]+$"),
    re.compile(r"^/category/[^/]+$"),
    re.compile(r"^/faq/[^/]+$"),
    re.compile(r"^/date/\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^/monthly/\d{4}-\d{2}$"),
    re.compile(r"^/sitemap-[^/]+$"),
    re.compile(r"^/(?:icon|apple-icon|opengraph-image|twitter-image)"),
    re.compile(r"^/(?:_next|api)/"),
]
# next.config.ts の redirects()。404ではないが、AI側に残っているのは古いURL。
REDIRECT_PATTERNS = [
    re.compile(r"^/en(?:/|$)"),
    re.compile(r"^/ranking$"),
    re.compile(r"^/ranking/(?:buys|sells|filings|trending)$"),
    re.compile(r"^/disclosures$"),
]

PAGE_GROUPS = [
    ("トップ", re.compile(r"^/$")),
    ("記事", re.compile(r"^/articles/")),
    ("銘柄", re.compile(r"^/stocks")),
    ("投資家", re.compile(r"^/investors")),
    ("ランキング", re.compile(r"^/(?:ranking|trending)")),
    ("週次・月次", re.compile(r"^/(?:weekly|monthly|date)")),
    ("アクティビスト", re.compile(r"^/activists")),
    ("自社株買い", re.compile(r"^/buybacks")),
    ("分類別", re.compile(r"^/category/")),
    ("FAQ", re.compile(r"^/faq")),
    ("運営・規約", re.compile(r"^/(?:about|contact|privacy|terms)")),
    ("機械向け", re.compile(r"^/(?:robots\.txt|sitemap|feed\.xml|llms|ads\.txt)")),
]


def page_group(path: str) -> str:
    for label, pattern in PAGE_GROUPS:
        if pattern.search(path):
            return label
    return "その他"


def path_status(path: str) -> str:
    """"ok" / "redirect" / "missing"。クエリ・末尾スラッシュは落として判定する。"""
    clean = unquote(path.split("?")[0])
    if len(clean) > 1:
        clean = clean.rstrip("/")
    if clean in EXACT_PATHS or clean == "":
        return "ok"
    if any(p.search(clean) for p in REDIRECT_PATTERNS):
        return "redirect"
    if any(p.search(clean) for p in PREFIX_PATTERNS):
        return "ok"
    return "missing"


def fetch_ai_rows(start: datetime, end: datetime) -> list[dict]:
    """指定期間のAIクローラーのアクセス行。bot_nameで絞ってから取るので数万行で収まる。"""
    bots = ",".join(f'"{b}"' for b in AI_BOTS)
    return sb.select(
        "blog_crawler_log",
        f"bot_name=in.({bots})"
        f"&occurred_at=gte.{quote(start.isoformat(), safe='')}"
        f"&occurred_at=lt.{quote(end.isoformat(), safe='')}"
        "&select=occurred_at,path,bot_name&order=occurred_at.desc",
    )


def print_counter(now: Counter, before: Counter, limit: int, unit: str = "回") -> None:
    if not now:
        print("  （データなし）")
        return
    for key, count in now.most_common(limit):
        print(f"  {count:>7,}{unit}  {key}{delta(count, before.get(key, 0))}")


def crawler_sections(days: int, limit: int) -> None:
    now_end = datetime.now(timezone.utc)
    now_start = now_end - timedelta(days=days)
    prev_start = now_start - timedelta(days=days)

    now_rows = fetch_ai_rows(now_start, now_end)
    prev_rows = fetch_ai_rows(prev_start, now_start)
    if not now_rows and not prev_rows:
        print("■ AIクローラーの巡回\n  （blog_crawler_logにAIクローラーの記録がありません）")
        return

    print("■ AIクローラー別の巡回数")
    print_counter(Counter(r["bot_name"] for r in now_rows),
                  Counter(r["bot_name"] for r in prev_rows), limit)

    print("\n■ AI巡回のページ種別")
    print_counter(Counter(page_group(r["path"]) for r in now_rows),
                  Counter(page_group(r["path"]) for r in prev_rows), limit)

    print(f"\n■ 回答生成のためにその場で取りに来たページ（{' / '.join(ON_DEMAND_BOTS)}）＝引用の代理指標")
    on_demand = [r for r in now_rows if r["bot_name"] in ON_DEMAND_BOTS]
    prev_on_demand = [r for r in prev_rows if r["bot_name"] in ON_DEMAND_BOTS]
    print_counter(Counter(r["path"] for r in on_demand),
                  Counter(r["path"] for r in prev_on_demand), limit)

    print("\n■ AIクローラーが当たっている存在しないURL（404の可能性）")
    missing = Counter(r["path"] for r in now_rows if path_status(r["path"]) == "missing")
    print_counter(missing, Counter(), limit)

    print("\n■ AIクローラーが当たっている旧URL（リダイレクト。AI側に古い参照が残っている）")
    redirected = Counter(r["path"] for r in now_rows if path_status(r["path"]) == "redirect")
    print_counter(redirected, Counter(), limit)


def is_ai_source(source: str, medium: str) -> bool:
    if medium == AI_REFERRAL_MEDIUM:
        return True
    return any(host in source for host in AI_REFERRAL_HOSTS)


def ga4_section(days: int, limit: int) -> None:
    property_id = os.getenv("GA4_PROPERTY_ID", "").strip()
    token = access_token()
    if not property_id or not token:
        print("\n■ AI経由の実訪問（GA4）\n"
              "  （GA4_PROPERTY_ID または gcp_key.json が無いためスキップ。"
              "詳細は tools/ga4_clicks.py のヘッダーを参照）")
        return

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)

    def fetch(dimensions: list, metrics: list, s: date, e: date) -> dict:
        rows, err = run_report(token, property_id, {
            "dateRanges": [{"startDate": s.isoformat(), "endDate": e.isoformat()}],
            "dimensions": [{"name": d} for d in dimensions],
            "metrics": [{"name": m} for m in metrics],
            "limit": 500,
        })
        if err:
            print(f"  [ga4] {err}")
            return {}
        return parse_rows(rows)

    print("\n■ AI経由の実訪問（GA4・セッション）")
    now = fetch(["sessionSource", "sessionMedium"], ["sessions", "engagementRate"], start, end)
    prev = fetch(["sessionSource", "sessionMedium"], ["sessions", "engagementRate"], prev_start, prev_end)
    now_ai = {k: v for k, v in now.items() if is_ai_source(k[0], k[1])}
    prev_ai = {k: v for k, v in prev.items() if is_ai_source(k[0], k[1])}
    if not now_ai and not prev_ai:
        print("  （AI経由のセッションは0件）")
    else:
        for key, vals in sorted(now_ai.items(), key=lambda kv: -kv[1][0])[:limit]:
            before = prev_ai.get(key, [0, 0])[0]
            print(f"  {int(vals[0]):>7,}件  エンゲージ {vals[1] * 100:>5.1f}%  "
                  f"{key[0]} / {key[1]}{delta(vals[0], before)}")
        total_now = sum(v[0] for v in now_ai.values())
        total_prev = sum(v[0] for v in prev_ai.values())
        all_now = sum(v[0] for v in now.values())
        share = total_now / all_now * 100 if all_now else 0
        print(f"  合計 {int(total_now):,}件（全セッションの{share:.1f}%）{delta(total_now, total_prev)}")

    print("\n■ AI経由の着地ページ")
    land_now = fetch(["landingPagePlusQueryString", "sessionSource", "sessionMedium"],
                     ["sessions"], start, end)
    land_ai: Counter = Counter()
    for key, vals in land_now.items():
        if is_ai_source(key[1], key[2]):
            land_ai[key[0]] += int(vals[0])
    if not land_ai:
        print("  （AI経由の着地ページは0件）")
    else:
        for path, count in land_ai.most_common(limit):
            print(f"  {count:>7,}件  {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GEO（AI検索での引用）レポート")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"集計日数（既定{DEFAULT_DAYS}）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"表示行数（既定{DEFAULT_LIMIT}）")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)
    print(f"GEOレポート: {start}〜{end}（比較: その前の{args.days}日）\n")

    if not sb.is_configured():
        print("[supabase] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です")
    else:
        crawler_sections(args.days, args.limit)
    ga4_section(args.days, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

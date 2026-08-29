"""
tools/gsc_report.py

Google Search Console（GSC）の検索パフォーマンスをAPIで取り出し、
「次にどのページのどこを直せば流入が増えるか」の形にして表示する（手動実行専用）。

なぜ作ったか:
  SEO施策はバックテストできず、判定材料はGSCの前後比較しかない（CLAUDE.md §4 / seo_geo_playbook §8）。
  にもかかわらず、これまでGSCの数字はユーザーのスクリーンショット共有に頼っており、
  「どのクエリで表示されているのに押されていないか」を継続的に見る手段が無かった。

出すもの（すべて前期間との差分付き）:
  1. 全体: クリック / 表示回数 / CTR / 平均掲載順位
  2. ページ種別（記事・銘柄・投資家・ランキング等）別の内訳 … どの型が効いているか
  3. 上位クエリ
  4. CTR改善候補: 10位以内に入っているのにCTRがサイト平均未満のクエリ → titleの書き換え対象
  5. あと一歩: 11〜20位で表示回数が多いクエリ → 加筆・内部リンクで1ページ目を狙う
  6. 上位ページ / 表示のあったURL数（インデックスされて競争できているページ数の下限）

必要な設定（どちらも1回だけ。未設定なら実行時に手順を表示する）:
  - GCPプロジェクトで Google Search Console API を有効化
  - GSC > 設定 > ユーザーと権限 で、サービスアカウントのメールアドレスを「制限付き」で追加
  - 任意: .env の GSC_SITE_URL（既定は sc-domain:kujira-watch.com。URLプレフィックス型なら
    https://kujira-watch.com/ を指定する）

実行:
  python3 tools/gsc_report.py                 # 直近28日 vs その前の28日
  python3 tools/gsc_report.py --days 7        # 期間を変える
  python3 tools/gsc_report.py --limit 30      # 表示行数
  python3 tools/gsc_report.py --sites         # 権限のあるプロパティ一覧（設定確認用）
"""
import argparse
import json
import os
import sys
import urllib.parse
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

from lib import gcp_auth

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

API_ROOT = "https://searchconsole.googleapis.com/webmasters/v3"
SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
DEFAULT_SITE = "sc-domain:kujira-watch.com"
DEFAULT_DAYS = 28
DEFAULT_LIMIT = 20
# GSCの集計は2〜3日遅れる。直近日を入れると「減った」と誤読するので既定で3日前までを見る。
DATA_LAG_DAYS = 3
# APIの1リクエストあたりの上限。ページ数の集計はこの上限に当たると頭打ちになる。
MAX_ROWS = 25000
# CTR改善候補・あと一歩の足切り。表示回数が一桁のクエリはCTRがぶれて判断材料にならない。
MIN_IMPRESSIONS = 20

# ページ種別。ga4_clicks.PAGE_GROUPS と似ているが、SEOではランキング・週次などの
# ハブページを一覧と分けて見たいので別に持つ。
PAGE_GROUPS = (
    ("/articles/", "記事"),
    ("/stocks/", "銘柄ページ"),
    ("/investors/", "投資家ページ"),
    ("/date/", "日付ページ"),
    ("/category/", "カテゴリページ"),
    ("/ranking", "ランキング"),
    ("/weekly", "週次まとめ"),
    ("/monthly", "月次まとめ"),
    ("/trending", "トレンド"),
    ("/activists", "アクティビスト"),
    ("/buybacks", "自社株買い"),
    ("/faq", "FAQ"),
)


def page_group(url: str) -> str:
    path = urllib.parse.urlsplit(url).path or "/"
    if path == "/":
        return "TOP"
    for prefix, label in PAGE_GROUPS:
        if path.startswith(prefix):
            return label
    return "その他"


def explain_error(status: int, payload: dict, site: str) -> str:
    """APIの失敗を「次に何をすれば直るか」に翻訳する。"""
    err = payload.get("error") or {}
    reason = ""
    for d in err.get("details") or []:
        reason = d.get("reason") or reason
    if reason == "SERVICE_DISABLED" or "has not been used in project" in (err.get("message") or ""):
        return ("Google Search Console API がGCPプロジェクトで無効です。\n"
                "  → https://console.cloud.google.com/apis/library/searchconsole.googleapis.com "
                "で有効化してください（数分で反映）")
    if status in (403, 401):
        return (f"プロパティ {site} を読む権限がありません。\n"
                f"  → GSC > 設定 > ユーザーと権限 > ユーザーを追加 で `{gcp_auth.client_email()}` を"
                "「制限付き」で追加してください")
    if status == 404:
        return (f"プロパティ {site} が見つかりません。\n"
                "  → `python3 tools/gsc_report.py --sites` で権限のあるプロパティ名を確認し、"
                ".env の GSC_SITE_URL に設定してください")
    return f"HTTP {status}: {err.get('message') or json.dumps(payload)[:300]}"


def _call(method: str, path: str, token: str, site: str, body: "dict | None" = None) -> tuple:
    """(payload, error_message)。"""
    resp = requests.request(method, f"{API_ROOT}{path}",
                            headers={"Authorization": f"Bearer {token}"}, json=body, timeout=60)
    if resp.status_code != 200:
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": {"message": resp.text[:300]}}
        return {}, explain_error(resp.status_code, payload, site)
    return resp.json(), ""


def list_sites(token: str) -> tuple:
    payload, err = _call("GET", "/sites", token, "-")
    return (payload.get("siteEntry") or []), err


def search_analytics(token: str, site: str, start: date, end: date,
                     dimensions: list, limit: int = 1000) -> tuple:
    """検索アナリティクスの行を取り出す。行は{keys, clicks, impressions, ctr, position}。"""
    body = {"startDate": start.isoformat(), "endDate": end.isoformat(),
            "dimensions": dimensions, "rowLimit": min(limit, MAX_ROWS)}
    payload, err = _call("POST", f"/sites/{urllib.parse.quote(site, safe='')}/searchAnalytics/query",
                         token, site, body)
    return (payload.get("rows") or []), err


def totals(rows: list) -> dict:
    """行を合計して全体指標にする。CTRと平均掲載順位は表示回数で加重する
    （行ごとの単純平均にすると、表示1回のクエリが上位表示のページと同じ重みになる）。"""
    clicks = sum(r.get("clicks", 0) for r in rows)
    impressions = sum(r.get("impressions", 0) for r in rows)
    weighted_pos = sum(r.get("position", 0) * r.get("impressions", 0) for r in rows)
    return {"clicks": clicks, "impressions": impressions,
            "ctr": clicks / impressions if impressions else 0.0,
            "position": weighted_pos / impressions if impressions else 0.0}


def group_totals(rows: list) -> dict:
    """URL行をページ種別に畳む。{種別: 全体指標}。"""
    buckets = {}
    for r in rows:
        buckets.setdefault(page_group(r["keys"][0]), []).append(r)
    return {k: totals(v) for k, v in buckets.items()}


def ctr_opportunities(rows: list, site_ctr: float, min_impressions: int = MIN_IMPRESSIONS) -> list:
    """1ページ目に入っているのにCTRがサイト平均未満の行。titleの書き換えで最も早く効く層。"""
    picked = [r for r in rows
              if r.get("impressions", 0) >= min_impressions
              and r.get("position", 99) <= 10.0
              and r.get("ctr", 0) < site_ctr]
    return sorted(picked, key=lambda r: -(r["impressions"] * (site_ctr - r["ctr"])))


def almost_first_page(rows: list, min_impressions: int = MIN_IMPRESSIONS) -> list:
    """11〜20位で表示回数が多い行。加筆・内部リンクで1ページ目に入れば流入になる。"""
    picked = [r for r in rows
              if r.get("impressions", 0) >= min_impressions
              and 10.0 < r.get("position", 99) <= 20.0]
    return sorted(picked, key=lambda r: -r["impressions"])


def delta(now: float, before: float) -> str:
    """前期間との差。0→正の増加は「新規」と出す（%にすると∞になり読めない）。"""
    if before == 0:
        return "  (新規)" if now else ""
    return f"  ({(now - before) / before * 100:+.0f}%)"


def _row_line(label: str, r: dict, width: int = 44) -> str:
    return (f"  {r['clicks']:5,.0f}クリック  表示{r['impressions']:7,.0f}  "
            f"CTR{r['ctr'] * 100:5.1f}%  {r['position']:5.1f}位  {label[:width]}")


def report(days: int, limit: int) -> int:
    site = os.getenv("GSC_SITE_URL", "").strip() or DEFAULT_SITE
    try:
        token = gcp_auth.access_token(SCOPE)
    except Exception as e:
        print(f"[gsc] サービスアカウント鍵の読み込みに失敗: {e}")
        return 1
    if not token:
        print(f"[gsc] サービスアカウント鍵が見つかりません: {gcp_auth.credentials_path()}")
        return 1

    end = date.today() - timedelta(days=DATA_LAG_DAYS)
    start = end - timedelta(days=days - 1)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    print(f"GSC 検索パフォーマンス: {start}〜{end}（比較: {prev_start}〜{prev_end}） site: {site}")

    queries, err = search_analytics(token, site, start, end, ["query"], MAX_ROWS)
    if err:
        print(f"[gsc] {err}")
        return 1
    prev_queries, _ = search_analytics(token, site, prev_start, prev_end, ["query"], MAX_ROWS)
    pages, _ = search_analytics(token, site, start, end, ["page"], MAX_ROWS)
    prev_pages, _ = search_analytics(token, site, prev_start, prev_end, ["page"], MAX_ROWS)

    now, before = totals(queries), totals(prev_queries)
    print("\n■ 全体")
    print(f"  クリック      {now['clicks']:8,.0f}{delta(now['clicks'], before['clicks'])}")
    print(f"  表示回数      {now['impressions']:8,.0f}{delta(now['impressions'], before['impressions'])}")
    print(f"  CTR             {now['ctr'] * 100:6.2f}%  (前期 {before['ctr'] * 100:.2f}%)")
    print(f"  平均掲載順位    {now['position']:6.1f}   (前期 {before['position']:.1f}) ※小さいほど上位")
    print(f"  表示のあったURL数 {len(pages):6,}{delta(len(pages), len(prev_pages))}"
          "  ※インデックスされて検索に出ているページ数の下限"
          + ("（API上限に到達。実数はこれ以上）" if len(pages) >= MAX_ROWS else ""))

    print("\n■ ページ種別")
    prev_groups = group_totals(prev_pages)
    for label, t in sorted(group_totals(pages).items(), key=lambda kv: -kv[1]["impressions"]):
        print(_row_line(label, t) + delta(t["clicks"], (prev_groups.get(label) or {}).get("clicks", 0)))

    print(f"\n■ 上位クエリ（クリック順・上位{limit}）")
    prev_q = {r["keys"][0]: r for r in prev_queries}
    for r in sorted(queries, key=lambda r: (-r["clicks"], -r["impressions"]))[:limit]:
        print(_row_line(r["keys"][0], r) + delta(r["clicks"], (prev_q.get(r["keys"][0]) or {}).get("clicks", 0)))

    print(f"\n■ CTR改善候補（10位以内・表示{MIN_IMPRESSIONS}回以上・CTRがサイト平均{now['ctr'] * 100:.1f}%未満）"
          "→ titleとdescriptionの書き換え対象")
    opportunities = ctr_opportunities(queries, now["ctr"])[:limit]
    for r in opportunities:
        gain = r["impressions"] * (now["ctr"] - r["ctr"])
        print(_row_line(r["keys"][0], r) + f"  → 平均CTRなら+{gain:.0f}クリック")
    if not opportunities:
        print("  該当なし")

    print(f"\n■ あと一歩（11〜20位・表示回数順・上位{limit}）→ 加筆・内部リンクで1ページ目を狙う")
    nearly = almost_first_page(queries)[:limit]
    for r in nearly:
        print(_row_line(r["keys"][0], r))
    if not nearly:
        print("  該当なし")

    print(f"\n■ 上位ページ（クリック順・上位{limit}）")
    prev_p = {r["keys"][0]: r for r in prev_pages}
    for r in sorted(pages, key=lambda r: (-r["clicks"], -r["impressions"]))[:limit]:
        url = urllib.parse.unquote(urllib.parse.urlsplit(r["keys"][0]).path)
        print(_row_line(url, r) + delta(r["clicks"], (prev_p.get(r["keys"][0]) or {}).get("clicks", 0)))
    return 0


def show_sites() -> int:
    try:
        token = gcp_auth.access_token(SCOPE)
    except Exception as e:
        print(f"[gsc] サービスアカウント鍵の読み込みに失敗: {e}")
        return 1
    sites, err = list_sites(token)
    if err:
        print(f"[gsc] {err}")
        return 1
    if not sites:
        print(f"[gsc] 権限のあるプロパティがありません。GSC > 設定 > ユーザーと権限 で "
              f"`{gcp_auth.client_email()}` を「制限付き」で追加してください")
        return 1
    for s in sites:
        print(f"  {s.get('permissionLevel', '?'):20} {s.get('siteUrl', '')}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Search Consoleの検索パフォーマンスを取得して前期間と比較する")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--sites", action="store_true", help="権限のあるプロパティ一覧を表示する")
    a = p.parse_args()
    sys.exit(show_sites() if a.sites else report(a.days, a.limit))


if __name__ == "__main__":
    main()

"""
tools/seo_loop.py

毎朝のSEOループの「Read」段: Search Console を6つのフラグだけで読み、
GA4 の着地ページ別の行動（CTAクリック）と突き合わせて「直す／書く／捨てる」の候補を出す。

なぜ作ったか（2026-09-25）:
  `tools/gsc_report.py` は28日の全体像を見る手動レポートで、毎朝見るには情報が多すぎた。
  毎日回すには「動くべき行だけ」に絞る必要がある。フラグ以外は出さない（ignore everything else）。
  加えてGSCはクリックまでしか見えず、着地後に何かを押したかは分からない。クリックが多いのに
  誰もCTAを押さないページは「勝ち」ではなく「漏れ」なので、GA4の landingPage 別クリックと並べる。

6つのフラグ（しきい値は下の定数）:
  almost_there : 3〜20位・非ブランドのクエリ → そのページに完全一致フレーズを1文足す
  no_clicks    : 表示500回超かつCTR0.5%未満のページ → title/description/H1/冒頭に完全一致クエリ
  decaying     : ページのクリックが前週比30%以上減 → 原因確認（順位低下 or 表示減）
  untargeted   : 上位に出ているのにページのtitleにクエリ語が1つも無い → 既存ページに加筆 or 新規ページ
  wrong_intent : 比較/計算/定義クエリなのにページ種別が合っていない → ページ型を変える
  ai_mode      : 7語以上（日本語は空白区切りが無いので20字以上）のクエリ → その言い回しで見出しを作る
  ＋ leak / converts : GSCクリックに対するGA4 CTAクリック（着地セッション起点）の多寡

実行:
  python3 tools/seo_loop.py                 # 直近7日 vs その前の7日
  python3 tools/seo_loop.py --limit 10      # 各フラグの表示行数
  python3 tools/seo_loop.py --no-titles     # untargeted 用のtitle取得（HTTP）を省く
  python3 tools/seo_loop.py --out logs/seo_loop.md   # Markdownでファイルにも書く（Actionsのサマリ用）

必要な設定は gsc_report.py / ga4_clicks.py と同じ（GA4_PROPERTY_ID が無ければ突き合わせ節だけ省く）。
Anthropic API は使わない。
"""
import argparse
import html
import os
import re
import sys
import urllib.parse
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from lib import gcp_auth
from tools import ga4_clicks, gsc_report

WINDOW_DAYS = 7
DEFAULT_LIMIT = 15
# almost_there: 1〜2位は既に取れている。21位以下は1文足しても届かない。
ALMOST_MIN_POS, ALMOST_MAX_POS = 3.0, 20.0
ALMOST_MIN_IMPRESSIONS = 10
# no_clicks: 元記事の値そのまま。サイト規模が小さい間は該当なしが普通。
NO_CLICKS_MIN_IMPRESSIONS = 500
NO_CLICKS_MAX_CTR = 0.005
# decaying: 前週5クリック未満の増減は誤差なので見ない。
DECAY_RATIO = 0.30
DECAY_MIN_PREV_CLICKS = 5
# untargeted: 10位以内に出ているクエリだけ見る（それ以下は「狙って取れていない」だけ）。
UNTARGETED_MAX_POS = 10.0
UNTARGETED_MIN_IMPRESSIONS = 10
UNTARGETED_MAX_FETCH = 30
# ai_mode: 元記事は (\b\w+\b\s){7,}。日本語クエリは空白が少ないので文字数でも拾う。
AI_MODE_MIN_WORDS = 7
AI_MODE_MIN_CHARS = 20
# leak: GSCで週10クリック以上あるのにCTAが1回も押されていない着地ページ。
LEAK_MIN_CLICKS = 10

BRAND_TERMS = ("クジラウォッチ", "くじらウォッチ", "kujira", "kujira-watch", "kujira watch")

# クエリの意図 → 受け止められるページ種別（gsc_report.page_group のラベル）。
# どれにも当たらないクエリは意図判定しない。
INTENT_RULES = (
    ("comparison", ("比較", "違い", " vs ", "vs.", "どっち", "おすすめ", "ランキング"),
     ("ランキング", "記事", "FAQ", "アクティビスト", "トレンド")),
    ("calculator", ("いくら", "計算", "シミュレーション", "何円", "何株"),
     ("銘柄ページ", "FAQ")),
    ("definition", ("とは", "意味", "読み方", "やり方", "見方", "方法"),
     ("FAQ", "記事")),
)

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


# ---------- 判定（純関数。テスト対象） ----------

def is_brand(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in BRAND_TERMS)


def is_ai_mode(query: str) -> bool:
    q = query.strip()
    if len(q.split()) >= AI_MODE_MIN_WORDS:
        return True
    # 文字数の基準は日本語（非ASCIIを含む）クエリだけ。英語は語数で足りる。
    return not q.isascii() and len(q.replace(" ", "").replace("\u3000", "")) >= AI_MODE_MIN_CHARS


def query_intent(query: str) -> "str | None":
    q = f" {query.lower()} "
    for name, words, _ in INTENT_RULES:
        if any(w in q for w in words):
            return name
    return None


def intent_mismatch(query: str, page_url: str) -> "str | None":
    """意図に合わないページ種別なら意図名を返す。"""
    intent = query_intent(query)
    if not intent:
        return None
    allowed = next(pages for name, _, pages in INTENT_RULES if name == intent)
    return intent if gsc_report.page_group(page_url) not in allowed else None


def best_page_per_query(rows: list) -> dict:
    """query×page行から、クエリごとに表示回数最大のページを1つ選ぶ。{query: row}"""
    best = {}
    for r in rows:
        q = r["keys"][0]
        if q not in best or r["impressions"] > best[q]["impressions"]:
            best[q] = r
    return best


def almost_there(qp_rows: list) -> list:
    picked = [r for r in best_page_per_query(qp_rows).values()
              if ALMOST_MIN_POS <= r["position"] <= ALMOST_MAX_POS
              and r["impressions"] >= ALMOST_MIN_IMPRESSIONS
              and not is_brand(r["keys"][0])]
    return sorted(picked, key=lambda r: (r["position"] > 10, -r["impressions"]))


def no_clicks(page_rows: list) -> list:
    picked = [r for r in page_rows
              if r["impressions"] > NO_CLICKS_MIN_IMPRESSIONS and r["ctr"] < NO_CLICKS_MAX_CTR]
    return sorted(picked, key=lambda r: -r["impressions"])


def decaying(page_rows: list, prev_page_rows: list) -> list:
    """[(row_now_or_empty, prev_row, 減少率)]。今週表示が無いページも減少100%として拾う。"""
    now = {r["keys"][0]: r for r in page_rows}
    out = []
    for p in prev_page_rows:
        if p["clicks"] < DECAY_MIN_PREV_CLICKS:
            continue
        cur = now.get(p["keys"][0]) or {"keys": p["keys"], "clicks": 0, "impressions": 0,
                                         "ctr": 0.0, "position": 0.0}
        drop = (p["clicks"] - cur["clicks"]) / p["clicks"]
        if drop >= DECAY_RATIO:
            out.append((cur, p, drop))
    return sorted(out, key=lambda t: -(t[1]["clicks"] - t[0]["clicks"]))


def query_terms(query: str) -> list:
    return [t for t in re.split(r"\s+", query.lower().strip()) if len(t) >= 2]


def untargeted(qp_rows: list, titles: dict) -> list:
    """10位以内なのにページtitleにクエリ語が1つも含まれない（=狙っていないのに出ている）行。
    titleが取れなかったページは判定しない（誤検知より見送りを選ぶ）。"""
    picked = []
    for r in best_page_per_query(qp_rows).values():
        q, page = r["keys"]
        title = titles.get(page)
        if not title or is_brand(q) or r["position"] > UNTARGETED_MAX_POS \
                or r["impressions"] < UNTARGETED_MIN_IMPRESSIONS:
            continue
        terms = query_terms(q)
        if terms and not any(t in title.lower() for t in terms):
            picked.append(r)
    return sorted(picked, key=lambda r: -r["impressions"])


def wrong_intent(qp_rows: list) -> list:
    out = []
    for r in best_page_per_query(qp_rows).values():
        intent = intent_mismatch(r["keys"][0], r["keys"][1])
        if intent and r["impressions"] >= ALMOST_MIN_IMPRESSIONS:
            out.append((r, intent))
    return sorted(out, key=lambda t: -t[0]["impressions"])


def ai_mode(query_rows: list) -> list:
    return sorted([r for r in query_rows if is_ai_mode(r["keys"][0])],
                  key=lambda r: -r["impressions"])


def page_path(url: str) -> str:
    return urllib.parse.unquote(urllib.parse.urlsplit(url).path) or "/"


def join_conversions(page_rows: list, landing: dict) -> list:
    """GSCのページ別クリックとGA4の着地ページ別 [sessions, cta_clicks] を並べる。
    [(path, gsc_clicks, sessions, cta_clicks)]。GA4の landingPage はパスのみなのでパスで突き合わせる。"""
    gsc = {}
    for r in page_rows:
        path = page_path(r["keys"][0]).rstrip("/") or "/"
        gsc[path] = gsc.get(path, 0) + r["clicks"]
    ga = {}
    for path, vals in landing.items():
        key = urllib.parse.unquote(path.split("?")[0]).rstrip("/") or "/"
        s, c = ga.get(key, (0.0, 0.0))
        ga[key] = (s + vals[0], c + (vals[1] if len(vals) > 1 else 0.0))
    return [(p, clicks, *ga.get(p, (0.0, 0.0))) for p, clicks in gsc.items() if clicks > 0]


def leaks(joined: list) -> list:
    return sorted([j for j in joined if j[1] >= LEAK_MIN_CLICKS and j[3] == 0],
                  key=lambda j: -j[1])


def converters(joined: list) -> list:
    """GSCクリック1件あたりのCTAクリックが多い順。「こういうページをもっと書く」の根拠。"""
    return sorted([j for j in joined if j[3] > 0], key=lambda j: -(j[3] / j[1]))


# ---------- 取得 ----------

def fetch_titles(urls: list) -> dict:
    out = {}
    for u in urls[:UNTARGETED_MAX_FETCH]:
        try:
            resp = requests.get(u, timeout=15, headers={"User-Agent": "kujira-seo-loop/1.0"})
            m = _TITLE.search(resp.text) if resp.status_code == 200 else None
            if m:
                out[u] = html.unescape(m.group(1)).strip()
        except requests.RequestException:
            continue
    return out


def fetch_landing(start: date, end: date) -> "tuple[dict, str]":
    """GA4: 着地ページ別の [sessions, CTAクリック数]。CTAクリックは着地セッション内のclickイベント。"""
    property_id = os.getenv("GA4_PROPERTY_ID", "").strip()
    if not property_id:
        return {}, "GA4_PROPERTY_ID が未設定"
    token = ga4_clicks.access_token()
    if not token:
        return {}, "サービスアカウント鍵が見つからない"
    sessions, err = ga4_clicks.run_report(token, property_id, ga4_clicks._period_body(
        ["landingPage"], ["sessions"], start, end, 5000))
    if err:
        return {}, err
    clicks, err = ga4_clicks.run_report(token, property_id, ga4_clicks._period_body(
        ["landingPage"], ["eventCount"], start, end, 5000, ga4_clicks._click_filter()))
    if err:
        return {}, err
    s, c = ga4_clicks.parse_rows(sessions), ga4_clicks.parse_rows(clicks)
    return {p: [v[0], (c.get(p) or [0.0])[0]] for p, v in s.items()}, ""


# ---------- 出力 ----------

def _line(r: dict, label: str) -> str:
    return (f"- {label} — 表示{r['impressions']:,.0f} / クリック{r['clicks']:,.0f} / "
            f"CTR{r['ctr'] * 100:.1f}% / {r['position']:.1f}位")


def render(sections: list) -> str:
    out = []
    for title, action, lines in sections:
        out.append(f"\n## {title}（{len(lines)}件）\n→ {action}")
        out.extend(lines or ["- 該当なし"])
    return "\n".join(out)


def run(limit: int, with_titles: bool, out_path: "str | None") -> int:
    site = os.getenv("GSC_SITE_URL", "").strip() or gsc_report.DEFAULT_SITE
    try:
        token = gcp_auth.access_token(gsc_report.SCOPE)
    except Exception as e:
        print(f"[seo_loop] サービスアカウント鍵の読み込みに失敗: {e}")
        return 1
    if not token:
        print(f"[seo_loop] サービスアカウント鍵が見つかりません: {gcp_auth.credentials_path()}")
        return 1

    end = date.today() - timedelta(days=gsc_report.DATA_LAG_DAYS)
    start = end - timedelta(days=WINDOW_DAYS - 1)
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=WINDOW_DAYS - 1)
    sa = gsc_report.search_analytics
    qp, err = sa(token, site, start, end, ["query", "page"], gsc_report.MAX_ROWS)
    if err:
        print(f"[seo_loop] {err}")
        return 1
    queries, _ = sa(token, site, start, end, ["query"], gsc_report.MAX_ROWS)
    pages, _ = sa(token, site, start, end, ["page"], gsc_report.MAX_ROWS)
    prev_pages, _ = sa(token, site, prev_start, prev_end, ["page"], gsc_report.MAX_ROWS)

    titles = {}
    if with_titles:
        cand = [r for r in best_page_per_query(qp).values()
                if r["position"] <= UNTARGETED_MAX_POS and r["impressions"] >= UNTARGETED_MIN_IMPRESSIONS]
        cand.sort(key=lambda r: -r["impressions"])
        titles = fetch_titles(list(dict.fromkeys(r["keys"][1] for r in cand)))

    sections = [
        ("almost_there（3〜20位・非ブランド）",
         "ランクしているページにクエリそのままの1文を足す。合わなければ新規ページ＋そこからリンク",
         [_line(r, f"「{r['keys'][0]}」 {page_path(r['keys'][1])}") for r in almost_there(qp)[:limit]]),
        (f"no_clicks（表示>{NO_CLICKS_MIN_IMPRESSIONS}・CTR<{NO_CLICKS_MAX_CTR * 100:.1f}%）",
         "title/description/H1/冒頭1文に完全一致クエリ、答えを最初の2行に",
         [_line(r, page_path(r["keys"][0])) for r in no_clicks(pages)[:limit]]),
        (f"decaying（クリック前週比-{DECAY_RATIO * 100:.0f}%以上）",
         "順位が落ちたのか表示が減ったのかをGSCで確認。削除・リダイレクトの巻き込みも疑う",
         [f"- {page_path(p['keys'][0])} — {p['clicks']:,.0f}→{c['clicks']:,.0f}クリック（-{d * 100:.0f}%）"
          f" / 順位 {p['position']:.1f}→{c['position']:.1f}"
          for c, p, d in decaying(pages, prev_pages)[:limit]]),
        ("untargeted（10位以内だがtitleにクエリ語なし）",
         "既存ページに見出しを足すか、そのクエリ専用ページを作って内部リンク",
         [_line(r, f"「{r['keys'][0]}」 {page_path(r['keys'][1])} title: {titles[r['keys'][1]][:40]}")
          for r in untargeted(qp, titles)[:limit]]
         if with_titles else ["- --no-titles のため省略"]),
        ("wrong_intent（クエリ意図とページ種別が不一致）",
         "比較なら比較表、計算なら計算機、定義ならFAQ型。titleを変えても型違いは直らない",
         [_line(r, f"[{intent}]「{r['keys'][0]}」 {page_path(r['keys'][1])}")
          for r, intent in wrong_intent(qp)[:limit]]),
        (f"ai_mode（{AI_MODE_MIN_WORDS}語以上 or {AI_MODE_MIN_CHARS}字以上）",
         "その言い回しのままH2・FAQにする（AI検索の裏クエリ）",
         [_line(r, f"「{r['keys'][0]}」") for r in ai_mode(queries)[:limit]]),
    ]

    # GSCと同じ期間で揃える（期間がずれると同じページのクリックとCTAを並べる意味が無くなる）。
    try:
        landing, ga_err = fetch_landing(start, end)
    except Exception as e:
        landing, ga_err = {}, f"GA4取得に失敗: {e}"
    if ga_err:
        sections.append(("leak / converts（GA4突き合わせ）", "GA4の設定後に有効", [f"- 省略: {ga_err}"]))
    else:
        joined = join_conversions(pages, landing)
        sections.append((f"leak（GSC週{LEAK_MIN_CLICKS}クリック以上・CTAクリック0）",
                         "クリックは勝ちではない。着地直後の導線（CTA位置・関連リンク）を直す",
                         [f"- {p} — GSC{g:,.0f}クリック / 着地{s:,.0f}セッション / CTA 0"
                          for p, g, s, _ in leaks(joined)[:limit]]))
        sections.append(("converts（GSCクリックあたりCTAクリックが多い）",
                         "この型のページを増やす（書く順番はここで決める）",
                         [f"- {p} — GSC{g:,.0f}クリック / CTA{c:,.0f}（{c / g:.2f}/クリック）"
                          for p, g, _, c in converters(joined)[:limit]]))

    head = (f"# SEOループ Read {end}（{start}〜{end} vs {prev_start}〜{prev_end}）\n"
            f"site: {site} / 6フラグ以外は出さない")
    text = head + render(sections)
    print(text)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return 0


def main():
    p = argparse.ArgumentParser(description="Search Consoleを6フラグで読み、GA4の着地後行動と突き合わせる")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--no-titles", action="store_true", help="untargeted用のtitle取得を省く")
    p.add_argument("--out", help="Markdownの出力先")
    a = p.parse_args()
    sys.exit(run(a.limit, not a.no_titles, a.out))


if __name__ == "__main__":
    main()

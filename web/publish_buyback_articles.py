"""
web/publish_buyback_articles.py

自社株買い「決定」開示（tdnet_buybacks。lib/buyback.py が原文PDFから抽出した取得枠）のうち
規模の大きいものを、microCMSブログ「大口投資家の監視ブログ」の記事として自動投稿する
（edinet_blog.yml、平日9:00-21:00 JST毎時。大量保有報告書の記事化 publish_blog_articles.py の後段）。

- 対象: 上限金額 MIN_AMOUNT_OKU 億円以上 または 発行済株式比率 MIN_RATIO_PCT %以上の決定開示。
  月次の取得状況報告（進捗）は tdnet_buybacks に入らないので対象外。
- dealType は「自社株買い」（提出者分類ではなく発行体自身の取得。microCMSのセレクト値・
  kujira-watch の DealType 双方に 2026-08-23 追加）。filerName は付けない（kujira-watch の
  記事ページは filerName があると EDINET の保有比率ブロックを出すため）。
- dealAmount には上限金額（億円）、ratioChangePct には発行済株式比率（上限）を入れる。
  記事ページ側は dealType=自社株買い のとき「（上限）」「発行済比率（上限）」と表示を切り替える。
- 本文は publish_blog_articles.py と同じく事実のみからHaikuに書かせ、推測は「※推測:」で分ける。
  X への個別投稿はしない（平日19:00の web/x_buyback.py「本日の自社株買い決定」が担う）。
- --backfill: 通常運転の窓（DEFAULT_DAYS=3日）を超えて生成が止まると、その期間の決定開示は
  二度と記事化されない（実例: 2026-08-13〜08-20の12件。API月次上限と稼働開始前が重なり
  2026-08-27に手作業で拾った）。--backfill は BACKFILL_DAYS 日まで遡り、既報インデックスに
  無い＝まだ記事が無い開示だけを古い順に拾い直す。edinet_blog.yml の当日最終便が1日1回叩く。
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import requests  # noqa: E402

from lib import api_budget  # noqa: E402
from lib import supabase_client as sb  # noqa: E402
from lib.writing_style import EN_STYLE_RULES, JA_STYLE_RULES, find_ai_tells  # noqa: E402
from web.publish_blog_articles import (  # noqa: E402
    ANTHROPIC_API_KEY,
    BACKFILL_DAYS,
    BACKFILL_MAX_ARTICLES,
    CLAUDE_MODEL,
    MAX_TITLE_LEN,
    MicroCMSPermissionError,
    _microcms_base_url,
    _microcms_headers,
    attach_figures,
    body_char_count,
    body_quality_key,
    build_eyecatch_for_article,
    build_price_chart_for_article,
    fetch_published_index,
    get_company_description,
    get_pit_ranking_snapshot,
    publish_article,
)
from web.article_figures import buyback_article_figures  # noqa: E402

DEAL_TYPE = "自社株買い"
DEFAULT_DAYS = 3
MIN_AMOUNT_OKU = 10.0
MIN_RATIO_PCT = 3.0
MIN_BODY_CHARS = 600


def is_worth_publishing(amount_oku: "float | None", ratio_pct: "float | None") -> bool:
    """記事にする規模か。上限金額か発行済比率のどちらかが閾値以上なら対象。
    小型株は金額が小さくても比率が大きい（需給インパクトが大きい）ため比率でも拾う。"""
    return (amount_oku is not None and amount_oku >= MIN_AMOUNT_OKU) or (
        ratio_pct is not None and ratio_pct >= MIN_RATIO_PCT
    )


def fetch_candidates(days: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    rows = sb.select(
        "tdnet_buybacks",
        f"disclosed_at=gte.{since}&extract_ok=is.true&select=*&order=disclosed_at.desc",
    )
    out = []
    for r in rows:
        amount_oku = round(r["max_amount_yen"] / 1e8, 1) if r.get("max_amount_yen") else None
        ratio = float(r["ratio_pct"]) if r.get("ratio_pct") is not None else None
        if is_worth_publishing(amount_oku, ratio):
            out.append({**r, "amount_oku": amount_oku, "ratio": ratio})
    return out


def published_deal_keys(days: int) -> "set | None":
    """直近days日ぶんの自社株買い記事の (銘柄コード, 開示日) 集合。取得失敗時は None。
    backfillで候補1件ずつ already_published() を叩かずに済ませるための既報一覧。"""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = fetch_published_index(since, extra_filter=f"tags[contains]{DEAL_TYPE}",
                                 fields="stockCode,dealDate")
    if rows is None:
        return None
    return {(str(r.get("stockCode") or ""), str(r.get("dealDate") or "")[:10]) for r in rows}


def already_published(stock_code: str, disc_date: str) -> bool:
    """同じ銘柄・同じ開示日の自社株買い記事が既にあればTrue。同日に同銘柄が決定開示を2本出す
    ことは実務上ない（取得枠と立会外買付を1本で出す）ので、銘柄×日付で十分。

    既報かどうかの判定に dealType を使ってはいけない。microCMSのdealTypeはセレクト型で、
    選択肢に無い値をPOSTしても**エラーにならず空配列で保存される**。「自社株買い」は
    CMS側の選択肢に登録されていないため、dealType[contains]自社株買い は常に0件を返し、
    毎時のedinet_blog.ymlが同じ開示を何度でも新規投稿してしまう
    （実害: 2026-08-25、コンヴァノ6574の同一開示から13本の重複記事が公開された）。
    tags はテキスト型でそのまま保存されるので、そちらで突き合わせる。
    """
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"stockCode[equals]{stock_code}[and]tags[contains]{DEAL_TYPE}"
                           f"[and]dealDate[begins_with]{disc_date}",
                "fields": "id",
                "limit": 1,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            # 判定できないときは重複より取りこぼしを選ばない（既報扱いにしてスキップ）
            print(f"    ⚠ 既報確認に失敗 HTTP {resp.status_code}（スキップ）")
            return True
        return resp.json().get("totalCount", 0) > 0
    except Exception as e:
        print(f"    ⚠ 既報確認に失敗（スキップ）: {e}")
        return True


def stock_name_of(code: str) -> str:
    """銘柄名。jpx_stock_list に無ければTDnetの会社名で補う。

    jpx_stock_list はJPX（東証）の上場銘柄一覧から作っているため、福証・名証単独上場の
    銘柄が載っていない（実例: 8560 宮崎太陽銀行、3066 JBイレブン）。TDnetはそれらの決定開示も
    配信するので、銘柄名が引けないというだけで記事化を永久に見送っていた
    （2026-08-27のbackfillで8560をスキップ）。TDnet側の会社名は取引所の略称（「宮崎太銀」）だが、
    開示を丸ごと落とすよりは略称で載せる。"""
    row = sb.select_one("jpx_stock_list", f"code=eq.{code}&select=name")
    name = (row or {}).get("name") or ""
    if name:
        return name
    from lib.tdnet import fetch_company_name
    name = fetch_company_name(code)
    if name:
        print(f"    ℹ {code}: jpx_stock_list に銘柄名が無いためTDnetの会社名「{name}」を使う")
    return name


def prior_buybacks(code: str, before: str) -> list[dict]:
    """同じ銘柄の過去の決定開示（before より前、新しい順・最大3件）。連続実施の文脈に使う。"""
    # disclosed_at のタイムゾーン部「+00:00」はURLで空白に化けて400になるため秒までで切る
    return sb.select(
        "tdnet_buybacks",
        f"code=eq.{code}&disclosed_at=lt.{before[:19]}&extract_ok=is.true"
        "&select=disclosed_at,max_amount_yen,ratio_pct&order=disclosed_at.desc",
        limit=3,
    )


def format_amount(amount_oku: "float | None") -> str:
    if amount_oku is None:
        return ""
    return f"{amount_oku:,.0f}億円" if amount_oku >= 100 else f"{amount_oku:.1f}".rstrip("0").rstrip(".") + "億円"


def build_titles(fact: dict, stock_name_en: str = "") -> dict:
    """検索クエリ型タイトル（ja/en）。「銘柄名（コード） 自社株買い」で検索されるため
    その語を必ず含め、規模（上限金額・比率）を続ける。publish_blog_articles.build_article_titles と同じく
    LLMに任せず決定的に組む。"""
    name, code = fact["stock_name"], fact["stock_code"]
    amount = format_amount(fact["amount_oku"])
    ratio = fact["ratio"]
    if amount and ratio is not None:
        scale = f"上限{amount}・発行済の{ratio:g}%"
    elif amount:
        scale = f"上限{amount}"
    else:
        scale = f"発行済の{ratio:g}%"
    title = f"{name}（{code}）、{scale}の自社株買いを決定｜TDnet適時開示"
    if len(title) > MAX_TITLE_LEN:
        title = f"{name}（{code}）、{scale}の自社株買いを決定"

    name_en = stock_name_en or name
    yen = fact.get("max_amount_yen")
    if yen:
        amount_en = f"¥{yen/1e9:.1f} billion" if yen >= 1e9 else f"¥{yen/1e6:.0f} million"
        title_en = f"{name_en} ({code}) Approves Share Buyback of Up to {amount_en}"
        if ratio is not None:
            title_en += f" ({ratio:g}% of Shares Outstanding)"
    else:
        title_en = f"{name_en} ({code}) Approves Share Buyback of Up to {ratio:g}% of Shares Outstanding"
    return {"title": title, "titleEn": title_en + " | TDnet Disclosure"}


def build_fact_sheet(row: dict) -> dict:
    code = row["code"]
    disc_date = row["disclosed_at"][:10]
    name = stock_name_of(code)
    snapshot = get_pit_ranking_snapshot(code, disc_date) or {}
    close = snapshot.get("close")
    yen = row.get("max_amount_yen")
    shares = row.get("max_shares")
    # 1株あたりの想定取得単価（上限金額÷上限株数）。開示日終値との比較で「どの水準まで買う前提か」が分かる
    implied_price = round(yen / shares) if yen and shares else None
    return {
        "stock_code": code,
        "stock_name": name,
        "disc_date": disc_date,
        "board_date": row.get("board_date"),
        "amount_oku": row["amount_oku"],
        "max_amount_yen": yen,
        "max_shares": shares,
        "ratio": row["ratio"],
        "period_from": row.get("period_from"),
        "period_to": row.get("period_to"),
        "method": row.get("method"),
        "purpose": row.get("purpose"),
        "will_cancel": bool(row.get("will_cancel")),
        "doc_url": row.get("doc_url"),
        "company_description": get_company_description(code, name) if name else "",
        "context_close": close,
        "implied_price": implied_price,
        "prior": prior_buybacks(code, row["disclosed_at"]),
    }


def _answer_sentence(f: dict) -> str:
    amount = format_amount(f["amount_oku"])
    parts = []
    if amount:
        parts.append(f"上限{amount}")
    if f["max_shares"]:
        parts.append(f"{f['max_shares']:,}株")
    if f["ratio"] is not None:
        parts.append(f"発行済株式総数（自己株式を除く）の{f['ratio']:g}%")
    scale = "・".join(parts)
    return (
        f"{f['stock_name']}（{f['stock_code']}）が{f['disc_date']}、{scale}を上限とする自己株式の取得"
        f"（自社株買い）を決議したことがTDnetの適時開示で分かりました。"
    )


def generate_body(f: dict) -> "dict | None":
    """事実だけからHaikuに本文（ja/en）を書かせる。JSON {body, bodyEn, stockNameEn}。失敗時None。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    lines = [
        f"- 対象銘柄: {f['stock_name']}（{f['stock_code']}）",
        f"- 開示日: {f['disc_date']}" + (f"（取締役会決議日: {f['board_date']}）" if f.get("board_date") and f["board_date"] != f["disc_date"] else ""),
    ]
    if f["amount_oku"] is not None:
        lines.append(f"- 取得価額の総額（上限）: {format_amount(f['amount_oku'])}")
    if f["max_shares"]:
        lines.append(f"- 取得する株式の総数（上限）: {f['max_shares']:,}株")
    if f["ratio"] is not None:
        lines.append(f"- 発行済株式総数（自己株式を除く）に対する割合: {f['ratio']:g}%")
    if f["period_from"] and f["period_to"]:
        period = f["period_from"] if f["period_from"] == f["period_to"] else f"{f['period_from']}〜{f['period_to']}"
        lines.append(f"- 取得期間: {period}")
    if f["method"]:
        lines.append(f"- 取得方法: {f['method']}")
    if f["purpose"]:
        lines.append(f"- 取得の理由（開示の要約）: {f['purpose']}")
    lines.append(f"- 取得した自己株式の消却: {'同時に決議・予定' if f['will_cancel'] else '開示に記載なし'}")
    if f["company_description"]:
        lines.append(f"- {f['stock_name']}の事業内容: {f['company_description']}")
    if f["context_close"] is not None:
        lines.append(f"- 開示日時点の株価: {f['context_close']:,.0f}円")
    if f["implied_price"] and f["context_close"]:
        lines.append(f"- 上限金額÷上限株数の想定取得単価: {f['implied_price']:,}円（開示日終値の{f['implied_price']/f['context_close']*100:.0f}%）")
    for p in f["prior"]:
        amt = f"上限{p['max_amount_yen']/1e8:.1f}億円" if p.get("max_amount_yen") else ""
        rto = f"・{float(p['ratio_pct']):g}%" if p.get("ratio_pct") is not None else ""
        lines.append(f"- 同社の過去の自社株買い決定: {p['disclosed_at'][:10]} {amt}{rto}")

    answer = _answer_sentence(f)
    prompt = f"""以下は日本株の自社株買い（自己株式取得）に関するTDnet適時開示に基づく事実です。この事実だけを根拠に、
投資家向けの解説記事を書いてください。事実にない金額・意図・背景は絶対に創作しないでください。
「上限」はあくまで取締役会が決議した枠であり、実際の取得額がこれを下回ることがある点は1回だけ触れてください。
自社株買い制度そのものの一般的な説明（会社法の条文、ROE向上の一般論など）や、「今後の動向を注視する必要がある」
といった定型的な結びの文は書かないでください。

{JA_STYLE_RULES}

事実:
{chr(10).join(lines)}

本文の1文目は、必ず次の文をそのまま使ってください（検索してきた読者への直答として最初に置く）:
「{answer}」
事業内容の事実がある場合は、1文目の直後にその会社が何をしている会社かを1文で補足してください。
発行済株式比率の事実がある場合は、その比率が株主にとってどれくらいの規模か（例: 発行済の1割なら1株当たり利益が
約1割押し上がる計算）が実感できるよう自然に触れてください。想定取得単価の事実がある場合は、開示日の株価との
関係（現在値より高い水準まで買う前提か等）に1文で触れてください。過去の自社株買い決定の事実があれば、
今回が継続的な株主還元の一環であることが伝わるよう1文で触れてください。
ToSTNeT-3（立会外買付）の場合は、翌営業日の取引開始前に前日終値で一括して買い付ける手法で、
大株主からの売却の受け皿になることが多い、という点を1文で補足してください。

最後に1文だけ、この自社株買いが今後の同社株や株主にとってどんな意味を持ちうるかの推測を加えてください。
必ず文頭に「※推測:」を付けて事実の記述と明確に分け、上記の事実から自然に読み取れる範囲に留めてください。

bodyEnには、上と同じ事実・トーンを保った自然な英語訳を書いてください（英語ネイティブの投資ニュース記事として
自然な文章。1文目は日本語の1文目と同じ内容の直答で始める。※推測の文は "*Speculation:" で始める。
金額は円建てのまま、例 "¥30 billion"）。
{EN_STYLE_RULES}

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください（タイトルは別途テンプレートで
組み立てるため出力しない）。stockNameEnには英語タイトル用のローマ字表記（例: "Lintec"）を短く書いてください:
{{"body": "<p>...</p>形式のHTML本文（600〜900字程度、3〜4段落。最後の段落に※推測文を含む）", "bodyEn": "<p>...</p> HTML body in English, same structure as body", "stockNameEn": "..."}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text, strict=False)
        return data if data.get("body") else None
    except Exception as e:
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
        else:
            print(f"    ⚠ 記事生成失敗: {e}")
        return None


def generate_body_checked(f: dict) -> "dict | None":
    """本文が短すぎる・AI常套句（lib/writing_style.py）を含む場合は1回だけ再生成し、
    マシな方を採用する（publish_blog_articles.generate_article_body_checked と同じ方針）。"""
    first = generate_body(f)
    if first is None:
        return None
    if body_char_count(first["body"]) >= MIN_BODY_CHARS and not find_ai_tells(first["body"]):
        return first
    second = generate_body(f)
    if second is None:
        return first
    if body_quality_key(second["body"], MIN_BODY_CHARS) > body_quality_key(first["body"], MIN_BODY_CHARS):
        return second
    return first


def build_payload(f: dict, article: dict) -> dict:
    titles = build_titles(f, article.get("stockNameEn") or "")
    tags = ["自社株買い"] + (["消却"] if f["will_cancel"] else [])
    payload = {
        "title": titles["title"],
        "body": article["body"],
        "stockName": f["stock_name"],
        "stockCode": f["stock_code"],
        "dealType": [DEAL_TYPE],
        "dealDate": f"{f['disc_date']}T00:00:00.000Z",
        "dealAmount": f["amount_oku"] if f["amount_oku"] is not None else 0,
        "tags": ",".join(tags),
    }
    if f["ratio"] is not None:
        payload["ratioChangePct"] = f["ratio"]
    if f.get("doc_url"):
        payload["sourceUrl"] = f["doc_url"]
    if article.get("bodyEn"):
        payload["titleEn"] = titles["titleEn"]
        payload["bodyEn"] = article["bodyEn"]
    return payload


def build_and_publish(days: int = DEFAULT_DAYS, max_articles: "int | None" = None,
                      dry_run: bool = False, backfill: bool = False) -> list[dict]:
    known: "set | None" = None
    if backfill:
        days = max(days, BACKFILL_DAYS)
        if max_articles is None:
            max_articles = BACKFILL_MAX_ARTICLES
        known = published_deal_keys(days)
        if known is None:
            print("[buyback_blog] 既報インデックスを取得できないため backfill を中止（重複投稿を避ける）")
            return []
    candidates = fetch_candidates(days)
    print(f"[buyback_blog] 記事候補: {len(candidates)}件（上限{MIN_AMOUNT_OKU:g}億円以上 or 発行済{MIN_RATIO_PCT:g}%以上）")
    if backfill:
        candidates = [c for c in candidates if (c["code"], c["disclosed_at"][:10]) not in known]
        # 古い開示ほど窓（BACKFILL_DAYS）から外れて永久に失われるので先に消化する
        candidates.sort(key=lambda c: c["disclosed_at"])
        print(f"[buyback_blog] backfill: 直近{days}日の未記事化候補 {len(candidates)}件")
    published: list[dict] = []
    for row in candidates:
        if max_articles is not None and len(published) >= max_articles:
            break
        code, disc_date = row["code"], row["disclosed_at"][:10]
        if already_published(code, disc_date):
            continue
        f = build_fact_sheet(row)
        if not f["stock_name"]:
            print(f"  - {code} {disc_date}: 銘柄名が取れないためスキップ")
            continue
        amount_label = format_amount(f["amount_oku"]) or "-"
        ratio_label = f"{f['ratio']:g}%" if f["ratio"] is not None else "-"
        print(f"  🏦 {f['stock_name']}({code}) {disc_date} 上限{amount_label} / {ratio_label}")

        article = generate_body_checked(f)
        if not article:
            continue
        payload = build_payload(f, article)
        if dry_run:
            print(f"    [dry-run] title: {payload['title']}\n    {re.sub('<[^>]+>', '', payload['body'])[:200]}…")
            published.append({**payload, "id": None})
            continue

        eyecatch = build_eyecatch_for_article(DEAL_TYPE, {
            "filer_name": f["stock_name"],
            "stock_name": f["stock_name"],
            "holding_ratio": f["ratio"] if f["ratio"] is not None else 0.0,
            "badge_label": "🏦 自社株買い",
            "disc_date": disc_date,
        })
        if eyecatch:
            payload["eyecatch"] = eyecatch
        figure_count = attach_figures(payload, buyback_article_figures(f))

        chart_url = build_price_chart_for_article(code, f["stock_name"])
        if chart_url:
            name = f["stock_name"]
            payload["body"] += (
                f'<figure><img src="{chart_url}" alt="{name}（{code}）株価推移（直近3ヶ月）">'
                f'<figcaption>{name}（{code}）の株価推移（直近3ヶ月・終値ベース）</figcaption></figure>'
            )
            if payload.get("bodyEn"):
                payload["bodyEn"] += (
                    f'<figure><img src="{chart_url}" alt="{name} ({code}) share price, last three months">'
                    f'<figcaption>Share price of {name} ({code}) over the last three months (closing prices).</figcaption></figure>'
                )
            figure_count += 1
        try:
            content_id = publish_article(payload)
        except MicroCMSPermissionError as e:
            print(f"    ⚠ microCMSの権限エラーのため以降の候補をスキップ: {e}")
            break
        if content_id:
            print(f"    ✅ 投稿: {content_id}（図{figure_count}枚）")
            published.append({**payload, "id": content_id})
    return published


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="決定開示を見る直近日数")
    p.add_argument("--max-articles", type=int, default=None, help="1回の実行で投稿する上限件数")
    p.add_argument("--dry-run", action="store_true", help="microCMSへ投稿せず内容を表示するのみ")
    p.add_argument("--backfill", action="store_true",
                   help=f"直近{BACKFILL_DAYS}日まで遡り、記事化されていない決定開示だけを拾い直す")
    args = p.parse_args()
    results = build_and_publish(days=args.days, max_articles=args.max_articles,
                                dry_run=args.dry_run, backfill=args.backfill)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(results)}件処理しました。")


if __name__ == "__main__":
    main()

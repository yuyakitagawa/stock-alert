"""
web/publish_blog_articles.py
EDINET大量保有報告書（買い方向のみ）を基に、microCMSブログ（microcms-blog-demo）へ
解説記事を自動生成・即時公開する（人間は後からmicroCMS管理画面で修正する運用）。

データ源: lib.db.get_edinet_large_holdings_recent（tools/scan_large_holdings.py が
          daily_alert.yml Step 2c で日次蓄積）を web.market_timing_alert 経由でノイズ除外
          （自己申告・過半数超を除外済み）して取得。売り方向（譲渡/売却）は対象外。

金額規模(dealAmount, 億円)の扱い:
  EDINET大量保有報告書は保有"比率(%)"のみで金額は開示されない。このスクリプトでは
  yfinanceの発行済株式数×株価×比率変化で概算し、記事本文にも「推定」と明記する。
  概算できない銘柄（株価・株式数が取得できない）はスキップする（架空の金額を出さない）。

記事生成: Claude（ANTHROPIC_API_KEY）に与えた事実のみから title/body を生成させる
          （事実にない金額・意図の創作を禁止するプロンプト）。

必要な環境変数: MICROCMS_SERVICE_DOMAIN, MICROCMS_API_KEY（書き込み権限）, ANTHROPIC_API_KEY
"""
import os
import re
import sys
import json
import argparse
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from lib.db import get_edinet_large_holdings_recent
from lib.utils import get_price_at_date
from tools.scan_large_holdings import is_sell_disclosure
from web.market_timing_alert import get_recent_large_holdings, LARGE_HOLDINGS_DAYS

load_dotenv()

MICROCMS_DOMAIN = os.getenv("MICROCMS_SERVICE_DOMAIN", "")
MICROCMS_KEY = os.getenv("MICROCMS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MAX_ARTICLES_PER_RUN = 3
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

DOC_TYPE_LABELS = {"350": "大量保有報告書", "360": "変更報告書（保有比率の変更）"}

# EDINET大量保有報告書の提出者からこのパイプラインが選びうるdealType。
# 自社株買い/ETFフローはこのデータ源では判定不能なため含めない（手動投稿用に別途存在）。
FILER_DEAL_TYPES = (
    "インサイダー買い",
    "日系ファンド買い",
    "外資系ファンド買い",
    "ベンチャーキャピタル買い",
    "財団買い",
    "日系企業買い",
    "外資系企業買い",
    "その他",
)

def category_from_deal_type(deal_type: str) -> str:
    """category はdealTypeから「買い」を除いた値（例: 日系ファンド買い→日系ファンド）。
    サイト上部のカテゴリフィルターがdealTypeと同じ粒度で絞り込めるようにするため。"""
    return deal_type[:-2] if deal_type.endswith("買い") else deal_type


def _microcms_base_url() -> str:
    return f"https://{MICROCMS_DOMAIN}.microcms.io/api/v1/articles"


def _microcms_headers() -> dict:
    return {"X-MICROCMS-API-KEY": MICROCMS_KEY, "Content-Type": "application/json"}


def already_published(stock_code: str, disc_date: str) -> bool:
    """同一銘柄・同一開示日の記事が既にmicroCMSにあればTrue（重複投稿防止）。"""
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={"filters": f"stockCode[equals]{stock_code}", "fields": "id,dealDate", "limit": 20},
            timeout=15,
        )
        if resp.status_code != 200:
            return False
        contents = resp.json().get("contents", [])
        return any(str(c.get("dealDate", ""))[:10] == disc_date for c in contents)
    except Exception:
        return False


def shares_outstanding(code: str) -> "float | None":
    import yfinance as yf
    try:
        info = yf.Ticker(f"{code}.T").info
        shares = info.get("sharesOutstanding")
        return float(shares) if shares else None
    except Exception:
        return None


def ratio_change_pct(code: str, filer_name: str, current_ratio: float, disc_date: str) -> float:
    """同一銘柄・同一提出者の過去開示から直近の比率を探し、変化幅(%)を返す。
    過去開示が無ければ、今回の比率をそのまま新規取得分として扱う。"""
    history = get_edinet_large_holdings_recent(days=400, codes=[code])
    past = [
        h for h in history
        if h.get("filer_name") == filer_name
        and h.get("disc_date", "") < disc_date
        and h.get("holding_ratio") is not None
    ]
    if not past:
        return current_ratio
    past.sort(key=lambda h: h["disc_date"])
    prev_ratio = past[-1]["holding_ratio"]
    return abs(current_ratio - prev_ratio)


def estimate_deal_amount_oku(code: str, ratio_change: float, disc_date: str) -> "float | None":
    """推定取得金額（億円） = 比率変化(%) × 発行済株式数 × 株価 ÷ 100 ÷ 1億。
    株式数・株価のいずれかが取得できなければ None（呼び出し側でスキップする）。"""
    if ratio_change <= 0:
        return None
    shares = shares_outstanding(code)
    price = get_price_at_date(code, date.fromisoformat(disc_date))
    if not shares or not price:
        return None
    amount_yen = shares * price * (ratio_change / 100)
    return round(amount_yen / 1e8, 1)


def generate_article_body(fact_sheet: dict) -> "dict | None":
    """Claudeに与えた事実のみからtitle/body/dealTypeを生成させる。
    JSONで {"title", "body", "dealType"} を返す。パース失敗時はNone（記事は投稿しない）。
    dealTypeは提出者名から実体（個人/日系ファンド/外資系ファンド/VC/財団/日系企業/外資系企業）を
    Claudeの一般知識で判断させる。キーワード一致だけでは日系/外資の区別や
    スペース無し個人名（例:「金親晋午」）を正しく判定できないため。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    deal_type_options = "\n".join(f"- {t}" for t in FILER_DEAL_TYPES)
    prompt = f"""以下は日本株の大量保有報告書（EDINET開示）に基づく事実です。この事実だけを根拠に、
投資家向けの解説記事を書いてください。事実にない金額・意図・背景は絶対に創作しないでください。
金額は発行済株式数と株価からの概算であり、実際の取得価格ではないことを本文中で明記してください。

事実:
- 対象銘柄: {fact_sheet['stock_name']}（{fact_sheet['stock_code']}）
- 提出者: {fact_sheet['filer_name']}
- 報告書種別: {fact_sheet['doc_type_label']}
- 保有比率: {fact_sheet['holding_ratio']}%
- 開示日: {fact_sheet['disc_date']}
- 推定取得金額: {fact_sheet['deal_amount_oku']}億円（発行済株式数と株価からの概算）

さらに、提出者名（{fact_sheet['filer_name']}）が何者かを一般知識から判断し、
以下のうち最も当てはまるものを1つだけ選んでください（不明な場合は「その他」）:
{deal_type_options}

判断基準の例:
- 個人名（役員・大株主等）→ インサイダー買い
- 日本国内に拠点を持つ投資ファンド・アセットマネジメント会社 → 日系ファンド買い
- 海外に拠点を持つ投資ファンド・アセットマネジメント会社 → 外資系ファンド買い
- ベンチャーキャピタル（VC、Ventures等）→ ベンチャーキャピタル買い
- 財団・育英会など非営利法人 → 財団買い
- ファンドではない日本の事業会社（自己勘定投資等）→ 日系企業買い
- ファンドではない海外の事業会社 → 外資系企業買い

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"title": "記事タイトル（40字以内）", "body": "<p>...</p>形式のHTML本文（250〜400字程度、2〜3段落）", "dealType": "上記リストから1つ"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if not data.get("title") or not data.get("body"):
            return None
        if data.get("dealType") not in FILER_DEAL_TYPES:
            data["dealType"] = "その他"
        return data
    except Exception as e:
        print(f"    ⚠ 記事生成失敗: {e}")
        return None


class MicroCMSPermissionError(Exception):
    """APIキーの権限不足など、リトライしても直らない投稿エラー。"""


_UNEXPECTED_TYPE_RE = re.compile(r"'(\w+)' has unexpected data type")


def _post_once(payload: dict) -> requests.Response:
    return requests.post(_microcms_base_url(), headers=_microcms_headers(), json=payload, timeout=20)


MAX_TYPE_MISMATCH_RETRIES = 5


def publish_article(payload: dict) -> "str | None":
    """microCMSへPOSTし、成功時はcontent idを返す（失敗時はNone）。
    権限不足（キーにPOST権限が無い等）は MicroCMSPermissionError を送出し、
    呼び出し側で以降の候補すべてをスキップさせる（無駄なClaude呼び出しを防ぐ）。
    セレクトフィールドが複数選択（配列）設定の場合、'has unexpected data type' を
    フィールドごとに検知して配列に包み直し、直るまで（最大 MAX_TYPE_MISMATCH_RETRIES 回）
    再送信する。1回で1フィールドしか教えてくれないAPIなので、複数フィールドが
    ズレていても順番に直していける。"""
    try:
        working_payload = dict(payload)
        fixed_fields = set()
        for _ in range(MAX_TYPE_MISMATCH_RETRIES + 1):
            resp = _post_once(working_payload)
            if resp.status_code in (401, 403) or (
                resp.status_code == 400 and "forbidden" in resp.text.lower()
            ):
                raise MicroCMSPermissionError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            if resp.status_code not in (200, 201) and resp.status_code == 400:
                match = _UNEXPECTED_TYPE_RE.search(resp.text)
                field = match.group(1) if match else None
                if (
                    field and field not in fixed_fields
                    and field in working_payload
                    and isinstance(working_payload[field], str)
                ):
                    working_payload[field] = [working_payload[field]]
                    fixed_fields.add(field)
                    print(f"    ↻ '{field}' を配列形式に変えて再送信します")
                    continue

            if resp.status_code not in (200, 201):
                print(f"    ⚠ 投稿失敗 HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json().get("id")
    except MicroCMSPermissionError:
        raise
    except Exception as e:
        print(f"    ⚠ 投稿例外: {e}")
        return None


def build_and_publish(days: int = LARGE_HOLDINGS_DAYS, max_articles: int = MAX_ARTICLES_PER_RUN,
                       dry_run: bool = False) -> list:
    if not dry_run and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("[publish_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return []

    holdings = get_recent_large_holdings(days=days)
    candidates = [
        h for h in holdings
        if h.get("issuer_code") and h.get("holding_ratio") is not None
        and not is_sell_disclosure(h.get("doc_description") or "")
    ]
    candidates.sort(key=lambda h: abs(h["holding_ratio"]), reverse=True)

    published = []
    for h in candidates:
        if len(published) >= max_articles:
            break
        code = str(h["issuer_code"])
        disc_date = h["disc_date"]
        filer_name = h.get("filer_name", "")
        name = h.get("name") or code

        if already_published(code, disc_date):
            continue

        change = ratio_change_pct(code, filer_name, h["holding_ratio"], disc_date)
        deal_amount = estimate_deal_amount_oku(code, change, disc_date)
        if deal_amount is None:
            print(f"  ⏭ {name}({code}): 金額を概算できないためスキップ")
            continue

        fact_sheet = {
            "stock_name": name,
            "stock_code": code,
            "filer_name": filer_name,
            "doc_type_label": DOC_TYPE_LABELS.get(h.get("doc_type_code", ""), "大量保有関連報告書"),
            "holding_ratio": h["holding_ratio"],
            "disc_date": disc_date,
            "deal_amount_oku": deal_amount,
        }
        article = generate_article_body(fact_sheet)
        if article is None:
            print(f"  ⏭ {name}({code}): 記事生成に失敗したためスキップ")
            continue

        deal_type = article["dealType"]
        payload = {
            "title": article["title"],
            "body": article["body"],
            "stockName": name,
            "stockCode": code,
            "dealType": deal_type,
            "dealDate": f"{disc_date}T00:00:00.000Z",
            "dealAmount": deal_amount,
            "category": category_from_deal_type(deal_type),
            "tags": "EDINET,自動生成",
        }

        if dry_run:
            print(f"  [dry-run] {name}({code}) {disc_date} 推定{deal_amount}億円\n    title: {payload['title']}")
            published.append({**payload, "id": None})
            continue

        try:
            content_id = publish_article(payload)
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため以降の候補もスキップして終了します: {e}")
            break
        if content_id:
            print(f"  ✅ 投稿: {name}({code}) {disc_date} 推定{deal_amount}億円 → id={content_id}")
            published.append({**payload, "id": content_id})
        else:
            print(f"  ⚠ {name}({code}): 投稿に失敗")

    return published


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=LARGE_HOLDINGS_DAYS, help="EDINET開示を見る直近日数")
    p.add_argument("--max-articles", type=int, default=MAX_ARTICLES_PER_RUN, help="1回の実行で投稿する上限件数")
    p.add_argument("--dry-run", action="store_true", help="microCMSへ投稿せず内容を表示するのみ")
    args = p.parse_args()

    results = build_and_publish(days=args.days, max_articles=args.max_articles, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(results)}件処理しました。")


if __name__ == "__main__":
    main()

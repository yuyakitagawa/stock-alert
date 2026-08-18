"""
web/publish_blog_articles.py
EDINET大量保有報告書（買い・売り双方向）を基に、microCMSブログ「大口投資家の監視ブログ」
（kujira-watch/、https://kujira-watch.com/ ）へ解説記事を自動生成・即時公開する（人間は後から
microCMS管理画面で修正する運用）。

データ源: lib.db.get_edinet_large_holdings_recent（tools/scan_large_holdings.py が
          daily_alert.yml Step 2c で日次蓄積）を web.market_timing_alert 経由でノイズ除外
          （自己申告・過半数超・訂正報告書を除外済み）して取得。売り方向（譲渡/売却、
          保有比率の減少）も買いと同様に記事化する（tools.scan_large_holdings.is_sell_disclosure()
          で方向を判定し、売り記事には payload["tags"] に "売り" を付与して区別する。
          microCMSのスキーマ変更を避けるため、既存の自由記述tagsフィールドで方向を表現する）。

金額規模(dealAmount, 億円)の扱い:
  EDINET大量保有報告書は保有"比率(%)"のみで金額は開示されない。このスクリプトでは
  yfinanceの発行済株式数×株価×比率変化で概算し、記事本文にも「推定」と明記する。
  概算できない銘柄（株価・株式数が取得できない）はスキップする（架空の金額を出さない）。

記事生成: Claude（ANTHROPIC_API_KEY）に与えた事実のみから title/body を生成させる
          （事実にない金額・意図の創作を禁止するプロンプト）。対象企業の事業内容
          （get_company_description、web検索で確認しjpx_stock_list.descriptionにキャッシュ）が
          わかれば冒頭紹介と保有比率の規模感に織り込む。本文末尾には「※推測:」ラベル付きの
          推測文を必ず1文加えさせる（事実と明確に区別し、存在しない具体的計画等は創作させない）。

株価チャート: build_price_chart_for_article が直近3ヶ月の終値（yahoo_price_cache）から
          簡易な折れ線チャートPNGをPillowのみで描画し、microCMSへアップロードして
          本文HTML末尾に<img>タグとして埋め込む（失敗時はチャート無しで記事のみ投稿）。

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

import lib.supabase_client as sb
from lib.db import get_edinet_large_holdings_recent
from lib.edinet import disclosure_doc_label, disclosure_kind_label
from lib.utils import get_price_at_date
from tools.scan_large_holdings import is_sell_disclosure
from web.market_timing_alert import get_recent_large_holdings, LARGE_HOLDINGS_DAYS

load_dotenv()

MICROCMS_DOMAIN = os.getenv("MICROCMS_SERVICE_DOMAIN", "")
MICROCMS_KEY = os.getenv("MICROCMS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# EDINET大量保有報告書の提出者分類。edinet_filer_classification（tools/backtest系の
# 投資家分類マスター）と1対1で対応する13分類。自社株買い/ETFフローはこのデータ源では
# 判定不能なため含めない（手動投稿用に別途存在）。
FILER_DEAL_TYPES = (
    "個人",
    "創業家の資産管理会社",
    "公益/一般財団法人",
    "プライムブローカー",
    "アクティビスト",
    "VC",
    "PE・メザニンファンド",
    "独立系ブティックAM",
    "国内アセットマネジメント",
    "外資系伝統運用会社",
    "日系証券銀行",
    "事業会社",
    "その他",
)


def classify_filer(filer_name: str) -> dict:
    """提出者名から投資家カテゴリを判定する。まずedinet_filer_classification
    （Web検索で確認済みの投資家マスター、tools/backtest系の分析でも共用）を引き、
    無ければClaudeの一般知識で判定して結果をマスターに保存する（confidence='low'として、
    後で人手やWeb検索での確認・上書きに備える）。キーワード一致だけでは日系/外資の区別や
    スペース無し個人名（例:「金親晋午」）を正しく判定できないため。"""
    cached = sb.select_one(
        "edinet_filer_classification",
        f"filer_name=eq.{requests.utils.quote(filer_name)}&select=category,is_foreign,description",
    )
    if cached:
        return cached

    import anthropic

    if not ANTHROPIC_API_KEY:
        return {"category": "その他", "is_foreign": False, "description": ""}
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    category_options = "\n".join(f"- {t}" for t in FILER_DEAL_TYPES)
    prompt = f"""以下の投資家（EDINET大量保有報告書の提出者）が何者かを一般知識から判断してください。

提出者名: {filer_name}

以下のうち最も当てはまるカテゴリを1つだけ選んでください（不明な場合は「その他」）:
{category_options}

判断基準の例:
- 個人名（役員・大株主等、スペースの有無に関わらず）→ 個人
- 創業家・オーナー一族の資産管理会社（非上場・実業なし）→ 創業家の資産管理会社
- 公益/一般財団法人 → 公益/一般財団法人
- プライムブローカー業務での保有（Goldman Sachs International等） → プライムブローカー
- エンゲージメント/株主提案/イベントドリブン戦略のファンド → アクティビスト
- ベンチャーキャピタル → VC
- プライベートエクイティ/バイアウト/メザニンファンド → PE・メザニンファンド
- megabank/保険系列以外の国内独立系登録運用会社 → 独立系ブティックAM
- megabank/保険系列の伝統的資産運用会社（信託銀行の信託口含む）→ 国内アセットマネジメント
- 海外拠点の大手分散型資産運用会社（プライムブローカー業務以外） → 外資系伝統運用会社
- 日本の証券会社・銀行本体（資産運用会社ではない） → 日系証券銀行
- 非金融の一般事業会社（内外問わず） → 事業会社

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"category": "上記リストから1つ", "is_foreign": true または false, "description": "何をしている先か1行で"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if data.get("category") not in FILER_DEAL_TYPES:
            data["category"] = "その他"
        result = {
            "category": data["category"],
            "is_foreign": bool(data.get("is_foreign", False)),
            "description": data.get("description", ""),
        }
    except Exception as e:
        # API障害等の一時的な失敗まで「その他」として永続キャッシュすると誤分類が固定化される
        # （実運用で発生: 2026-08-14、課金切れでVC/個人の提出者が軒並み「その他」に上書きされた）。
        # キャッシュせず、次回呼び出し時に再判定させる。
        print(f"    ⚠ 投資家分類に失敗（今回はその他扱い、キャッシュはしない）: {e}")
        return {"category": "その他", "is_foreign": False, "description": ""}

    sb.upsert(
        "edinet_filer_classification",
        [{"filer_name": filer_name, "confidence": "low", **result}],
        on_conflict="filer_name",
    )
    return result


# 投資家分類ごとのPexels検索クエリ（英語の方がPexelsの検索精度が高い）。
# 銘柄固有の写真は現実的に存在しないため、分類のイメージに合う汎用的な金融系写真を使う。
EYECATCH_QUERY_BY_CATEGORY = {
    "個人": "confident businessperson office",
    "創業家の資産管理会社": "family business heritage",
    "公益/一般財団法人": "foundation charity building",
    "プライムブローカー": "wall street trading floor",
    "アクティビスト": "boardroom meeting negotiation",
    "VC": "startup office technology team",
    "PE・メザニンファンド": "corporate merger handshake",
    "独立系ブティックAM": "modern office finance",
    "国内アセットマネジメント": "tokyo financial district",
    "外資系伝統運用会社": "new york stock exchange skyline",
    "日系証券銀行": "bank building japan",
    "事業会社": "corporate headquarters building",
}
EYECATCH_DEFAULT_QUERY = "stock market finance city"

EYECATCH_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
EYECATCH_W, EYECATCH_H = 1200, 630


def search_pexels_photo(query: str) -> "dict | None":
    """Pexels検索APIで1枚選び、{"bytes": 画像本体, "photographer": 撮影者名} を返す。
    未設定・取得失敗時はNone。撮影者名はPexelsのAPI利用ガイドラインが推奨するクレジット表記
    （"Photo by <撮影者> on Pexels"）を画像に焼き込むために保持する。"""
    if not PEXELS_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 5, "orientation": "landscape"},
            timeout=15,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if not photos:
            return None
        photo = photos[0]
        photo_resp = requests.get(photo["src"]["large"], timeout=20)
        photo_resp.raise_for_status()
        return {"bytes": photo_resp.content, "photographer": photo.get("photographer") or "Pexels"}
    except Exception as e:
        print(f"    ⚠ Pexels写真取得失敗: {e}")
        return None


def _wrap_text_lines(draw, text: str, font, max_width: int, max_lines: int = 3) -> list:
    """1文字ずつ幅を測って折り返す（CJKは単語区切りが無いため文字単位で判定する）。"""
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(current)
            current = ch
            if len(lines) >= max_lines:
                break
        else:
            current = trial
    else:
        if current:
            lines.append(current)
    return lines[:max_lines]


def generate_eyecatch_image(category: str, card: dict) -> "bytes | None":
    """投資家分類とニュースカード情報（提出者名・銘柄名・保有比率・売買方向・開示日）から、
    Pexels写真+黒帯+3段組みテキストのアイキャッチPNG(bytes)を生成する。文章タイトルではなく
    「誰が／何を／どれだけ／いつ」を一目で読める構造化カードにすることで、Google Discoverの
    カード面での視認性を上げる狙い。Pexels未設定・取得失敗・合成失敗時はNone
    （呼び出し側は画像なしで記事を投稿する）。"""
    from PIL import Image, ImageDraw, ImageFont
    import io

    query = EYECATCH_QUERY_BY_CATEGORY.get(category, EYECATCH_DEFAULT_QUERY)
    photo = search_pexels_photo(query)
    if not photo:
        return None

    try:
        ss = 2
        w, h = EYECATCH_W * ss, EYECATCH_H * ss
        img = Image.open(io.BytesIO(photo["bytes"])).convert("RGB")
        sw, sh = img.size
        scale = max(w / sw, h / sh)
        nw, nh = int(sw * scale + 0.5), int(sh * scale + 0.5)
        img = img.resize((nw, nh), Image.LANCZOS).crop(
            ((nw - w) // 2, (nh - h) // 2, (nw - w) // 2 + w, (nh - h) // 2 + h)
        )
        draw = ImageDraw.Draw(img, "RGBA")
        pad_left = 80 * ss
        max_text_width = w - pad_left - 60 * ss

        badge_font = ImageFont.truetype(EYECATCH_FONT_PATH, 30 * ss, index=0)
        filer_font = ImageFont.truetype(EYECATCH_FONT_PATH, 52 * ss, index=0)
        stock_font = ImageFont.truetype(EYECATCH_FONT_PATH, 44 * ss, index=0)

        badge_text = f"{card['badge_label']}　{card['disc_date']}"
        filer_lines = _wrap_text_lines(draw, card["filer_name"], filer_font, max_text_width, max_lines=2)
        stock_text = f"{card['stock_name']}　{card['holding_ratio']:.2f}%"
        stock_lines = _wrap_text_lines(draw, stock_text, stock_font, max_text_width, max_lines=2)

        badge_h = int(30 * ss * 1.5)
        filer_line_h = int(52 * ss * 1.3)
        stock_line_h = int(44 * ss * 1.3)
        gap = 16 * ss
        band_h = (
            badge_h + gap
            + filer_line_h * len(filer_lines) + gap
            + stock_line_h * len(stock_lines)
            + 60 * ss
        )
        band_y = (h - band_h) // 2
        band = Image.new("RGBA", (w, band_h), (10, 9, 8, 200))
        img.paste(band, (0, band_y), band)

        y = band_y + 30 * ss
        draw.text((pad_left, y), badge_text, font=badge_font, fill=(255, 200, 80, 255))
        y += badge_h + gap
        for line in filer_lines:
            draw.text((pad_left, y), line, font=filer_font, fill=(255, 255, 255, 255))
            y += filer_line_h
        y += gap
        for line in stock_lines:
            draw.text((pad_left, y), line, font=stock_font, fill=(255, 255, 255, 255))
            y += stock_line_h

        # Pexels API利用ガイドラインが推奨するクレジット表記を右下に小さく焼き込む
        # （ライセンス上は必須ではないが、APIレート制限緩和申請等でも使用実績として示せるようにする）
        credit_font = ImageFont.truetype(EYECATCH_FONT_PATH, 18 * ss, index=0)
        credit_text = f"Photo: {photo['photographer']} / Pexels"
        credit_w = draw.textbbox((0, 0), credit_text, font=credit_font)[2]
        draw.text(
            (w - credit_w - 24 * ss, h - 38 * ss),
            credit_text, font=credit_font, fill=(255, 255, 255, 170),
        )

        img = img.resize((EYECATCH_W, EYECATCH_H), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"    ⚠ アイキャッチ合成失敗: {e}")
        return None


def _upload_media(image_bytes: bytes, filename: str) -> "str | None":
    """microCMSのメディアアップロードAPI(management API)へPNGを送りURLを返す。
    APIキーに「メディアアップロード」権限が無い場合は失敗するので、その場合はNoneを返し
    呼び出し側は画像なしで記事を投稿する。アイキャッチ・株価チャートの両方が使う共通処理。"""
    try:
        resp = requests.post(
            f"https://{MICROCMS_DOMAIN}.microcms-management.io/api/v1/media",
            headers={"X-MICROCMS-API-KEY": MICROCMS_KEY},
            files={"file": (filename, image_bytes, "image/png")},
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"    ⚠ 画像アップロード失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("url")
    except Exception as e:
        print(f"    ⚠ 画像アップロード例外: {e}")
        return None


def upload_eyecatch(image_bytes: bytes) -> "str | None":
    return _upload_media(image_bytes, "eyecatch.png")


def upload_price_chart(image_bytes: bytes) -> "str | None":
    return _upload_media(image_bytes, "chart.png")


CHART_DAYS = 63  # 3ヶ月（営業日）相当
CHART_FONT_PATH = EYECATCH_FONT_PATH


def generate_price_chart_image(code: str, name: str) -> "bytes | None":
    """直近3ヶ月の終値推移をシンプルな折れ線チャートPNGにする（PIL直描画のみ、
    matplotlib等の新規依存を避けるため既存依存のPillowだけで完結させる）。
    株価取得失敗・データ不足時はNone（呼び出し側はチャートなしで本文を使う）。"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    from lib.utils import get_prices

    try:
        prices = get_prices(code, days=100)
        if prices is None or len(prices) < 20:
            return None
        closes = [float(c) for c in prices["Close"].values[-CHART_DAYS:]]
    except Exception as e:
        print(f"    ⚠ チャート用株価取得失敗: {e}")
        return None

    try:
        w, h = 1000, 500
        pad_l, pad_r, pad_t, pad_b = 90, 40, 60, 60
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(CHART_FONT_PATH, 24, index=0)
        small_font = ImageFont.truetype(CHART_FONT_PATH, 18, index=0)

        draw.text((pad_l, 15), f"{name}（{code}） 株価推移（直近3ヶ月）", font=font, fill=(20, 20, 20))

        lo, hi = min(closes), max(closes)
        if hi == lo:
            hi = lo + 1
        plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
        n = len(closes)
        points = []
        for i, c in enumerate(closes):
            x = pad_l + (i / (n - 1)) * plot_w if n > 1 else pad_l
            y = pad_t + plot_h - (c - lo) / (hi - lo) * plot_h
            points.append((x, y))
        draw.line(points, fill=(30, 90, 200), width=3)

        draw.text((10, pad_t - 12), f"{hi:,.0f}円", font=small_font, fill=(90, 90, 90))
        draw.text((10, pad_t + plot_h - 12), f"{lo:,.0f}円", font=small_font, fill=(90, 90, 90))
        draw.text((pad_l, h - pad_b + 20), "3ヶ月前", font=small_font, fill=(90, 90, 90))
        end_label = "直近"
        end_w = draw.textbbox((0, 0), end_label, font=small_font)[2]
        draw.text((w - pad_r - end_w, h - pad_b + 20), end_label, font=small_font, fill=(90, 90, 90))

        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"    ⚠ チャート画像生成失敗: {e}")
        return None


def build_price_chart_for_article(code: str, name: str) -> "str | None":
    """直近3ヶ月の株価チャートを生成・アップロードし、記事本文に埋め込む<img>用URLを返す。
    生成・アップロードのどこかで失敗すればNone（呼び出し側はチャートなしで本文を使う）。"""
    image_bytes = generate_price_chart_image(code, name)
    if not image_bytes:
        return None
    return upload_price_chart(image_bytes)


def build_eyecatch_for_article(category: str, card: dict) -> "dict | None":
    """投資家分類・ニュースカード情報からアイキャッチを生成・アップロードし、microCMSのimage型
    フィールドにそのまま設定できる {"url": ...} を返す。どこかで失敗すればNone
    （画像なしで記事を投稿する）。"""
    if not PEXELS_API_KEY:
        return None
    image_bytes = generate_eyecatch_image(category, card)
    if not image_bytes:
        return None
    url = upload_eyecatch(image_bytes)
    return {"url": url} if url else None


def _microcms_base_url() -> str:
    return f"https://{MICROCMS_DOMAIN}.microcms.io/api/v1/articles"


def _microcms_headers() -> dict:
    return {"X-MICROCMS-API-KEY": MICROCMS_KEY, "Content-Type": "application/json"}


# kujira-watch/src/lib/microcms.ts の FEATURED_POOL_SIZE/FEATURED_COUNT
# （ホームページ「注目」枠 getFeaturedArticles()）と同じ値。ここを変える場合は
# あちらも合わせて変更すること。
FEATURED_POOL_SIZE = 20
FEATURED_COUNT = 3


def get_featured_article_ids(pool_size: int = FEATURED_POOL_SIZE, count: int = FEATURED_COUNT) -> set:
    """kujira-watch側 getFeaturedArticles() と同じロジック（直近pool_size件のプールから
    推定取引金額dealAmountが大きい順に先頭count件を採用）を
    Python側で再現し、現在ホームページで「注目」表示されている記事のidセットを返す。
    X投稿をこれと一致させることで、サイトで目立っていない小粒な開示がXにだけ投稿される
    事態を防ぐ。取得失敗時は空集合（この場合X投稿は0件になる）。"""
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "orders": "-dealDate,-dealAmount",
                "limit": pool_size,
                "fields": "id,dealAmount",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return set()
        contents = resp.json().get("contents", [])
    except Exception as e:
        print(f"  ⚠ 注目記事プール取得失敗: {e}")
        return set()

    contents.sort(key=lambda a: a.get("dealAmount", 0), reverse=True)
    return {a["id"] for a in contents[:count]}


def already_published(stock_code: str, disc_date: str, deal_amount: "float | None" = None,
                      filer_name: str = "", ratio_change: "float | None" = None) -> bool:
    """同一開示の記事が既にmicroCMSにあればTrue（重複投稿防止）。
    突き合わせキーは 銘柄コード＋開示日＋提出者名＋比率変化幅(ratioChangePct)。
    いずれも開示データから決まる値なので、何度再実行しても同じ開示は同じキーになる。

    dealAmountでの突き合わせは、推定金額が株価から都度概算されるため株価キャッシュ更新を
    またぐと全銘柄で±0.05億円を超えてズレ、既報の開示が全て別イベント扱いになる
    （実害: 2026-08-17、daily_alert.ymlの株価更新直後の便で17件が重複投稿）。
    filerName未保存の旧記事（2026-08-16以前）へのフォールバックとしてのみ残す。

    同一銘柄・同日・同一提出者の複数開示も実在する（実例: 2936 2025-08-13に橋本舜が
    9:50と10:07に別々の変更報告書を提出）ため、提出者一致だけでは重複と断定せず
    ratioChangePctの一致まで確認して別イベントを区別する。"""
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"stockCode[equals]{stock_code}",
                "fields": "id,dealDate,dealAmount,filerName,ratioChangePct",
                "limit": 50,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return False
        contents = resp.json().get("contents", [])
        for c in contents:
            if str(c.get("dealDate", ""))[:10] != disc_date:
                continue
            if filer_name and c.get("filerName"):
                if c["filerName"] != filer_name:
                    continue
                if ratio_change is None or c.get("ratioChangePct") is None:
                    return True
                if abs(c["ratioChangePct"] - ratio_change) < 0.01:
                    return True
                continue
            # 旧記事フォールバック: filerNameが無い記事は金額規模で突き合わせる
            if deal_amount is None or c.get("dealAmount") is None:
                return True
            if abs(c["dealAmount"] - deal_amount) < 0.05:
                return True
        return False
    except Exception:
        return False


def shares_outstanding(code: str) -> "float | None":
    """yfinanceの.infoは一時的なレート制限で単発失敗することが多く（2026-08-06の実行では
    価格データが揃っている銘柄（サンリオ等）でもこれが原因で「金額を概算できない」スキップに
    なっていた）、最大3回まで短い間隔でリトライする。J-REIT（投資口）はsharesOutstandingが
    空でimpliedSharesOutstandingに口数が入ることがあるためフォールバックで見る。"""
    import time
    import yfinance as yf
    for attempt in range(3):
        try:
            info = yf.Ticker(f"{code}.T").info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if shares:
                return float(shares)
            return None
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1.5 * (attempt + 1))
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


def get_company_description(code: str, name: str) -> str:
    """対象企業の事業内容を1文程度で返す。jpx_stock_list.descriptionにキャッシュがあれば
    それを使い、無ければClaudeのweb_searchで会社概要を確認して生成しキャッシュする。
    ※一般知識だけで書かせると中小型株の約2/3が「不明（空文字）」になり
    （2026-08-15のバックフィルで生成1,134件に対し不明1,507件）、
    /trendingや/stocks/[code]で事業内容が出ない銘柄が大量に残るため、
    web検索で裏取りさせる。創作を防ぐガード（不明なら空文字）はそのまま維持する。"""
    cached = sb.select_one("jpx_stock_list", f"code=eq.{code}&select=description")
    if cached and cached.get("description"):
        return cached["description"]

    import anthropic

    if not ANTHROPIC_API_KEY:
        return ""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""日本の上場企業「{name}」（証券コード{code}）の事業内容を調べ、
会社四季報の【特色】欄と同程度の密度で説明してください。web_searchで会社概要
（会社公式サイト・IR資料など）を確認してから答えること。

書き方:
- 2〜3文、90〜130字程度。
- 主力事業と、それが売上のどのくらいを占めるか（分かる場合のみ）。
- 主力製品・サービス・ブランドの具体名。
- 特徴（国内シェア・主要販売先・展開地域・設立や上場の経緯など、裏が取れたものだけ）。
- 検索で裏が取れなかった事柄は書かない。数値・シェア・順位を推測で書くことは禁止。
- 株価や投資判断には触れない。

検索しても事業内容が特定できない場合のみ空文字を返してください。

最後に、他のテキストを含まずJSONのみを出力してください:
{{"description": "事業内容の説明、または空文字"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": prompt}],
        )
        # web_search使用時は検索結果ブロックとテキストブロックが交互に並ぶため、
        # 全テキストを連結して末尾のJSONだけを取り出す。
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        matches = re.findall(r'\{[^{}]*"description"[^{}]*\}', text, re.DOTALL)
        # 説明文の途中に生の改行が入ったJSONを返すことがあり、strict=Falseでないと
        # "Invalid control character" で丸ごと落ちる（2026-08-18のバックフィルで8件発生）。
        description = json.loads(matches[-1], strict=False).get("description", "") if matches else ""
        description = " ".join(description.split())
        description = description or ""
    except Exception as e:
        print(f"    ⚠ 事業内容取得に失敗: {e}")
        description = ""

    if description:
        sb.upsert("jpx_stock_list", [{"code": code, "description": description}], on_conflict="code")
    return description


def get_filer_profile(filer_name: str, category: str) -> str:
    """kujira-watch側 /investors/[filer] に表示する投資家プロフィール(日本語800〜1000字程度)を
    返す。edinet_filer_classification.profileにキャッシュがあればそれを使い、無ければClaudeの
    一般知識で生成してキャッシュする（get_company_descriptionと同じ方針。設立時期・運用方針・
    著名な投資事例など、確信が持てる範囲のみ記述させ、役員名や具体的な運用資産額等の検証不能な
    事実は創作させない。一般個人など公開情報が乏しい提出者は空文字を返す想定）。"""
    cached = sb.select_one(
        "edinet_filer_classification",
        f"filer_name=eq.{requests.utils.quote(filer_name)}&select=profile",
    )
    if cached and cached.get("profile"):
        return cached["profile"]

    import anthropic

    if not ANTHROPIC_API_KEY:
        return ""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""日本の株式市場でEDINET大量保有報告書（5%ルール）を提出している投資家
「{filer_name}」（当サイトでの分類: {category}）について、一般知識の範囲で分かることを
800〜1000字程度で説明してください。設立時期・拠点・運用方針や投資スタイル・過去の著名な
投資事例・業界内での位置づけなど、確信が持てる情報のみを記述してください。個人名義の
提出者や情報が乏しい提出者の場合、無理に埋めず分かる範囲だけで構いません（それでも
何も書けない場合は空文字を返してください）。存在しない具体的な事実（役員名・具体的な
運用資産額・未公開の投資判断の理由等）は絶対に創作しないでください。

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"profile": "説明文、または空文字"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1500, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        profile = json.loads(text).get("profile", "") or ""
    except Exception as e:
        print(f"    ⚠ 投資家プロフィール取得に失敗: {e}")
        profile = ""

    if profile:
        # categoryもペイロードに含める: on_conflict時のUPDATEでは不要だが、PostgreSQLは
        # ON CONFLICTのUPDATE分岐に関わらずINSERT側の候補行構築時点でNOT NULL制約を
        # 評価するため、category(NOT NULL)を欠いたペイロードだとUPDATEのみのつもりでも
        # 「null value in column "category"」で失敗する（実運用で発生: 2026-08-10）。
        sb.upsert(
            "edinet_filer_classification",
            [{"filer_name": filer_name, "category": category, "profile": profile}],
            on_conflict="filer_name",
        )
    return profile


def dp_level_label(drop_prob: float) -> str:
    """README記載の5段階表示（高30/やや高22/中14/やや低7）と同じ閾値でdrop_probを
    ラベル化する。Isotonic較正の特性上、小数%そのままだと多数の銘柄が同値に見えるため、
    Web/メール/LINEと同じ粒度で記事にも文脈を渡す。"""
    if drop_prob >= 30:
        return "高"
    if drop_prob >= 22:
        return "やや高"
    if drop_prob >= 14:
        return "中"
    if drop_prob >= 7:
        return "やや低"
    return "低"


def get_pit_ranking_snapshot(code: str, as_of: str) -> "dict | None":
    """開示日(as_of)時点で直近のclose・drop_probをgen_rankingsから取得する。
    記事公開時点（post-hoc、開示後に株価が既に動いた後）のスナップショットを
    「開示当時の文脈」として出すと先読みバイアスになるため、as_of以前の最新行のみを見る
    （CLAUDE.md PIT規律）。"""
    return sb.select_one(
        "gen_rankings",
        f"code=eq.{code}&date=lte.{as_of}&order=date.desc&select=close,drop_prob",
    )


def format_ratio(ratio: float) -> str:
    """保有比率を表示用文字列にする（20.93→"20.93"、5.00→"5"、末尾ゼロは落とす）。"""
    return f"{ratio:.2f}".rstrip("0").rstrip(".")


def is_new_holding(fact_sheet: dict) -> bool:
    """実質的な新規保有か（直近400日に同一提出者の過去開示が無く、変化幅=今回比率になるケース）。"""
    change = fact_sheet.get("ratio_change_pct")
    return change is not None and change >= fact_sheet["holding_ratio"]


# 検索結果で全文が見えるよう、テンプレタイトルはこの長さに収める（超過時は提出者名を短縮する）
MAX_TITLE_LEN = 60


def build_article_titles(fact_sheet: dict, stock_name_en: str = "", filer_name_en: str = "") -> dict:
    """検索クエリ型の記事タイトル（ja/en）を決定的テンプレートで組み立てる（LLM不使用）。
    「銘柄名（コード）」「保有比率」「大量保有報告書」という検索語が必ずタイトルに入ることを
    保証するため、LLMの自由生成をやめてテンプレ化した（SEO/AIO 30日計画 P1）。
    stock_name_en/filer_name_enは英語タイトル用のローマ字名（generate_article_body()が
    本文英訳と一緒に返す。空なら日本語名のまま使う）。"""
    name = fact_sheet["stock_name"]
    code = fact_sheet["stock_code"]
    filer = fact_sheet["filer_name"]
    name_en = stock_name_en or name
    filer_en = filer_name_en or filer
    ratio = format_ratio(fact_sheet["holding_ratio"])
    is_sell = fact_sheet.get("direction") == "sell"

    def ja_title(filer_name: str) -> str:
        if is_new_holding(fact_sheet):
            action = f"{filer_name}が{ratio}%を新規保有"
        else:
            action = f"{filer_name}が保有比率{ratio}%に{'引き下げ' if is_sell else '引き上げ'}"
        return f"{name}（{code}）、{action}｜大量保有報告書"

    title = ja_title(filer)
    if len(title) > MAX_TITLE_LEN:
        excess = len(title) - MAX_TITLE_LEN
        keep = max(4, len(filer) - excess - 1)
        title = ja_title(filer[:keep] + "…")

    if is_new_holding(fact_sheet):
        title_en = f"{filer_en} Takes {ratio}% Stake in {name_en} ({code}) | Large Shareholding Report"
    elif is_sell:
        title_en = f"{filer_en} Cuts Stake in {name_en} ({code}) to {ratio}% | Large Shareholding Report"
    else:
        title_en = f"{filer_en} Raises Stake in {name_en} ({code}) to {ratio}% | Large Shareholding Report"
    return {"title": title, "titleEn": title_en}


def generate_article_body(fact_sheet: dict) -> "dict | None":
    """Claudeに与えた事実のみからbodyを生成させる。JSONで
    {"body", "bodyEn"} を返す（タイトルはbuild_article_titles()で別途組み立てる）。
    本文の1文目は検索クエリへの直答文（銘柄名・コード・提出者・保有比率を含む）に固定する。
    パース失敗時はNone（記事は投稿しない）。
    投資家分類（dealType）は事前にclassify_filer()で判定済み（edinet_filer_classification
    マスター参照＋未登録時のみClaude判定、記事本文生成とは別の呼び出しに分離してキャッシュ
    可能にした）で、その分類の一言説明がfact_sheet['filer_description']としてあれば本文に
    自然に織り込む。titleEn/bodyEnはkujira-watch（/en）の英語版用の英訳で、日本語版と同じ
    事実・トーンを保った自然な英語にする（1回のClaude呼び出しでJA/EN両方を生成し、
    翻訳による事実のズレとAPI呼び出し回数の増加を防ぐ）。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    is_sell = fact_sheet.get("direction") == "sell"
    deal_verb = "売却" if is_sell else "取得"
    deal_amount_label = fact_sheet.get("deal_amount_label") or (
        "推定売却金額" if is_sell else "推定取得金額"
    )
    context_close = fact_sheet.get("context_close")
    context_dp_level = fact_sheet.get("context_dp_level")
    if context_close is not None and context_dp_level is not None:
        context_line = f"- 開示日時点の株価: {context_close:,.0f}円 / 弊社モデルの下落リスク水準: {context_dp_level}\n"
        if is_sell:
            risk_hint = "下落リスクが高まる局面でのリスク回避的な売却か、値上がり後の利益確定売りか、といった観点も交えつつ、"
        else:
            risk_hint = "下落リスクが低い局面での買い増しか、リスクが高い局面での打診買いか、といった観点も交えつつ、"
    else:
        context_line = ""
        risk_hint = ""
    filer_description = fact_sheet.get("filer_description") or ""
    filer_description_line = f"- 提出者について: {filer_description}\n" if filer_description else ""
    company_description = fact_sheet.get("company_description") or ""
    company_description_line = f"- {fact_sheet['stock_name']}の事業内容: {company_description}\n" if company_description else ""
    ratio_change_pct = fact_sheet.get("ratio_change_pct")
    is_new = is_new_holding(fact_sheet)
    if ratio_change_pct is not None:
        if is_new:
            change_line = "- 変化: 直近400日以内に同一提出者による当銘柄の開示が確認できず、今回が実質的な新規保有（または大幅な保有再開）とみられる\n"
        else:
            change_line = f"- 変化: 保有比率はこれまでの開示から{ratio_change_pct:.2f}ポイント{'減少' if is_sell else '増加'}し、今回{fact_sheet['holding_ratio']}%になった\n"
    else:
        change_line = ""
    ratio_str = format_ratio(fact_sheet["holding_ratio"])
    if is_new:
        answer_sentence = (
            f"{fact_sheet['stock_name']}（{fact_sheet['stock_code']}）について、{fact_sheet['filer_name']}が"
            f"同社株式の{ratio_str}%を保有していることが大量保有報告書（EDINET）で分かりました。"
        )
    else:
        answer_sentence = (
            f"{fact_sheet['stock_name']}（{fact_sheet['stock_code']}）について、{fact_sheet['filer_name']}が"
            f"保有比率を{ratio_str}%まで{'引き下げ' if is_sell else '引き上げ'}たことが大量保有報告書（EDINET）で分かりました。"
        )
    prompt = f"""以下は日本株の大量保有報告書（EDINET開示）に基づく事実です。この事実だけを根拠に、
投資家向けの解説記事を書いてください。事実にない金額・意図・背景は絶対に創作しないでください。
この取引は保有比率が{"減少した売却（譲渡等を含む）" if is_sell else "増加した取得（買い増し・新規取得）"}です。
金額が概算であることは見出しの「{deal_amount_label}」表記のみで十分伝わるため、本文中で改めて
「概算であり実際の{deal_verb}価格ではない」等の注記を繰り返さないでください。
大量保有報告書制度そのものの一般的な説明（5%ルールの趣旨、市場透明性・投資家保護目的など）や、
「今後の動向を注視する必要がある」といった、この取引固有ではない定型的な結びの文も書かないでください。

事実:
- 対象銘柄: {fact_sheet['stock_name']}（{fact_sheet['stock_code']}）
- 提出者: {fact_sheet['filer_name']}
- 報告書種別: {fact_sheet['doc_type_label']}
- 保有比率: {fact_sheet['holding_ratio']}%
- 開示日: {fact_sheet['disc_date']}
- {deal_amount_label}: {fact_sheet['deal_amount_oku']}億円（発行済株式数と株価からの概算）
{filer_description_line}{company_description_line}{context_line}{change_line}
本文の1文目は、必ず次の文をそのまま使ってください（検索してきた読者への直答として最初に置く）:
「{answer_sentence}」
提出者について の事実がある場合は、それがどんな種類の投資家かを1文で読者に補足してください
（例: 提出者が海外の資産運用会社なら「海外の資産運用会社による{deal_verb}」等）。無い場合は無理に触れなくてよいです。
{fact_sheet['stock_name']}の事業内容 の事実がある場合は、1文目の直後にその会社が何をしている会社かを
1文で読者に補足してください。また保有比率{fact_sheet['holding_ratio']}%という数字が、対象企業の
株式のどれくらいの規模を占めるかが実感できるよう自然に触れてください（時価総額の一角を占める大株主、等）。
変化 の事実がある場合は、今回の保有比率が一回限りの数字ではなく前回からの推移であることが
伝わるよう、1文で自然に触れてください。

最後に1文だけ、{risk_hint}この{deal_verb}が今後の同社や当該投資家にとってどんな意味を持ちうるかの推測を
加えてください。ただし事実として断定せず、必ず文頭に「※推測:」を付けて、事実の記述とは明確に
分けてください（例: 「※推測: 海外ファンドとの関係強化を通じて新市場開拓を模索している可能性がある」）。
上記の事実（事業内容・提出者の属性・保有比率の規模等）から自然に読み取れる範囲の推測に留め、
事実として存在しない具体的な計画やコメントの引用は創作しないでください。

bodyEnには、上と同じ事実・トーンを保った自然な英語訳を書いてください（直訳調は避け、
英語ネイティブの投資ニュース記事として自然な文章にする）。英語も1文目は日本語の1文目と同じ内容の
直答（例: "A large shareholding report (EDINET filing) shows that ... raised its stake in ... to ...%."）で
始めてください。※推測の文はEnglishでは
"*Speculation:" という接頭辞で始めてください。金額は円建てのまま（例: "¥3.34 billion"）でよいです。

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください（タイトルは別途
テンプレートで組み立てるため出力しない）。stockNameEn/filerNameEnには英語タイトル用の
ローマ字表記（例: "Ain Holdings", "Simplex Asset Management"）を短く書いてください:
{{"body": "<p>...</p>形式のHTML本文（650〜900字程度、3〜4段落。最後の段落に※推測文を含む）", "bodyEn": "<p>...</p> HTML body in English, same structure as body", "stockNameEn": "...", "filerNameEn": "..."}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if not data.get("body"):
            return None
        return data
    except Exception as e:
        print(f"    ⚠ 記事生成失敗: {e}")
        return None


# プロンプトでは650〜900字を指示しているが、実測では中央値445字・全911本が800字未満と
# 指示がまったく守られていなかった（2026-08-18に公開済み記事を全件計測して判明）。
# 指示するだけでは効かないので、生成後に実測して不足なら書き直させる。
MIN_BODY_CHARS = 650


def body_char_count(html: str) -> int:
    """HTMLタグと空白を除いた本文の実文字数（日本語なので文字数がそのまま分量になる）。"""
    return len(re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html or "")))


def generate_article_body_checked(fact_sheet: dict) -> "dict | None":
    """generate_article_body()の結果が短すぎたら1回だけ再生成する。
    2回目も不足していたら、より長い方を採用する（記事を落とすよりは公開する）。"""
    first = generate_article_body(fact_sheet)
    if first is None:
        return None
    first_len = body_char_count(first.get("body", ""))
    if first_len >= MIN_BODY_CHARS:
        return first
    print(f"    ↻ 本文{first_len}字（下限{MIN_BODY_CHARS}字）のため再生成")
    second = generate_article_body(fact_sheet)
    if second is None:
        return first
    return second if body_char_count(second.get("body", "")) > first_len else first


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
    ズレていても順番に直していける。文字列以外（eyecatch等のオブジェクト値）で型不一致に
    なった場合は配列化では直せないため、そのフィールドを除外して再送信する（記事自体を
    投稿失敗させるより、画像等の付随情報なしで本文だけ投稿する方を優先する）。"""
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
                if field and field not in fixed_fields and field in working_payload:
                    if isinstance(working_payload[field], str):
                        working_payload[field] = [working_payload[field]]
                        fixed_fields.add(field)
                        print(f"    ↻ '{field}' を配列形式に変えて再送信します")
                        continue
                    else:
                        del working_payload[field]
                        fixed_fields.add(field)
                        print(f"    ↻ '{field}' の型が不一致のため除外して再送信します")
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


def _patch_once(content_id: str, payload: dict) -> requests.Response:
    return requests.patch(
        f"{_microcms_base_url()}/{content_id}", headers=_microcms_headers(), json=payload, timeout=20,
    )


def update_article(content_id: str, payload: dict) -> bool:
    """既存記事をPATCHで更新する（publish_article()と同じ型不一致リトライを流用）。
    tools/reclassify_blog_articles.py の一括再分類・tools/rewrite_thin_blog_articles.py の
    本文リライトで使う。以前はPATCH権限が無いAPIキーでも動くようPUTを使っていたが、
    2026-08-14にAPIキーの権限が変わりPUTが「Content is already exists. If you want update,
    please use PATCH request.」で拒否されるようになったため切り替えた。PATCHは差分更新のため、
    呼び出し側は変更したいフィールドだけをpayloadに含めればよい（全フィールド送付でも問題ない）。"""
    try:
        working_payload = dict(payload)
        fixed_fields = set()
        for _ in range(MAX_TYPE_MISMATCH_RETRIES + 1):
            resp = _patch_once(content_id, working_payload)
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
                print(f"    ⚠ 更新失敗 HTTP {resp.status_code}: {resp.text[:200]}")
                return False
            return True
    except MicroCMSPermissionError:
        raise
    except Exception as e:
        print(f"    ⚠ 更新例外: {e}")
        return False


# 記事化の足切り。EDINETの変更報告書には「保有比率0.04%・推定取得額0億円」のような
# 実質ニュース価値の無い開示が大量に含まれ、これを記事化するとGoogleに /articles/
# テンプレート全体を低品質と判断され、新規記事がクロールすらされなくなる
# （2026-08-18のGSC「検出 - インデックス未登録」の主因）。
# 表示側の判定は kujira-watch/src/lib/articleIndexability.ts にあり、しきい値は必ず揃えること
# （ずれると「サイトマップに載っているのにnoindex」という矛盾した指示をGoogleに送る）。
MIN_DEAL_AMOUNT_OKU = 3.0
MIN_RATIO_CHANGE_PT = 1.0


def is_worth_publishing(deal_amount_oku: float, ratio_change_pt: float) -> bool:
    """推定金額か保有比率の変化幅のどちらかが基準を超える開示だけを記事にする。"""
    if deal_amount_oku >= MIN_DEAL_AMOUNT_OKU:
        return True
    return abs(ratio_change_pt) >= MIN_RATIO_CHANGE_PT


def build_and_publish(days: int = LARGE_HOLDINGS_DAYS, max_articles: "int | None" = None,
                       dry_run: bool = False) -> list:
    if not dry_run and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("[publish_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return []

    holdings = get_recent_large_holdings(days=days)
    candidates = [
        h for h in holdings
        if h.get("issuer_code") and h.get("holding_ratio") is not None
    ]
    candidates.sort(key=lambda h: abs(h["holding_ratio"]), reverse=True)

    published = []
    for h in candidates:
        if max_articles is not None and len(published) >= max_articles:
            break
        code = str(h["issuer_code"])
        disc_date = h["disc_date"]
        filer_name = h.get("filer_name", "")
        name = h.get("name") or code

        change = ratio_change_pct(code, filer_name, h["holding_ratio"], disc_date)
        deal_amount = estimate_deal_amount_oku(code, change, disc_date)
        if deal_amount is None:
            print(f"  ⏭ {name}({code}): 金額を概算できないためスキップ")
            continue

        is_sell = is_sell_disclosure(
            h.get("doc_description") or "", h.get("holding_ratio"), h.get("holding_ratio_prior")
        )
        direction = "sell" if is_sell else "buy"
        # 保有比率の変化幅(ポイント)。売りは負値で持たせ、フロントのファクトボックスで±表示する。
        # microCMSのratioChangePctと同じ値なので重複判定の突き合わせキーにも使う
        signed_change = round(-change if is_sell else change, 2)

        # 足切りはClaude呼び出し（事業内容・本文生成）より前に置く。API費用も同時に減る。
        if not is_worth_publishing(deal_amount, signed_change):
            print(f"  ⏭ {name}({code}): 推定{deal_amount}億円・比率変化{signed_change}ptで基準未満のためスキップ")
            continue

        if already_published(code, disc_date, deal_amount, filer_name, signed_change):
            continue

        snapshot = get_pit_ranking_snapshot(code, disc_date)
        context_close = snapshot.get("close") if snapshot else None
        context_dp = snapshot.get("drop_prob") if snapshot else None
        filer_info = classify_filer(filer_name)
        company_description = get_company_description(code, name)
        get_filer_profile(filer_name, filer_info["category"])

        fact_sheet = {
            "stock_name": name,
            "stock_code": code,
            "filer_name": filer_name,
            "doc_type_label": disclosure_doc_label(h.get("doc_description"), h.get("doc_type_code", "")),
            "holding_ratio": h["holding_ratio"],
            "disc_date": disc_date,
            "deal_amount_oku": deal_amount,
            "direction": direction,
            "deal_amount_label": "推定売却金額" if is_sell else "推定取得金額",
            "context_close": context_close,
            "context_dp_level": dp_level_label(context_dp) if context_dp is not None else None,
            "filer_description": filer_info.get("description") or "",
            "company_description": company_description,
            "ratio_change_pct": change,
        }
        article = generate_article_body_checked(fact_sheet)
        if article is None:
            print(f"  ⏭ {name}({code}): 記事生成に失敗したためスキップ")
            continue
        titles = build_article_titles(
            fact_sheet,
            stock_name_en=article.get("stockNameEn") or "",
            filer_name_en=article.get("filerNameEn") or "",
        )

        deal_type = filer_info["category"]
        tags = "EDINET,自動生成,売り" if is_sell else "EDINET,自動生成"
        payload = {
            "title": titles["title"],
            "body": article["body"],
            "stockName": name,
            "stockCode": code,
            "dealType": deal_type,
            "dealDate": f"{disc_date}T00:00:00.000Z",
            "dealAmount": deal_amount,
            "ratioChangePct": signed_change,
            "tags": tags,
            "filerName": filer_name,
        }
        if article.get("bodyEn"):
            payload["titleEn"] = titles["titleEn"]
            payload["bodyEn"] = article["bodyEn"]

        direction_mark = "📉売り" if is_sell else "📈買い"
        if dry_run:
            print(f"  [dry-run] {direction_mark} {name}({code}) {disc_date} 推定{deal_amount}億円\n    title: {payload['title']}")
            published.append({**payload, "id": None})
            continue

        if is_sell:
            badge_label = "📉 売却"
        elif disclosure_kind_label(h.get("doc_description"), h.get("doc_type_code", "")) == "新規":
            badge_label = "📈 新規取得"
        else:
            badge_label = "📈 買い増し"
        eyecatch = build_eyecatch_for_article(deal_type, {
            "filer_name": filer_name,
            "stock_name": name,
            "holding_ratio": h["holding_ratio"],
            "badge_label": badge_label,
            "disc_date": disc_date,
        })
        if eyecatch:
            payload["eyecatch"] = eyecatch

        chart_url = build_price_chart_for_article(code, name)
        if chart_url:
            payload["body"] += f'<p><img src="{chart_url}" alt="{name}（{code}）株価推移（直近3ヶ月）"></p>'

        try:
            content_id = publish_article(payload)
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため以降の候補もスキップして終了します: {e}")
            break
        if content_id:
            print(f"  ✅ 投稿: {direction_mark} {name}({code}) {disc_date} 推定{deal_amount}億円 → id={content_id}")
            published.append({**payload, "id": content_id})
        else:
            print(f"  ⚠ {name}({code}): 投稿に失敗")

    return published


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=LARGE_HOLDINGS_DAYS, help="EDINET開示を見る直近日数")
    p.add_argument("--max-articles", type=int, default=None, help="1回の実行で投稿する上限件数（未指定なら上限なし）")
    p.add_argument("--dry-run", action="store_true", help="microCMSへ投稿せず内容を表示するのみ")
    args = p.parse_args()

    results = build_and_publish(days=args.days, max_articles=args.max_articles, dry_run=args.dry_run)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(results)}件処理しました。")

    if not args.dry_run:
        from web.x_client import post_daily_summary, post_top_articles
        featured_ids = get_featured_article_ids()
        posted = post_top_articles(results, featured_ids)
        if posted:
            print(f"🐦 X投稿: {posted}件")
        # 「本日のクジラ」日次サマリー(21時JSTの最終便のみ投稿される。時刻ガードはx_client側)
        post_daily_summary()


if __name__ == "__main__":
    main()

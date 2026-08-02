"""
web/publish_blog_articles.py
EDINET大量保有報告書（買い方向のみ）を基に、microCMSブログ「大口投資家の監視ブログ」
（kujira-watch/、https://kujira-watch.com/ ）へ解説記事を自動生成・即時公開する（人間は後から
microCMS管理画面で修正する運用）。

データ源: lib.db.get_edinet_large_holdings_recent（tools/scan_large_holdings.py が
          daily_alert.yml Step 2c で日次蓄積）を web.market_timing_alert 経由でノイズ除外
          （自己申告・過半数超を除外済み）して取得。売り方向（譲渡/売却）は対象外。

金額規模(dealAmount, 億円)の扱い:
  EDINET大量保有報告書は保有"比率(%)"のみで金額は開示されない。このスクリプトでは
  yfinanceの発行済株式数×株価×比率変化で概算し、記事本文にも「推定」と明記する。
  概算できない銘柄（株価・株式数が取得できない）はスキップする（架空の金額を出さない）。

記事生成: Claude（ANTHROPIC_API_KEY）に与えた事実のみから title/body を生成させる
          （事実にない金額・意図の創作を禁止するプロンプト）。対象企業の事業内容
          （get_company_description、一般知識ベースでjpx_stock_list.descriptionにキャッシュ）が
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
from lib.utils import get_price_at_date
from tools.scan_large_holdings import is_sell_disclosure
from web.market_timing_alert import get_recent_large_holdings, LARGE_HOLDINGS_DAYS

load_dotenv()

MICROCMS_DOMAIN = os.getenv("MICROCMS_SERVICE_DOMAIN", "")
MICROCMS_KEY = os.getenv("MICROCMS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

DOC_TYPE_LABELS = {"350": "大量保有報告書", "360": "変更報告書（保有比率の変更）"}

# EDINET大量保有報告書の提出者分類。edinet_filer_classification（tools/backtest系の
# 投資家分類マスター）と1対1で対応する14分類。自社株買い/ETFフローはこのデータ源では
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
        print(f"    ⚠ 投資家分類に失敗（その他扱い）: {e}")
        result = {"category": "その他", "is_foreign": False, "description": ""}

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


def generate_eyecatch_image(title: str, category: str) -> "bytes | None":
    """タイトル文字列と投資家分類から、Pexels写真+黒帯+太字白文字のアイキャッチPNG(bytes)を
    生成する。Pexels未設定・取得失敗・合成失敗時はNone（呼び出し側は画像なしで記事を投稿する）。"""
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
        font = ImageFont.truetype(EYECATCH_FONT_PATH, 56 * ss, index=0)
        pad_left = 80 * ss
        max_text_width = w - pad_left - 60 * ss
        lines = _wrap_text_lines(draw, title, font, max_text_width)

        line_h = int(56 * ss * 1.35)
        band_h = line_h * len(lines) + 60 * ss
        band_y = (h - band_h) // 2
        band = Image.new("RGBA", (w, band_h), (10, 9, 8, 200))
        img.paste(band, (0, band_y), band)

        y = band_y + 30 * ss
        for line in lines:
            draw.text((pad_left, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_h

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


def build_eyecatch_for_article(title: str, category: str) -> "dict | None":
    """記事タイトル・分類からアイキャッチを生成・アップロードし、microCMSのimage型フィールドに
    そのまま設定できる {"url": ...} を返す。どこかで失敗すればNone（画像なしで記事を投稿する）。"""
    if not PEXELS_API_KEY:
        return None
    image_bytes = generate_eyecatch_image(title, category)
    if not image_bytes:
        return None
    url = upload_eyecatch(image_bytes)
    return {"url": url} if url else None


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


def get_company_description(code: str, name: str) -> str:
    """対象企業の事業内容を1文程度で返す。jpx_stock_list.descriptionにキャッシュがあれば
    それを使い、無ければClaudeの一般知識で生成してキャッシュする（classify_filerと同じ方針。
    Web検索確認済みの投資家分類マスターと異なり事業内容は検証手段が無いため、
    プロンプト側で「不明なら空文字」と念押しし、断定できない場合に創作させない）。"""
    cached = sb.select_one("jpx_stock_list", f"code=eq.{code}&select=description")
    if cached and cached.get("description"):
        return cached["description"]

    import anthropic

    if not ANTHROPIC_API_KEY:
        return ""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""日本の上場企業「{name}」（証券コード{code}）の主な事業内容を、一般知識の範囲で
1文（40字程度）で簡潔に説明してください。確信が持てない場合は空文字を返してください。

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"description": "事業内容の説明、または空文字"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=200, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        description = json.loads(text).get("description", "") or ""
    except Exception as e:
        print(f"    ⚠ 事業内容取得に失敗: {e}")
        description = ""

    if description:
        sb.upsert("jpx_stock_list", [{"code": code, "description": description}], on_conflict="code")
    return description


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


def generate_article_body(fact_sheet: dict) -> "dict | None":
    """Claudeに与えた事実のみからtitle/bodyを生成させる。JSONで {"title", "body"} を返す。
    パース失敗時はNone（記事は投稿しない）。投資家分類（dealType）は事前にclassify_filer()で
    判定済み（edinet_filer_classificationマスター参照＋未登録時のみClaude判定、記事本文生成とは
    別の呼び出しに分離してキャッシュ可能にした）で、その分類の一言説明が
    fact_sheet['filer_description']としてあれば本文に自然に織り込む。"""
    import anthropic

    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    context_close = fact_sheet.get("context_close")
    context_dp_level = fact_sheet.get("context_dp_level")
    if context_close is not None and context_dp_level is not None:
        context_line = f"- 開示日時点の株価: {context_close:,.0f}円 / 弊社モデルの下落リスク水準: {context_dp_level}\n"
        risk_hint = "下落リスクが低い局面での買い増しか、リスクが高い局面での打診買いか、といった観点も交えつつ、"
    else:
        context_line = ""
        risk_hint = ""
    filer_description = fact_sheet.get("filer_description") or ""
    filer_description_line = f"- 提出者について: {filer_description}\n" if filer_description else ""
    company_description = fact_sheet.get("company_description") or ""
    company_description_line = f"- {fact_sheet['stock_name']}の事業内容: {company_description}\n" if company_description else ""
    prompt = f"""以下は日本株の大量保有報告書（EDINET開示）に基づく事実です。この事実だけを根拠に、
投資家向けの解説記事を書いてください。事実にない金額・意図・背景は絶対に創作しないでください。
金額が概算であることは見出しの「推定取得金額」表記のみで十分伝わるため、本文中で改めて
「概算であり実際の取得価格ではない」等の注記を繰り返さないでください。
大量保有報告書制度そのものの一般的な説明（5%ルールの趣旨、市場透明性・投資家保護目的など）や、
「今後の動向を注視する必要がある」といった、この取引固有ではない定型的な結びの文も書かないでください。

事実:
- 対象銘柄: {fact_sheet['stock_name']}（{fact_sheet['stock_code']}）
- 提出者: {fact_sheet['filer_name']}
- 報告書種別: {fact_sheet['doc_type_label']}
- 保有比率: {fact_sheet['holding_ratio']}%
- 開示日: {fact_sheet['disc_date']}
- 推定取得金額: {fact_sheet['deal_amount_oku']}億円（発行済株式数と株価からの概算）
{filer_description_line}{company_description_line}{context_line}
提出者について の事実がある場合は、それがどんな種類の投資家かを1文で読者に補足してください
（例: 提出者が海外の資産運用会社なら「海外の資産運用会社による取得」等）。無い場合は無理に触れなくてよいです。
{fact_sheet['stock_name']}の事業内容 の事実がある場合は、冒頭でその会社が何をしている会社かを
1文で読者に補足してください。また保有比率{fact_sheet['holding_ratio']}%という数字が、対象企業の
株式のどれくらいの規模を占めるかが実感できるよう自然に触れてください（時価総額の一角を占める大株主、等）。

最後に1文だけ、{risk_hint}この取得が今後の同社や当該投資家にとってどんな意味を持ちうるかの推測を
加えてください。ただし事実として断定せず、必ず文頭に「※推測:」を付けて、事実の記述とは明確に
分けてください（例: 「※推測: 海外ファンドとの関係強化を通じて新市場開拓を模索している可能性がある」）。
上記の事実（事業内容・提出者の属性・保有比率の規模等）から自然に読み取れる範囲の推測に留め、
事実として存在しない具体的な計画やコメントの引用は創作しないでください。

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください:
{{"title": "記事タイトル（40字以内）", "body": "<p>...</p>形式のHTML本文（500〜700字程度、3〜4段落。最後の段落に※推測文を含む）"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text)
        if not data.get("title") or not data.get("body"):
            return None
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


def _put_once(content_id: str, payload: dict) -> requests.Response:
    return requests.put(
        f"{_microcms_base_url()}/{content_id}", headers=_microcms_headers(), json=payload, timeout=20,
    )


def update_article(content_id: str, payload: dict) -> bool:
    """既存記事をPUTで更新する（publish_article()と同じ型不一致リトライを流用）。
    tools/reclassify_blog_articles.py の一括再分類で使う。PUTは完全上書きのため、
    payloadには変更したいフィールドだけでなく必須項目（title/body/stockName/stockCode/
    dealType/dealDate/dealAmount）を全て含めること（呼び出し側が既存レコード全体を
    取得してから該当フィールドだけ書き換えて渡す想定）。
    PATCH権限が無いAPIキー（'PATCH is forbidden'）でも動くようPUTを使っている。"""
    try:
        working_payload = dict(payload)
        fixed_fields = set()
        for _ in range(MAX_TYPE_MISMATCH_RETRIES + 1):
            resp = _put_once(content_id, working_payload)
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


def build_and_publish(days: int = LARGE_HOLDINGS_DAYS, max_articles: "int | None" = None,
                       dry_run: bool = False) -> list:
    if not dry_run and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("[publish_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return []

    holdings = get_recent_large_holdings(days=days)
    candidates = [
        h for h in holdings
        if h.get("issuer_code") and h.get("holding_ratio") is not None
        and not is_sell_disclosure(
            h.get("doc_description") or "", h.get("holding_ratio"), h.get("holding_ratio_prior")
        )
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

        if already_published(code, disc_date):
            continue

        change = ratio_change_pct(code, filer_name, h["holding_ratio"], disc_date)
        deal_amount = estimate_deal_amount_oku(code, change, disc_date)
        if deal_amount is None:
            print(f"  ⏭ {name}({code}): 金額を概算できないためスキップ")
            continue

        snapshot = get_pit_ranking_snapshot(code, disc_date)
        context_close = snapshot.get("close") if snapshot else None
        context_dp = snapshot.get("drop_prob") if snapshot else None
        filer_info = classify_filer(filer_name)
        company_description = get_company_description(code, name)

        fact_sheet = {
            "stock_name": name,
            "stock_code": code,
            "filer_name": filer_name,
            "doc_type_label": DOC_TYPE_LABELS.get(h.get("doc_type_code", ""), "大量保有関連報告書"),
            "holding_ratio": h["holding_ratio"],
            "disc_date": disc_date,
            "deal_amount_oku": deal_amount,
            "context_close": context_close,
            "context_dp_level": dp_level_label(context_dp) if context_dp is not None else None,
            "filer_description": filer_info.get("description") or "",
            "company_description": company_description,
        }
        article = generate_article_body(fact_sheet)
        if article is None:
            print(f"  ⏭ {name}({code}): 記事生成に失敗したためスキップ")
            continue

        deal_type = filer_info["category"]
        payload = {
            "title": article["title"],
            "body": article["body"],
            "stockName": name,
            "stockCode": code,
            "dealType": deal_type,
            "dealDate": f"{disc_date}T00:00:00.000Z",
            "dealAmount": deal_amount,
            "tags": "EDINET,自動生成",
        }

        if dry_run:
            print(f"  [dry-run] {name}({code}) {disc_date} 推定{deal_amount}億円\n    title: {payload['title']}")
            published.append({**payload, "id": None})
            continue

        eyecatch = build_eyecatch_for_article(article["title"], deal_type)
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
            print(f"  ✅ 投稿: {name}({code}) {disc_date} 推定{deal_amount}億円 → id={content_id}")
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


if __name__ == "__main__":
    main()

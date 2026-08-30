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

解説図: attach_figures が web.article_figures で保有比率の推移・株主構成・提出者の
          ポートフォリオを図にし、本文のその話をしている段落の直後に差し込む
          （文字だけの記事を減らす目的。Pillow直描画なのでAPI課金は発生しない）。

取りこぼしのbackfill: 通常運転は直近LARGE_HOLDINGS_DAYS(3)日の開示しか見ないため、API上限や
          ワークフロー障害で3日を超えて生成が止まると、その期間の開示は二度と記事化されない。
          --backfill は BACKFILL_DAYS(30)日まで遡り、既報インデックス（fetch_published_index）に
          無い＝まだ記事が無い開示だけを古い順に拾い直す。30日窓の候補は1,000件を超えるので、
          推定売買金額ビュー(edinet_holding_amounts)で足切りしてから株価・発行済株式数を引く。
          edinet_blog.yml の当日最終便が1日1回叩く。

必要な環境変数: MICROCMS_SERVICE_DOMAIN, MICROCMS_API_KEY（書き込み権限）, ANTHROPIC_API_KEY
"""
import os
import re
import sys
import json
import argparse
import hashlib
from collections import Counter
from functools import lru_cache
from datetime import date, datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import lib.supabase_client as sb
from lib import api_budget
from lib import api_usage
from lib.db import get_edinet_large_holdings_recent, mark_article_published
from lib.edinet import disclosure_doc_label, disclosure_kind_label, summarize_disposals
from lib.publish_ledger import PublishLedger  # noqa: E402
from lib import publish_ledger as pl  # noqa: E402
from lib.utils import get_price_at_date
from lib.writing_style import JA_STYLE_RULES, find_ai_tells
from web.article_figures import build_article_figures, figure_html, insert_figures_into_body
from tools.scan_large_holdings import is_correction_report, is_sell_disclosure
from web.market_timing_alert import get_recent_large_holdings, LARGE_HOLDINGS_DAYS

load_dotenv()

MICROCMS_DOMAIN = os.getenv("MICROCMS_SERVICE_DOMAIN", "")
MICROCMS_KEY = os.getenv("MICROCMS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# 事業内容・投資家プロフィールを「空文字（不明）」で取りこぼした対象を、何日おきに
# 再挑戦するか。web_searchは $10/1,000検索 + 検索結果が入力トークンとして課金され、
# 1社あたり約$0.05かかる。空振りを記録せず毎回叩き直していたため、バックフィルの
# たびに同じ「不明」社群にフル課金していた（2026-08-15〜18で4回、月次上限に到達）。
# 一度不明だった先が急に判明することは稀なので、四半期に1度だけ再挑戦する。
RECHECK_DAYS = 90

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

    if not ANTHROPIC_API_KEY or api_budget.reached():
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
        api_usage.record(resp, task="classify_filer")
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        data = json.loads(text, strict=False)
        if data.get("category") not in FILER_DEAL_TYPES:
            data["category"] = "その他"
        result = {
            "category": data["category"],
            "is_foreign": bool(data.get("is_foreign", False)),
            "description": data.get("description", ""),
        }
        # 個人の提出者に説明文を持たせない。氏名だけを手がかりにClaudeの一般知識で書かせると、
        # 同姓同名や似た名前の有名人と取り違えた経歴が生成され、それが記事本文に載る。
        # 実害（2026-08-27に発見・是正）: 加藤公一レオ氏（売れるネット広告社の創業者）を
        # 「立憲民主党の衆議院議員」、仲暁子氏を「衆議院議員」、南部靖之氏を「セコム創業者」、
        # 細川馨氏を「元首相」と記載した説明が1,370件中45件に保存され、記事3本に載っていた。
        # 実在の個人についての誤った経歴であり、提出者名以上の情報は載せない。
        if result["category"] == "個人":
            result["description"] = ""
    except Exception as e:
        # API障害等の一時的な失敗まで「その他」として永続キャッシュすると誤分類が固定化される
        # （実運用で発生: 2026-08-14、課金切れでVC/個人の提出者が軒並み「その他」に上書きされた）。
        # キャッシュせず、次回呼び出し時に再判定させる。
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
        else:
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
# 「個人」は以前 confident businessperson office で人物ポートレートを引いていたが、
# 実在の別人の顔写真が個人投資家の氏名と並んで出て本人の写真に見えるため、街並みに変更した。
EYECATCH_QUERY_BY_CATEGORY = {
    "個人": "tokyo business district street",
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
    "自社株買い": "corporate finance treasury office",  # web/publish_buyback_articles.py が使う
}
EYECATCH_DEFAULT_QUERY = "stock market finance city"

EYECATCH_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
EYECATCH_W, EYECATCH_H = 1200, 630


# 検索クエリ→候補写真リストのプロセス内キャッシュ。同じ分類の記事を連続で処理しても
# Pexels検索APIは1クエリにつき1回しか叩かない（無料枠200req/時を使い切らないため）。
_PEXELS_CANDIDATE_CACHE: dict = {}
PEXELS_CANDIDATES = 80  # Pexels APIのper_page上限


def search_pexels_photo(query: str, seed: "str | None" = None) -> "dict | None":
    """Pexels検索APIの候補から1枚選び、{"bytes": 画像本体, "photographer": 撮影者名} を返す。
    未設定・取得失敗時はNone。撮影者名はPexelsのAPI利用ガイドラインが推奨するクレジット表記
    （"Photo by <撮影者> on Pexels"）を画像に焼き込むために保持する。

    seed（記事の識別子）で候補80枚から決定的に1枚を選ぶ。以前は常に photos[0] を使っており、
    同じ分類の記事が全部同じ写真になっていた（実測40記事中8種類、1枚が25%を占有）。
    seedを固定にしているのは、同じ記事を再生成したときに画像が入れ替わらないようにするため。"""
    if not PEXELS_API_KEY:
        return None
    try:
        photos = _PEXELS_CANDIDATE_CACHE.get(query)
        if photos is None:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": PEXELS_API_KEY},
                params={"query": query, "per_page": PEXELS_CANDIDATES, "orientation": "landscape"},
                timeout=15,
            )
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            _PEXELS_CANDIDATE_CACHE[query] = photos
        if not photos:
            return None
        idx = int(hashlib.md5((seed or "").encode("utf-8")).hexdigest(), 16) % len(photos) if seed else 0
        photo = photos[idx]
        photo_resp = requests.get(photo["src"]["large"], timeout=20)
        photo_resp.raise_for_status()
        return {"bytes": photo_resp.content, "photographer": photo.get("photographer") or "Pexels"}
    except Exception as e:
        print(f"    ⚠ Pexels写真取得失敗: {e}")
        return None


# 折り返しで分断したくない文字（半角数字・小数点・カンマ・%）。
# 「13.41%」が「13.」「41%」に割れて別の数字に読めるのを防ぐ。
_UNBREAKABLE_CHARS = set("0123456789.,%")


def _wrap_text_lines(draw, text: str, font, max_width: int, max_lines: int = 3) -> list:
    """1文字ずつ幅を測って折り返す（CJKは単語区切りが無いため文字単位で判定する）。
    数値トークン（13.41% など）の途中では折り返さず、トークンごと次の行へ送る。"""
    lines, current = [], ""
    for ch in text:
        trial = current + ch
        if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            head, tail = current, ch
            # 折り返し位置が数値トークンの内側なら、トークンの先頭まで戻して丸ごと次行へ送る
            if ch in _UNBREAKABLE_CHARS:
                cut = len(head)
                while cut > 0 and head[cut - 1] in _UNBREAKABLE_CHARS:
                    cut -= 1
                if cut > 0:  # 行全体が数値のときは戻さない（無限に送れないため）
                    head, tail = head[:cut], head[cut:] + ch
            lines.append(head)
            current = tail
            if len(lines) >= max_lines:
                break
        else:
            current = trial
    else:
        if current:
            lines.append(current)
    return lines[:max_lines]


# EDINETの提出者名・銘柄名は英数字が全角（例:「ＢＣＰＥ　Ｐａｎｇｅａ　Ｃａｙｍａｎ，　Ｌ．Ｐ．」）で
# 登録されており、そのまま画像に焼き込むと字間が間延びして読みにくく、素人臭い見た目になる。
# kujira-watch/src/lib/format.ts の displayText() と同じ規則の写し。片方だけ変えないこと。
# NFKCは使わない（全角括弧・句読点まで半角化してしまい和文の見た目が崩れるため）。
# 変換するのは表示文字列だけで、DB照合・APIに渡す値は原文のまま使う。
_FULLWIDTH_SYMBOLS = {"．": ".", "，": ",", "＆": "&", "　": " "}


def display_text(text: str) -> str:
    """全角英数字と英文文脈の記号だけを半角へ寄せる。日本語の句読点・中黒・全角括弧は変換しない。"""
    out = []
    for ch in text or "":
        if "０" <= ch <= "９" or "Ａ" <= ch <= "Ｚ" or "ａ" <= ch <= "ｚ":
            out.append(chr(ord(ch) - 0xFEE0))
        else:
            out.append(_FULLWIDTH_SYMBOLS.get(ch, ch))
    return re.sub(r" {2,}", " ", "".join(out)).strip()


def _stock_line_text(card: dict, badge_label: str) -> str:
    """アイキャッチ2段目（銘柄名＋保有比率）の文字列を組み立てる。

    保有比率0%は「全株売却して保有ゼロ」（実データ上ほとんどがこれ）と、自社株買い記事で
    比率を取れなかった場合の既定値0.0の2通りがある。前者は事実として意味があるので
    「全株売却」と書き、後者は数字を出さず銘柄名だけにする。素の「0.00%」はどちらも
    データ欠損に見えるため焼き込まない。badge_labelは絵文字置換後の文字列を受け取る。"""
    ratio = card.get("holding_ratio")
    stock_name = display_text(card["stock_name"])
    if ratio:
        return f"{stock_name}　{ratio:.2f}%"
    if "売却" in badge_label:
        return f"{stock_name}　全株売却"
    return stock_name


def generate_eyecatch_image(category: str, card: dict) -> "bytes | None":
    """投資家分類とニュースカード情報（提出者名・銘柄名・保有比率・売買方向・開示日）から、
    Pexels写真+黒帯+3段組みテキストのアイキャッチPNG(bytes)を生成する。文章タイトルではなく
    「誰が／何を／どれだけ／いつ」を一目で読める構造化カードにすることで、Google Discoverの
    カード面での視認性を上げる狙い。Pexels未設定・取得失敗・合成失敗時はNone
    （呼び出し側は画像なしで記事を投稿する）。"""
    from PIL import Image, ImageDraw, ImageFont
    import io

    query = EYECATCH_QUERY_BY_CATEGORY.get(category, EYECATCH_DEFAULT_QUERY)
    # 記事ごとに候補から別の写真を引くためのseed。同じ記事なら常に同じ写真になる。
    seed = f"{card.get('filer_name', '')}|{card.get('stock_name', '')}|{card.get('disc_date', '')}"
    photo = search_pexels_photo(query, seed=seed)
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

        # バッジ先頭の絵文字（📈📉📝）はNoto Sans CJKに無く豆腐（☒）で描画されるため、
        # フォントが持つ記号に置き換えて焼き込む。
        badge_label = card["badge_label"]
        for emoji, symbol in (("📈", "▲"), ("📉", "▼"), ("📝", "■")):
            badge_label = badge_label.replace(emoji, symbol)
        badge_text = f"{badge_label}　{card['disc_date']}"
        filer_lines = _wrap_text_lines(
            draw, display_text(card["filer_name"]), filer_font, max_text_width, max_lines=2
        )
        stock_text = _stock_line_text(card, badge_label)
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
        # 帯はニュートラルな黒ではなくブランドネイビー(--surface-inverse #16213a)。
        # サイト上の注目カード（ダーク地）と同じ色にして、一覧に並んだときに浮かないようにする。
        band = Image.new("RGBA", (w, band_h), (22, 33, 58, 205))
        img.paste(band, (0, band_y), band)

        y = band_y + 30 * ss
        # バッジ文字はブランドの金(--brand-gold-bright #d9a44f)。以前の(255,200,80)は
        # どのブランド色でもない黄色で、サイトの配色から浮いていた。
        draw.text((pad_left, y), badge_text, font=badge_font, fill=(217, 164, 79, 255))
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
        # 写真ベースなのでPNG（約1MB）ではなくJPEGで保存する（約100〜200KB。
        # microCMSのメディア容量と配信帯域を抑える。表示側はnext/imageが再最適化する）。
        img.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"    ⚠ アイキャッチ合成失敗: {e}")
        return None


def _upload_media(image_bytes: bytes, filename: str, content_type: str = "image/png") -> "str | None":
    """microCMSのメディアアップロードAPI(management API)へ画像を送りURLを返す。
    APIキーに「メディアアップロード」権限が無い場合は失敗するので、その場合はNoneを返し
    呼び出し側は画像なしで記事を投稿する。アイキャッチ・株価チャートの両方が使う共通処理。"""
    try:
        resp = requests.post(
            f"https://{MICROCMS_DOMAIN}.microcms-management.io/api/v1/media",
            headers={"X-MICROCMS-API-KEY": MICROCMS_KEY},
            files={"file": (filename, image_bytes, content_type)},
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
    return _upload_media(image_bytes, "eyecatch.jpg", "image/jpeg")


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


def build_eyecatch_for_article(category: str, card: dict) -> "str | None":
    """投資家分類・ニュースカード情報からアイキャッチを生成・アップロードし、microCMSのimage型
    フィールドに設定するメディアURL（文字列）を返す。どこかで失敗すればNone
    （画像なしで記事を投稿する）。
    microCMSのコンテンツAPIは画像フィールドに {"url": ...} のオブジェクトではなく
    メディアURL文字列を要求する。2026-08-15〜08-22はオブジェクトで送っていたため毎回
    「'eyecatch' has unexpected data type」で除外され、全記事が画像なしで投稿されていた。"""
    if not PEXELS_API_KEY:
        return None
    image_bytes = generate_eyecatch_image(category, card)
    if not image_bytes:
        return None
    return upload_eyecatch(image_bytes)


def _microcms_base_url() -> str:
    return f"https://{MICROCMS_DOMAIN}.microcms.io/api/v1/articles"


def _microcms_headers() -> dict:
    return {"X-MICROCMS-API-KEY": MICROCMS_KEY, "Content-Type": "application/json"}


# microCMSのlimit上限（1リクエストあたり100件）と、既報インデックスで辿るページ数の上限。
# 100×20=2,000記事。30日ぶんの記事は500件前後なので通常は5〜6ページで終わる。
MICROCMS_PAGE_LIMIT = 100
MICROCMS_MAX_PAGES = 20

# 稼働が止まっていた期間の取りこぼしを拾い直すときに遡る日数。
# 通常運転の窓（LARGE_HOLDINGS_DAYS=3日）を超えて生成が止まると、その期間の開示は
# 二度と記事化されない（実例: 2026-08-13〜08-20の自社株買い決定12件。API月次上限と
# 機能の稼働開始前が重なり、2026-08-27に手作業でbackfillした）。
BACKFILL_DAYS = 30

# backfill 1回あたりの投稿上限（--max-articles 未指定時の既定値）。
# 上限なしで走らせるとAnthropic APIの月次上限に一撃で到達し、同じ日に大量の古い記事が並ぶ。
# 15件にしているのは 2026-08-27 の実測から: 直近30日の取りこぼしは約142件（60件サンプルで
# 30件が「本物の取りこぼし」）で、1日1便×15件なら10日で窓（BACKFILL_DAYS=30日）から
# 外れる前に消化しきれる。取りこぼしが解消した後の定常状態では数件しか残らず上限に当たらない。
BACKFILL_MAX_ARTICLES = 15


def fetch_published_index(since_date: str, extra_filter: str = "",
                          fields: str = "stockCode,dealDate,filerName") -> "list[dict] | None":
    """dealDateが since_date 以降の記事をまとめて取得する（取りこぼしbackfill用の既報一覧）。

    already_published() は候補1件につきmicroCMSへ1リクエスト投げる。直近3日の通常運転
    （候補100件強）なら問題ないが、取りこぼしを拾うために窓を30日へ広げると候補が1,000件を
    超え、毎回それだけ叩くことになる。既報の一覧を先に1回だけ取り、既に記事がある開示は
    リクエスト無しで落とす。

    取得に失敗した／ページ上限で打ち切った場合は None を返す。呼び出し側は
    「既報が分からないまま30日分を投稿し直す」ことを避けるため backfill 自体を中止すること
    （dealTypeでの既報判定が常に0件を返して同一開示を13本投稿した 2026-08-25 の再発防止）。
    """
    filters = f"dealDate[greater_than]{since_date}T00:00:00.000Z"
    if extra_filter:
        filters = f"{extra_filter}[and]{filters}"
    out: list[dict] = []
    for page in range(MICROCMS_MAX_PAGES):
        try:
            resp = requests.get(
                _microcms_base_url(), headers=_microcms_headers(),
                params={"filters": filters, "fields": fields, "orders": "-dealDate",
                        "limit": MICROCMS_PAGE_LIMIT, "offset": page * MICROCMS_PAGE_LIMIT},
                timeout=30,
            )
        except Exception as e:
            print(f"  ⚠ 既報インデックスの取得に失敗: {e}")
            return None
        if resp.status_code != 200:
            print(f"  ⚠ 既報インデックスの取得に失敗 HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        body = resp.json()
        out.extend(body.get("contents", []))
        if len(out) >= body.get("totalCount", 0):
            return out
    print(f"  ⚠ 既報インデックスが{MICROCMS_MAX_PAGES}ページを超えました（{len(out)}件で打ち切り）")
    return None


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
                      filer_name: str = "", ratio_change: "float | None" = None,
                      unique_filing: bool = False) -> bool:
    """同一開示の記事が既にmicroCMSにあればTrue（重複投稿防止）。
    突き合わせキーは 銘柄コード＋開示日＋提出者名＋比率変化幅(ratioChangePct)。
    いずれも開示データから決まる値なので、何度再実行しても同じ開示は同じキーになる。

    dealAmountでの突き合わせは、推定金額が株価から都度概算されるため株価キャッシュ更新を
    またぐと全銘柄で±0.05億円を超えてズレ、既報の開示が全て別イベント扱いになる
    （実害: 2026-08-17、daily_alert.ymlの株価更新直後の便で17件が重複投稿）。
    filerName未保存の旧記事（2026-08-16以前）へのフォールバックとしてのみ残す。

    同一銘柄・同日・同一提出者の複数開示も実在する（実例: 2936 2025-08-13に橋本舜が
    9:50と10:07に別々の変更報告書を提出）ため、提出者一致だけでは重複と断定せず
    ratioChangePctの一致まで確認して別イベントを区別する。

    ただしその日その提出者の開示が1件しか無い場合（unique_filing=True）は、比率変化幅が
    一致しなくても同一開示とみなす。変化幅の算出ロジックを変えると既報記事のキーとずれ、
    同じ開示がもう一度投稿されてしまうため（cleanup_duplicate_blog_articles.pyも
    ratioChangePctまで一致した重複しか回収しない）。

    照会は開示日(dealDate)まで絞り込む。銘柄コードだけで引いてlimit=50を被せていた頃は、
    記事が50件を超える銘柄で既報が応答に入らず重複と判定できない穴があった。
    そして照会に失敗したときは**既報扱い(True)にして投稿を見送る**（publish_buyback_articles
    の同名関数と同じ方針）。判定不能のまま投稿すると重複記事がサイトに恒久的に残り、
    filerNameを持たない世代では回収もできない（実例: 9706に同一記事が11件）。
    見送っても本スクリプトは平日9:00-21:00 JSTに毎時走り、直近LARGE_HOLDINGS_DAYS日の
    開示を毎回見直すので、次の便で取り直せる。"""
    try:
        resp = requests.get(
            _microcms_base_url(),
            headers=_microcms_headers(),
            params={
                "filters": f"stockCode[equals]{stock_code}[and]dealDate[begins_with]{disc_date}",
                "fields": "id,dealDate,dealAmount,filerName,ratioChangePct",
                "limit": 50,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            print(f"    ⚠ 既報確認に失敗 HTTP {resp.status_code}（重複を避けて投稿を見送り）")
            return True
        contents = resp.json().get("contents", [])
        for c in contents:
            if str(c.get("dealDate", ""))[:10] != disc_date:
                continue
            if filer_name and c.get("filerName"):
                if c["filerName"] != filer_name:
                    continue
                if unique_filing:
                    return True
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
    except Exception as e:
        print(f"    ⚠ 既報確認に失敗（重複を避けて投稿を見送り）: {e}")
        return True


@lru_cache(maxsize=None)
def shares_outstanding(code: str) -> "float | None":
    """yfinanceの.infoは一時的なレート制限で単発失敗することが多く（2026-08-06の実行では
    価格データが揃っている銘柄（サンリオ等）でもこれが原因で「金額を概算できない」スキップに
    なっていた）、最大3回まで短い間隔でリトライする。J-REIT（投資口）はsharesOutstandingが
    空でimpliedSharesOutstandingに口数が入ることがあるためフォールバックで見る。

    同一銘柄が同じ実行内で複数回判定されることがある（同日・同一提出者の開示が2件出る等、
    実例: 2026-08-27 ハリマ共和物産7444）ため、実行内で結果をメモ化する。上場廃止・新規コードで
    404になる銘柄（8190.T / 5953.T / 9170.T）の3回リトライを毎回やり直さないためでもある。"""
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


# 変更報告書なのに直前保有割合が取れていない開示を「その日のうちに」記事化しないための待機日数。
# EDINETはメタデータ公開とXBRL本文の可用性にラグがあり、提出直後の便では
# fetch_xbrl_details() が holding_ratio_prior を拾えないことがある。その状態で記事化すると
# ratio_change_pct() が「今回比率の全量＝新規取得」とみなし、
#   - タイトルが「X%を新規保有」（実際は変更報告書）
#   - estimate_deal_amount_oku() の推定金額が比率全量ぶんに膨らむ
# という二重の誤りが公開されたまま残る（次の便でDB側の前回比率は埋まるが、記事は初回値のまま）。
# 実測（2026-08-19、直近14日の照合可能56件）で13件=23%がこの誤りを抱えていた。
# この日数を過ぎても前回比率が埋まらない開示は、XBRLの書式差で恒久的に取れないと判断し、
# 従来どおり過去開示からの再導出にフォールバックして記事化する（取りこぼしを作らない）。
PRIOR_RATIO_WAIT_DAYS = 2


def is_change_report(doc_description: str) -> bool:
    """変更報告書（訂正報告書は除く）かどうか。変更報告書は必ず直前保有割合を持つ。"""
    desc = doc_description or ""
    return "変更報告書" in desc and "訂正" not in desc


def should_wait_for_prior_ratio(doc_description: str, prior_ratio: "float | None",
                                disc_date: str, today: "date | None" = None) -> bool:
    """変更報告書なのに直前保有割合が未取得なら、次の便まで記事化を見送るかどうか。"""
    if prior_ratio is not None or not is_change_report(doc_description):
        return False
    today = today or date.today()
    try:
        elapsed = (today - date.fromisoformat(disc_date[:10])).days
    except ValueError:
        return False
    return elapsed < PRIOR_RATIO_WAIT_DAYS


def ratio_change_pct(code: str, filer_name: str, current_ratio: float, disc_date: str,
                     prior_ratio: "float | None" = None,
                     is_amendment: bool = False) -> "float | None":
    """今回開示の保有比率が前回からどれだけ動いたか（変化幅、%ポイント）を返す。

    EDINET開示自体が持つ直前保有割合(prior_ratio)があればそれを使う。DB蓄積分の履歴から
    前回比率を再導出すると、履歴に同じ比率の行が残っている場合や履歴が無い全売却
    （比率0%）で変化幅が0と算出され、記事化されずに落ちる（実例: 2026-08-17、三菱商事の
    ＴＯＹＯ ＴＩＲＥ 20%→0%「短期大量譲渡」が「金額を概算できない」として不投稿）。
    prior_ratioが無い開示のみ、従来通り過去開示から直近の比率を探す。

    is_amendmentは変更報告書かどうか（is_change_report()の結果）。前回比率も過去開示も
    無いときに「今回比率の全量＝今回動いた分」とみなせるのは**新規の大量保有報告書だけ**。
    変更報告書は提出者が既に5%以上を保有している届出なので、全量ぶんの変化幅を返すと
    変化幅も推定金額も実態の数十倍に膨らむ。should_wait_for_prior_ratio()はXBRLの遅延を
    PRIOR_RATIO_WAIT_DAYSだけ待つが、待っても直前保有割合が入らない開示（特例報告に多い。
    2026-08-19の実測で直近90日に7件。変更報告書のprior充填率は99.6%）はそこを通過して
    しまうため、ここで変化幅を「不明」としてNoneを返し、呼び出し側で記事化を見送る。"""
    if prior_ratio is not None:
        return abs(current_ratio - prior_ratio)
    history = get_edinet_large_holdings_recent(days=400, codes=[code])
    past = [
        h for h in history
        if h.get("filer_name") == filer_name
        and h.get("disc_date", "") < disc_date
        and h.get("holding_ratio") is not None
    ]
    if not past:
        return None if is_amendment else current_ratio
    past.sort(key=lambda h: h["disc_date"])
    prev_ratio = past[-1]["holding_ratio"]
    return abs(current_ratio - prev_ratio)


@lru_cache(maxsize=None)
def close_price_from_yfinance(code: str, target_date: date) -> "float | None":
    """yahoo_price_cacheに無い銘柄の終値をyfinanceから直接取る（target_date以前の直近終値）。

    価格キャッシュはスクリーニング対象ユニバースしか埋めておらず、新規上場銘柄や
    ユニバース外の銘柄はEDINET開示が出ても株価が引けない（実例: 2026-08-18、
    アイ・グリッド・ソリューションズ603A・ビート・ホールディングス9399・デンタス6174が
    「金額を概算できない」として不投稿）。発行済株式数は既にyfinanceから取っているので、
    株価も同じ経路でフォールバックする。shares_outstanding と同じ理由で実行内メモ化する。"""
    import yfinance as yf
    try:
        hist = yf.Ticker(f"{code}.T").history(period="1mo")
        if hist is None or hist.empty:
            return None
        closes = [
            (idx.date() if hasattr(idx, "date") else idx, float(row))
            for idx, row in zip(hist.index, hist["Close"])
        ]
        past = [c for d, c in closes if d <= target_date]
        return past[-1] if past else closes[0][1]
    except Exception:
        return None


def disclosure_close_price(code: str, disc_date: str) -> "float | None":
    """開示日の終値（yahoo_price_cache優先、無ければyfinance）。

    推定金額の計算と、記事本文へ渡す「開示日時点の株価」の両方でこの1つの値を使う。
    以前は本文用の株価を gen_rankings（日次ランキング、開示日にまだ当日行が無ければ数日前の
    行を拾う）から取っていたため、記事本文の「2026年8月18日時点の株価11,180円」と
    ファクトボックスの「基準終値（2026/08/18）10,760円」が同じ画面で食い違っていた
    （2026-08-19の監査で検出）。"""
    target = date.fromisoformat(disc_date[:10])
    return get_price_at_date(code, target) or close_price_from_yfinance(code, target)


def estimate_deal_amount_oku(code: str, ratio_change: float, disc_date: str) -> "float | None":
    """推定取得金額（億円） = 比率変化(%) × 発行済株式数 × 株価 ÷ 100 ÷ 1億。
    株式数・株価のいずれかが取得できなければ None（呼び出し側でスキップする）。"""
    if ratio_change <= 0:
        return None
    shares = shares_outstanding(code)
    price = disclosure_close_price(code, disc_date)
    if not shares or not price:
        return None
    amount_yen = shares * price * (ratio_change / 100)
    return round(amount_yen / 1e8, 1)


def deal_amount_label(is_sell: bool, is_exact: bool) -> str:
    """金額の見出し語。短期大量譲渡で開示単価が取れたときだけ「推定」を外す。"""
    verb = "売却" if is_sell else "取得"
    return f"{verb}金額" if is_exact else f"推定{verb}金額"


def format_transfer_facts(transfers: dict) -> str:
    """短期大量譲渡の「譲渡の相手方・単価」をプロンプトの事実行にする。

    EDINETは通常「誰に売ったか」を開示しないが、短期大量譲渡に該当する変更報告書だけは
    相手方と単価が表で載る。記事の一次情報としての価値が最も高い部分なので必ず本文に出す。
    """
    if not transfers or not transfers.get("counterparties"):
        return ""
    names = "、".join(transfers["counterparties"][:3])
    if len(transfers["counterparties"]) > 3:
        names += f"ほか{len(transfers['counterparties']) - 3}者"
    line = f"- 譲渡の相手方（開示された事実）: {names}\n"
    unit_price = transfers.get("unit_price")
    shares = transfers.get("shares")
    if unit_price and shares:
        line += f"- 譲渡の単価と数量（開示された事実）: 1株{unit_price:,.0f}円 × {shares:,}株\n"
    if transfers.get("venue"):
        line += f"- 取引の場: {transfers['venue']}（{'取引所外の相対取引' if transfers['venue'] == '市場外' else '取引所内'}）\n"
    return line


def checked_recently(checked_at: "str | None", days: int = RECHECK_DAYS) -> bool:
    """ネガティブキャッシュの判定。checked_at（ISO8601）が days 日以内なら True。

    未設定・解析不能なら False（＝まだ試していない扱いにして生成を許す）。
    """
    if not checked_at:
        return False
    try:
        ts = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts > datetime.now(timezone.utc) - timedelta(days=days)


def get_company_description(code: str, name: str) -> str:
    """対象企業の事業内容を1文程度で返す。jpx_stock_list.descriptionにキャッシュがあれば
    それを使い、無ければClaudeのweb_searchで会社概要を確認して生成しキャッシュする。
    ※一般知識だけで書かせると中小型株の約2/3が「不明（空文字）」になり
    （2026-08-15のバックフィルで生成1,134件に対し不明1,507件）、
    /trendingや/stocks/[code]で事業内容が出ない銘柄が大量に残るため、
    web検索で裏取りさせる。創作を防ぐガード（不明なら空文字）はそのまま維持する。

    空文字だった場合も description_checked_at に試行日時を刻み、RECHECK_DAYS 以内は
    再試行しない（web_searchは1社あたり約$0.05かかるため、同じ「不明」社群を
    バックフィルのたびに叩き直すと一気に月次上限を食う）。"""
    cached = sb.select_one(
        "jpx_stock_list", f"code=eq.{code}&select=description,description_checked_at"
    )
    if cached and cached.get("description"):
        return cached["description"]
    if cached and checked_recently(cached.get("description_checked_at")):
        return ""

    import anthropic

    if not ANTHROPIC_API_KEY or api_budget.reached():
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
            # max_usesは検索料（$10/1,000検索）と入力トークン（1検索≒6,000トークン）に
            # 直結する。会社四季報【特色】相当の一文に複数回の検索は過剰なので1に抑える
            # （2026-08-29に2→1。1件あたり約$0.034→$0.017）。
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            messages=[{"role": "user", "content": prompt}],
        )
        api_usage.record(resp, task="company_description")
        # web_search使用時は検索結果ブロックとテキストブロックが交互に並ぶため、
        # 全テキストを連結して末尾のJSONだけを取り出す。
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        matches = re.findall(r'\{[^{}]*"description"[^{}]*\}', text, re.DOTALL)
        # 説明文の途中に生の改行が入ったJSONを返すことがあり、strict=Falseでないと
        # "Invalid control character" で丸ごと落ちる（2026-08-18のバックフィルで8件発生）。
        description = json.loads(matches[-1], strict=False).get("description", "") if matches else ""
        # web_searchの引用を伴う回答は本文に<cite index="7-1">…</cite>を挟んで返すことがあり、
        # そのまま保存するとkujira-watchの銘柄ページ・/trendingにタグが文字として表示される
        # （2026-08-25の点検で64銘柄が汚染。DB側はtools/strip_html_from_descriptions.pyで一括修正）。
        description = re.sub(r"<[^>]+>", "", description)
        description = " ".join(description.split())
        description = description or ""
    except Exception as e:
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
            return ""  # 上限エラーは「試行済み」に含めない（課金されていないため）
        print(f"    ⚠ 事業内容取得に失敗: {e}")
        description = ""

    # 空文字（不明）でもchecked_atを刻む。これが無いと同じ銘柄に何度でも課金される。
    payload = {"code": code, "description_checked_at": datetime.now(timezone.utc).isoformat()}
    if description:
        payload["description"] = description
    sb.upsert("jpx_stock_list", [payload], on_conflict="code")
    return description


def get_filer_profile(filer_name: str, category: str) -> str:
    """kujira-watch側 /investors/[filer] に表示する投資家プロフィール(日本語800〜1000字程度)を
    返す。edinet_filer_classification.profileにキャッシュがあればそれを使い、無ければClaudeの
    一般知識で生成してキャッシュする（get_company_descriptionと同じ方針。設立時期・運用方針・
    著名な投資事例など、確信が持てる範囲のみ記述させ、役員名や具体的な運用資産額等の検証不能な
    事実は創作させない）。

    ただし分類が「個人」の提出者にはプロフィールを持たせない。氏名だけを手がかりに
    Claudeの一般知識で書かせると、同姓同名や似た名前の有名人と取り違えた経歴が生成され、
    それが実在の個人の紹介文として /investors/[filer] に表示される。
    実害（2026-08-29に発見・是正）: 138件のうち、永守重信氏の会社名を「ニッドー」と書いたもの、
    藤巻米隆氏について別人の名前を挙げて迷う文章、正垣泰彦氏（サイゼリヤ創業者）を
    「すかいらーくグループの創業者」と書いたものなどが保存され、投資家ページに出ていた。
    個人については提出者名と開示の事実だけを載せる（解説文が無い投資家ページは
    kujira-watch側の pageIndexability が noindex にする）。"""
    if category == "個人":
        return ""
    cached = sb.select_one(
        "edinet_filer_classification",
        f"filer_name=eq.{requests.utils.quote(filer_name)}&select=profile,profile_checked_at",
    )
    if cached and cached.get("profile"):
        return cached["profile"]
    if cached and checked_recently(cached.get("profile_checked_at")):
        return ""

    import anthropic

    if not ANTHROPIC_API_KEY or api_budget.reached():
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
        api_usage.record(resp, task="filer_profile")
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        # プロフィールは800〜1000字の地の文で生の改行が混じるため strict=False
        profile = json.loads(text, strict=False).get("profile", "") or ""
    except Exception as e:
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
            return ""  # 上限エラーは「試行済み」に含めない（課金されていないため）
        print(f"    ⚠ 投資家プロフィール取得に失敗: {e}")
        profile = ""

    # categoryもペイロードに含める: on_conflict時のUPDATEでは不要だが、PostgreSQLは
    # ON CONFLICTのUPDATE分岐に関わらずINSERT側の候補行構築時点でNOT NULL制約を
    # 評価するため、category(NOT NULL)を欠いたペイロードだとUPDATEのみのつもりでも
    # 「null value in column "category"」で失敗する（実運用で発生: 2026-08-10）。
    # profileが空でもchecked_atは刻む（公開情報の乏しい提出者を毎回引き直さないため）。
    payload = {
        "filer_name": filer_name,
        "category": category,
        "profile_checked_at": datetime.now(timezone.utc).isoformat(),
    }
    if profile:
        payload["profile"] = profile
    sb.upsert("edinet_filer_classification", [payload], on_conflict="filer_name")
    return profile


def get_pit_ranking_snapshot(code: str, as_of: str) -> "dict | None":
    """開示日(as_of)時点で直近のcloseをgen_rankingsから取得する。
    記事公開時点（post-hoc、開示後に株価が既に動いた後）のスナップショットを
    「開示当時の文脈」として出すと先読みバイアスになるため、as_of以前の最新行のみを見る
    （CLAUDE.md PIT規律）。

    drop_prob（下落モデルの予測値）はかつてここから取って記事本文の「弊社モデルの
    下落リスク水準」に使っていたが、モデルの説明ページがサイトに無いまま検証不能な
    独自指標をYMYLの判断材料として提示する形になっていたため2026-08-25に取りやめた。"""
    return sb.select_one(
        "gen_rankings",
        f"code=eq.{code}&date=lte.{as_of}&order=date.desc&select=close",
    )


def format_ratio(ratio: float) -> str:
    """保有比率を表示用文字列にする（20.93→"20.93"、5.00→"5"、末尾ゼロは落とす）。"""
    return f"{ratio:.2f}".rstrip("0").rstrip(".")


def is_new_holding(fact_sheet: dict) -> bool:
    """実質的な新規保有か。

    開示の直前保有割合(prior_ratio)が分かる場合はそれが0のときだけ新規とする。変化幅と
    今回比率の比較だけで判定すると、全売却（比率0%・変化幅=前回比率）が「0%を新規保有」に
    化ける。prior_ratioが無い開示のみ、直近400日に同一提出者の過去開示が無く変化幅=今回比率に
    なるケースを新規とみなす従来のヒューリスティックにフォールバックする。"""
    if fact_sheet.get("is_correction"):
        return False
    # 変更報告書は「既に5%以上を保有している提出者」しか出せない届出なので、直前保有割合が
    # 取れなくても新規保有ではありえない。ヒューリスティックより報告書種別を優先する。
    if fact_sheet.get("doc_type_label") == "変更報告書":
        return False
    prior = fact_sheet.get("prior_ratio")
    if prior is not None:
        return prior == 0 and fact_sheet["holding_ratio"] > 0
    change = fact_sheet.get("ratio_change_pct")
    return change is not None and change >= fact_sheet["holding_ratio"]


# 検索結果で全文が見えるよう、テンプレタイトルはこの長さに収める（超過時は提出者名を短縮する）
# 検索結果に出るのは全角30〜32字程度で、この上限に収まる記事はほとんど無い。それでも
# 短くしないのは、切るとしたら提出者名（＝検索されている語そのもの。GSC実測で上位クエリの
# 大半が提出者の人名・法人名）を削ることになるため。表示順が「銘柄名→提出者名→保有比率→
# 大量保有報告書」なので、切れるのは後ろの補足だけで済んでいる。
MAX_TITLE_LEN = 60


def build_article_titles(fact_sheet: dict) -> dict:
    """検索クエリ型の記事タイトルを決定的テンプレートで組み立てる（LLM不使用）。
    「銘柄名（コード）」「保有比率」「大量保有報告書」という検索語が必ずタイトルに入ることを
    保証するため、LLMの自由生成をやめてテンプレ化した（SEO/AIO 30日計画 P1）。"""
    name = fact_sheet["stock_name"]
    code = fact_sheet["stock_code"]
    filer = fact_sheet["filer_name"]
    ratio = format_ratio(fact_sheet["holding_ratio"])
    is_sell = fact_sheet.get("direction") == "sell"
    is_correction = bool(fact_sheet.get("is_correction"))

    def ja_title(filer_name: str) -> str:
        if is_correction:
            return f"{name}（{code}）、{filer_name}が保有比率を{ratio}%に訂正｜訂正報告書"
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

    return {"title": title}


# 記事に独自性を与える周辺事実。EDINET開示1件の数字だけを書くと、どの記事も
# 「誰が何%取得・推定何億円・※推測」の3段落テンプレートになり（実測2026-08-24:
# 全976記事が1,000字未満、中央値455字）、Googleの「質の低いコンテンツ」ガイドラインが
# 挙げる cookie cutter pages に該当した。AdSense審査も同じ理由で不承認になっている。
#
# ここで集めるのは、EDINET原本を1件見ただけでは分からず、開示を全期間ぶん横断して
# 初めて書ける事実（同じ提出者がその銘柄を何回に分けて買い進めたか、他に何を持っているか、
# その銘柄の株主構成に他に誰がいるか）。競合の大量保有報告書データベースには無い切り口で、
# 「独自性のある質の高いコンテンツ」の要件に直接効く。
#
# 重要: すべて point-in-time で取る（CLAUDE.mdのSQL PIT規律）。開示日より後のデータを
# 混ぜると「その時点では知り得なかった事実」を記事に書くことになる。
MAX_RELATED_ITEMS = 5


def _latest_by(rows: list, key: str) -> list:
    """開示日降順で渡された行から、keyごとに最初に出現した行（=最新の開示）だけを残す。"""
    seen, out = set(), []
    for row in rows:
        value = row.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(row)
    return out


def build_context_facts(code: str, filer_name: str, disc_date: str) -> dict:
    """開示日時点で判明している周辺事実を集める。取得に失敗した項目は空で返す
    （記事生成そのものは止めない）。"""
    facts: dict = {}
    if not filer_name:
        return facts
    try:
        # 1. この提出者×この銘柄の開示履歴（何回目の開示か、いつから買い進めているか）
        history = sb.select(
            "edinet_large_holdings",
            f"select=disc_date,holding_ratio&issuer_code=eq.{code}"
            f"&filer_name=eq.{requests.utils.quote(filer_name)}&disc_date=lte.{disc_date}"
            "&order=disc_date.asc",
        )
        if len(history) >= 2:
            first = history[0]
            facts["holding_history"] = {
                "count": len(history),
                "first_date": first.get("disc_date"),
                "first_ratio": first.get("holding_ratio"),
                # 本文用には初回と回数だけで足りるが、解説図（保有比率の推移）は各回の数字を使う
                "points": [
                    {"date": r.get("disc_date"), "ratio": r.get("holding_ratio")} for r in history
                ],
            }

        # 2. 同じ提出者が同時点で持っている他の銘柄（保有比率の高い順）
        others = sb.select(
            "edinet_large_holdings",
            f"select=issuer_code,issuer_name,holding_ratio,disc_date"
            f"&filer_name=eq.{requests.utils.quote(filer_name)}&disc_date=lte.{disc_date}"
            "&order=disc_date.desc",
            limit=500,
        )
        latest_others = [
            r for r in _latest_by(others, "issuer_code")
            if str(r.get("issuer_code")) != str(code) and r.get("holding_ratio") is not None
        ]
        latest_others.sort(key=lambda r: r["holding_ratio"], reverse=True)
        if latest_others:
            top_others = latest_others[:MAX_RELATED_ITEMS]
            # 業種は必ずマスターから引く。社名から推測させると事実に無い記述が混ざる。
            codes = ",".join(str(r["issuer_code"]) for r in top_others if r.get("issuer_code"))
            sectors = {}
            if codes:
                for row in sb.select("jpx_stock_list", f"select=code,sector&code=in.({codes})"):
                    if row.get("sector"):
                        sectors[str(row["code"])] = row["sector"]
            facts["filer_other_holdings"] = [
                {
                    "name": r.get("issuer_name") or r.get("issuer_code"),
                    "code": r.get("issuer_code"),
                    "ratio": r["holding_ratio"],
                    "sector": sectors.get(str(r.get("issuer_code"))) or "",
                }
                for r in top_others
            ]
            facts["filer_holding_total"] = len(latest_others) + 1

        # 3. 同じ銘柄の他の大株主（この開示が株主構成の中でどの位置づけか）
        peers = sb.select(
            "edinet_large_holdings",
            f"select=filer_name,holding_ratio,disc_date&issuer_code=eq.{code}"
            f"&disc_date=lte.{disc_date}&order=disc_date.desc",
            limit=500,
        )
        latest_peers = [
            r for r in _latest_by(peers, "filer_name")
            if r.get("filer_name") != filer_name and r.get("holding_ratio") is not None
        ]
        latest_peers.sort(key=lambda r: r["holding_ratio"], reverse=True)
        if latest_peers:
            facts["stock_other_filers"] = [
                {"name": r["filer_name"], "ratio": r["holding_ratio"]}
                for r in latest_peers[:MAX_RELATED_ITEMS]
            ]

        # 4. 銘柄の指標（開示日以前で最新の行。開示日より後の株価は使わない）
        metrics = sb.select_one(
            "gen_rankings",
            f"select=date,per,pbr,pos52&code=eq.{code}&date=lte.{disc_date}"
            "&order=date.desc",
        )
        if metrics:
            facts["stock_metrics"] = {
                k: metrics.get(k) for k in ("per", "pbr", "pos52") if metrics.get(k) is not None
            }

        # 5. 業種（同業の中での位置づけに触れられるようにする）
        master = sb.select_one("jpx_stock_list", f"select=sector&code=eq.{code}")
        if master and master.get("sector"):
            facts["sector"] = master["sector"]
    except Exception as e:  # 周辺事実は「あれば厚くなる」ものなので、失敗しても記事は出す
        print(f"    ⚠ 周辺事実の取得に失敗（記事は続行）: {e}")
    return facts


def format_context_facts(facts: dict, stock_name: str, filer_name: str) -> str:
    """build_context_facts()の結果をプロンプトの「事実」ブロックに足す行に整形する。"""
    if not facts:
        return ""
    lines = []
    history = facts.get("holding_history")
    if history:
        lines.append(
            f"- 過去の開示: {filer_name}はこの銘柄について今回を含め{history['count']}回の開示を出しており、"
            f"最初の開示は{history['first_date']}（保有比率{format_ratio(history['first_ratio'])}%）"
        )
    others = facts.get("filer_other_holdings")
    if others:
        listed = "、".join(
            f"{o['name']}（{o['code']}、{o['sector'] + '、' if o.get('sector') else ''}"
            f"{format_ratio(o['ratio'])}%）"
            for o in others
        )
        total = facts.get("filer_holding_total")
        lines.append(
            f"- {filer_name}が同時点で5%以上を保有している他の銘柄（保有比率の高い順、全{total}銘柄のうち上位）: {listed}"
        )
    peers = facts.get("stock_other_filers")
    if peers:
        listed = "、".join(f"{p['name']}（{format_ratio(p['ratio'])}%）" for p in peers)
        lines.append(f"- {stock_name}に大量保有報告書を出している他の投資家（保有比率の高い順）: {listed}")
    sector = facts.get("sector")
    if sector:
        lines.append(f"- {stock_name}の業種: {sector}")
    metrics = facts.get("stock_metrics") or {}
    if metrics:
        parts = []
        if metrics.get("per") is not None:
            parts.append(f"PER {metrics['per']:.1f}倍")
        if metrics.get("pbr") is not None:
            parts.append(f"PBR {metrics['pbr']:.2f}倍")
        if metrics.get("pos52") is not None:
            parts.append(f"52週レンジ内の位置 {metrics['pos52']*100:.0f}%（0%が安値、100%が高値）")
        if parts:
            lines.append(f"- 開示日時点の{stock_name}の指標: " + " / ".join(parts))
    return "\n".join(lines) + "\n" if lines else ""


def generate_article_body(fact_sheet: dict) -> "dict | None":
    """Claudeに与えた事実のみからbodyを生成させる。JSONで
    {"body"} を返す（タイトルはbuild_article_titles()で別途組み立てる）。
    本文の1文目は検索クエリへの直答文（銘柄名・コード・提出者・保有比率を含む）に固定する。
    パース失敗時はNone（記事は投稿しない）。
    投資家分類（dealType）は事前にclassify_filer()で判定済み（edinet_filer_classification
    マスター参照＋未登録時のみClaude判定、記事本文生成とは別の呼び出しに分離してキャッシュ
    可能にした）で、その分類の一言説明がfact_sheet['filer_description']としてあれば本文に
    自然に織り込む。"""
    import anthropic

    if not ANTHROPIC_API_KEY or api_budget.reached():
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    is_sell = fact_sheet.get("direction") == "sell"
    is_correction = bool(fact_sheet.get("is_correction"))
    deal_verb = "訂正" if is_correction else ("売却" if is_sell else "取得")
    label = fact_sheet.get("deal_amount_label") or deal_amount_label(is_sell, False)
    # 下落モデル（drop_prob）の水準は記事に出さない。モデルの説明ページがサイトに無いまま
    # 「弊社モデルでは下落リスク水準を◯◯と評価」と書くと、YMYL（金融）で検証不能な独自指標を
    # 判断材料として提示していることになり、AdSense/E-E-A-Tの信頼性評価で減点される
    # （2026-08-25の再監査で全記事の30%が該当）。開示日時点の株価は開示原本と突き合わせ可能な
    # 事実なので残す。
    context_close = fact_sheet.get("context_close")
    context_line = f"- 開示日時点の株価: {context_close:,.0f}円\n" if context_close is not None else ""
    filer_description = fact_sheet.get("filer_description") or ""
    filer_description_line = f"- 提出者について: {filer_description}\n" if filer_description else ""
    company_description = fact_sheet.get("company_description") or ""
    company_description_line = f"- {fact_sheet['stock_name']}の事業内容: {company_description}\n" if company_description else ""
    ratio_change_pct = fact_sheet.get("ratio_change_pct")
    is_new = is_new_holding(fact_sheet)
    prior_ratio = fact_sheet.get("prior_ratio")
    if is_correction:
        change_line = (
            f"- 変化: 訂正前に届け出ていた保有比率{prior_ratio}%を{fact_sheet['holding_ratio']}%へ訂正した"
            f"（{ratio_change_pct:.2f}ポイントの{'下方' if is_sell else '上方'}修正）\n"
        )
    elif ratio_change_pct is not None:
        if is_new:
            change_line = "- 変化: 直近400日以内に同一提出者による当銘柄の開示が確認できず、今回が実質的な新規保有（または大幅な保有再開）とみられる\n"
        else:
            change_line = f"- 変化: 保有比率はこれまでの開示から{ratio_change_pct:.2f}ポイント{'減少' if is_sell else '増加'}し、今回{fact_sheet['holding_ratio']}%になった\n"
    else:
        change_line = ""
    context_facts_line = format_context_facts(
        fact_sheet.get("context_facts") or {}, fact_sheet["stock_name"], fact_sheet["filer_name"]
    )
    ratio_str = format_ratio(fact_sheet["holding_ratio"])
    if is_correction:
        answer_sentence = (
            f"{fact_sheet['stock_name']}（{fact_sheet['stock_code']}）について、{fact_sheet['filer_name']}が"
            f"これまで届け出ていた保有比率{format_ratio(prior_ratio)}%を{ratio_str}%に訂正したことが"
            "訂正報告書（EDINET）で分かりました。"
        )
    elif is_new:
        answer_sentence = (
            f"{fact_sheet['stock_name']}（{fact_sheet['stock_code']}）について、{fact_sheet['filer_name']}が"
            f"同社株式の{ratio_str}%を保有していることが大量保有報告書（EDINET）で分かりました。"
        )
    else:
        answer_sentence = (
            f"{fact_sheet['stock_name']}（{fact_sheet['stock_code']}）について、{fact_sheet['filer_name']}が"
            f"保有比率を{ratio_str}%まで{'引き下げ' if is_sell else '引き上げ'}たことが大量保有報告書（EDINET）で分かりました。"
        )
    # 訂正報告書は「既報の数字が誤っていた」という開示であり、今回売買があったとは限らない。
    # 推定金額も出さない（売買を伴わない訂正に金額を付けると、実在しない取引を報じることになる）。
    if is_correction:
        nature_instruction = (
            "この開示は、既に届け出ていた保有比率を事後に訂正する訂正報告書です。"
            "今回新たに株式が売買されたことを意味するとは限らないため、売買・取得・売却があったと"
            "断定して書かないでください（「届け出ていた保有比率が訂正された」という書き方をすること）。\n"
            "訂正の理由・原因はこの開示からは分からないため、推測で理由を書かないでください。\n"
        )
        amount_fact_line = f"- 訂正前の届出比率: {prior_ratio}%\n"
        speculation_scope = (
            "推測してよいのは、この訂正が市場や投資家にとって持つ意味だけです。"
            "訂正が起きた原因・理由（ポジション調整、計算誤り等）の推測は書かないでください。"
        )
    else:
        transfers = fact_sheet.get("transfers") or {}
        is_exact_amount = transfers.get("amount_oku") is not None
        nature_instruction = (
            f"この取引は保有比率が{'減少した売却（譲渡等を含む）' if is_sell else '増加した取得（買い増し・新規取得）'}です。\n"
        )
        if is_exact_amount:
            # 開示に単価が載っている（短期大量譲渡）ケース。概算の注記を書かせない。
            nature_instruction += (
                "金額は開示された単価×株数から計算した実額です。「概算」「推定」とは書かないでください。\n"
            )
            amount_fact_line = (
                f"- {label}: {fact_sheet['deal_amount_oku']}億円"
                f"（開示された譲渡単価×株数から算出した実額）\n"
            )
        else:
            nature_instruction += (
                f"金額が概算であることは見出しの「{label}」表記のみで十分伝わるため、本文中で改めて\n"
                f"「概算であり実際の{deal_verb}価格ではない」等の注記を繰り返さないでください。\n"
            )
            amount_fact_line = f"- {label}: {fact_sheet['deal_amount_oku']}億円（発行済株式数と株価からの概算）\n"
        amount_fact_line += format_transfer_facts(transfers)
        speculation_scope = ""

    prompt = f"""以下は日本株の大量保有報告書（EDINET開示）に基づく事実です。この事実だけを根拠に、
投資家向けの解説記事を書いてください。事実にない金額・意図・背景は絶対に創作しないでください。
{nature_instruction}大量保有報告書制度そのものの一般的な説明（5%ルールの趣旨、市場透明性・投資家保護目的など）や、
「今後の動向を注視する必要がある」といった、この取引固有ではない定型的な結びの文も書かないでください。

{JA_STYLE_RULES}

事実:
- 対象銘柄: {fact_sheet['stock_name']}（{fact_sheet['stock_code']}）
- 提出者: {fact_sheet['filer_name']}
- 報告書種別: {fact_sheet['doc_type_label']}
- 保有比率: {fact_sheet['holding_ratio']}%
- 開示日: {fact_sheet['disc_date']}
{amount_fact_line}{filer_description_line}{company_description_line}{context_line}{change_line}{context_facts_line}
本文の1文目は、必ず次の文をそのまま使ってください（検索してきた読者への直答として最初に置く）:
「{answer_sentence}」
提出者について の事実がある場合は、それがどんな種類の投資家かを1文で読者に補足してください
（例: 提出者が海外の資産運用会社なら「海外の資産運用会社による{deal_verb}」等）。無い場合は無理に触れなくてよいです。
{fact_sheet['stock_name']}の事業内容 の事実がある場合は、1文目の直後にその会社が何をしている会社かを
1文で読者に補足してください。また保有比率{fact_sheet['holding_ratio']}%という数字が、対象企業の
株式のどれくらいの規模を占めるかが実感できるよう自然に触れてください（時価総額の一角を占める大株主、等）。
変化 の事実がある場合は、今回の保有比率が一回限りの数字ではなく前回からの推移であることが
伝わるよう、1文で自然に触れてください。

以下は、事実に該当する項目があるときだけ、それぞれ独立したセクションとして書いてください
（無い項目は飛ばす。無理に触れて薄い文を足さないこと）。各セクションは<h2>見出し</h2>で始め、
本文は3文以上・200字以上にしてください。見出しはその段落で何が分かるかを具体的に示す
10〜20字の日本語にし、「まとめ」「考察」のような中身の無い語は使わないでください
（例:「2019年からの保有推移」「サイバーエージェントが持つ他の銘柄」「株主構成での位置づけ」）。

- 過去の開示: 今回の{deal_verb}が単発ではなく、いつから何回に分けて積み上げてきたポジションなのか。
  初回開示時の保有比率から今回までで何ポイント動いたかを数字で示す。
- 同時点で保有している他の銘柄: その投資家がどんな銘柄を選ぶ投資家なのか（どの業種に集中しているか、
  保有比率の傾向）。必ず具体的な銘柄名・業種・比率を挙げて論じる。
- 他の投資家: 今回の{deal_verb}が対象企業の株主構成の中でどういう位置づけか（筆頭級か、他の大株主と
  比べてどれだけ大きいか）。具体名と比率を挙げて比較する。
- 業種・指標: PER・PBR・52週レンジ内の位置がそれぞれ何を示しているか。ただし「割安だから買い」のような
  投資判断の断定はしないでください。

他社の業種・事業内容は、事実に業種が明記されている銘柄についてだけ言及してください。社名から業種を
推測して書くことは禁止です（例: 事実に業種の無い銘柄を「電機メーカー」と決めつけない）。

最後に1文だけ、この{deal_verb}が今後の同社や当該投資家にとってどんな意味を持ちうるかの推測を
加えてください。{speculation_scope}ただし事実として断定せず、必ず文頭に「※推測:」を付けて、事実の記述とは明確に
分けてください（例: 「※推測: 海外ファンドとの関係強化を通じて新市場開拓を模索している可能性がある」）。
上記の事実（事業内容・提出者の属性・保有比率の規模等）から自然に読み取れる範囲の推測に留め、
事実として存在しない具体的な計画やコメントの引用は創作しないでください。

出力はJSON形式のみとし、他のテキストやコードフェンスは含めないでください（タイトルは別途
テンプレートで組み立てるため出力しない）:
{{"body": "HTML本文（1,300〜1,700字。冒頭の直答段落は<p>で始め、以降は<h2>見出し</h2>と<p>本文</p>を3〜5セクション。最後に※推測文の<p>）"}}
"""
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        api_usage.record(resp, task="blog_body")
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        # strict=False は必須。本文はHTMLなのでモデルがJSON文字列の中に生の改行を入れてくる。
        # 既定の strict=True だと "Invalid control character at ..." で丸ごと落ち、記事が1本消える
        # （2026-08-27のbackfill便では22回の生成のうち7回がこれで失敗し、再生成で拾い直していた）。
        data = json.loads(text, strict=False)
        if not data.get("body"):
            return None
        return data
    except Exception as e:
        if api_budget.note(e):
            print(api_budget.SKIP_MESSAGE)
        else:
            print(f"    ⚠ 記事生成失敗: {e}")
        return None


# プロンプトでは650〜900字を指示しているが、実測では中央値445字・全911本が800字未満と
# 指示がまったく守られていなかった（2026-08-18に公開済み記事を全件計測して判明）。
# 指示するだけでは効かないので、生成後に実測して不足なら書き直させる。
MIN_BODY_CHARS = 650


def body_char_count(html: str) -> int:
    """HTMLタグと空白を除いた本文の実文字数（日本語なので文字数がそのまま分量になる）。"""
    return len(re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html or "")))


def body_quality_key(body: str, min_chars: int) -> tuple:
    """本文の採用優先度（大きいほど良い）。字数充足 > AI常套句の少なさ > 字数の順で比較する。
    publish_buyback_articles.py の再生成判定でも使う。"""
    return (body_char_count(body) >= min_chars, -len(find_ai_tells(body)), body_char_count(body))


def generate_article_body_checked(fact_sheet: dict) -> "dict | None":
    """generate_article_body()の結果が短すぎる・AI常套句（lib/writing_style.py）を含む場合は
    1回だけ再生成する。2回目も直らなければマシな方を採用する（記事を落とすよりは公開する）。

    文末の単調さ（「ます。」4連続など）は再生成のきっかけにしない（find_ai_tells の
    include_monotone=False）。2026-08-28の実測で再生成412回中335回がこれだけを理由にしており、
    引き直しても164記事中63記事で解消していなかった。検出は残してログに出し、抑制は
    プロンプト（JA_STYLE_RULES）側で行う。"""
    first = generate_article_body(fact_sheet)
    if first is None:
        return None
    first_len = body_char_count(first.get("body", ""))
    tells = find_ai_tells(first.get("body", ""), include_monotone=False)
    if first_len >= MIN_BODY_CHARS and not tells:
        monotone = [t for t in find_ai_tells(first.get("body", "")) if t.startswith("文末単調")]
        if monotone:
            print(f"    ・{monotone[0]}（再生成はしない）")
        return first
    reason = f"本文{first_len}字（下限{MIN_BODY_CHARS}字）" if first_len < MIN_BODY_CHARS else f"AI常套句{tells}"
    print(f"    ↻ {reason}のため再生成")
    second = generate_article_body(fact_sheet)
    if second is None:
        return first
    if body_quality_key(second.get("body", ""), MIN_BODY_CHARS) > body_quality_key(first.get("body", ""), MIN_BODY_CHARS):
        return second
    return first


class MicroCMSPermissionError(Exception):
    """APIキーの権限不足など、リトライしても直らない投稿エラー。"""


_UNEXPECTED_TYPE_RE = re.compile(r"'(\w+)' has unexpected data type")


def attach_figures(payload: dict, figures: list) -> int:
    """web.article_figures が作った解説図をmicroCMSへアップロードし、本文（日本語・英語の
    両方）の該当段落の直後に差し込む。差し込めた枚数を返す。アップロードに失敗した図は黙って
    飛ばす（図が1枚も無くても記事は従来どおり公開する）。英語本文は段落と図の対応を語句で
    探せないため、日本語と同じ順序で均等に配置する。"""
    ja = []
    for fig in figures:
        url = _upload_media(fig["bytes"], fig["filename"])
        if not url:
            continue
        ja.append({"html": figure_html(url, fig["alt"], fig["caption"]), "anchors": fig["anchors"]})
    if not ja:
        return 0
    payload["body"] = insert_figures_into_body(payload["body"], ja)
    return len(ja)


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
    tools/reclassify_blog_articles.py の一括再分類・tools/apply_rewritten_articles.py の
    本文差し替えで使う。以前はPATCH権限が無いAPIキーでも動くようPUTを使っていたが、
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
# この数値は kujira-watch/src/lib/faqData.tsx のFAQ「すべての大量保有報告書が記事に
# なっていますか？」で読者にも公開している。変更時は両方を同じコミットで直すこと。
# 2026-08-29に 3.0億円/1.0pt から引き上げ（Anthropic APIの消化削減。直近30日の開示851件で
# 通過が693件→529件＝-24%。記事1本あたり約$0.013なので月-$2.2）。
MIN_DEAL_AMOUNT_OKU = 5.0
MIN_RATIO_CHANGE_PT = 1.5

# **公開済み記事**をindexするか・掃除で消すかの基準。表示側 kujira-watch/src/lib/
# articleIndexability.ts の INDEXABLE_MIN_* と必ず同じ値にすること（ずれると
# 「サイトマップに載っているのにnoindex」という矛盾した指示をGoogleに送る）。
# 新規記事の足切り(MIN_*)を2026-08-29に引き上げた後も、こちらは据え置く。
# 引き上げに合わせて下げると、既に順位が付いている既存記事の24%をnoindexに落とすことになり、
# 節約する月$2.2に対して失うものが大きすぎる。新規記事は必ずMIN_*≥INDEXABLE_MIN_*なので
# 「出したのにnoindex」は起きない。
INDEXABLE_MIN_DEAL_AMOUNT_OKU = 3.0
INDEXABLE_MIN_RATIO_CHANGE_PT = 1.0


def is_worth_publishing(deal_amount_oku: float, ratio_change_pt: float) -> bool:
    """推定金額か保有比率の変化幅のどちらかが基準を超える開示だけを記事にする（新規記事用）。"""
    if deal_amount_oku >= MIN_DEAL_AMOUNT_OKU:
        return True
    return abs(ratio_change_pt) >= MIN_RATIO_CHANGE_PT


def is_indexable_article(deal_amount_oku: float, ratio_change_pt: float) -> bool:
    """公開済み記事がインデックス対象か（＝サイトに残す価値があるか）。
    既存記事の掃除・アイキャッチ補完はこちらで判定する。新規記事の足切りは
    is_worth_publishing() で、こちらより厳しい。"""
    if deal_amount_oku >= INDEXABLE_MIN_DEAL_AMOUNT_OKU:
        return True
    return abs(ratio_change_pt) >= INDEXABLE_MIN_RATIO_CHANGE_PT


def published_holding_keys(days: int) -> "set | None":
    """直近days日ぶんの既報記事の (銘柄コード, 開示日, 提出者名) 集合。取得失敗時は None。"""
    since = (date.today() - timedelta(days=days)).isoformat()
    rows = fetch_published_index(since, fields="stockCode,dealDate,filerName")
    if rows is None:
        return None
    return {
        (str(r.get("stockCode") or ""), str(r.get("dealDate") or "")[:10], r.get("filerName") or "")
        for r in rows
    }


def estimated_amounts(days: int) -> dict:
    """edinet_holding_amounts（開示1件ごとの推定売買金額ビュー）を doc_id 引きの dict で返す。
    取得できなければ空dict（=足切りせずに全件見る）。"""
    since = (date.today() - timedelta(days=days)).isoformat()
    try:
        rows = sb.select("edinet_holding_amounts",
                         f"disc_date=gte.{since}&select=doc_id,deal_amount_oku,ratio_change_pt")
    except Exception as e:
        print(f"  ⚠ 推定金額ビューを取得できないため足切りなしで続行: {e}")
        return {}
    return {r["doc_id"]: r for r in rows}


def is_backfill_target(h: dict, published_keys: set, amounts: dict) -> bool:
    """backfillの事前足切り。記事を作ったことがある開示と、推定金額ビューの時点で足切り基準に
    届かない開示を、yfinance（発行済株式数・終値）とmicroCMSを叩く前に落とす。

    30日窓の候補は1,000件を超え、1件ずつ株数と終値を引くと1回の便では終わらない。
    ビューに行が無い開示（株数が取れない銘柄・ビュー更新前の開示）は判定できないので残し、
    従来どおりループ内で概算する。"""
    # microCMSに記事が無くても、過去に作ったことがあるなら作り直さない。
    # 低品質・リライト不能・誤報として意図的に削除した記事（2026-08-18に129件、08-25に74件、
    # 08-27に12件）を、取りこぼしと誤認して復活させないため。
    if h.get("article_published_at"):
        return False
    key = (str(h["issuer_code"]), str(h["disc_date"])[:10], h.get("filer_name") or "")
    if key in published_keys:
        return False
    v = amounts.get(h.get("doc_id"))
    if v is None:
        return True
    return is_worth_publishing(v.get("deal_amount_oku") or 0.0, v.get("ratio_change_pt") or 0.0)


def build_and_publish(days: int = LARGE_HOLDINGS_DAYS, max_articles: "int | None" = None,
                       dry_run: bool = False, backfill: bool = False,
                       ledger: "PublishLedger | None" = None) -> list:
    # ledger は候補1件ごとの結末を記録する台帳。「候補はあったのに公開0件」の原因
    # （正常な見送りか、生成・投稿の失敗か）を分類して終了コードに出す。
    ledger = ledger if ledger is not None else PublishLedger("publish_blog_articles")
    if not dry_run and (not MICROCMS_DOMAIN or not MICROCMS_KEY):
        print("[publish_blog_articles] MICROCMS_SERVICE_DOMAIN / MICROCMS_API_KEY 未設定のためスキップ")
        return []

    published_keys, amounts = None, {}
    if backfill:
        days = max(days, BACKFILL_DAYS)
        if max_articles is None:
            max_articles = BACKFILL_MAX_ARTICLES
        published_keys = published_holding_keys(days)
        if published_keys is None:
            print("[publish_blog_articles] 既報インデックスを取得できないため backfill を中止（重複投稿を避ける）")
            return []
        amounts = estimated_amounts(days)

    holdings = get_recent_large_holdings(days=days)
    candidates = [
        h for h in holdings
        if h.get("issuer_code") and h.get("holding_ratio") is not None
    ]
    if backfill:
        candidates = [h for h in candidates if is_backfill_target(h, published_keys, amounts)]
        # 古い開示ほど窓（BACKFILL_DAYS）から外れて永久に失われるので先に消化する
        candidates.sort(key=lambda h: (h["disc_date"], -abs(h["holding_ratio"])))
        print(f"[publish_blog_articles] backfill: 直近{days}日の未記事化候補 {len(candidates)}件")
    else:
        candidates.sort(key=lambda h: abs(h["holding_ratio"]), reverse=True)
    ledger.start(len(candidates))
    # 同一銘柄・同一開示日・同一提出者の開示が何件あるか（重複判定の緩和条件に使う）
    filing_counts = Counter(
        (str(h["issuer_code"]), h["disc_date"], h.get("filer_name", "")) for h in candidates
    )

    published = []
    for h in candidates:
        if max_articles is not None and len(published) >= max_articles:
            ledger.stop_early(pl.SKIP_MAX_ARTICLES)
            break
        code = str(h["issuer_code"])
        disc_date = h["disc_date"]
        filer_name = h.get("filer_name", "")
        name = h.get("name") or code

        prior_ratio = h.get("holding_ratio_prior")
        is_correction = is_correction_report(h.get("doc_description") or "")
        if should_wait_for_prior_ratio(h.get("doc_description") or "", prior_ratio, disc_date):
            print(f"  ⏭ {name}({code}): 変更報告書だが直前保有割合が未取得のため次の便へ持ち越し")
            ledger.skip(pl.SKIP_WAIT_NEXT_RUN, f"{name}({code})")
            continue
        change = ratio_change_pct(
            code, filer_name, h["holding_ratio"], disc_date, prior_ratio,
            is_change_report(h.get("doc_description") or ""),
        )
        if change is None:
            # 待っても直前保有割合が入らなかった変更報告書。全量を動いたとみなすと
            # 「X%を新規保有」＋過大な推定金額になるため記事化しない。
            print(f"  ⏭ {name}({code}): 変更報告書だが直前保有割合を取得できず変化幅を確定できないためスキップ")
            ledger.skip(pl.SKIP_NO_PRIOR_RATIO, f"{name}({code})")
            continue
        if change <= 0:
            print(f"  ⏭ {name}({code}): 前回開示から保有比率が変わっていないためスキップ")
            ledger.skip(pl.SKIP_NO_RATIO_CHANGE, f"{name}({code})")
            continue
        # 短期大量譲渡の開示には「譲渡の相手方・単価」が載っており、単価×数量で実額が出せる。
        # 実額が出せた開示では概算を使わない（実例: 日立製作所→日立建機は開示日終値ベースの
        # 概算1,274.9億円に対し、開示単価5,227円ベースの実額は1,121.8億円）。
        transfers = summarize_disposals(h.get("short_term_transfers") or [], change)
        if is_correction:
            # 訂正は売買を伴わないため推定金額を出さない（記事化の可否は比率変化幅で判断する）
            deal_amount = 0.0
        elif transfers["amount_oku"] is not None:
            deal_amount = transfers["amount_oku"]
        else:
            deal_amount = estimate_deal_amount_oku(code, change, disc_date)
            if deal_amount is None:
                print(f"  ⏭ {name}({code}): 株価または発行済株式数を取得できず金額を概算できないためスキップ")
                ledger.skip(pl.SKIP_NO_AMOUNT, f"{name}({code})")
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
            ledger.skip(pl.SKIP_BELOW_THRESHOLD, f"{name}({code})")
            continue

        if h.get("article_published_at"):
            # 既に記事を作った開示。microCMS上に無いのは意図的に削除したから（低品質・
            # リライト不能・誤報）なので、作り直さない。
            ledger.skip(pl.SKIP_ALREADY_ARTICLED, f"{name}({code})")
            continue
        unique_filing = filing_counts[(code, disc_date, filer_name)] == 1
        if already_published(code, disc_date, deal_amount, filer_name, signed_change, unique_filing):
            ledger.skip(pl.SKIP_ALREADY_PUBLISHED, f"{name}({code})")
            continue

        # 株価は金額の概算に使ったものと同じ値を本文へ渡す（サイトの「基準終値」とも同じ源）。
        context_close = disclosure_close_price(code, disc_date)
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
            "deal_amount_label": deal_amount_label(is_sell, transfers["amount_oku"] is not None),
            "transfers": transfers,
            "context_close": context_close,
            "filer_description": filer_info.get("description") or "",
            "company_description": company_description,
            "ratio_change_pct": change,
            "prior_ratio": prior_ratio,
            "is_correction": is_correction,
            # 開示1件の数字だけでは記事がテンプレートになるため、開示を横断して初めて書ける
            # 事実（保有の積み上げ履歴・他の保有銘柄・他の大株主・指標）を足す。
            "context_facts": build_context_facts(code, filer_name, disc_date),
        }
        article = generate_article_body_checked(fact_sheet)
        if article is None:
            print(f"  ⏭ {name}({code}): 記事生成に失敗したためスキップ")
            ledger.skip(pl.FAIL_GENERATION, f"{name}({code})")
            continue
        titles = build_article_titles(fact_sheet)

        deal_type = filer_info["category"]
        # 訂正記事はフロント側（isCorrectionArticle）で金額の代わりに「訂正」と表示するため、
        # tagsで方向（売り）とは別に区別できるようにする（microCMSのスキーマ変更を避ける方針）。
        tag_list = ["EDINET", "自動生成"]
        if is_correction:
            tag_list.append("訂正")
        if is_sell:
            tag_list.append("売り")
        tags = ",".join(tag_list)
        payload = {
            "title": titles["title"],
            "body": article["body"],
            "stockName": name,
            "stockCode": code,
            "dealType": [deal_type],  # microCMSのセレクト(複数選択)は配列。文字列だと400→再送信が毎回発生していた
            "dealDate": f"{disc_date}T00:00:00.000Z",
            "dealAmount": deal_amount,
            "ratioChangePct": signed_change,
            "tags": tags,
            "filerName": filer_name,
        }

        direction_mark = "📝訂正" if is_correction else ("📉売り" if is_sell else "📈買い")
        amount_estimated = transfers["amount_oku"] is None
        amount_mark = (f"{signed_change:+.2f}pt" if is_correction
                       else f"{'推定' if amount_estimated else '開示単価'}{deal_amount}億円")
        # holdingRatioはX投稿の1行目（「〇%まで買い増し」）と数字カード画像で使う。
        # microCMSのスキーマは変えたくないので、送信するpayloadではなく戻り値にだけ足す。
        if dry_run:
            print(f"  [dry-run] {direction_mark} {name}({code}) {disc_date} {amount_mark}\n    title: {payload['title']}")
            published.append({**payload, "id": None, "holdingRatio": h["holding_ratio"]})
            ledger.publish(f"{name}({code})")
            continue

        if is_correction:
            badge_label = "📝 訂正"
        elif is_sell:
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

        figure_count = attach_figures(payload, build_article_figures(fact_sheet))

        chart_url = build_price_chart_for_article(code, name)
        if chart_url:
            payload["body"] += (
                f'<figure><img src="{chart_url}" alt="{name}（{code}）株価推移（直近3ヶ月）">'
                f'<figcaption>{name}（{code}）の株価推移（直近3ヶ月・終値ベース）</figcaption></figure>'
            )
            figure_count += 1

        try:
            content_id = publish_article(payload)
        except MicroCMSPermissionError as e:
            print(f"  ✖ 権限エラーのため以降の候補もスキップして終了します: {e}")
            ledger.stop_early(pl.FAIL_PERMISSION, f"{name}({code})")
            break
        if content_id:
            print(f"  ✅ 投稿: {direction_mark} {name}({code}) {disc_date} {amount_mark} 図{figure_count}枚 → id={content_id}")
            if h.get("doc_id"):
                mark_article_published(h["doc_id"])
            published.append({**payload, "id": content_id, "holdingRatio": h["holding_ratio"]})
            ledger.publish(f"{name}({code})")
        else:
            print(f"  ⚠ {name}({code}): 投稿に失敗")
            ledger.skip(pl.FAIL_PUBLISH, f"{name}({code})")

    return published


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=LARGE_HOLDINGS_DAYS, help="EDINET開示を見る直近日数")
    p.add_argument("--max-articles", type=int, default=None, help="1回の実行で投稿する上限件数（未指定なら上限なし）")
    p.add_argument("--dry-run", action="store_true", help="microCMSへ投稿せず内容を表示するのみ")
    p.add_argument("--backfill", action="store_true",
                   help=f"直近{BACKFILL_DAYS}日まで遡り、記事化されていない開示だけを拾い直す")
    args = p.parse_args()

    ledger = PublishLedger("publish_blog_articles")
    results = build_and_publish(days=args.days, max_articles=args.max_articles,
                                dry_run=args.dry_run, backfill=args.backfill, ledger=ledger)
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{len(results)}件処理しました。")

    # backfillは数日前の開示を拾い直す便なので、Xには流さない（タイムラインに古い開示が並ぶ）
    if not args.dry_run and not args.backfill:
        from web.x_client import post_daily_summary, post_top_articles
        featured_ids = get_featured_article_ids()
        posted = post_top_articles(results, featured_ids)
        if posted:
            print(f"🐦 X投稿: {posted}件")
        # 「本日のクジラ」日次サマリー(21時JSTの最終便のみ投稿される。時刻ガードはx_client側)
        post_daily_summary()

    # 内訳は毎回出す（日次ログレビューが「候補はあったが全部基準未満」を読めるように）。
    # 生成・投稿の失敗があったときだけLINE通知＋終了コード4でワークフローを赤くする。
    return ledger.finish()


if __name__ == "__main__":
    sys.exit(main() or 0)

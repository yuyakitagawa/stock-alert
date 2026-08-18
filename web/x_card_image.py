"""
web/x_card_image.py

X投稿に添付する「数字カード」画像（1200x675, 16:9）を生成する。

これまでXに添付していたのは記事本文と同じ株価チャート
（publish_blog_articles.generate_price_chart_image）だけで、投稿の主張である
「誰が・どの銘柄を・保有比率何%・いくら」を画像が1つも伝えていなかった。
タイムラインでは画像がほぼ唯一の視認要素になるため、1枚目を数字カード、
2枚目をチャートにする（post_top_articlesが2枚組でアップロードする）。

依存は既存のPillowのみ（matplotlib等は増やさない）。フォントが見つからない場合は
Noneを返し、呼び出し側は画像なしで投稿を続行する。
"""
import io
import os

# 本番(GitHub Actions)はワークフローでNoto Sans CJKを入れている。ローカル(mac)確認用に
# ヒラギノもフォールバックに入れる（どちらも無ければカード生成をあきらめる）。
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
]

CARD_W, CARD_H = 1200, 675
NAVY = (16, 42, 67)
WHITE = (255, 255, 255)
INK = (23, 30, 38)
MUTED = (110, 122, 134)
GREEN = (21, 128, 79)      # 買い（色覚多様性に配慮して青緑寄り）
RED = (190, 45, 45)        # 売り
AMBER = (176, 106, 12)     # 訂正
HAIRLINE = (223, 228, 233)

BRAND = "KUJIRA WATCH"
SITE_LABEL = "kujira-watch.com"
DISCLAIMER = "EDINET提出書類より作成。投資助言ではありません"


def _font_path() -> "str | None":
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _font(size: int):
    from PIL import ImageFont

    path = _font_path()
    if path is None:
        return None
    return ImageFont.truetype(path, size, index=0)


def _fit(draw, text: str, font, max_w: int) -> str:
    """max_wに収まるまで末尾を削って「…」を付ける。社名・ファンド名は長さが読めないため。"""
    if draw.textlength(text, font=font) <= max_w:
        return text
    ellipsis = "…"
    out = ""
    for char in text:
        if draw.textlength(out + char + ellipsis, font=font) > max_w:
            break
        out += char
    return (out + ellipsis) if out else ellipsis


def _base_canvas():
    """ヘッダー帯・フッターだけ入った共通の下地。戻り値は (img, draw)。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CARD_W, CARD_H), WHITE)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, CARD_W, 74], fill=NAVY)
    brand_font = _font(30)
    small_font = _font(20)
    if brand_font:
        draw.text((48, 22), BRAND, font=brand_font, fill=WHITE)
    if small_font:
        draw.text((CARD_W - 48 - draw.textlength(SITE_LABEL, font=small_font), 28),
                  SITE_LABEL, font=small_font, fill=(178, 196, 214))
        draw.line([(48, CARD_H - 74), (CARD_W - 48, CARD_H - 74)], fill=HAIRLINE, width=2)
        draw.text((48, CARD_H - 54), DISCLAIMER, font=small_font, fill=MUTED)
    return img, draw


def _to_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_deal_card(stock_name: str, stock_code: str, filer_name: str, badge: str,
                    ratio: "float | None", ratio_prior: "float | None",
                    amount_label: str, date_label: str, kind: str = "buy") -> "bytes | None":
    """記事投稿の1枚目。銘柄・提出者・保有比率の変化・金額を1枚で読ませる。
    kindは buy / sell / correction（バッジと数字の色に使う）。"""
    if _font_path() is None:
        print("  ⚠ カード画像用のフォントが見つからないため画像なしで投稿します")
        return None
    try:
        accent = {"sell": RED, "correction": AMBER}.get(kind, GREEN)
        img, draw = _base_canvas()

        badge_font = _font(26)
        filer_font = _font(30)
        stock_font = _font(66)
        ratio_font = _font(80)
        label_font = _font(24)
        amount_font = _font(46)

        # バッジ（新規取得 / 買い増し / 売却 / 訂正）
        badge_w = draw.textlength(badge, font=badge_font) + 40
        draw.rounded_rectangle([48, 110, 48 + badge_w, 158], radius=24, fill=accent)
        draw.text((68, 121), badge, font=badge_font, fill=WHITE)

        # 提出者（主語なので銘柄の上に置く）
        draw.text((48 + badge_w + 20, 124),
                  _fit(draw, filer_name or "大口投資家", filer_font, CARD_W - 150 - badge_w),
                  font=filer_font, fill=MUTED)

        # 銘柄名（最大サイズ）＋証券コード
        stock_label = f"{stock_name}（{stock_code}）" if stock_code else stock_name
        draw.text((48, 190), _fit(draw, stock_label, stock_font, CARD_W - 96),
                  font=stock_font, fill=INK)

        # 保有比率の変化（この投稿の主張）
        draw.text((48, 310), "保有比率", font=label_font, fill=MUTED)
        if ratio is not None:
            prior_text = f"{ratio_prior:.2f}%  →  " if ratio_prior is not None else ""
            draw.text((48, 344), prior_text, font=ratio_font, fill=MUTED)
            draw.text((48 + draw.textlength(prior_text, font=ratio_font), 344),
                      f"{ratio:.2f}%", font=ratio_font, fill=accent)
        else:
            draw.text((48, 344), "―", font=ratio_font, fill=MUTED)

        # 金額（副次情報として右下）
        draw.text((48, 470), amount_label, font=amount_font, fill=INK)
        if date_label:
            draw.text((CARD_W - 48 - draw.textlength(date_label, font=label_font), 490),
                      date_label, font=label_font, fill=MUTED)
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ カード画像生成に失敗: {e}")
        return None


def build_list_card(title: str, subtitle: str, rows: list, footer: str = "") -> "bytes | None":
    """日次サマリー・週次ランキング用の一覧カード。rowsは (左ラベル, 右の値, 色種別) のタプル。
    色種別は buy / sell / none。"""
    if _font_path() is None:
        return None
    try:
        img, draw = _base_canvas()
        title_font = _font(44)
        sub_font = _font(28)
        row_font = _font(34)
        value_font = _font(34)

        draw.text((48, 110), _fit(draw, title, title_font, CARD_W - 96), font=title_font, fill=INK)
        if subtitle:
            draw.text((48, 168), _fit(draw, subtitle, sub_font, CARD_W - 96), font=sub_font, fill=MUTED)

        y = 230
        for left, right, tone in rows[:6]:
            color = {"buy": GREEN, "sell": RED}.get(tone, INK)
            right_w = draw.textlength(right, font=value_font)
            draw.text((48, y), _fit(draw, left, row_font, CARD_W - 140 - right_w),
                      font=row_font, fill=INK)
            draw.text((CARD_W - 48 - right_w, y), right, font=value_font, fill=color)
            y += 58
            draw.line([(48, y - 12), (CARD_W - 48, y - 12)], fill=HAIRLINE, width=1)

        if footer:
            draw.text((48, CARD_H - 108), _fit(draw, footer, sub_font, CARD_W - 96),
                      font=sub_font, fill=MUTED)
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ 一覧カード画像生成に失敗: {e}")
        return None

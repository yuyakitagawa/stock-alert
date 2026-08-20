"""
web/x_card_image.py

X投稿に添付する「数字カード」画像（1200x675, 16:9）を生成する。

これまでXに添付していたのは記事本文と同じ株価チャートだけで、投稿の主張である
「誰が・どの銘柄を・保有比率何%・いくら」を画像が1つも伝えていなかった。
タイムラインでは画像がほぼ唯一の視認要素になるため、1枚目を数字カード、
2枚目をチャートにする（post_top_articlesが2枚組でアップロードする）。

配色は kujira-watch/src/app/globals.css と video/remotion/src/theme.ts と同じ
ブランドトークンのみを使う（媒体をまたいで色が食い違うと、Xから来た読者が
サイトを別物と感じるため）。独自色は増やさない。

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
PAD = 64
HEADER_H = 78
FOOTER_Y = CARD_H - 76

# --- ブランドトークン（globals.css / remotion theme と同値。新規色の追加は禁止） ---
NAVY = (22, 33, 58)        # --brand-navy #16213a
NAVY_DEEP = (13, 21, 38)   # navyDeep #0d1526
PAPER = (255, 253, 248)    # --paper #fffdf8
TINT = (241, 236, 225)     # --section-tint #f1ece1
RULE = (222, 213, 192)     # --rule #ded5c0
GOLD = (184, 134, 58)      # --brand-gold #b8863a
INK = (32, 29, 26)         # --foreground #201d1a
MUTED = (122, 112, 96)     # INKを紙面に馴染ませたミュート（--ruleと同系）
WHITE = (255, 255, 255)
BUY = (4, 120, 87)         # success #047857（買い・上昇）
SELL = (190, 18, 60)       # error #be123c（売り・下落）

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
    out = ""
    for char in text:
        if draw.textlength(out + char + "…", font=font) > max_w:
            break
        out += char
    return (out + "…") if out else "…"


def _stock_line(draw, stock_name: str, stock_code: str, max_w: int):
    """「社名（コード）」を証券コードを落とさずに収め、(表示文字列, フォント) を返す。

    社名が長いと末尾から削られてコードごと「…」に消えていた（実例: セブン&アイ・
    ホールディングス（3382）→「セブン&アイ・ホールディングス（3…」）。コードは銘柄検索で
    拾われる手掛かりなので、まずフォントを段階的に下げて全体を入れ、それでも入らない場合は
    社名側だけを削る。"""
    suffix = f"（{stock_code}）" if stock_code else ""
    for size in (60, 54, 48, 42):
        font = _font(size)
        if draw.textlength(stock_name + suffix, font=font) <= max_w:
            return stock_name + suffix, font
    font = _font(42)
    name = _fit(draw, stock_name, font, max_w - draw.textlength(suffix, font=font))
    return name + suffix, font


def _accent(kind: str):
    return {"sell": SELL, "correction": NAVY_DEEP}.get(kind, BUY)


def _base_canvas():
    """ヘッダー帯・フッターだけ入った共通の下地。戻り値は (img, draw)。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CARD_W, CARD_H), PAPER)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, CARD_W, HEADER_H], fill=NAVY)
    # ヘッダー下の金の細線。サイトのアクセント色を1本だけ入れてブランドの手掛かりにする。
    draw.rectangle([0, HEADER_H, CARD_W, HEADER_H + 4], fill=GOLD)

    brand_font, small_font = _font(28), _font(21)
    if brand_font:
        draw.text((PAD, HEADER_H // 2), BRAND, font=brand_font, fill=WHITE, anchor="lm")
    if small_font:
        draw.text((CARD_W - PAD, HEADER_H // 2), SITE_LABEL, font=small_font,
                  fill=(178, 196, 214), anchor="rm")
        draw.line([(PAD, FOOTER_Y), (CARD_W - PAD, FOOTER_Y)], fill=RULE, width=2)
        draw.text((PAD, FOOTER_Y + 30), DISCLAIMER, font=small_font, fill=MUTED, anchor="lm")
    return img, draw


def _badge(draw, x: int, y_center: int, text: str, color) -> int:
    """左端xに角丸のバッジを描き、右端のxを返す。"""
    font = _font(25)
    w = draw.textlength(text, font=font) + 38
    draw.rounded_rectangle([x, y_center - 23, x + w, y_center + 23], radius=23, fill=color)
    draw.text((x + w / 2, y_center), text, font=font, fill=WHITE, anchor="mm")
    return int(x + w)


def _to_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_deal_card(stock_name: str, stock_code: str, filer_name: str, badge: str,
                    ratio: "float | None", ratio_prior: "float | None",
                    amount_label: str, date_label: str, kind: str = "buy") -> "bytes | None":
    """記事投稿の1枚目。銘柄・提出者・保有比率の変化・金額を1枚で読ませる。
    kindは buy / sell / correction（バッジと数字の色に使う）。

    保有比率の変化がこのカードの主役なので、そこだけ地に色を敷いて他と段を分ける。
    金額は同じ帯の右側に置き、右半分が空白のまま残らないようにする。"""
    if _font_path() is None:
        print("  ⚠ カード画像用のフォントが見つからないため画像なしで投稿します")
        return None
    try:
        accent = _accent(kind)
        img, draw = _base_canvas()

        # 1段目: バッジ＋提出者名
        row1_y = 150
        badge_right = _badge(draw, PAD, row1_y, badge, accent)
        filer_font = _font(29)
        draw.text((badge_right + 20, row1_y),
                  _fit(draw, filer_name or "大口投資家", filer_font, CARD_W - PAD - badge_right - 40),
                  font=filer_font, fill=MUTED, anchor="lm")

        # 2段目: 銘柄名（このカードで一番大きい固有名詞）
        stock_label, stock_font = _stock_line(draw, stock_name, stock_code, CARD_W - PAD * 2)
        draw.text((PAD, 236), stock_label, font=stock_font, fill=INK, anchor="lm")

        # 3段目: 主役の帯（保有比率の変化 ＋ 推定金額）
        band_top, band_bottom = 300, 524
        draw.rounded_rectangle([PAD, band_top, CARD_W - PAD, band_bottom], radius=12, fill=TINT)
        label_font, ratio_font, prior_font = _font(22), _font(76), _font(46)
        amount_font = _font(46)
        inner_x = PAD + 36
        draw.text((inner_x, band_top + 46), "保有比率", font=label_font, fill=MUTED, anchor="lm")

        ratio_y = band_top + 140
        if ratio is not None:
            if ratio_prior is not None:
                prior_text = f"{ratio_prior:.2f}%"
                draw.text((inner_x, ratio_y), prior_text, font=prior_font, fill=MUTED, anchor="lm")
                arrow_x = inner_x + draw.textlength(prior_text, font=prior_font) + 18
                draw.text((arrow_x, ratio_y), "→", font=prior_font, fill=MUTED, anchor="lm")
                ratio_x = arrow_x + draw.textlength("→", font=prior_font) + 18
            else:
                ratio_x = inner_x
            draw.text((ratio_x, ratio_y), f"{ratio:.2f}%", font=ratio_font, fill=accent, anchor="lm")
        else:
            draw.text((inner_x, ratio_y), "―", font=ratio_font, fill=MUTED, anchor="lm")

        # 帯の右側。売買記事は推定金額、訂正報告書は金額を持たないので訂正幅を出す
        # （右半分が空のまま残るのを避けつつ、訂正の大きさを一目で伝える）。
        right_x = CARD_W - PAD - 36
        if kind == "correction" and ratio is not None and ratio_prior is not None:
            delta = ratio - ratio_prior
            draw.text((right_x, band_top + 46), "訂正幅", font=label_font, fill=MUTED, anchor="rm")
            draw.text((right_x, ratio_y), f"{delta:+.2f}pt", font=amount_font,
                      fill=SELL if delta < 0 else BUY, anchor="rm")
        elif amount_label and amount_label != "―":
            amount_title = "推定売却金額" if kind == "sell" else "推定取得金額"
            draw.text((right_x, band_top + 46), amount_title, font=label_font, fill=MUTED, anchor="rm")
            draw.text((right_x, ratio_y), amount_label, font=amount_font, fill=INK, anchor="rm")

        if date_label:
            draw.text((CARD_W - PAD, FOOTER_Y + 30), date_label, font=_font(21),
                      fill=MUTED, anchor="rm")
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ カード画像生成に失敗: {e}")
        return None


def build_list_card(title: str, subtitle: str, rows: list, footer: str = "") -> "bytes | None":
    """日次サマリー・週次ランキング用の一覧カード。rowsは (左ラベル, 右の値, 色種別) のタプル。
    色種別は buy / sell / none。

    行数は日によって2〜6件と振れるため、行の高さを残り高さから決めて縦の空白を作らない。"""
    if _font_path() is None:
        return None
    try:
        img, draw = _base_canvas()
        title_font, sub_font = _font(42), _font(26)
        row_font, value_font = _font(33), _font(33)

        draw.text((PAD, 138), _fit(draw, title, title_font, CARD_W - PAD * 2),
                  font=title_font, fill=INK, anchor="lm")
        if subtitle:
            draw.text((PAD, 194), _fit(draw, subtitle, sub_font, CARD_W - PAD * 2),
                      font=sub_font, fill=MUTED, anchor="lm")

        rows = rows[:6]
        if rows:
            area_top, area_bottom = 236, FOOTER_Y - (46 if footer else 12)
            row_h = min(84, max(52, (area_bottom - area_top) // len(rows)))
            # 2件しか無い日でも下半分が空かないよう、行ブロックごと領域の中央に置く
            y = area_top + max(0, (area_bottom - area_top - row_h * len(rows)) // 2)
            for i, (left, right, tone) in enumerate(rows):
                color = {"buy": BUY, "sell": SELL}.get(tone, INK)
                if i % 2 == 0:
                    draw.rounded_rectangle([PAD, y, CARD_W - PAD, y + row_h - 6], radius=8, fill=TINT)
                center = y + (row_h - 6) / 2
                right_w = draw.textlength(right, font=value_font)
                draw.text((PAD + 24, center), _fit(draw, left, row_font, CARD_W - PAD * 2 - 72 - right_w),
                          font=row_font, fill=INK, anchor="lm")
                draw.text((CARD_W - PAD - 24, center), right, font=value_font, fill=color, anchor="rm")
                y += row_h

        if footer:
            draw.text((PAD, FOOTER_Y - 26), _fit(draw, footer, sub_font, CARD_W - PAD * 2),
                      font=sub_font, fill=MUTED, anchor="lm")
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ 一覧カード画像生成に失敗: {e}")
        return None

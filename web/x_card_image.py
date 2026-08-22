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
PAD = 56
SPLIT_X = 540            # 左ネイビーパネルの右端（Canva案3の分割比 45:55 を踏襲）
FOOTER_H = 64            # 下端の濃紺フッター帯
FOOTER_Y = CARD_H - FOOTER_H

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

BRAND = "大口投資家の監視ブログ"
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


def _wrap(draw, text: str, font, max_w: int) -> list:
    lines, cur = [], ""
    for ch in text:
        if cur and draw.textlength(cur + ch, font=font) > max_w:
            lines.append(cur)
            cur = ""
        cur += ch
    return lines + [cur] if cur else lines


def _stock_block(draw, stock_name: str, stock_code: str, x: int, y: int, max_w: int):
    """「社名（コード）」を左パネルに最大2行で描く。証券コードは銘柄検索で拾われる手掛かりなので、
    「（7 / 203）」のように行またぎで割らず、社名が長い場合も最終行の末尾に必ず丸ごと残す。
    フォントを段階的に下げ、2行でも入らなければ社名側だけを削る。"""
    suffix = f"（{stock_code}）" if stock_code else ""
    for size in (52, 46, 40):
        font = _font(size)
        lines = _wrap(draw, stock_name, font, max_w)
        if suffix:
            if lines and draw.textlength(lines[-1] + suffix, font=font) <= max_w:
                lines[-1] += suffix
            else:
                lines.append(suffix)
        if len(lines) <= 2:
            break
    else:
        name = _fit(draw, stock_name, font, max_w * 2 - draw.textlength(suffix, font=font) - 8)
        lines = _wrap(draw, name, font, max_w)
        if suffix:
            if lines and draw.textlength(lines[-1] + suffix, font=font) <= max_w:
                lines[-1] += suffix
            else:
                lines.append(suffix)
        lines = lines[:2]
    line_h = int(font.size * 1.25)
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=font, fill=WHITE, anchor="lm")
    return lines, font


def _shrink_to_fit(draw, text: str, sizes, max_w: int):
    """フォントを段階的に下げて全体を収める。最小サイズでも入らなければ末尾を「…」にする。"""
    for size in sizes:
        font = _font(size)
        if draw.textlength(text, font=font) <= max_w:
            return text, font
    return _fit(draw, text, font, max_w), font


def _accent(kind: str):
    return {"sell": SELL, "correction": NAVY_DEEP}.get(kind, BUY)


def _base_canvas(split: bool = True):
    """共通の下地。Canvaで出した案（左: ネイビーのパネル、右: 紙色、下: 濃紺の帯）を移植。
    split=False は一覧カード用で、上部をネイビーのタイトル帯にする。戻り値は (img, draw)。

    以前はヘッダー帯＋白地のみで、スクロール中のタイムラインで他のテキストカードに埋もれていた。
    面積の半分近くを濃色にして、数字側の紙色との明暗差で主役の数字へ視線を誘導する。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (CARD_W, CARD_H), PAPER)
    draw = ImageDraw.Draw(img)
    if split:
        draw.rectangle([0, 0, SPLIT_X, FOOTER_Y], fill=NAVY)
        draw.rectangle([SPLIT_X, 0, SPLIT_X + 5, FOOTER_Y], fill=GOLD)
    draw.rectangle([0, FOOTER_Y, CARD_W, CARD_H], fill=NAVY_DEEP)
    draw.rectangle([0, FOOTER_Y, CARD_W, FOOTER_Y + 3], fill=GOLD)

    small_font = _font(21)
    if small_font:
        draw.text((PAD, FOOTER_Y + FOOTER_H // 2), DISCLAIMER, font=small_font,
                  fill=(178, 196, 214), anchor="lm")
    return img, draw


def _footer_right(draw, text: str):
    """フッター帯の右端（日付など）。"""
    if text:
        draw.text((CARD_W - PAD, FOOTER_Y + FOOTER_H // 2), text, font=_font(21),
                  fill=(178, 196, 214), anchor="rm")


def _whale(draw, cx: int, cy: int, color):
    """ネイビーパネルの隅に置く控えめなクジラのシルエット（サイトのモチーフ）。
    胴体の楕円＋腹側の膨らみ＋尾びれだけの幾何図形で、装飾というよりブランドの手掛かり。"""
    draw.ellipse([cx - 160, cy - 40, cx + 80, cy + 40], fill=color)
    draw.ellipse([cx - 120, cy - 10, cx + 40, cy + 56], fill=color)
    draw.polygon([(cx + 60, cy - 4), (cx + 150, cy - 62), (cx + 128, cy + 2), (cx + 150, cy + 60)], fill=color)


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

    左のネイビーパネルに「誰が・どの銘柄を」、右の紙面に「何%→何%・いくら」。
    保有比率の数字をカード内で最大（112px）にして、タイムラインの縮小表示でも数字だけは読める。"""
    if _font_path() is None:
        print("  ⚠ カード画像用のフォントが見つからないため画像なしで投稿します")
        return None
    try:
        accent = _accent(kind)
        img, draw = _base_canvas(split=True)
        left_w = SPLIT_X - PAD * 2

        # --- 左パネル: ブランド → バッジ → 提出者 → 銘柄名 ---
        _whale(draw, SPLIT_X - 190, FOOTER_Y - 90, (30, 44, 74))
        draw.text((PAD, 52), BRAND, font=_font(24), fill=(178, 196, 214), anchor="lm")
        # 訂正はネイビー地に沈むので、バッジ色は金にして地と分ける
        _badge(draw, PAD, 130, badge, GOLD if kind == "correction" else accent)
        filer_text, filer_font = _shrink_to_fit(draw, filer_name or "大口投資家", (26, 23, 20), left_w)
        draw.text((PAD, 196), filer_text, font=filer_font, fill=(178, 196, 214), anchor="lm")
        # 銘柄名は左パネル幅に2行まで折り返す（コードは最終行の末尾に必ず残す）
        _stock_block(draw, stock_name, stock_code, PAD, 250, left_w)

        # --- 右パネル: 保有比率（主役）→ 金額 ---
        rx = SPLIT_X + 5 + 48
        right_w = CARD_W - PAD - rx
        label_font, big_font, prior_font, amount_font = _font(24), _font(112), _font(40), _font(54)
        draw.text((rx, 64), "保有比率", font=label_font, fill=MUTED, anchor="lm")
        if ratio is not None:
            if ratio_prior is not None:
                draw.text((rx, 118), f"{ratio_prior:.2f}% →", font=prior_font, fill=MUTED, anchor="lm")
            ratio_text = f"{ratio:.2f}%"
            big = big_font if draw.textlength(ratio_text, font=big_font) <= right_w else _font(92)
            draw.text((rx - 4, 230), ratio_text, font=big, fill=accent, anchor="lm")
        else:
            draw.text((rx, 230), "―", font=big_font, fill=MUTED, anchor="lm")

        # 下段: 売買記事は推定金額、訂正報告書は金額を持たないので訂正幅を出す
        draw.line([(rx, 330), (CARD_W - PAD, 330)], fill=RULE, width=2)
        if kind == "correction" and ratio is not None and ratio_prior is not None:
            delta = ratio - ratio_prior
            draw.text((rx, 376), "訂正幅", font=label_font, fill=MUTED, anchor="lm")
            draw.text((rx, 446), f"{delta:+.2f}pt", font=amount_font,
                      fill=SELL if delta < 0 else BUY, anchor="lm")
        elif amount_label and amount_label != "―":
            amount_title = "推定売却金額" if kind == "sell" else "推定取得金額"
            draw.text((rx, 376), amount_title, font=label_font, fill=MUTED, anchor="lm")
            draw.text((rx, 446), _fit(draw, amount_label, amount_font, right_w),
                      font=amount_font, fill=INK, anchor="lm")
        draw.text((CARD_W - PAD, 64), SITE_LABEL, font=_font(21), fill=MUTED, anchor="rm")

        _footer_right(draw, date_label)
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ カード画像生成に失敗: {e}")
        return None


def build_list_card(title: str, subtitle: str, rows: list, footer: str = "") -> "bytes | None":
    """日次サマリー・週次ランキング用の一覧カード。rowsは (左ラベル, 右の値, 色種別) のタプル。
    色種別は buy / sell / none。

    上部をネイビーの見出し帯（記事カードの左パネルと同じ色）にして、下の紙面に行を並べる。
    行数は日によって2〜6件と振れるため、行の高さを残り高さから決めて縦の空白を作らない。"""
    if _font_path() is None:
        return None
    try:
        img, draw = _base_canvas(split=False)
        head_h = 150
        draw.rectangle([0, 0, CARD_W, head_h], fill=NAVY)
        draw.rectangle([0, head_h, CARD_W, head_h + 5], fill=GOLD)
        _whale(draw, CARD_W - 170, 78, (30, 44, 74))
        title_font, sub_font = _font(44), _font(24)
        row_font, value_font = _font(33), _font(36)

        draw.text((PAD, 30), BRAND, font=_font(21), fill=(178, 196, 214), anchor="lm")
        draw.text((PAD, 82), _fit(draw, title, title_font, CARD_W - PAD * 2 - 260),
                  font=title_font, fill=WHITE, anchor="lm")
        if subtitle:
            draw.text((PAD, 126), _fit(draw, subtitle, sub_font, CARD_W - PAD * 2 - 260),
                      font=sub_font, fill=(178, 196, 214), anchor="lm")

        rows = rows[:6]
        if rows:
            area_top, area_bottom = head_h + 5 + 20, FOOTER_Y - (48 if footer else 16)
            row_h = min(80, max(52, (area_bottom - area_top) // len(rows)))
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
        _footer_right(draw, SITE_LABEL)
        return _to_png(img)
    except Exception as e:
        print(f"  ⚠ 一覧カード画像生成に失敗: {e}")
        return None

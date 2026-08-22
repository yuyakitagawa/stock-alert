"""
video/thumbnail.py

YouTube用カスタムサムネイル（1280x720）の合成。

台紙 `video/assets/thumbnail_base.png` は Canva で作ったブランド素材（ネイビー地・
クジラのラインアート・上昇チャート、下半分は無地）。ここに Pillow で
「銘柄名（証券コード）」と「買い/売り ◯億円」を重ねるだけにして、毎回の生成で
外部デザインツールを呼ばない（CI から Canva は叩けない）。

Shorts フィードでは動画の1フレームが使われるが、検索結果・チャンネルページ・
おすすめ（横長枠）ではカスタムサムネイルが使われるため、銘柄と金額が一目でわかる
絵にしておくとクリック率が上がる。
"""
import os

from web.x_card_image import _font, _font_path, _fit

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BASE_IMAGE = os.path.join(ASSETS_DIR, "thumbnail_base.png")

WIDTH, HEIGHT = 1280, 720
# Canva台紙のチャート線は y≈480 まで伸びている。その下に半透明の下地を敷き、文字は下地の中だけに置く
PLATE_TOP = 470
TEXT_TOP = 486
MARGIN_X = 64
PLATE_BG = (6, 10, 20, 200)

WHITE = (255, 255, 255)
CREAM = (250, 247, 240)
GOLD = (217, 164, 79)
RED = (244, 63, 94)


def format_amount(oku: float) -> str:
    """Remotion側 formatAmount と同じ丸め（100億以上は整数、それ未満は小数1桁）。"""
    if oku >= 100:
        return f"{round(oku):,}"
    return f"{oku:.1f}".replace(".0", "")


def compose(props: dict, out_path: str) -> "str | None":
    """propsからサムネイルPNGを書き出し、パスを返す。台紙やフォントが無ければNone。"""
    if not os.path.exists(BASE_IMAGE):
        print(f"[thumbnail] 台紙が見つかりません: {BASE_IMAGE}")
        return None
    if _font_path() is None:
        print("[thumbnail] 日本語フォントが見つからないためサムネイル生成をスキップします")
        return None

    from PIL import Image, ImageDraw

    im = Image.open(BASE_IMAGE).convert("RGB")
    if im.size != (WIDTH, HEIGHT):
        im = im.resize((WIDTH, HEIGHT))
    # 文字の可読性は影ではなく下地で担保する（動画本編の PLATE_BG と同じ考え方）
    plate = Image.new("RGBA", (WIDTH, HEIGHT - PLATE_TOP), PLATE_BG)
    im.paste(plate, (0, PLATE_TOP), plate)
    draw = ImageDraw.Draw(im)
    max_w = WIDTH - MARGIN_X * 2

    direction = props.get("direction", "buy")
    accent = GOLD if direction == "buy" else RED
    verb = "買い" if direction == "buy" else "売り"

    # 1行目: 銘柄名（証券コード）。長い社名はフォントを落として丸ごと入れる
    name = f"{props.get('stockName', '')}（{props.get('stockCode', '')}）"
    font = None
    for size in (60, 52, 46, 40):
        font = _font(size)
        if draw.textlength(name, font=font) <= max_w:
            break
    else:
        name = _fit(draw, name, font, max_w)
    draw.text((MARGIN_X, TEXT_TOP), name, font=font, fill=WHITE, anchor="la")

    # 2行目: 買い/売り + 金額。数字が主役なので最大サイズ
    amount = f"{format_amount(float(props.get('dealAmountOku', 0)))}億円"
    verb_font = _font(56)
    amount_font = _font(108)
    y = TEXT_TOP + 76
    draw.text((MARGIN_X, y + 44), verb, font=verb_font, fill=accent, anchor="la")
    x = MARGIN_X + draw.textlength(verb, font=verb_font) + 24
    draw.text((x, y), amount, font=amount_font, fill=accent, anchor="la")

    # 3行目: 提出者。空なら分類ラベル
    filer = props.get("filerName") or props.get("dealTypeLabel") or ""
    if filer:
        f3 = _font(34)
        draw.text((MARGIN_X, HEIGHT - 22), _fit(draw, filer, f3, max_w), font=f3,
                  fill=CREAM, anchor="ls")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    im.save(out_path, "PNG")
    return out_path

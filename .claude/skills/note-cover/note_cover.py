#!/usr/bin/env python3
"""note表紙ジェネレータ: 絶景写真に黒帯＋極太白文字を合成して1280x670のPNGを出力する。

参考フォーマット:
  - 背景: 全面写真（cover-crop）
  - 中央に横いっぱいの半透明黒帯
  - 帯内に左寄せの極太ゴシック（ヒラギノ W6）2行

使い方:
  python3 note_cover.py \
    --photo /tmp/scenery.jpg \
    --line1 "その「アドバイス」、明日には無価値。" \
    --line2 "価値があるのは”続く仕組み”だけ。" \
    --out ~/Desktop/note_cover.png

写真は事前にローカルへ用意する（フリー素材は Pexels 等の商用可ライセンスを使う）。
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

# ヒラギノ角ゴ ttc: index 2 = W6（太字）。index 0 は W3 なので使わないこと。
FONT_PATH = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_INDEX = 2
SS = 2  # スーパーサンプリング倍率（縁をくっきりさせる）


def build(photo, line1, line2, out, pad_left=110, opacity=222,
          max_font=96, base_w=1280, base_h=670):
    W, H = base_w * SS, base_h * SS
    img = Image.open(photo).convert("RGB")
    sw, sh = img.size
    s = max(W / sw, H / sh)
    nw, nh = int(sw * s + 0.5), int(sh * s + 0.5)
    img = img.resize((nw, nh), Image.LANCZOS).crop(
        ((nw - W) // 2, (nh - H) // 2, (nw - W) // 2 + W, (nh - H) // 2 + H))

    draw = ImageDraw.Draw(img, "RGBA")

    def font(px):
        return ImageFont.truetype(FONT_PATH, px, index=FONT_INDEX)

    def tsize(t, f):
        b = draw.textbbox((0, 0), t, font=f)
        return b[2] - b[0], b[3] - b[1]

    PAD_L = pad_left * SS
    maxw = W - PAD_L - 70 * SS
    size = 40 * SS
    f = font(size)
    while True:
        n = font(size + 2 * SS)
        w = max(tsize(line1, n)[0], tsize(line2, n)[0])
        if w > maxw or size + 2 * SS > max_font * SS:
            break
        size += 2 * SS
        f = n

    w1, h1 = tsize(line1, f)
    w2, h2 = tsize(line2, f)
    gap = int(size * 0.40)
    pad = int(size * 0.62)
    band_h = h1 + gap + h2 + pad * 2
    band_y = (H - band_h) // 2

    band = Image.new("RGBA", (W, band_h), (10, 9, 8, opacity))
    img.paste(band, (0, band_y), band)

    y = band_y + pad
    draw.text((PAD_L, y), line1, font=f, fill=(255, 255, 255, 255))
    draw.text((PAD_L, y + h1 + gap), line2, font=f, fill=(255, 255, 255, 255))

    img = img.resize((base_w, base_h), Image.LANCZOS)
    img.save(out, "PNG")
    return out, size // SS


def main():
    ap = argparse.ArgumentParser(description="note表紙ジェネレータ")
    ap.add_argument("--photo", required=True, help="背景写真のローカルパス")
    ap.add_argument("--line1", required=True, help="1行目コピー")
    ap.add_argument("--line2", required=True, help="2行目コピー")
    ap.add_argument("--out", required=True, help="出力PNGパス")
    ap.add_argument("--pad-left", type=int, default=110, help="左パディング(px)")
    ap.add_argument("--opacity", type=int, default=222, help="黒帯の不透明度0-255")
    ap.add_argument("--max-font", type=int, default=96, help="最大フォントサイズ(px)")
    a = ap.parse_args()
    out, fs = build(a.photo, a.line1, a.line2, a.out,
                    pad_left=a.pad_left, opacity=a.opacity, max_font=a.max_font)
    print(f"saved {out} (fontsize {fs})")


if __name__ == "__main__":
    main()

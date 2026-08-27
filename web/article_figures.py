"""
web/article_figures.py
ブログ記事の本文に差し込む解説図（Pillow直描画・API課金なし）。

背景: 記事1本の画像がアイキャッチ＋末尾の株価チャートの2枚しかなく、本文が文字だけで
      読み進めづらかった。build_context_facts() が既に集めている事実（保有比率の推移・
      同じ銘柄の他の大株主・提出者のポートフォリオ）は、文章より図の方が速く伝わるため、
      同じデータをそのまま図にして「その話をしている段落の直後」に差し込む。

方針:
- 新規依存を増やさない（Pillowのみ）。フォントとブランド色は web/x_card_image.py と共有する。
- 記事固有の事実だけを描く（5%ルールの一般解説のような、どの記事でも同じ図は作らない）。
- データが足りない図は作らない（None）。図が0枚でも記事は従来どおり公開する。
"""
import io
import re

from web.x_card_image import _fit, _font, GOLD, INK, MUTED, NAVY, PAPER, RULE

SS = 2                      # 一旦2倍で描いて縮小する（文字と斜め線のジャギーを消す）
FIG_W = 1000
MAX_HISTORY_BARS = 8        # 開示回数が多い提出者でも横に潰れない上限
MAX_BAR_ROWS = 6
DEFAULT_SOURCE = "EDINET提出書類"   # 自社株買い記事はTDnet（適時開示）が出典になる


def _px(v: int) -> int:
    return v * SS


def _new_canvas(h: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (_px(FIG_W), _px(h)), PAPER)
    return img, ImageDraw.Draw(img)


def _draw_header(draw, title: str, subtitle: str):
    """見出しは提出者名＋銘柄名で長くなりがちなので、切り詰める前にフォントを段階的に下げる。"""
    sub_font = _font(_px(18))
    max_w = _px(FIG_W - 80)
    for size in (28, 25, 22):
        title_font = _font(_px(size))
        if draw.textlength(title, font=title_font) <= max_w:
            break
    draw.text((_px(40), _px(28)), _fit(draw, title, title_font, max_w), font=title_font, fill=INK)
    if subtitle:
        draw.text((_px(40), _px(70)), _fit(draw, subtitle, sub_font, max_w), font=sub_font, fill=MUTED)


def _draw_footer(draw, h: int, source: str = DEFAULT_SOURCE):
    font = _font(_px(16))
    draw.line([(_px(40), _px(h - 46)), (_px(FIG_W - 40), _px(h - 46))], fill=RULE, width=_px(1))
    draw.text((_px(40), _px(h - 36)), f"出典: {source}より作成 / kujira-watch.com", font=font, fill=MUTED)


def _finish(img, h: int) -> bytes:
    from PIL import Image

    img = img.resize((FIG_W, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


_CORP_SUFFIX_RE = re.compile(r"株式会社|合同会社|有限会社|\(株\)|（株）|㈱")


def _short_name(name: str) -> str:
    """図の中では法人格を落とす（「株式会社」だけで6文字使い、見出しも凡例も入らなくなる）。
    本文・alt・キャプションでは正式名称のまま使う。"""
    short = _CORP_SUFFIX_RE.sub("", str(name or "")).strip("　 ")
    return short or str(name or "")


def _ratio_label(value: float) -> str:
    return f"{value:.2f}%"


def _short_date(iso_date: str) -> str:
    """2024-03-15 → 24/3/15（軸ラベルは横幅が限られるので年は下2桁にする）。"""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso_date or ""))
    if not m:
        return str(iso_date or "")
    return f"{m.group(1)[2:]}/{int(m.group(2))}/{int(m.group(3))}"


def _vertical_bar_figure(title: str, subtitle: str, points: list, unit: str,
                         value_fmt=None, source: str = DEFAULT_SOURCE) -> "bytes | None":
    """時系列の縦棒グラフ共通処理。points は {"date", "value"} の古い順リストで、
    最後（＝今回の開示）だけゴールドで塗る。棒の長さを素直に比べさせたいので0起点にする。"""
    points = [p for p in (points or []) if p.get("value") is not None]
    if len(points) < 2 or _font(10) is None:
        return None
    points = points[-MAX_HISTORY_BARS:]
    value_fmt = value_fmt or (lambda v: f"{v:.2f}{unit}")

    h = 460
    pad_l, pad_r, pad_t, pad_b = 118, 40, 130, 96   # 左は「15,000億円」級の目盛りラベルが入る幅
    img, draw = _new_canvas(h)
    _draw_header(draw, title, subtitle)

    plot_w, plot_h = FIG_W - pad_l - pad_r, h - pad_t - pad_b
    top = max(p["value"] for p in points) * 1.25 or 1.0
    base_y = pad_t + plot_h

    # 目盛りは0と上端の2本だけ（棒の長さを素直に比較させたいので0起点にする）
    label_font = _font(_px(16))
    for value, text in ((0.0, "0"), (top, value_fmt(top))):
        y = base_y - (value / top) * plot_h
        draw.line([(_px(pad_l), _px(y)), (_px(FIG_W - pad_r), _px(y))], fill=RULE, width=_px(1))
        tw = draw.textlength(text, font=label_font)
        draw.text((_px(pad_l - 12) - tw, _px(y - 10)), text, font=label_font, fill=MUTED)

    slot = plot_w / len(points)
    bar_w = min(84, slot * 0.55)
    value_font = _font(_px(20))
    for i, p in enumerate(points):
        cx = pad_l + slot * (i + 0.5)
        bar_h = (p["value"] / top) * plot_h
        is_latest = i == len(points) - 1
        draw.rectangle(
            [(_px(cx - bar_w / 2), _px(base_y - bar_h)), (_px(cx + bar_w / 2), _px(base_y))],
            fill=GOLD if is_latest else NAVY,
        )
        value_text = value_fmt(p["value"])
        vw = draw.textlength(value_text, font=value_font)
        draw.text(
            (_px(cx) - vw / 2, _px(base_y - bar_h - 30)),
            value_text, font=value_font, fill=INK if is_latest else MUTED,
        )
        date_text = _short_date(p.get("date"))
        dw = draw.textlength(date_text, font=label_font)
        draw.text((_px(cx) - dw / 2, _px(base_y + 12)), date_text, font=label_font, fill=MUTED)

    _draw_footer(draw, h, source)
    return _finish(img, h)


def holding_history_figure(stock_name: str, filer_name: str, points: list) -> "bytes | None":
    """提出者×銘柄の保有比率の推移（開示ごとの縦棒、今回の開示だけゴールド）。
    開示が2回未満なら推移として意味がないのでNone。"""
    points = [{"date": p.get("date"), "value": p.get("ratio")} for p in (points or [])]
    usable = [p for p in points if p["value"] is not None]
    return _vertical_bar_figure(
        f"{_short_name(filer_name)}による{_short_name(stock_name)}の保有比率の推移",
        f"大量保有報告書の開示ごとの保有比率（直近{min(len(usable), MAX_HISTORY_BARS)}回）",
        points, "%", _ratio_label,
    )


def buyback_history_figure(stock_name: str, points: list) -> "bytes | None":
    """同じ会社の自社株買い決定（取得上限金額）の推移。過去の決定が無ければNone。"""
    return _vertical_bar_figure(
        f"{_short_name(stock_name)}の自社株買い（取得上限金額）の推移",
        "取締役会決議の開示ごとの取得上限金額（直近の決定）",
        points, "億円", lambda v: f"{v:,.0f}億円" if v >= 100 else f"{v:,.1f}億円",
        source="適時開示（TDnet）",
    )


def _horizontal_bar_figure(title: str, subtitle: str, rows: list) -> "bytes | None":
    """横棒グラフ共通処理。rows は {"label", "value", "highlight"} のリスト（並び順どおりに描く）。"""
    rows = [r for r in (rows or []) if r.get("value") is not None][:MAX_BAR_ROWS]
    if len(rows) < 2 or _font(10) is None:
        return None

    pad_t, row_h, bar_h = 128, 58, 28
    label_w, value_w = 392, 96
    h = pad_t + row_h * len(rows) + 70
    img, draw = _new_canvas(h)
    _draw_header(draw, title, subtitle)

    label_font = _font(_px(19))
    value_font = _font(_px(20))
    x0 = 40 + label_w
    bar_area = FIG_W - 40 - value_w - x0
    top = max(r["value"] for r in rows) or 1.0
    for i, r in enumerate(rows):
        y = pad_t + row_h * i
        cy = y + row_h / 2
        highlight = bool(r.get("highlight"))
        label = _fit(draw, str(r["label"]), label_font, _px(label_w - 16))
        draw.text((_px(40), _px(cy - 13)), label, font=label_font, fill=INK if highlight else MUTED)
        bar_len = max(4.0, (r["value"] / top) * bar_area)
        draw.rectangle(
            [(_px(x0), _px(cy - bar_h / 2)), (_px(x0 + bar_len), _px(cy + bar_h / 2))],
            fill=GOLD if highlight else NAVY,
        )
        draw.text(
            (_px(x0 + bar_len + 12), _px(cy - 13)),
            _ratio_label(r["value"]), font=value_font, fill=INK if highlight else MUTED,
        )

    _draw_footer(draw, h)
    return _finish(img, h)


def shareholders_figure(stock_name: str, filer_name: str, holding_ratio: float, peers: list) -> "bytes | None":
    """同じ銘柄に大量保有報告書を出している投資家の比較（今回の提出者をゴールドで強調）。"""
    rows = [{"label": _short_name(filer_name), "value": holding_ratio, "highlight": True}]
    rows += [{"label": _short_name(p.get("name")), "value": p.get("ratio")} for p in (peers or [])]
    rows = [r for r in rows if r["value"] is not None]
    rows.sort(key=lambda r: r["value"], reverse=True)
    return _horizontal_bar_figure(
        f"{_short_name(stock_name)}に大量保有報告書を出している投資家",
        "各提出者が直近の開示で届け出た保有比率",
        rows,
    )


def portfolio_figure(filer_name: str, stock_name: str, stock_code: str, holding_ratio: float,
                     others: list) -> "bytes | None":
    """提出者が5%以上を保有している銘柄の一覧（今回の銘柄をゴールドで強調）。"""
    this_label = f"{_short_name(stock_name)}（{stock_code}）" if stock_code else _short_name(stock_name)
    rows = [{"label": this_label, "value": holding_ratio, "highlight": True}]
    rows += [
        {"label": f"{_short_name(o.get('name'))}（{o.get('code')}）" if o.get("code")
                  else _short_name(o.get("name")),
         "value": o.get("ratio")}
        for o in (others or [])
    ]
    rows = [r for r in rows if r["value"] is not None]
    rows.sort(key=lambda r: r["value"], reverse=True)
    return _horizontal_bar_figure(
        f"{_short_name(filer_name)}が5%以上を保有する主な銘柄",
        "直近の大量保有報告書ベースの保有比率（上位）",
        rows,
    )


def build_article_figures(fact_sheet: dict) -> list:
    """fact_sheetから作れる解説図を作り、[{bytes, filename, alt, caption, alt_en,
    caption_en, anchors}] を本文での登場順に返す。anchorsは「この図を差し込む段落」を
    本文から探すための手掛かり語（見つからなければ均等配置にフォールバックする）。"""
    facts = fact_sheet.get("context_facts") or {}
    stock_name = fact_sheet.get("stock_name") or ""
    filer_name = fact_sheet.get("filer_name") or ""
    ratio = fact_sheet.get("holding_ratio")
    figures = []

    history = facts.get("holding_history") or {}
    points = history.get("points") or []
    image = holding_history_figure(stock_name, filer_name, points) if ratio is not None else None
    if image:
        first_date = str(history.get("first_date") or "")
        anchors = ["回の開示", "回目", "初回", "積み上げ", "買い増し"]
        m = re.match(r"(\d{4})-(\d{2})", first_date)
        if m:
            anchors += [f"{m.group(1)}年", f"{m.group(1)}年{int(m.group(2))}月"]
        figures.append({
            "bytes": image,
            "filename": "holding-history.png",
            "alt": f"{filer_name}による{stock_name}の保有比率の推移",
            "caption": f"{filer_name}が{stock_name}について提出した大量保有報告書の保有比率推移（EDINET開示ベース）",
            "alt_en": f"{stock_name}: reported stake held by {filer_name} over time",
            "caption_en": f"Reported stake in {stock_name} held by {filer_name}, by EDINET filing date.",
            "anchors": anchors,
        })

    peers = facts.get("stock_other_filers") or []
    image = shareholders_figure(stock_name, filer_name, ratio, peers) if ratio is not None else None
    if image:
        figures.append({
            "bytes": image,
            "filename": "shareholders.png",
            "alt": f"{stock_name}に大量保有報告書を出している投資家の保有比率比較",
            "caption": f"{stock_name}に大量保有報告書を出している投資家の保有比率（今回の{filer_name}を強調）",
            "alt_en": f"Large shareholders of {stock_name} by reported stake",
            "caption_en": f"Investors that have filed large shareholding reports on {stock_name} ({filer_name} highlighted).",
            "anchors": ["大株主", "株主構成", "筆頭"] + [p.get("name", "") for p in peers[:3]],
        })

    others = facts.get("filer_other_holdings") or []
    image = (portfolio_figure(filer_name, stock_name, fact_sheet.get("stock_code") or "", ratio, others)
             if ratio is not None else None)
    if image:
        figures.append({
            "bytes": image,
            "filename": "filer-portfolio.png",
            "alt": f"{filer_name}が5%以上を保有する主な銘柄",
            "caption": f"{filer_name}が同時点で5%以上を保有している主な銘柄（今回の{stock_name}を強調）",
            "alt_en": f"Other Japanese equities in which {filer_name} reports a stake above 5%",
            "caption_en": f"Other holdings where {filer_name} reports a stake above 5% ({stock_name} highlighted).",
            "anchors": ["他の銘柄", "ポートフォリオ", "保有銘柄"] + [o.get("name", "") for o in others[:3]],
        })

    return figures


def buyback_article_figures(fact: dict) -> list:
    """自社株買い記事用の解説図（過去の決議と今回の取得上限金額の比較）。
    過去の決議が1件も無ければ図なし（1本だけの棒グラフは情報量が無い）。"""
    prior = fact.get("prior") or []
    amount_oku = fact.get("amount_oku")
    if not prior or amount_oku is None:
        return []
    points = [
        {"date": str(r.get("disclosed_at") or "")[:10],
         "value": (r["max_amount_yen"] / 1e8) if r.get("max_amount_yen") else None}
        for r in reversed(prior)
    ]
    points.append({"date": fact.get("disc_date"), "value": amount_oku})
    stock_name = fact.get("stock_name") or ""
    image = buyback_history_figure(stock_name, points)
    if not image:
        return []
    anchors = ["過去", "前回", "これまで", "自己株式の取得"]
    anchors += [f"{p['date'][:4]}年" for p in points[:-1] if len(p.get("date") or "") >= 4]
    return [{
        "bytes": image,
        "filename": "buyback-history.png",
        "alt": f"{stock_name}の自社株買い（取得上限金額）の推移",
        "caption": f"{stock_name}が決議した自己株式取得の上限金額の推移（適時開示ベース、今回の決定を強調）",
        "alt_en": f"{stock_name}: announced share buyback ceilings over time",
        "caption_en": f"Buyback ceilings announced by {stock_name} (latest announcement highlighted).",
        "anchors": anchors,
    }]


def figure_html(url: str, alt: str, caption: str) -> str:
    """本文に差し込む<figure>。width/heightは入れない（縦幅が図ごとに違うため、
    prose側のmax-width:100%に任せる）。"""
    return f'<figure><img src="{url}" alt="{alt}"><figcaption>{caption}</figcaption></figure>'


def _anchor_paragraph_index(paragraphs: list, anchors: list) -> "int | None":
    """anchorsの語を最も多く含む段落の1始まりindexを返す（どこにも無ければNone）。
    1段落目は検索クエリへの直答文なので図は差し込まない。"""
    best_idx, best_hits = None, 0
    for i, p in enumerate(paragraphs[1:], start=2):
        hits = sum(1 for a in anchors if a and a in p)
        if hits > best_hits:
            best_idx, best_hits = i, hits
    return best_idx


def insert_figures_into_body(body: str, figures: list) -> str:
    """figures（{"html", "anchors"}のリスト）を本文の該当段落の直後に差し込む。
    最終段落は「※推測」の締めなので、その後ろには入れない（末尾は株価チャートの位置）。"""
    if not figures:
        return body
    paragraphs = [p for p in re.split(r"(?<=</p>)", body or "") if p.strip()]
    if len(paragraphs) < 3:
        return (body or "") + "".join(f["html"] for f in figures)

    last = len(paragraphs) - 1  # 最終段落の前までに収める
    used, placements = set(), {}
    for i, fig in enumerate(figures):
        idx = _anchor_paragraph_index(paragraphs, fig.get("anchors") or [])
        if idx is None:
            idx = int(round((i + 1) * len(paragraphs) / (len(figures) + 1)))
        idx = max(2, min(idx, last))
        while idx in used and idx < last:
            idx += 1
        while idx in used and idx > 2:
            idx -= 1
        used.add(idx)
        placements.setdefault(idx, []).append(fig["html"])

    out = []
    for i, p in enumerate(paragraphs, start=1):
        out.append(p)
        out.extend(placements.get(i, []))
    return "".join(out)

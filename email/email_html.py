"""純粋なHTML/SVG生成関数。I/O・DB・ネットワーク呼び出しなし。"""

EMAIL_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     max-width:700px;margin:0 auto;padding:16px;color:#222;background:#f5f5f5}
.card{background:#fff;border-radius:10px;padding:16px;margin-bottom:16px;
      box-shadow:0 1px 4px rgba(0,0,0,.08)}
h2{margin:0 0 12px;font-size:16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f0f0f0;padding:7px 5px;text-align:center;font-weight:600;
   border-bottom:2px solid #ddd;white-space:nowrap}
td{padding:7px 5px;border-bottom:1px solid #eee;vertical-align:middle}
.net-pos{color:#0a7a0a;font-weight:700}
.net-neg{color:#c0392b;font-weight:700}
.net-neu{color:#888;font-weight:700}
.rel-pos{color:#0a7a0a}
.rel-neg{color:#c0392b}
.badge-new{display:inline-block;background:#ff6b35;color:white;
           font-size:10px;font-weight:700;padding:1px 5px;border-radius:8px;margin-left:4px;
           vertical-align:middle}
"""

_DEFENSIVE_SECTORS = {"電気・ガス業", "食料品", "医薬品", "情報・通信業",
                      "陸運業", "不動産業", "銀行業", "保険業", "その他金融業",
                      "倉庫・運輸関連業", "水産・農林業", "鉱業"}
_CYCLICAL_SECTORS = {"輸送用機器", "機械", "鉄鋼", "非鉄金属", "化学",
                     "海運業", "金属製品", "ガラス・土石製品", "繊維製品",
                     "石油・石炭製品", "パルプ・紙", "ゴム製品"}
_GROWTH_SECTORS = {"電気機器", "精密機器", "サービス業", "その他製品",
                   "卸売業", "小売業", "証券、商品先物取引業"}


def volatility_label(vol):
    if vol < 20:
        return "🟢低"
    if vol < 40:
        return "🟡中"
    if vol < 60:
        return "🟠高"
    return "🔴超高"


def get_judgment(net):
    if net >= 15:
        return "🟢強気買い", "#1a7a1a"
    elif net >= 5:
        return "🔵やや強気", "#1a4a8a"
    elif net >= -5:
        return "🟡中立", "#7a6a00"
    elif net >= -15:
        return "🟠やや弱気", "#b05000"
    else:
        return "🔴売り検討", "#c0392b"


def net_cls(n):
    return "net-pos" if n >= 5 else ("net-neg" if n < -5 else "net-neu")


def rel_cls(r):
    return "" if r is None else ("rel-pos" if r >= 0 else "rel-neg")


def rel_str(r):
    return f"{r:+.1f}%" if r is not None else "-"


def classify_sector(sector):
    if sector in _DEFENSIVE_SECTORS:
        return "defensive"
    if sector in _CYCLICAL_SECTORS:
        return "cyclical"
    if sector in _GROWTH_SECTORS:
        return "growth"
    return "other"


def fundamentals_suffix(row):
    parts = []
    per, pbr = row.get("PER"), row.get("PBR")
    if per and str(per) not in ("nan", "None", "-"):
        parts.append(f"PER{float(per):.0f}")
    if pbr and str(pbr) not in ("nan", "None", "-"):
        parts.append(f"PBR{float(pbr):.1f}")
    return " ".join(parts)


def stop_loss_cell_html(row, close_val):
    stop_val = row.get("損切り価格(円)")
    stop_pct = row.get("損切り幅(%)")
    if not stop_val or str(stop_val) in ("nan", "None"):
        return "-"
    pct_part = f" ({stop_pct:+.1f}%)" if stop_pct and str(stop_pct) not in ("nan", "None") else ""
    return (f"現値 ¥{close_val:,}<br>"
            f"<span style='color:#c0392b;font-weight:700'>↓ ¥{int(stop_val):,}</span>"
            f"<span style='font-size:11px;color:#c0392b'>{pct_part}</span>")


def build_sparkline_svg(prices_close, width=80, height=28):
    data = [p for p in prices_close[-60:] if p is not None]
    if len(data) < 2:
        return ""
    mn, mx = min(data), max(data)
    if mx == mn:
        return ""
    n = len(data)
    pts = []
    for i, v in enumerate(data):
        x = round(i / (n - 1) * width, 1)
        y = round((1 - (v - mn) / (mx - mn)) * (height - 4) + 2, 1)
        pts.append(f"{x},{y}")
    color = "#0a7a0a" if data[-1] >= data[0] else "#c0392b"
    return (f'<svg width="{width}" height="{height}" '
            f'style="display:inline-block;vertical-align:middle;margin-top:2px">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'</svg>')


def build_candidate_observation(candidates):
    if not candidates:
        return ""
    nets = [c["net"] for c in candidates]
    avg_net = sum(nets) / len(nets)
    sec_classes = [classify_sector(c["sector"]) for c in candidates]
    n = len(candidates)
    cnt_def = sec_classes.count("defensive")
    cnt_cyc = sec_classes.count("cyclical")
    cnt_grw = sec_classes.count("growth")

    if cnt_def >= max(3, n * 0.6):
        sector_msg = "🛡️ <b>防御的セクター中心</b>（電力・食品・不動産など）。市場が荒れている時の典型パターン。"
    elif cnt_cyc >= max(3, n * 0.6):
        sector_msg = "⚙️ <b>景気敏感株中心</b>（鉄鋼・機械・化学など）。景気回復期待の表れ。"
    elif cnt_grw >= max(3, n * 0.6):
        sector_msg = "🚀 <b>成長株中心</b>（電気機器・精密機器・サービスなど）。リスクオン局面。"
    else:
        sector_msg = "🌐 <b>セクター分散</b>。特定の市場テーマなし。"

    if avg_net >= 11.0:
        conf_msg = f"📊 平均ネット {avg_net:.1f}% — モデル確信度<b>高め</b>"
    elif avg_net >= 9.5:
        conf_msg = f"📊 平均ネット {avg_net:.1f}% — モデル確信度<b>中</b>"
    else:
        conf_msg = f"📊 平均ネット {avg_net:.1f}% — モデル確信度<b>低め</b>"

    return (f"<div style='background:#f0f7ff;border-left:3px solid #2980b9;"
            f"padding:8px 12px;margin:8px 0 12px;border-radius:4px;font-size:13px;line-height:1.6'>"
            f"<b style='color:#2980b9'>📍 今日の傾向</b><br>"
            f"{sector_msg}<br>{conf_msg}"
            f"</div>")


def build_priority_section(priority_actions):
    if not priority_actions:
        return ""
    items = ""
    for a in priority_actions:
        items += (f"<div style='display:flex;align-items:flex-start;gap:10px;"
                  f"padding:10px 0;border-bottom:1px solid #f0f0f0'>"
                  f"<div style='font-size:20px;line-height:1.2'>{a['emoji']}</div>"
                  f"<div><div style='font-weight:700;font-size:14px'>{a['title']}</div>"
                  f"<div style='color:#666;font-size:12px;margin-top:2px'>{a['detail']}</div>"
                  f"</div></div>")
    return (f"<div class='card' style='border-left:4px solid #f39c12'>"
            f"<h2>🎯 今日の優先アクション</h2>{items}</div>")


def _holding_days_cell(holding_days):
    """保有日数を色付きセルで返す。閾値を超えると警告色。"""
    if holding_days is None:
        return "<span style='color:#aaa'>-</span>"
    if holding_days > 63:
        return (f"<span style='color:#c0392b;font-weight:700'>{holding_days}d</span>"
                f"<br><span style='font-size:10px;color:#c0392b'>期限超</span>")
    if holding_days > 30:
        return (f"<span style='color:#b05000;font-weight:700'>{holding_days}d</span>"
                f"<br><span style='font-size:10px;color:#b05000'>中期</span>")
    return f"<span style='color:#888'>{holding_days}d</span>"


def build_sell_section(results):
    sells = sorted([r for r in results if r["signal"] == "sell"], key=lambda x: x["net"])
    if not sells:
        return (f"<div class='card' style='border-left:4px solid #27ae60'>"
                f"<h2>✅ 売り検討なし</h2>"
                f"<p style='color:#666;margin:0'>全チェック銘柄がポジティブ/中立判定です。</p></div>"), sells
    rows = ""
    for r in sells:
        hd_cell = _holding_days_cell(r.get("holding_days"))
        drop_str = f"{r['drop_prob']:.1f}%" if r.get("drop_prob") is not None else "-"
        rows += (f"<tr>"
                 f"<td><b>{r['name']}</b><br>"
                 f"<span style='color:#888;font-size:12px'>{r['code']} ¥{r['close']:,.0f}</span></td>"
                 f"<td style='text-align:center'>{r['prob']:.1f}%</td>"
                 f"<td style='text-align:center'>{drop_str}</td>"
                 f"<td class='{net_cls(r['net'])}' style='text-align:center'>{r['net']:+.1f}%</td>"
                 f"<td style='text-align:center;font-size:11px'>{r.get('recommend','')}</td>"
                 f"<td class='{rel_cls(r.get('rel20'))}' style='text-align:center'>{rel_str(r.get('rel20'))}</td>"
                 f"<td style='text-align:center;color:#888;font-size:12px'>{r.get('vol',0):.0f}%{r.get('vol_label','')}</td>"
                 f"<td style='text-align:center;font-size:12px'>{hd_cell}</td>"
                 f"</tr>")
    section = (f"<div class='card' style='border-left:4px solid #c0392b'>"
               f"<h2>🔴 売り検討 ({len(sells)}銘柄)</h2>"
               f"<p style='color:#666;font-size:13px;margin:0 0 10px'>"
               f"ネット低下または保有期限超過。理由を確認してください。</p>"
               f"<table><tr style='background:#fde8e8'>"
               f"<th>銘柄</th><th>上昇確率</th><th>下落確率</th><th>ネット</th><th>推奨</th><th>日経差(20日)</th><th>ボラ</th><th>保有</th></tr>"
               f"{rows}</table></div>")
    return section, sells


def build_all_rows(results, earnings_map=None):
    earnings_map = earnings_map or {}
    rows = ""
    for idx, r in enumerate(sorted(results, key=lambda x: x["net"], reverse=True), 1):
        drop_str   = f"{r['drop_prob']:.1f}%" if r.get("drop_prob") is not None else "-"
        spark      = build_sparkline_svg(r.get("prices_close", []))
        spark_html = f"<br>{spark}" if spark else ""
        days = earnings_map.get(str(r["code"]))
        earn_badge = (f"<span style='display:inline-block;background:#e74c3c;color:white;"
                      f"font-size:9px;font-weight:700;padding:1px 4px;border-radius:6px;"
                      f"margin-left:3px;vertical-align:middle'>決算{days}日前</span>"
                      if days is not None else "")
        cut_badge = (f"<span style='display:inline-block;background:#6c3483;color:white;"
                     f"font-size:9px;font-weight:700;padding:1px 4px;border-radius:6px;"
                     f"margin-left:3px;vertical-align:middle'>⚡即切</span>"
                     if r.get("ret20", 0) < -10.0 else "")
        hd = r.get("holding_days")
        hold_badge = ""
        if hd is not None and hd > 63:
            hold_badge = (f"<span style='display:inline-block;background:#c0392b;color:white;"
                          f"font-size:9px;font-weight:700;padding:1px 4px;border-radius:6px;"
                          f"margin-left:3px;vertical-align:middle'>⏰{hd}d</span>")
        elif hd is not None and hd > 30:
            hold_badge = (f"<span style='display:inline-block;background:#e67e22;color:white;"
                          f"font-size:9px;font-weight:700;padding:1px 4px;border-radius:6px;"
                          f"margin-left:3px;vertical-align:middle'>{hd}d</span>")
        qty = r.get("qty")
        qty_str = f"<span style='color:#555;font-size:10px'>×{qty:,}株</span>" if qty else ""
        rows += (f"<tr>"
                 f"<td style='text-align:center;color:#aaa;font-size:12px'>{idx}</td>"
                 f"<td><b>{r['name']}</b>{earn_badge}{cut_badge}{hold_badge}"
                 f"<span style='color:#888;font-size:11px'><br>{r['code']} ¥{r['close']:,.0f} {qty_str}</span>"
                 f"{spark_html}</td>"
                 f"<td style='text-align:center'>{r['prob']:.1f}%</td>"
                 f"<td style='text-align:center'>{drop_str}</td>"
                 f"<td class='{net_cls(r['net'])}' style='text-align:center'>{r['net']:+.1f}%</td>"
                 f"<td style='text-align:center;font-size:11px'>{r.get('recommend', '')}</td>"
                 f"<td class='{rel_cls(r.get('rel20'))}' style='text-align:center'>{rel_str(r.get('rel20'))}</td>"
                 f"<td style='text-align:center;color:#888;font-size:11px'>{r.get('vol',0):.0f}%{r.get('vol_label','')}</td>"
                 f"</tr>")
    return rows


def bear_market_banner_html(is_bear, nk20):
    if not is_bear or nk20 is None:
        return ""
    return (
        f"<div style='background:#c0392b;border-radius:8px;padding:16px;margin-bottom:16px'>"
        f"<div style='color:white;font-size:18px;font-weight:700;margin-bottom:6px'>"
        f"🚨 下落相場 — 新規買いは見送り推奨</div>"
        f"<div style='color:#fdd;font-size:13px;line-height:1.6'>"
        f"日経225の20日リターンが {nk20:+.1f}% と急落しています。<br>"
        f"下落相場ではモデルの精度が落ち、買いシグナルの信頼性が低下します。<br>"
        f"<b>既存ポジションの損切りラインを確認し、新規買いは相場が落ち着くまで待ってください。</b>"
        f"</div></div>"
    )


def index_etf_banner_html(is_bear, candidate_count, nk20):
    if is_bear or nk20 is None or nk20 <= 3.0 or candidate_count > 3:
        return ""
    return (
        f"<div style='background:#f39c12;border-radius:8px;padding:16px;margin-bottom:16px'>"
        f"<div style='color:white;font-size:16px;font-weight:700;margin-bottom:6px'>"
        f"💡 日経225 ETFの検討推奨</div>"
        f"<div style='color:#fff8e1;font-size:13px;line-height:1.6'>"
        f"新規候補が {candidate_count} 銘柄しかなく、日経225は20日で {nk20:+.1f}% と好調。<br>"
        f"個別株が指数に追いついていない可能性があります。<br>"
        f"<b>個別株より日経225 ETF（1321 / 1330 / 1346 等）の方が効率的かもしれません。</b>"
        f"</div></div>"
    )


def hot_market_banner_html(is_hot, nk60):
    if not is_hot or nk60 is None:
        return ""
    return (
        f"<div style='background:#e67e22;border-radius:8px;padding:16px;margin-bottom:16px'>"
        f"<div style='color:white;font-size:18px;font-weight:700;margin-bottom:6px'>"
        f"🚀 日経急騰中 — 新規エントリーは慎重に</div>"
        f"<div style='color:#fef9e7;font-size:13px;line-height:1.6'>"
        f"日経225の60日リターンが {nk60:+.1f}% と急騰しています。<br>"
        f"大型株主導の急騰相場では中小型株主体のモデル候補が指数に追いつけない傾向があります。<br>"
        f"<b>新規候補は参考程度に留め、個別株より日経225 ETF（1321/1330/1346）も選択肢として検討してください。</b>"
        f"</div></div>"
    )


def summary_stat_cards_html(n_sell, n_buy):
    box = ("<div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;"
           "box-shadow:0 1px 4px rgba(0,0,0,.08)'>"
           "<div style='font-size:26px;font-weight:700;color:{color}'>{val}</div>"
           "<div style='font-size:12px;color:#888'>{label}</div></div>")
    return (
        "<div style='display:flex;gap:8px;margin-bottom:16px'>"
        + box.format(val=n_sell, label="売り検討", color="#c0392b")
        + box.format(val=n_buy, label="買い継続", color="#0a7a0a")
        + "</div>"
    )

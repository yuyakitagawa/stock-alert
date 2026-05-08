import numpy as np
import os
import glob
import re
import json
import smtplib
import joblib
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from utils import get_prices, get_nikkei_returns, extract_features, add_cs_rank_features, IsotonicCalibrated

load_dotenv(os.path.expanduser("~/stock-alert/.env"))

GMAIL_ADDRESS         = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD    = os.getenv("GMAIL_APP_PASSWORD")
NET_SELL_THRESHOLD    = -5
BEAR_MARKET_THRESHOLD = -5.0
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9"}


# ───────────────────────────── データ読み込み ──────────────────────────────

def load_held_stocks():
    """Google SheetsまたはCSVからチェック銘柄を読み込む"""
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/stock-alert"))
        from sheets_helper import load_watch_list
        return load_watch_list()
    except Exception as e:
        print(f"[WARN] sheets_helper失敗、CSVで代替: {e}")
        csv_path = os.path.expanduser("~/stock-alert/watch_list.csv")
        if not os.path.exists(csv_path):
            print("ERROR: watch_list.csvが見つかりません")
            return {}
        df = pd.read_csv(csv_path, dtype=str)
        return dict(zip(df["コード"].str.strip(), df["銘柄名"].str.strip()))


def load_top_ranking(n=15):
    """最新のランキングCSVから上位N銘柄を読み込む"""
    files = glob.glob(os.path.expanduser("~/stock-alert/ranking_*.csv"))
    if not files:
        return None
    df = pd.read_csv(max(files, key=os.path.getmtime))
    return df.head(n)


def load_prev_ranking_codes():
    """前回（最新より1つ古い）のランキングCSVから銘柄コードセットを取得"""
    files = sorted(glob.glob(os.path.expanduser("~/stock-alert/ranking_*.csv")),
                   key=os.path.getmtime)
    if len(files) < 2:
        return set()
    try:
        df = pd.read_csv(files[-2])
        return set(df["銘柄コード"].astype(str).tolist())
    except Exception:
        return set()


def load_prev_results():
    """昨日のアラート結果をJSONから読み込む（差分計算用）"""
    path = os.path.expanduser("~/stock-alert/alert_results_prev.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        return {str(r["code"]): r for r in items}
    except Exception:
        return {}


def save_results_for_tomorrow(results):
    """今日のアラート結果をJSONで保存（明日の差分計算用）"""
    path = os.path.expanduser("~/stock-alert/alert_results_prev.json")
    to_save = [{"code": r["code"], "name": r["name"], "net": round(r["net"], 2), "signal": r["signal"]}
               for r in results]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 結果保存失敗: {e}")


# ───────────────────────────── スパークライン ──────────────────────────────

def build_sparkline_svg(prices_close, width=80, height=28):
    """終値リストからインラインSVGスパークラインを生成（Gmail非対応クライアント向け）"""
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


# ───────────────────────────── 優先アクション ─────────────────────────────

def build_priority_actions(results):
    """今日の優先アクション（最大3件）: 売り検討→強気買い の順で選出"""
    actions = []
    sells = sorted([r for r in results if r["signal"] == "sell"], key=lambda x: x["net"])
    for r in sells[:2]:
        dp = r.get("drop_prob")
        detail = f"ネット {r['net']:+.1f}%　下落確率 {dp:.1f}%" if dp is not None else f"ネット {r['net']:+.1f}%"
        actions.append({"emoji": "🔴", "title": f"{r['name']}（{r['code']}）を売り検討", "detail": detail})
    for r in sorted([r for r in results if r["net"] >= 15], key=lambda x: -x["net"]):
        if len(actions) >= 3:
            break
        actions.append({"emoji": "🟢",
                         "title": f"{r['name']}（{r['code']}）は強気シグナル",
                         "detail": f"ネット {r['net']:+.1f}%　ボラ {r.get('vol', 0):.0f}%"})
    return actions[:3]


# ───────────────────────────── 差分セクション ─────────────────────────────

def build_diff_section(results, prev_results):
    """昨日からネットスコアが±3%以上変動した銘柄を表示"""
    if not prev_results:
        return ""
    significant = []
    for r in results:
        prev = prev_results.get(str(r["code"]))
        if prev is None:
            continue
        delta = r["net"] - prev.get("net", 0)
        if abs(delta) >= 3:
            significant.append((r, prev.get("net", 0), delta))
    significant.sort(key=lambda x: abs(x[2]), reverse=True)
    if not significant:
        return ""
    rows = ""
    for r, prev_net, delta in significant[:6]:
        arrow = "▲" if delta > 0 else "▼"
        color = "#0a7a0a" if delta > 0 else "#c0392b"
        rows += (f"<tr>"
                 f"<td>{r['name']}（{r['code']}）</td>"
                 f"<td style='text-align:center'>{prev_net:+.1f}%</td>"
                 f"<td style='text-align:center;color:{color};font-weight:700'>{delta:+.1f}% {arrow}</td>"
                 f"<td style='text-align:center'>{r['net']:+.1f}%</td>"
                 f"</tr>")
    return (f"<div class='card' style='border-left:4px solid #8e44ad'>"
            f"<h2>📅 昨日との差分（±3%以上変動）</h2>"
            f"<table><tr style='background:#f5eef8'>"
            f"<th>銘柄</th><th>昨日</th><th>変化</th><th>今日</th></tr>{rows}</table>"
            f"</div>")


# ───────────────────────────── セクター警告 ───────────────────────────────

def _fetch_sector_kabutan(code):
    """kabutan.jpから業種文字列を取得（失敗時は空文字）"""
    try:
        resp = requests.get(f"https://kabutan.jp/stock/?code={code}",
                            headers=_HEADERS, timeout=8)
        if resp.status_code == 200:
            m = re.search(r'業種.*?<a[^>]*>([^<]+)</a>', resp.text)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return ""


def get_next_earnings_cached(code):
    """次回決算発表予定日をkabutan.jpから取得してJSONキャッシュに保存。失敗時はNone。"""
    cache_path = os.path.expanduser("~/stock-alert/earnings_cache.json")
    today_str  = datetime.now().strftime("%Y-%m-%d")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass
    key = str(code)
    entry = cache.get(key, {})
    if entry.get("fetched") == today_str:
        date_str = entry.get("date")
        return datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    try:
        resp = requests.get(f"https://kabutan.jp/stock/?code={code}",
                            headers=_HEADERS, timeout=8)
        date = None
        if resp.status_code == 200:
            m = re.search(r'決算発表予定日[^<]*<[^>]+>(\d{4}/\d{2}/\d{2})', resp.text)
            if m:
                date = datetime.strptime(m.group(1), "%Y/%m/%d").date()
        cache[key] = {"fetched": today_str, "date": date.isoformat() if date else None}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        return date
    except Exception:
        return None


def get_sector_cached(code):
    """業種をキャッシュから取得（なければkabutan.jpから取得してCSVキャッシュに保存）"""
    cache_path = os.path.expanduser("~/stock-alert/sector_cache.csv")
    cache = {}
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, dtype=str)
            cache = dict(zip(df["code"], df["sector"]))
        except Exception:
            pass
    key = str(code)
    if key in cache:
        return cache[key]
    sector = _fetch_sector_kabutan(code) or "不明"
    cache[key] = sector
    try:
        pd.DataFrame(list(cache.items()), columns=["code", "sector"]).to_csv(cache_path, index=False)
    except Exception:
        pass
    return sector


def build_sector_warning(results):
    """同一業種に3銘柄以上集中している場合に警告HTMLを生成"""
    sector_map = {r["code"]: get_sector_cached(r["code"]) for r in results}
    counts = Counter(v for v in sector_map.values() if v and v != "不明")
    warnings = sorted([(s, c) for s, c in counts.items() if c >= 3], key=lambda x: -x[1])
    if not warnings:
        return ""
    rows = ""
    for sector, count in warnings:
        names_in = [r["name"] for r in results if sector_map.get(r["code"]) == sector]
        rows += (f"<tr><td>{sector}</td>"
                 f"<td style='text-align:center'>{count}銘柄</td>"
                 f"<td style='font-size:12px;color:#666'>{', '.join(names_in[:5])}</td></tr>")
    return (f"<div class='card' style='border-left:4px solid #e67e22'>"
            f"<h2>⚠️ セクター集中警告</h2>"
            f"<p style='color:#666;font-size:13px;margin:0 0 10px'>同一業種に3銘柄以上集中しています。分散を検討してください。</p>"
            f"<table><tr style='background:#fef9e7'><th>業種</th><th>銘柄数</th><th>銘柄名</th></tr>{rows}</table>"
            f"</div>")


# ────────────────────────────── 判定ロジック ──────────────────────────────

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


def _net_cls(n):
    return "net-pos" if n >= 5 else ("net-neg" if n < -5 else "net-neu")

def _rel_cls(r):
    return "" if r is None else ("rel-pos" if r >= 0 else "rel-neg")

def _rel_str(r):
    return f"{r:+.1f}%" if r is not None else "-"


_EMAIL_CSS = """
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


# ──────────────────────────── HTMLセクション ──────────────────────────────

def _build_priority_section(priority_actions):
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


def _build_sell_section(results):
    sells = sorted([r for r in results if r["signal"] == "sell"], key=lambda x: x["net"])
    if not sells:
        return (f"<div class='card' style='border-left:4px solid #27ae60'>"
                f"<h2>✅ 売り検討なし</h2>"
                f"<p style='color:#666;margin:0'>全チェック銘柄がポジティブ/中立判定です。</p></div>"), sells
    rows = ""
    for r in sells:
        rows += (f"<tr>"
                 f"<td><b>{r['name']}</b><br>"
                 f"<span style='color:#888;font-size:12px'>{r['code']} ¥{r['close']:,.0f}</span></td>"
                 f"<td class='{_net_cls(r['net'])}' style='text-align:center'>{r['net']:+.1f}%</td>"
                 f"<td class='{_rel_cls(r.get('rel20'))}' style='text-align:center'>{_rel_str(r.get('rel20'))}</td>"
                 f"<td style='text-align:center;color:#888;font-size:12px'>{r.get('vol',0):.0f}%{r.get('vol_label','')}</td>"
                 f"</tr>")
    section = (f"<div class='card' style='border-left:4px solid #c0392b'>"
               f"<h2>🔴 売り検討 ({len(sells)}銘柄)</h2>"
               f"<p style='color:#666;font-size:13px;margin:0 0 10px'>ネットスコアがマイナス。ニュース・決算を確認してください。</p>"
               f"<table><tr style='background:#fde8e8'><th>銘柄</th><th>ネット</th><th>日経比20d</th><th>ボラ</th></tr>"
               f"{rows}</table></div>")
    return section, sells


def _build_ranking_section(results, prev_ranking_codes):
    held_codes = {str(r["code"]) for r in results}
    ranking = load_top_ranking(5)
    rows = ""
    count = 0
    if ranking is not None:
        for _, row in ranking.iterrows():
            code_str = str(int(row["銘柄コード"]))
            if code_str in held_codes or count >= 5:
                continue
            net    = row.get("ネット(%)", row["上昇確率(%)"])
            vol    = row.get("ボラ(%)", 0)
            vol_lb = row.get("ボラ水準", "")
            rel20_r = row.get("日経比20日(%)", None)
            rel20_v = (float(rel20_r) if isinstance(rel20_r, (int, float))
                       and not isinstance(rel20_r, bool) else None)
            per = row.get("PER"); pbr = row.get("PBR")
            fund_str = ""
            if per and str(per) not in ("nan", "None", "-"):
                fund_str += f"PER{float(per):.0f}"
            if pbr and str(pbr) not in ("nan", "None", "-"):
                fund_str += f" PBR{float(pbr):.1f}"
            stop_val = row.get("損切り価格(円)")
            stop_str = f"¥{int(stop_val):,}" if stop_val and str(stop_val) not in ("nan", "None") else "-"
            stop_pct = row.get("損切り幅(%)")
            stop_pct_str = f"({stop_pct:+.1f}%)" if stop_pct and str(stop_pct) not in ("nan", "None") else ""
            new_badge = "<span class='badge-new'>NEW</span>" if (prev_ranking_codes and code_str not in prev_ranking_codes) else ""
            rows += (f"<tr>"
                     f"<td><b>{row['銘柄名']}</b>{new_badge}<br>"
                     f"<span style='color:#888;font-size:12px'>{code_str} ¥{int(row['直近株価(円)']):,}"
                     f"{' ' + fund_str if fund_str else ''}</span></td>"
                     f"<td class='{_net_cls(net)}' style='text-align:center'>{net:+.1f}%</td>"
                     f"<td class='{_rel_cls(rel20_v)}' style='text-align:center'>{_rel_str(rel20_v)}</td>"
                     f"<td style='text-align:center;color:#888;font-size:12px'>{vol:.1f}%{vol_lb}</td>"
                     f"<td style='text-align:center;color:#c0392b;font-size:12px'>{stop_str}{stop_pct_str}</td>"
                     f"</tr>")
            count += 1
    if not rows:
        return ""
    return (f"<div class='card' style='border-left:4px solid #2980b9'>"
            f"<h2>📈 新規候補 Top{count}（ネットスコア順・未保有）</h2>"
            f"<p style='color:#666;font-size:13px;margin:0 0 10px'>"
            f"ネット = 上昇確率 − 下落確率 ／ 日経比20d = 過去20日の日経225比超過リターン</p>"
            f"<table><tr style='background:#e8f0fe'>"
            f"<th>銘柄</th><th>ネット</th><th>日経比20d</th><th>ボラ</th><th>損切り</th></tr>"
            f"{rows}</table></div>")


def _build_all_rows(results, earnings_map=None):
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
        rows += (f"<tr>"
                 f"<td style='text-align:center;color:#aaa;font-size:12px'>{idx}</td>"
                 f"<td><b>{r['name']}</b>{earn_badge}"
                 f"<span style='color:#888;font-size:11px'><br>{r['code']} ¥{r['close']:,.0f}</span>"
                 f"{spark_html}</td>"
                 f"<td style='text-align:center'>{r['prob']:.1f}%</td>"
                 f"<td style='text-align:center'>{drop_str}</td>"
                 f"<td class='{_net_cls(r['net'])}' style='text-align:center'>{r['net']:+.1f}%</td>"
                 f"<td style='text-align:center;font-size:11px'>{r.get('recommend', '')}</td>"
                 f"<td class='{_rel_cls(r.get('rel20'))}' style='text-align:center'>{_rel_str(r.get('rel20'))}</td>"
                 f"<td style='text-align:center;color:#888;font-size:11px'>{r.get('vol',0):.0f}%{r.get('vol_label','')}</td>"
                 f"</tr>")
    return rows


# ──────────────────────────── HTML組み立て ────────────────────────────────

def build_earnings_map(codes):
    """銘柄コードリスト → {code: days_until} の辞書。14日以内のみ返す。"""
    today = datetime.now().date()
    result = {}
    for code in codes:
        d = get_next_earnings_cached(code)
        if d is None:
            continue
        days = (d - today).days
        if 0 <= days <= 14:
            result[str(code)] = days
    return result


def build_html(results, today, is_bear=False, nk5=None, nk20=None, nk60=None,
               prev_ranking_codes=None, prev_results=None, priority_actions=None):
    prev_ranking_codes = prev_ranking_codes or set()
    prev_results       = prev_results or {}

    earnings_map = build_earnings_map([r["code"] for r in results])

    sell_result = _build_sell_section(results)
    if isinstance(sell_result, tuple):
        sell_section, sells = sell_result
    else:
        sell_section, sells = sell_result, []

    buy_cnt = sum(1 for r in results if r.get("recommend", "").startswith(("✅", "🔵")))
    neu_cnt = len(results) - len(sells) - buy_cnt
    nk_str  = (f"日経225: 5日{nk5:+.1f}% / 20日{nk20:+.1f}% / 60日{nk60:+.1f}%"
               if nk5 is not None else "")
    bear_banner = (
        f"<div style='background:#c0392b;border-radius:8px;padding:16px;margin-bottom:16px'>"
        f"<div style='color:white;font-size:18px;font-weight:700;margin-bottom:6px'>"
        f"🚨 下落相場 — 新規買いは見送り推奨</div>"
        f"<div style='color:#fdd;font-size:13px;line-height:1.6'>"
        f"日経225の20日リターンが {nk20:+.1f}% と急落しています。<br>"
        f"下落相場ではモデルの精度が落ち、買いシグナルの信頼性が低下します。<br>"
        f"<b>既存ポジションの損切りラインを確認し、新規買いは相場が落ち着くまで待ってください。</b>"
        f"</div></div>"
    ) if is_bear else ""

    return f"""<html><head>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<style>{_EMAIL_CSS}</style></head>
<body>
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;border-radius:10px;padding:18px;margin-bottom:16px'>
  <div style='font-size:20px;font-weight:700;margin-bottom:4px'>📊 チェック銘柄アラート</div>
  <div style='font-size:13px;color:#aaa'>{today} ／ {nk_str}</div>
</div>
{bear_banner}
<div style='display:flex;gap:8px;margin-bottom:16px'>
  <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
    <div style='font-size:26px;font-weight:700;color:#c0392b'>{len(sells)}</div>
    <div style='font-size:12px;color:#888'>売り検討</div>
  </div>
  <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
    <div style='font-size:26px;font-weight:700;color:#0a7a0a'>{buy_cnt}</div>
    <div style='font-size:12px;color:#888'>買い可能性</div>
  </div>
  <div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)'>
    <div style='font-size:26px;font-weight:700;color:#888'>{neu_cnt}</div>
    <div style='font-size:12px;color:#888'>様子見</div>
  </div>
</div>
{_build_priority_section(priority_actions or [])}
{sell_section}
{_build_ranking_section(results, prev_ranking_codes)}
<div class='card'>
  <h2>📋 チェック銘柄一覧（{len(results)}銘柄 / ネット順）</h2>
  <p style='color:#666;font-size:12px;margin:0 0 10px'>上昇/下落 = モデル確率 ／ ネット = 上昇−下落 ／ 日経比20d = 過去20日超過リターン</p>
  <table>
    <tr><th>#</th><th>銘柄</th><th>上昇</th><th>下落</th><th>ネット</th><th>推奨</th><th>日経比20d</th><th>ボラ</th></tr>
    {_build_all_rows(results, earnings_map)}
  </table>
</div>
{build_diff_section(results, prev_results)}
{build_sector_warning(results)}
<p style='color:#aaa;font-size:11px;text-align:center;margin-top:8px'>
  このメールは過去データに基づく参考情報です。投資判断はご自身の責任で行ってください。
</p>
</body></html>"""


# ──────────────────────────── メール送信 ──────────────────────────────────

def send_email(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = GMAIL_ADDRESS
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())


# ──────────────────────────── メイン処理 ──────────────────────────────────

def main():
    today = datetime.now().strftime("%Y年%m月%d日")
    print("=" * 50)
    print(f"チェック銘柄アラート  {today}")
    print("=" * 50)

    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print("ERROR: .envにGMAIL_ADDRESSとGMAIL_APP_PASSWORDを設定してください")
        return

    held_stocks = load_held_stocks()
    if not held_stocks:
        return
    print(f"チェック銘柄: {len(held_stocks)}銘柄")

    rise_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    drop_path = os.path.expanduser("~/stock-alert/rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pklが見つかりません。先にrf_predict.pyを実行してください")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None

    print("日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD
    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")

    prev_ranking_codes = load_prev_ranking_codes()
    prev_results       = load_prev_results()

    # フェーズ1: 全銘柄の特徴量を収集
    raw_data = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
        feat = extract_features(
            prices["Close"].values,
            prices["Volume"].tolist() if "Volume" in prices.columns else None,
            nk_rets,
        )
        if feat is None:
            continue
        if feat[12] > 0.15 or feat[10] < -0.15:
            continue
        raw_data.append((code, name, prices, feat))

    # フェーズ2: クロスセクショナルランク特徴量を付加（28→34次元）
    if not raw_data:
        print("有効銘柄なし"); return
    feats_matrix = np.array([d[3] for d in raw_data], dtype=float)
    feats_aug    = add_cs_rank_features(feats_matrix)

    # フェーズ3: スコア計算・結果集計
    results = []
    for idx, (code, name, prices, feat) in enumerate(raw_data):
        feat_aug  = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100 if drop_model else None
        close     = float(prices["Close"].iloc[-1])
        net       = rise_prob - drop_prob if drop_prob is not None else rise_prob
        signal    = "sell" if net < NET_SELL_THRESHOLD else "hold"
        judgment, _ = get_judgment(net)

        vol = round(feat_aug[7], 1)
        vol_label = ("🟢低" if vol < 20 else "🟡中" if vol < 40 else "🟠高" if vol < 60 else "🔴超高")

        if net >= 10:
            recommend = "✅ 買い"
        elif net >= 5:
            recommend = "🔵 買い候補"
        elif net < -10:
            recommend = "🔴 売り"
        elif net < -5:
            recommend = "⚠️ 売り候補"
        else:
            recommend = "⏳ 様子見"

        p   = prices["Close"].values
        s5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) >= 6  else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        rel5  = round(s5  - nk5,  1) if nk5  is not None else None
        rel20 = round(s20 - nk20, 1) if nk20 is not None else None

        dp_str   = f"{drop_prob:5.1f}%" if drop_prob is not None else "  N/A "
        rel20_str = f"{rel20:+.1f}%" if rel20 is not None else "N/A"
        print(f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}% 下落{dp_str} ネット{net:+.1f}% 日経比20d{rel20_str} ボラ{vol:.1f}%{vol_label}")

        results.append({
            "code": code, "name": name, "prob": rise_prob, "drop_prob": drop_prob,
            "net": net, "close": close, "signal": signal,
            "vol": vol, "vol_label": vol_label, "recommend": recommend,
            "rel5": rel5, "rel20": rel20,
            "prices_close": prices["Close"].values.tolist(),
        })

    save_results_for_tomorrow(results)

    sell_count      = sum(1 for r in results if r["signal"] == "sell")
    buy_count       = sum(1 for r in results if r.get("recommend", "").startswith(("✅", "🔵")))
    priority_actions = build_priority_actions(results)
    bear_prefix     = "⚠️下落相場 " if is_bear else ""
    nk_str          = f"日経{nk20:+.1f}%" if nk20 is not None else ""
    priority_str    = f"優先{len(priority_actions)} / " if priority_actions else ""
    subject = f"{bear_prefix}[{today[5:]}] {priority_str}売り{sell_count} / 買い候補{buy_count} / {nk_str} {len(results)}銘柄"

    html = build_html(
        results, today, is_bear=is_bear, nk5=nk5, nk20=nk20, nk60=nk60,
        prev_ranking_codes=prev_ranking_codes,
        prev_results=prev_results,
        priority_actions=priority_actions,
    )

    print(f"\nGmail送信中 → {GMAIL_ADDRESS}")
    try:
        send_email(subject, html)
        print("送信完了 ✅")
    except Exception as e:
        print(f"送信失敗: {e}")


if __name__ == "__main__":
    main()

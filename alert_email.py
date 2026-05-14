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
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from lib.utils import get_prices, get_nikkei_returns, extract_features, add_cs_rank_features, get_sector_cached

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getenv("STOCK_ALERT_HOME", PROJECT_DIR)
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.expanduser("~/stock-alert")

load_dotenv(os.path.join(BASE_DIR, ".env"))

GMAIL_ADDRESS         = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD    = os.getenv("GMAIL_APP_PASSWORD")
NET_SELL_THRESHOLD    = -5
BEAR_MARKET_THRESHOLD = -5.0
NEW_CANDIDATE_NET_MIN        = 8.0   # 新規候補のネットスコア下限
NEW_CANDIDATE_NET_MAX        = 13.0  # 新規候補のネットスコア上限（過熱銘柄を回避）
CANDIDATE_EARNINGS_SKIP_DAYS = 7    # 決算N日以内の新規候補を除外
CANDIDATE_DROP_PROB_MAX      = 10.0  # 下落確率N%超の新規候補を除外（【回避】ライン）
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9"}


def _held_codes(results):
    return {str(r["code"]) for r in results}


def _row_code_str(row):
    """ランキング行の銘柄コードを安全に文字列化。変換不能なら None。"""
    raw = row.get("銘柄コード") if hasattr(row, "get") else None
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return None


def _safe_float(val):
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)
    if isinstance(val, str):
        text = val.strip()
        if not text or text.lower() in ("nan", "none", "-"):
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _row_net_percent(row, *, use_rise_fallback):
    """ランキング行のネット(%)を数値化。欠損時は use_rise_fallback に応じて上昇確率 or 0 を使う。"""
    if use_rise_fallback:
        raw = row.get("ネット(%)", row.get("上昇確率(%)"))
    else:
        raw = row.get("ネット(%)", 0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _net_in_candidate_band(net):
    return NEW_CANDIDATE_NET_MIN <= net <= NEW_CANDIDATE_NET_MAX


def _is_new_candidate_skipped(code_str, drop_v):
    """新規候補スキップ判定: 高下落確率 or 決算前"""
    if drop_v is not None and drop_v > CANDIDATE_DROP_PROB_MAX:
        return True
    d = get_next_earnings_cached(code_str)
    if d is not None:
        days = (d - datetime.now().date()).days
        if 0 <= days <= CANDIDATE_EARNINGS_SKIP_DAYS:
            return True
    return False


def volatility_label(vol):
    if vol < 20:
        return "🟢低"
    if vol < 40:
        return "🟡中"
    if vol < 60:
        return "🟠高"
    return "🔴超高"


def _fundamentals_suffix(row):
    parts = []
    per, pbr = row.get("PER"), row.get("PBR")
    if per and str(per) not in ("nan", "None", "-"):
        parts.append(f"PER{float(per):.0f}")
    if pbr and str(pbr) not in ("nan", "None", "-"):
        parts.append(f"PBR{float(pbr):.1f}")
    return " ".join(parts)


def _stop_loss_cell_html(row, close_val):
    stop_val = row.get("損切り価格(円)")
    stop_pct = row.get("損切り幅(%)")
    if not stop_val or str(stop_val) in ("nan", "None"):
        return "-"
    pct_part = f" ({stop_pct:+.1f}%)" if stop_pct and str(stop_pct) not in ("nan", "None") else ""
    return (f"現値 ¥{close_val:,}<br>"
            f"<span style='color:#c0392b;font-weight:700'>↓ ¥{int(stop_val):,}</span>"
            f"<span style='font-size:11px;color:#c0392b'>{pct_part}</span>")


def _unheld_ranking_row_count(ranking_df, held_codes):
    if ranking_df is None:
        return 0
    n = 0
    for _, row in ranking_df.iterrows():
        code = _row_code_str(row)
        if code is None:
            continue
        if code not in held_codes:
            n += 1
    return n


def _new_candidates_for_sector_warning(ranking_df, held_codes, max_rows=100):
    """セクター集中警告用: 新規候補レンジかつ未保有の銘柄リスト。"""
    out = []
    if ranking_df is None:
        return out
    for _, row in ranking_df.head(max_rows).iterrows():
        code = _row_code_str(row)
        if code is None or code in held_codes:
            continue
        net = _row_net_percent(row, use_rise_fallback=False)
        if net is None or not _net_in_candidate_band(net):
            continue
        drop_v = _safe_float(row.get("下落確率(%)", None))
        if _is_new_candidate_skipped(code, drop_v):
            continue
        out.append({"code": code, "name": row["銘柄名"]})
    return out


# ───────────────────────────── データ読み込み ──────────────────────────────

def _ranking_glob_files():
    files = glob.glob(os.path.join(BASE_DIR, "ranking_*.csv"))
    if not files:
        files = glob.glob(os.path.join(BASE_DIR, "results", "ranking_*.csv"))
    return files


def load_held_stocks():
    """Google SheetsまたはCSVからチェック銘柄を読み込む"""
    try:
        from lib.sheets_helper import load_watch_list
        return load_watch_list()
    except Exception as e:
        print(f"[WARN] sheets_helper失敗、CSVで代替: {e}")
        csv_path = os.path.join(BASE_DIR, "watch_list.csv")
        if not os.path.exists(csv_path):
            print("ERROR: watch_list.csvが見つかりません")
            return {}
        df = pd.read_csv(csv_path, dtype=str)
        return dict(zip(df["コード"].str.strip(), df["銘柄名"].str.strip()))


def load_top_ranking(n=15):
    """最新のランキングCSVから上位N銘柄を読み込む（BASE_DIR → results/ の順で探す）"""
    files = _ranking_glob_files()
    if not files:
        return None
    df = pd.read_csv(max(files, key=os.path.getmtime))
    return df.head(n)


def load_prev_ranking_codes():
    """前回（最新より1つ古い）のランキングCSVから銘柄コードセットを取得"""
    files = sorted(_ranking_glob_files(), key=os.path.getmtime)
    if len(files) < 2:
        return set()
    try:
        df = pd.read_csv(files[-2])
        return set(df["銘柄コード"].astype(str).tolist())
    except Exception:
        return set()


def load_prev_results():
    """昨日のアラート結果をJSONから読み込む（差分計算用）"""
    path = os.path.join(BASE_DIR, "alert_results_prev.json")
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
    path = os.path.join(BASE_DIR, "alert_results_prev.json")
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

def build_priority_actions(results, ranking_df=None):
    """今日の優先アクション: 即切 > 売りシグナル > 新規候補の買いシグナル"""
    actions = []

    # 【即切】: 20日リターン < -10%（最優先）
    cuts = sorted([r for r in results if r.get("ret20", 0) < -10.0], key=lambda x: x.get("ret20", 0))
    for r in cuts[:2]:
        actions.append({"emoji": "⚡",
                         "title": f"{r['name']}（{r['code']}）— 即切シグナル",
                         "detail": f"20日リターン {r['ret20']:+.1f}%　ネット {r['net']:+.1f}%"})

    # 保有株の売りシグナル（ネット昇順）
    sells = sorted([r for r in results if r["signal"] == "sell" and r.get("ret20", 0) >= -10.0],
                   key=lambda x: x["net"])
    for r in sells[:2]:
        if len(actions) >= 3:
            break
        dp = r.get("drop_prob")
        label = "下降シグナル" if r["net"] < -10 else "弱気シグナル"
        detail = f"ネット {r['net']:+.1f}%　下落確率 {dp:.1f}%" if dp is not None else f"ネット {r['net']:+.1f}%"
        actions.append({"emoji": "🔴", "title": f"{r['name']}（{r['code']}）— {label}", "detail": detail})

    # 新規候補の買いシグナル（ランキングCSVからnet 8〜13%、未保有）
    if len(actions) < 3:
        held_codes = _held_codes(results)
        ranking = ranking_df.head(50) if ranking_df is not None else load_top_ranking(50)
        if ranking is not None:
            for _, row in ranking.iterrows():
                if len(actions) >= 3:
                    break
                code_str = _row_code_str(row)
                if code_str is None or code_str in held_codes:
                    continue
                net_v = _row_net_percent(row, use_rise_fallback=False)
                if net_v is None or not _net_in_candidate_band(net_v):
                    continue
                drop_v = _safe_float(row.get("下落確率(%)", None))
                if _is_new_candidate_skipped(code_str, drop_v):
                    continue
                actions.append({"emoji": "✅",
                                 "title": f"{row['銘柄名']}（{code_str}）— 新規買いシグナル",
                                 "detail": f"ネット {net_v:+.1f}%　ボラ {row.get('ボラ(%)', 0):.1f}%"})

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

def get_next_earnings_cached(code):
    """次回決算発表予定日をkabutan.jpから取得してJSONキャッシュに保存。失敗時はNone。"""
    cache_path = os.path.join(BASE_DIR, "earnings_cache.json")
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


def recommend_from_net(net):
    """ランキング・保有スコア共通の推奨ラベル（CSV表記に依存しない）"""
    if net > 13:
        return "🟡 高値警戒"
    if net >= 8:
        return "✅ 買い"
    if net >= 5:
        return "🔵 様子見"
    if net < -10:
        return "🔴 下降シグナル"
    if net < -5:
        return "⚠️ 弱気シグナル"
    return "⏳ 方向感なし"


def recommend_from_scores(net, drop_prob=None):
    """drop_prob も考慮した推奨ラベル。【推奨】条件: drop_prob<6% かつ net>=10%"""
    if drop_prob is not None and drop_prob < 6.0 and net >= 10.0:
        return "🌟 推奨"
    return recommend_from_net(net)


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
               f"<table><tr style='background:#fde8e8'><th>銘柄</th><th>ネット</th><th>日経差(20日)</th><th>ボラ</th></tr>"
               f"{rows}</table></div>")
    return section, sells


# セクター性質分類（観察文生成用）
_DEFENSIVE_SECTORS = {"電気・ガス業", "食料品", "医薬品", "情報・通信業",
                      "陸運業", "不動産業", "銀行業", "保険業", "その他金融業",
                      "倉庫・運輸関連業", "水産・農林業", "鉱業"}
_CYCLICAL_SECTORS = {"輸送用機器", "機械", "鉄鋼", "非鉄金属", "化学",
                     "海運業", "金属製品", "ガラス・土石製品", "繊維製品",
                     "石油・石炭製品", "パルプ・紙", "ゴム製品"}
_GROWTH_SECTORS = {"電気機器", "精密機器", "サービス業", "その他製品",
                   "卸売業", "小売業", "証券、商品先物取引業"}


def _classify_sector(sector):
    if sector in _DEFENSIVE_SECTORS:
        return "defensive"
    if sector in _CYCLICAL_SECTORS:
        return "cyclical"
    if sector in _GROWTH_SECTORS:
        return "growth"
    return "other"


def build_candidate_observation(candidates):
    """新規候補リストから「今日の傾向」観察文を自動生成"""
    if not candidates:
        return ""
    nets = [c["net"] for c in candidates]
    avg_net = sum(nets) / len(nets)
    sec_classes = [_classify_sector(c["sector"]) for c in candidates]
    n = len(candidates)
    cnt_def = sec_classes.count("defensive")
    cnt_cyc = sec_classes.count("cyclical")
    cnt_grw = sec_classes.count("growth")

    # セクター傾向
    if cnt_def >= max(3, n * 0.6):
        sector_msg = "🛡️ <b>防御的セクター中心</b>（電力・食品・不動産など）。市場が荒れている時の典型パターン。"
    elif cnt_cyc >= max(3, n * 0.6):
        sector_msg = "⚙️ <b>景気敏感株中心</b>（鉄鋼・機械・化学など）。景気回復期待の表れ。"
    elif cnt_grw >= max(3, n * 0.6):
        sector_msg = "🚀 <b>成長株中心</b>（電気機器・精密機器・サービスなど）。リスクオン局面。"
    else:
        sector_msg = "🌐 <b>セクター分散</b>。特定の市場テーマなし。"

    # 確信度
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


def _build_ranking_section(results, prev_ranking_codes, ranking_df=None):
    held_codes = _held_codes(results)
    ranking = (ranking_df.head(50) if ranking_df is not None else load_top_ranking(50))
    rows = ""
    count = 0
    selected_candidates = []  # 観察文生成用
    if ranking is not None:
        for _, row in ranking.iterrows():
            code_str = _row_code_str(row)
            if code_str in held_codes or count >= 5:
                continue
            rise = row.get("上昇確率(%)", None)
            drop_v = _safe_float(row.get("下落確率(%)", None))
            net_v = _row_net_percent(row, use_rise_fallback=True)
            if net_v is None or not _net_in_candidate_band(net_v):
                continue
            if _is_new_candidate_skipped(code_str, drop_v):
                continue
            selected_candidates.append({"net": net_v, "sector": get_sector_cached(code_str)})
            recommend = recommend_from_scores(net_v, drop_v)
            vol    = row.get("ボラ(%)", 0)
            vol_lb = row.get("ボラ水準", "")
            rel20_v = _safe_float(row.get("日経比20日(%)", None))
            fund_str = _fundamentals_suffix(row)
            close_val = int(row["直近株価(円)"])
            stop_cell = _stop_loss_cell_html(row, close_val)
            drop_str = f"{drop_v:.1f}%" if drop_v is not None else "-"
            new_badge = "<span class='badge-new'>NEW</span>" if (prev_ranking_codes and code_str not in prev_ranking_codes) else ""
            rows += (f"<tr>"
                     f"<td><b>{row['銘柄名']}</b>{new_badge}<br>"
                     f"<span style='color:#888;font-size:12px'>{code_str} ¥{int(row['直近株価(円)']):,}"
                     f"{' ' + fund_str if fund_str else ''}</span></td>"
                     f"<td style='text-align:center'>{rise:.1f}%</td>"
                     f"<td style='text-align:center'>{drop_str}</td>"
                     f"<td class='{_net_cls(net_v)}' style='text-align:center'>{net_v:+.1f}%</td>"
                     f"<td style='text-align:center;font-size:11px'>{recommend}</td>"
                     f"<td class='{_rel_cls(rel20_v)}' style='text-align:center'>{_rel_str(rel20_v)}</td>"
                     f"<td style='text-align:center;color:#888;font-size:12px'>{vol:.1f}%{vol_lb}</td>"
                     f"<td style='text-align:center;font-size:12px;line-height:1.5'>{stop_cell}</td>"
                     f"</tr>")
            count += 1
    if not rows:
        return (f"<div class='card' style='border-left:4px solid #2980b9'>"
                f"<h2>📈 新規候補</h2>"
                f"<p style='color:#666;margin:0;font-size:13px'>本日は条件を満たす新規候補がありません。"
                f"スクリーナー条件に合致する銘柄が市場に少ないか、保有銘柄と重複しています。</p>"
                f"</div>")
    observation = build_candidate_observation(selected_candidates)
    return (f"<div class='card' style='border-left:4px solid #2980b9'>"
            f"<h2>📈 新規候補 Top{count}（ネット {int(NEW_CANDIDATE_NET_MIN)}〜{int(NEW_CANDIDATE_NET_MAX)}%・未保有）</h2>"
            f"{observation}"
            f"<p style='color:#666;font-size:12px;margin:0 0 10px;line-height:1.6'>"
            f"上昇/下落 = 3ヶ月後に±15%以上動くモデル確率 ／ ネット = 上昇−下落 ／ 日経差(20日) = 過去20日で日経225より何%多く動いたか<br>"
            f"<b>PER</b> = 株価÷1株利益（低いほど割安）／ <b>PBR</b> = 株価÷1株純資産（1倍割れで資産価値以下）<br>"
            f"<b>損切り</b> = 現値 → ストップ目安価格（カッコ内はそこまでの下落率）。この価格を割ったら損切り検討</p>"
            f"<table><tr style='background:#e8f0fe'>"
            f"<th>銘柄</th><th>上昇</th><th>下落</th><th>ネット</th><th>推奨</th><th>日経差(20日)</th><th>ボラ</th><th>損切り</th></tr>"
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
        cut_badge = (f"<span style='display:inline-block;background:#6c3483;color:white;"
                     f"font-size:9px;font-weight:700;padding:1px 4px;border-radius:6px;"
                     f"margin-left:3px;vertical-align:middle'>⚡即切</span>"
                     if r.get("ret20", 0) < -10.0 else "")
        rows += (f"<tr>"
                 f"<td style='text-align:center;color:#aaa;font-size:12px'>{idx}</td>"
                 f"<td><b>{r['name']}</b>{earn_badge}{cut_badge}"
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


def _bear_market_banner_html(is_bear, nk20):
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


def _index_etf_banner_html(is_bear, candidate_count, nk20):
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


def _summary_stat_cards_html(n_sell, n_buy, n_neu):
    box = ("<div style='flex:1;background:#fff;border-radius:8px;padding:12px;text-align:center;"
           "box-shadow:0 1px 4px rgba(0,0,0,.08)'>"
           "<div style='font-size:26px;font-weight:700;color:{color}'>{val}</div>"
           "<div style='font-size:12px;color:#888'>{label}</div></div>")
    return (
        "<div style='display:flex;gap:8px;margin-bottom:16px'>"
        + box.format(val=n_sell, label="売り検討", color="#c0392b")
        + box.format(val=n_buy, label="買い可能性", color="#0a7a0a")
        + box.format(val=n_neu, label="様子見", color="#888")
        + "</div>"
    )


def build_html(results, today, is_bear=False, nk5=None, nk20=None, nk60=None,
               prev_ranking_codes=None, prev_results=None, priority_actions=None,
               ranking_df=None):
    prev_ranking_codes = prev_ranking_codes or set()
    prev_results       = prev_results or {}

    if ranking_df is None:
        ranking_df = load_top_ranking(1000)

    earnings_map = build_earnings_map([r["code"] for r in results])

    held_codes_set = _held_codes(results)
    _candidates_for_sector = _new_candidates_for_sector_warning(ranking_df, held_codes_set)

    sell_section, sells = _build_sell_section(results)

    buy_cnt = sum(1 for r in results if r.get("recommend", "").startswith("✅"))
    neu_cnt = len(results) - len(sells) - buy_cnt
    nk_str  = (f"日経225: 5日{nk5:+.1f}% / 20日{nk20:+.1f}% / 60日{nk60:+.1f}%"
               if nk5 is not None else "")
    bear_banner = _bear_market_banner_html(is_bear, nk20)
    candidate_count = _unheld_ranking_row_count(ranking_df, held_codes_set)
    index_banner = _index_etf_banner_html(is_bear, candidate_count, nk20)

    return f"""<html><head>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<style>{_EMAIL_CSS}</style></head>
<body>
<div style='background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;border-radius:10px;padding:18px;margin-bottom:16px'>
  <div style='font-size:20px;font-weight:700;margin-bottom:4px'>📊 チェック銘柄アラート</div>
  <div style='font-size:13px;color:#aaa'>{today} ／ {nk_str}</div>
</div>
{bear_banner}
{index_banner}
{_summary_stat_cards_html(len(sells), buy_cnt, neu_cnt)}
{_build_priority_section(priority_actions or [])}
{sell_section}
{_build_ranking_section(results, prev_ranking_codes, ranking_df)}
{build_sector_warning(_candidates_for_sector)}
<div class='card'>
  <h2>📋 チェック銘柄一覧（{len(results)}銘柄 / ネット順）</h2>
  <p style='color:#666;font-size:12px;margin:0 0 10px'>上昇/下落 = 3ヶ月後に±15%以上動くモデル確率 ／ ネット = 上昇−下落 ／ 日経差(20日) = 過去20日間で日経225より何%多く動いたか</p>
  <table>
    <tr><th>#</th><th>銘柄</th><th>上昇</th><th>下落</th><th>ネット</th><th>推奨</th><th>日経差(20日)</th><th>ボラ</th></tr>
    {_build_all_rows(results, earnings_map)}
  </table>
</div>
{build_diff_section(results, prev_results)}
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

def _load_alert_models_or_exit():
    rise_path = os.path.join(BASE_DIR, "rf_model.pkl")
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pklが見つかりません。先にrf_train_v3.pyを実行してください")
        return None, None
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    return rise_model, drop_model


def _gather_raw_feature_rows(held_stocks, nk5, nk20, nk60):
    raw_data = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            print(f"  ❓ {name}({code}): データ取得失敗")
            continue
        nk_rets = (nk5 / 100, nk20 / 100, nk60 / 100) if nk5 is not None else None
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
    return raw_data


def _augmented_feature_matrix(raw_data):
    feats_matrix = np.array([d[3] for d in raw_data], dtype=float)
    return add_cs_rank_features(feats_matrix)


def _held_results_from_models(raw_data, feats_aug, rise_model, drop_model, nk5, nk20):
    results = []
    for idx, (code, name, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100 if drop_model else None
        close = float(prices["Close"].iloc[-1])
        net = rise_prob - drop_prob if drop_prob is not None else rise_prob
        signal = "sell" if net < NET_SELL_THRESHOLD else "hold"
        judgment, _ = get_judgment(net)
        vol = round(feat_aug[7], 1)
        vol_label = volatility_label(vol)
        recommend = recommend_from_scores(net, drop_prob)
        p = prices["Close"].values
        s5 = (p[-1] - p[-6]) / p[-6] * 100 if len(p) >= 6 else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        rel5 = round(s5 - nk5, 1) if nk5 is not None else None
        rel20 = round(s20 - nk20, 1) if nk20 is not None else None
        dp_str = f"{drop_prob:5.1f}%" if drop_prob is not None else "  N/A "
        rel20_str = f"{rel20:+.1f}%" if rel20 is not None else "N/A"
        print(f"  {judgment}  {name}({code}): 上昇{rise_prob:5.1f}% 下落{dp_str} ネット{net:+.1f}% 日経差(20日){rel20_str} ボラ{vol:.1f}%{vol_label}")
        results.append({
            "code": code, "name": name, "prob": rise_prob, "drop_prob": drop_prob,
            "net": net, "close": close, "signal": signal,
            "vol": vol, "vol_label": vol_label, "recommend": recommend,
            "rel5": rel5, "rel20": rel20, "ret20": round(s20, 1),
            "prices_close": prices["Close"].values.tolist(),
        })
    return results


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

    rise_model, drop_model = _load_alert_models_or_exit()
    if rise_model is None:
        return

    print("日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD
    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")

    prev_ranking_codes = load_prev_ranking_codes()
    prev_results       = load_prev_results()

    raw_data = _gather_raw_feature_rows(held_stocks, nk5, nk20, nk60)
    if not raw_data:
        print("有効銘柄なし"); return
    feats_aug = _augmented_feature_matrix(raw_data)
    results = _held_results_from_models(raw_data, feats_aug, rise_model, drop_model, nk5, nk20)

    save_results_for_tomorrow(results)

    ranking_df = load_top_ranking(1000)
    sell_count      = sum(1 for r in results if r["signal"] == "sell")
    buy_count       = sum(1 for r in results if r.get("recommend", "").startswith("✅"))
    priority_actions = build_priority_actions(results, ranking_df=ranking_df)
    bear_prefix     = "⚠️下落相場 " if is_bear else ""
    nk_str          = f"日経{nk20:+.1f}%" if nk20 is not None else ""
    priority_str    = f"優先{len(priority_actions)} / " if priority_actions else ""
    subject = f"{bear_prefix}[{today[5:]}] {priority_str}売り{sell_count} / 買い候補{buy_count} / {nk_str} {len(results)}銘柄"

    html = build_html(
        results, today, is_bear=is_bear, nk5=nk5, nk20=nk20, nk60=nk60,
        prev_ranking_codes=prev_ranking_codes,
        prev_results=prev_results,
        priority_actions=priority_actions,
        ranking_df=ranking_df,
    )

    print(f"\nGmail送信中 → {GMAIL_ADDRESS}")
    try:
        send_email(subject, html)
        print("送信完了 ✅")
    except Exception as e:
        print(f"送信失敗: {e}")


if __name__ == "__main__":
    main()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import glob
import re
import smtplib
import joblib
import requests
import pandas as pd
from collections import Counter
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
from lib.utils import get_prices, get_nikkei_returns, extract_features, add_cs_rank_features, get_sector_cached, recommend_from_net, recommend_from_scores
from core.rank_stocks import passes_buy_filter, fetch_us_sector_etf_returns, get_sector_etf, _load_sector_cache, STRONG_EFFECT_ETFS
from lib.db import save_held_scores, get_earnings_cache, set_earnings_cache, CACHE_MISS, get_holding_days, get_yutai_cache
from config import (BASE_DIR, BEAR_MARKET_THRESHOLD, HOT_MARKET_THRESHOLD,
                    NEW_CANDIDATE_NET_MIN, NEW_CANDIDATE_NET_MAX,
                    CANDIDATE_DROP_PROB_MAX, CANDIDATE_EARNINGS_SKIP_DAYS,
                    CANDIDATE_CONFLICT_NET_MIN, CANDIDATE_CONFLICT_DROP_MIN,
                    SELL_DAYS_MID, SELL_DAYS_LATE,
                    NET_SELL_THRESHOLD_MID, NET_SELL_THRESHOLD_LATE)
from email_html import (
    EMAIL_CSS as _EMAIL_CSS,
    volatility_label, get_judgment,
    net_cls as _net_cls, rel_cls as _rel_cls, rel_str as _rel_str,
    fundamentals_suffix as _fundamentals_suffix,
    stop_loss_cell_html as _stop_loss_cell_html,
    build_sparkline_svg, build_candidate_observation,
    build_priority_section as _build_priority_section,
    build_sell_section as _build_sell_section,
    build_all_rows as _build_all_rows,
    bear_market_banner_html as _bear_market_banner_html,
    hot_market_banner_html as _hot_market_banner_html,
    index_etf_banner_html as _index_etf_banner_html,  # noqa: E501
    summary_stat_cards_html as _summary_stat_cards_html,
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "ja,en;q=0.9"}


def _etf_badge_html(code_str, etf_rets):
    """S買い候補の米国ETFリードラグバッジHTMLを返す（強相関セクターのみ）"""
    if not etf_rets:
        return ""
    etf = get_sector_etf(str(code_str))
    if not etf or etf not in STRONG_EFFECT_ETFS:
        return ""
    ret = etf_rets.get(etf)
    if ret is None:
        return ""
    bg  = "#e8f8f0" if ret >= 0 else "#fdf0f0"
    clr = "#1e8449" if ret >= 0 else "#c0392b"
    return (f"<span style='font-size:10px;color:{clr};background:{bg};"
            f"border:1px solid {clr}33;border-radius:3px;padding:1px 5px;margin-left:4px;"
            f"white-space:nowrap'>🇺🇸{etf}{ret:+.1f}%</span>")


def _held_codes(results):
    return {str(r["code"]) for r in results}


def _tiered_sell_signal(net, holding_days):
    """下降シグナル基準（net<-10）でのみ売り。BTで75%勝率+9.4%。"""
    if holding_days is not None and holding_days > SELL_DAYS_LATE:
        return "sell" if net < NET_SELL_THRESHOLD_LATE else "hold"
    return "sell" if net < NET_SELL_THRESHOLD_MID else "hold"


def _row_code_str(row):
    val = row.get("銘柄コード") if hasattr(row, "get") else row["銘柄コード"]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return str(int(float(val)))
    except (TypeError, ValueError):
        return None


def _safe_float(val):
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
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


def _is_new_candidate_skipped(code_str, net, drop_v):
    """新規候補スキップ判定: 高下落確率 or コンフリクト or 決算前"""
    if drop_v is not None and drop_v > CANDIDATE_DROP_PROB_MAX:
        return True
    if (drop_v is not None and net is not None
            and net >= CANDIDATE_CONFLICT_NET_MIN
            and drop_v >= CANDIDATE_CONFLICT_DROP_MIN):
        return True
    d = get_next_earnings_cached(code_str)
    if d is not None:
        days = (d - datetime.now().date()).days
        if 0 <= days <= CANDIDATE_EARNINGS_SKIP_DAYS:
            return True
    return False




def _unheld_ranking_row_count(ranking_df, held_codes):
    if ranking_df is None:
        return 0
    n = 0
    for _, row in ranking_df.iterrows():
        code = _row_code_str(row)
        if code is not None and code not in held_codes:
            n += 1
    return n


def _new_candidates_for_sector_warning(ranking_df, held_codes, max_rows=100):
    """セクター集中警告用: 新規候補レンジかつ未保有、推奨がS買いの銘柄リスト。"""
    out = []
    if ranking_df is None:
        return out
    for _, row in ranking_df.head(max_rows).iterrows():
        code = _row_code_str(row)
        if code is None or code in held_codes:
            continue
        rec_str = row.get("推奨", "") or ""
        if "S買い" not in rec_str:
            continue
        net = _row_net_percent(row, use_rise_fallback=False)
        if net is None or not _net_in_candidate_band(net):
            continue
        drop_v = _safe_float(row.get("下落確率(%)", None))
        if _is_new_candidate_skipped(code, net, drop_v):
            continue
        out.append({"code": code, "name": row["銘柄名"]})
    return out


# ───────────────────────────── データ読み込み ──────────────────────────────

def _ranking_glob_files():
    return glob.glob(os.path.join(BASE_DIR, "data", "rankings", "ranking_*.csv"))


def _db_ranking_to_df(rows):
    """DB行リスト → ranking CSV と同じ列名のDataFrame"""
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.rename(columns={
        "code": "銘柄コード", "name": "銘柄名", "close": "直近株価(円)",
        "rise_prob": "上昇確率(%)", "drop_prob": "下落確率(%)", "net": "ネット(%)",
        "vol": "ボラ(%)", "recommend": "推奨", "rel20": "日経比20日(%)",
        "stop_loss": "損切り価格(円)", "per": "PER", "pbr": "PBR",
    })
    df["ボラ水準"] = df["ボラ(%)"].apply(volatility_label)
    valid = df["損切り価格(円)"].notna() & df["直近株価(円)"].notna() & (df["直近株価(円)"] != 0)
    df.loc[valid, "損切り幅(%)"] = (
        (df.loc[valid, "損切り価格(円)"] - df.loc[valid, "直近株価(円)"]) / df.loc[valid, "直近株価(円)"] * 100
    ).round(1)
    return df


def load_held_stocks():
    """
    Google SheetsまたはCSVからチェック銘柄を読み込む。
    戻り値: (held_stocks, buy_date_map, qty_map)
      - held_stocks : {コード: 銘柄名}
      - buy_date_map: {コード: "YYYY-MM-DD"} — 購入日列がある場合のみ
      - qty_map     : {コード: int}          — 数量列がある場合のみ
    """
    try:
        from lib.sheets_helper import load_watch_list_df
        df = load_watch_list_df()
    except Exception as e:
        logger.warning("sheets_helper失敗、CSVで代替: %s", e)
        csv_path = os.path.join(BASE_DIR, "watch_list.csv")
        if not os.path.exists(csv_path):
            logger.error("watch_list.csvが見つかりません")
            return {}, {}, {}
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        df["コード"] = df["コード"].str.strip().str.zfill(4)
        df["銘柄名"] = df["銘柄名"].str.strip()

    held_stocks = dict(zip(df["コード"], df["銘柄名"]))

    buy_date_map = {}
    if "購入日" in df.columns:
        for _, row in df.iterrows():
            val = str(row["購入日"]).strip()
            if val and val.lower() not in ("nan", ""):
                buy_date_map[row["コード"]] = val

    qty_map = {}
    if "数量" in df.columns:
        for _, row in df.iterrows():
            try:
                qty_map[row["コード"]] = int(float(str(row["数量"]).strip()))
            except (ValueError, TypeError):
                pass

    return held_stocks, buy_date_map, qty_map


def load_top_ranking(n=15):
    """最新ランキングをDBから取得。DBにデータがなければCSVにフォールバック。"""
    import sqlite3
    from lib.db import DB_PATH, init_db
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT MAX(date) FROM daily_ranking").fetchone()
        if row and row[0]:
            rows = con.execute(
                "SELECT * FROM daily_ranking WHERE date=? ORDER BY net DESC LIMIT ?",
                (row[0], n)
            ).fetchall()
            df = _db_ranking_to_df(rows)
            if df is not None:
                return df
    files = _ranking_glob_files()
    if not files:
        return None
    return pd.read_csv(max(files, key=os.path.getmtime)).head(n)


def load_prev_ranking_codes():
    """前回ランキングの銘柄コードセットをDBから取得。"""
    import sqlite3
    from lib.db import DB_PATH, init_db
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        dates = con.execute(
            "SELECT DISTINCT date FROM daily_ranking ORDER BY date DESC LIMIT 2"
        ).fetchall()
        if len(dates) < 2:
            return set()
        rows = con.execute("SELECT code FROM daily_ranking WHERE date=?", (dates[1][0],)).fetchall()
        return {r[0] for r in rows}




# ───────────────────────────── スパークライン ──────────────────────────────

# ───────────────────────────── 優先アクション ─────────────────────────────

def build_priority_actions(results, ranking_df=None, etf_rets=None):
    """今日の優先アクション: 即切 > 売りシグナル > 買い増し > 新規候補の買いシグナル"""
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

    # 【買い増し】: 保有株かつ S買い（drop_prob<4% & net≥10%）
    if len(actions) < 3:
        add_candidates = sorted(
            [r for r in results
             if r["signal"] == "hold"
             and r.get("recommend", "").startswith("🥇")
             and (r.get("qty") is None or r.get("qty", 0) > 0)],
            key=lambda x: -x["net"]
        )
        for r in add_candidates:
            if len(actions) >= 3:
                break
            dp = r.get("drop_prob")
            actions.append({"emoji": "🟢",
                             "title": f"{r['name']}（{r['code']}）— 買い増しシグナル",
                             "detail": f"ネット {r['net']:+.1f}%　下落確率 {dp:.1f}%"})

    # 新規候補の買いシグナル（ランキングCSVからnet 8〜13%、未保有、推奨がS買いのみ）
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
                rec_str = row.get("推奨", "") or ""
                if "S買い" not in rec_str:
                    continue
                net_v = _row_net_percent(row, use_rise_fallback=False)
                if net_v is None or not _net_in_candidate_band(net_v):
                    continue
                drop_v = _safe_float(row.get("下落確率(%)", None))
                if _is_new_candidate_skipped(code_str, net_v, drop_v):
                    continue
                etf_str = ""
                if etf_rets:
                    etf = get_sector_etf(code_str)
                    if etf and etf in STRONG_EFFECT_ETFS:
                        ret = etf_rets.get(etf)
                        if ret is not None:
                            etf_str = f"　🇺🇸{etf}{ret:+.1f}%"
                actions.append({"emoji": "✅",
                                 "title": f"{row['銘柄名']}（{code_str}）— 新規買いシグナル",
                                 "detail": f"ネット {net_v:+.1f}%　ボラ {row.get('ボラ(%)', 0):.1f}%{etf_str}"})

    return actions[:3]


# ───────────────────────────── セクター警告 ───────────────────────────────

def get_next_earnings_cached(code):
    """次回決算発表予定日をkabutan.jpから取得してDBキャッシュに保存。失敗時はNone。"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    cached = get_earnings_cache(code, today_str)
    if cached is not CACHE_MISS:
        return datetime.strptime(cached, "%Y-%m-%d").date() if cached else None
    try:
        resp = requests.get(f"https://kabutan.jp/stock/?code={code}",
                            headers=_HEADERS, timeout=8)
        date = None
        if resp.status_code == 200:
            m = re.search(r'決算発表予定日[^<]*<[^>]+>(\d{4}/\d{2}/\d{2})', resp.text)
            if m:
                date = datetime.strptime(m.group(1), "%Y/%m/%d").date()
        set_earnings_cache(code, today_str, date.isoformat() if date else None)
        return date
    except (requests.RequestException, ValueError):
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




def build_yutai_rebound_section(today_str: str) -> str:
    """権利落ち後0〜14日のリバウンドチャンス銘柄をHTMLで返す。なければ空文字。"""
    import calendar
    from lib.db import DB_PATH, init_db
    import sqlite3
    today = datetime.strptime(today_str, "%Y-%m-%d").date()

    init_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        # yutai_cacheから権利確定月ごとに権利落ち日を計算し、0〜14日後の銘柄を抽出
        yutai_rows = con.execute(
            "SELECT code, record_month FROM yutai_cache WHERE has_yutai=1 AND record_month IS NOT NULL"
        ).fetchall()
    finally:
        con.close()

    rebound_codes = []
    for row in yutai_rows:
        rm = row["record_month"]
        for yr in [today.year - 1, today.year]:
            last_day = calendar.monthrange(yr, rm)[1]
            ex_date = datetime(yr, rm, last_day).date() - __import__("datetime").timedelta(days=2)
            days_since = (today - ex_date).days
            if 0 <= days_since <= 14:
                rebound_codes.append((row["code"], days_since))
                break

    if not rebound_codes:
        return ""

    # daily_rankingから今日の銘柄スコアを取得
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows_html = ""
        for code, days_since in sorted(rebound_codes, key=lambda x: x[1]):
            rec = con.execute(
                "SELECT name, close, net, drop_prob, vol FROM daily_ranking WHERE date=? AND code=? LIMIT 1",
                (today_str, str(code))
            ).fetchone()
            if rec is None:
                continue
            net_color = "#27ae60" if rec["net"] >= 5 else "#e74c3c" if rec["net"] < 0 else "#888"
            rows_html += (
                f"<tr>"
                f"<td>{rec['name']} <span style='color:#666;font-size:11px'>({code})</span></td>"
                f"<td style='text-align:right'>{int(rec['close']):,}円</td>"
                f"<td style='text-align:right;color:{net_color};font-weight:bold'>{rec['net']:+.1f}%</td>"
                f"<td style='text-align:right;color:#888'>{days_since}日前</td>"
                f"</tr>"
            )
    finally:
        con.close()

    if not rows_html:
        return ""

    return (
        "<div class='card' style='border-left:4px solid #9b59b6'>"
        "<h2>🔄 優待リバウンドチャンス</h2>"
        "<p style='color:#666;font-size:12px;margin:0 0 10px'>"
        "権利落ち後0〜14日以内の銘柄。優待クロス解消による反発が期待できます。購入前にファンダメンタルズを確認してください。</p>"
        "<table>"
        "<tr style='background:#f5eef8'><th>銘柄</th><th>株価</th><th>ネット</th><th>権利落ち</th></tr>"
        f"{rows_html}"
        "</table></div>"
    )


def _build_ranking_section(results, prev_ranking_codes, ranking_df=None, etf_rets=None):
    held_codes = _held_codes(results)
    ranking = (ranking_df.head(50) if ranking_df is not None else load_top_ranking(50))
    rows = ""
    count = 0
    selected_candidates = []  # 観察文生成用
    if ranking is not None:
        for _, row in ranking.iterrows():
            code_str = _row_code_str(row)
            if code_str is None or code_str in held_codes or count >= 5:
                continue
            rise = row.get("上昇確率(%)", None)
            drop_v = _safe_float(row.get("下落確率(%)", None))
            net_v = _row_net_percent(row, use_rise_fallback=True)
            if net_v is None or not _net_in_candidate_band(net_v):
                continue
            if _is_new_candidate_skipped(code_str, net_v, drop_v):
                continue
            rec_str = row.get("推奨", "") or ""
            if "S買い" not in rec_str:
                continue
            selected_candidates.append({"net": net_v, "sector": get_sector_cached(code_str)})
            recommend = recommend_from_scores(net_v, drop_v, allow_buy=True)
            vol    = row.get("ボラ(%)", 0)
            vol_lb = row.get("ボラ水準", "")
            vol_rank = _safe_float(row.get("ボラランク(%)", None))
            rel20_v = _safe_float(row.get("日経比20日(%)", None))
            fund_str = _fundamentals_suffix(row)
            close_val = int(row["直近株価(円)"])
            stop_cell = _stop_loss_cell_html(row, close_val)
            drop_str = f"{drop_v:.1f}%" if drop_v is not None else "-"
            vrank_str = (f"<span style='font-size:10px;color:#666'>ボラランク {vol_rank:.0f}%ile</span>"
                         if vol_rank is not None else "")
            new_badge = "<span class='badge-new'>NEW</span>" if (prev_ranking_codes and code_str not in prev_ranking_codes) else ""
            etf_badge = _etf_badge_html(code_str, etf_rets) if allow_buy else ""
            rows += (f"<tr>"
                     f"<td><b>{row['銘柄名']}</b>{new_badge}<br>"
                     f"<span style='color:#888;font-size:12px'>{code_str} ¥{int(row['直近株価(円)']):,}"
                     f"{' ' + fund_str if fund_str else ''}</span><br>{vrank_str}</td>"
                     f"<td style='text-align:center'>{rise:.1f}%</td>"
                     f"<td style='text-align:center'>{drop_str}</td>"
                     f"<td class='{_net_cls(net_v)}' style='text-align:center'>{net_v:+.1f}%</td>"
                     f"<td style='text-align:center;font-size:11px'>{recommend}{etf_badge}</td>"
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


def build_html(results, today, is_bear=False, is_hot=False, nk5=None, nk20=None, nk60=None,
               prev_ranking_codes=None, priority_actions=None,
               ranking_df=None, etf_rets=None):
    prev_ranking_codes = prev_ranking_codes or set()

    if ranking_df is None:
        ranking_df = load_top_ranking(1000)

    earnings_map = build_earnings_map([r["code"] for r in results])

    held_codes_set = _held_codes(results)
    _candidates_for_sector = _new_candidates_for_sector_warning(ranking_df, held_codes_set)

    sell_section, sells = _build_sell_section(results)

    buy_cnt = len(results) - len(sells)
    nk_str  = (f"日経225: 5日{nk5:+.1f}% / 20日{nk20:+.1f}% / 60日{nk60:+.1f}%"
               if nk5 is not None else "")
    bear_banner = _bear_market_banner_html(is_bear, nk20)
    hot_banner  = _hot_market_banner_html(is_hot, nk60)
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
{hot_banner}
{index_banner}
{_summary_stat_cards_html(len(sells), buy_cnt)}
{_build_priority_section(priority_actions or [])}
{sell_section}
{_build_ranking_section(results, prev_ranking_codes, ranking_df, etf_rets=etf_rets)}
{build_sector_warning(_candidates_for_sector)}
{build_yutai_rebound_section(datetime.now().strftime("%Y-%m-%d"))}
<div class='card'>
  <h2>📋 チェック銘柄一覧（{len(results)}銘柄 / ネット順）</h2>
  <p style='color:#666;font-size:12px;margin:0 0 10px'>上昇/下落 = 3ヶ月後に±15%以上動くモデル確率 ／ ネット = 上昇−下落 ／ 日経差(20日) = 過去20日間で日経225より何%多く動いたか</p>
  <table>
    <tr><th>#</th><th>銘柄</th><th>上昇</th><th>下落</th><th>ネット</th><th>推奨</th><th>日経差(20日)</th><th>ボラ</th></tr>
    {_build_all_rows(results, earnings_map)}
  </table>
</div>
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
        logger.error("rf_model.pklが見つかりません。先にrf_train_v3.pyを実行してください")
        return None, None
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    return rise_model, drop_model


def _gather_raw_feature_rows(held_stocks, nk5, nk20, nk60):
    raw_data = []
    for code, name in held_stocks.items():
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            logger.warning("データ取得失敗: %s(%s)", name, code)
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


def _held_results_from_models(raw_data, feats_aug, rise_model, drop_model, nk5, nk20,
                               holding_days_map=None, buy_date_map=None):
    holding_days_map = holding_days_map or {}
    buy_date_map = buy_date_map or {}
    results = []
    for idx, (code, name, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) * 100 if drop_model else None
        close = float(prices["Close"].iloc[-1])
        net = rise_prob - drop_prob if drop_prob is not None else rise_prob
        holding_days = holding_days_map.get(str(code))
        signal = _tiered_sell_signal(net, holding_days)
        judgment, _ = get_judgment(net)
        vol = round(feat_aug[7], 1)
        vol_label = volatility_label(vol)
        volumes = prices["Volume"].tolist() if "Volume" in prices.columns else []
        buy_ok = passes_buy_filter(feat, close, volumes, nk20=nk20)
        recommend = recommend_from_scores(net, drop_prob, allow_buy=buy_ok)
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
            "holding_days": holding_days,
        })
    return results


def main():
    today = datetime.now().strftime("%Y年%m月%d日")
    logger.info("チェック銘柄アラート開始  %s", today)

    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        logger.error(".envにGMAIL_ADDRESSとGMAIL_APP_PASSWORDを設定してください")
        return

    held_stocks, buy_date_map, qty_map = load_held_stocks()
    if not held_stocks:
        return
    logger.info("チェック銘柄: %d銘柄", len(held_stocks))
    today_date_str = datetime.now().strftime("%Y-%m-%d")

    rise_model, drop_model = _load_alert_models_or_exit()
    if rise_model is None:
        return

    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD
    is_hot  = nk60 is not None and nk60 >= HOT_MARKET_THRESHOLD
    if nk5 is not None:
        logger.info("日経225: 5日%+.2f%% / 20日%+.2f%% / 60日%+.2f%%", nk5, nk20, nk60)
        if is_bear:
            logger.warning("下落相場検知（日経20日: %+.1f%%）: 買いシグナルの信頼性低下", nk20)
        if is_hot:
            logger.warning("急騰相場検知（日経60日: %+.1f%%）: 中小型株は指数に追いつけない可能性あり", nk60)

    prev_ranking_codes = load_prev_ranking_codes()

    raw_data = _gather_raw_feature_rows(held_stocks, nk5, nk20, nk60)
    if not raw_data:
        logger.warning("有効銘柄なし"); return
    feats_aug = _augmented_feature_matrix(raw_data)

    # 購入日が取得できた銘柄は正確な保有日数を使用、それ以外はDBから推定
    if buy_date_map:
        from datetime import date as _date
        today_d = _date.fromisoformat(today_date_str)
        holding_days_map = {}
        for code in held_stocks:
            if code in buy_date_map:
                try:
                    buy_d = _date.fromisoformat(buy_date_map[code])
                    holding_days_map[code] = (today_d - buy_d).days
                except ValueError:
                    pass
        # 購入日がない銘柄はDBで補完
        missing = [c for c in held_stocks if c not in holding_days_map]
        if missing:
            holding_days_map.update(get_holding_days(missing, today_date_str))
    else:
        holding_days_map = get_holding_days(list(held_stocks.keys()), today_date_str)

    results = _held_results_from_models(raw_data, feats_aug, rise_model, drop_model, nk5, nk20,
                                        holding_days_map, buy_date_map)

    # 数量・含み損益を結果に付加
    prices_cache = {str(code): prices for code, name, prices, feat in raw_data}
    for r in results:
        code_str = str(r["code"])
        r["qty"] = qty_map.get(code_str)
        if r["qty"] == 0:
            r["signal"] = "hold"
        buy_date_str = buy_date_map.get(code_str)
        buy_price = None
        after_since_buy = None
        if buy_date_str and code_str in prices_cache:
            try:
                import pandas as pd
                # YY/MM/DD と YYYY-MM-DD の両形式を吸収
                parts = str(buy_date_str).strip().split("/")
                if len(parts) == 3:
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    if y < 100:
                        y += 2000
                    buy_ts = pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
                else:
                    buy_ts = pd.Timestamp(buy_date_str)
                hist = prices_cache[code_str]
                idx = hist.index
                # タイムゾーン有無を吸収
                cmp_ts = buy_ts.tz_localize(idx.tz) if idx.tz is not None else buy_ts
                after_since_buy = hist[idx >= cmp_ts]
                if not after_since_buy.empty:
                    buy_price = float(after_since_buy["Close"].iloc[0])
            except Exception:
                pass
        r["buy_price"] = buy_price
        qty = r["qty"]
        close = r.get("close")
        if buy_price and qty and close:
            r["pnl"]     = (close - buy_price) * qty
            r["pnl_pct"] = (close - buy_price) / buy_price * 100
        else:
            r["pnl"] = r["pnl_pct"] = None


    save_held_scores(today_date_str, results)

    ranking_df = load_top_ranking(1000)

    # 米国セクターETFリードラグ情報（メール表示用）
    _load_sector_cache()
    etf_rets = fetch_us_sector_etf_returns()
    if etf_rets:
        logger.info("US ETF: " + " ".join(f"{k}{v:+.1f}%" for k, v in sorted(etf_rets.items()) if k in STRONG_EFFECT_ETFS))

    sell_count      = sum(1 for r in results if r["signal"] == "sell")
    buy_count       = len(results) - sell_count
    priority_actions = build_priority_actions(results, ranking_df=ranking_df, etf_rets=etf_rets)
    bear_prefix     = "⚠️下落相場 " if is_bear else ("🚀急騰相場 " if is_hot else "")
    nk_str          = f"日経{nk20:+.1f}%" if nk20 is not None else ""
    priority_str    = f"優先{len(priority_actions)} / " if priority_actions else ""
    subject = f"{bear_prefix}[{today[5:]}] {priority_str}売り{sell_count} / 買い候補{buy_count} / {nk_str} {len(results)}銘柄"

    html = build_html(
        results, today, is_bear=is_bear, is_hot=is_hot, nk5=nk5, nk20=nk20, nk60=nk60,
        prev_ranking_codes=prev_ranking_codes,
        priority_actions=priority_actions,
        ranking_df=ranking_df,
        etf_rets=etf_rets,
    )

    logger.info("Gmail送信中 → %s", GMAIL_ADDRESS)
    try:
        send_email(subject, html)
        logger.info("送信完了")
    except smtplib.SMTPException as e:
        logger.error("送信失敗: %s", e)


if __name__ == "__main__":
    main()

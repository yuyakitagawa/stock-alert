"""
lib/alt_data.py
オルタナティブデータ取得モジュール（全て無料・公開データ）

1. 信用倍率 / 信用残    (kabutan.jp/stock/credit)
2. TDnet適時開示        (kabutan IR + TDnet)
3. 空売り残高           (JPX/TSE公開週次Excel)
4. Googleトレンド       (pytrends — 要pip install pytrends)

設計方針:
- DBキャッシュ優先（当日1回のみ外部アクセス）
- 失敗時は常に None / [] / 0.0 を返す（例外は伝播しない）
- rank_stocks.py の後処理で使用。モデル特徴量は次回再学習で追加予定。
"""
import re
import io
import time
import requests
from datetime import datetime, date, timedelta
from lib.db import (
    get_margin_data_latest, upsert_margin_data,
    get_tdnet_events_recent, upsert_tdnet_events,
    get_short_interest_latest, bulk_upsert_short_interest,
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── 1. 信用倍率 / 信用残 ─────────────────────────────────────────────────

def _parse_number(text: str) -> "float | None":
    """カンマ・全角数字を除去して float に変換。失敗時は None。"""
    try:
        cleaned = text.replace(",", "").replace("，", "").replace("倍", "").strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def get_margin_data(code: str) -> "dict | None":
    """信用倍率・信用買残・信用売残を返す。DBキャッシュ（当日）優先。

    Returns:
        {'ratio': float, 'buy': float, 'sell': float, 'date': str} or None
    """
    today_str = date.today().isoformat()
    cached = get_margin_data_latest(str(code))
    if cached and cached.get("fetched_date") == today_str:
        return {"ratio": cached["ratio"], "buy": cached["buy_balance"],
                "sell": cached["sell_balance"], "date": cached["week_date"]}

    try:
        resp = requests.get(
            f"https://kabutan.jp/stock/credit?code={code}",
            headers=_HEADERS, timeout=8
        )
        if resp.status_code != 200:
            return cached  # 古いキャッシュを返す
        text = resp.text.replace("\n", "").replace("\t", "").replace(" ", "")

        # 信用倍率のパターン: 数値 + 倍
        ratio_m = re.search(r'信用倍率[^<]{0,20}<td[^>]*>([\d,.]+)倍?</td>', text)
        buy_m   = re.search(r'信用買残[^<]{0,20}<td[^>]*>([\d,]+)', text)
        sell_m  = re.search(r'信用売残[^<]{0,20}<td[^>]*>([\d,]+)', text)

        # フォールバック: テーブル行から数値を抽出
        if not ratio_m:
            ratio_m = re.search(r'<td[^>]*>([\d]+\.[\d]+)</td>[^<]*<td[^>]*>倍</td>', text)
        if not ratio_m:
            # 最もシンプルな数値一致（倍の直前）
            ratio_m = re.search(r'>([\d]+\.[\d]{1,2})</td>', text)

        ratio = _parse_number(ratio_m.group(1)) if ratio_m else None
        buy   = _parse_number(buy_m.group(1))   if buy_m   else None
        sell  = _parse_number(sell_m.group(1))  if sell_m  else None

        # 最新週の確定日を取得（kabutan は YYYY/MM/DD 形式）
        date_m = re.search(r'(\d{4}/\d{2}/\d{2})', text[:3000])
        week_date = date_m.group(1).replace("/", "-") if date_m else today_str

        if ratio is not None:
            upsert_margin_data(
                code, week_date,
                buy   if buy   is not None else 0.0,
                sell  if sell  is not None else 0.0,
                ratio
            )
            return {"ratio": ratio, "buy": buy, "sell": sell, "date": week_date}

        # ratio が取れなくても古いキャッシュを返す
        return cached
    except Exception:
        return cached


# ── 2. TDnet適時開示 ──────────────────────────────────────────────────────

# イベントタイプ分類キーワード（前方優先＝具体的・信頼度の高い分類を先に。
# 汎用的な ma は最後＝「完全子会社化」等の具体イベントに先取りさせる）
_EVENT_KEYWORDS = {
    "buyback":     ["自己株式取得", "自社株買い", "自己株式の取得"],
    "dividend":    ["配当予想の修正", "増配", "配当金", "特別配当", "記念配当"],
    "upward":      ["業績予想の修正（上方）", "上方修正", "業績上方"],
    "downward":    ["業績予想の修正（下方）", "下方修正", "業績下方"],
    # ⑤ 需給・ガバナンス（ma より具体的なので先に判定）
    "tob":         ["公開買付", "TOB", "ＴＯＢ", "株式公開買付"],
    "parent":      ["完全子会社化", "親子上場", "上場廃止", "非公開化", "MBO"],
    "mgmt":        ["代表取締役の異動", "社長交代", "経営体制", "代表取締役異動"],
    "holding":     ["大量保有", "変更報告書"],
    # ③ 強いカタリスト（事業構造の変化）
    "alliance":    ["業務提携", "資本提携", "資本業務提携", "協業", "戦略的提携"],
    "order":       ["受注", "大型契約", "長期契約", "基本合意"],
    "newbiz":      ["新製品", "新サービス", "新規事業", "市場参入", "事業開始"],
    "restructure": ["構造改革", "事業再編", "事業譲渡", "撤退", "希望退職", "中期経営計画"],
    # 汎用M&A（最後＝catch-all）
    "ma":          ["合併", "買収", "M&A", "子会社化", "株式取得"],
}


def _classify_event(title: str) -> str:
    for etype, keywords in _EVENT_KEYWORDS.items():
        if any(k in title for k in keywords):
            return etype
    return "other"


def get_tdnet_events(code: str, days: int = 45) -> list:
    """直近 days 日以内の適時開示イベントを返す。DBキャッシュ（当日）優先。

    Returns:
        [{'announce_date': str, 'title': str, 'event_type': str}, ...]
    """
    today_str = date.today().isoformat()
    # 当日にすでにフェッチ済みか確認
    from lib.db import init_db, _conn
    init_db()
    with _conn() as con:
        already = con.execute(
            "SELECT COUNT(*) as n FROM tdnet_events WHERE code=? AND fetched_date=?",
            (str(code), today_str)
        ).fetchone()["n"]
    if already > 0:
        return get_tdnet_events_recent(str(code), days)

    events = []
    try:
        # kabutan のIRファイリングページ（TDnetデータを集約）
        resp = requests.get(
            f"https://kabutan.jp/stock/irfiling?code={code}",
            headers=_HEADERS, timeout=8
        )
        if resp.status_code == 200:
            text = resp.text
            # 日付パターン: YYYY/MM/DD と見出しを組み合わせて解析
            rows = re.findall(
                r'(\d{4}/\d{2}/\d{2})[^<]*</td>[^<]*<td[^>]*>[^<]*<a[^>]*>([^<]{5,120})</a>',
                text
            )
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            for date_str, title in rows:
                ann_date = date_str.replace("/", "-")
                if ann_date < cutoff:
                    continue
                etype = _classify_event(title)
                if etype != "other" or any(
                    k in title for k in ["決算", "配当", "株主", "増資", "分割"]
                ):
                    events.append({
                        "announce_date": ann_date,
                        "title": title.strip(),
                        "event_type": etype,
                    })

        # TDnetから直接も試みる
        if not events:
            resp2 = requests.get(
                f"https://www.release.tdnet.info/inbs/I_main_00.html?code={code}",
                headers=_HEADERS, timeout=8
            )
            if resp2.status_code == 200:
                rows2 = re.findall(
                    r'<td[^>]*>(\d{4}/\d{2}/\d{2})</td>[^<]*<td[^>]*>[^<]*</td>[^<]*<td[^>]*><a[^>]*>([^<]+)</a>',
                    resp2.text
                )
                cutoff = (date.today() - timedelta(days=days)).isoformat()
                for date_str, title in rows2[:20]:
                    ann_date = date_str.replace("/", "-")
                    if ann_date < cutoff:
                        continue
                    etype = _classify_event(title)
                    events.append({
                        "announce_date": ann_date,
                        "title": title.strip(),
                        "event_type": etype,
                    })

        if events:
            upsert_tdnet_events(str(code), events)
        else:
            # フェッチ済みマーク（空の場合でも）
            upsert_tdnet_events(str(code), [{"announce_date": today_str,
                                              "title": "_check", "event_type": "other"}])
    except Exception:
        pass

    return get_tdnet_events_recent(str(code), days)


def summarize_tdnet_events(events: list) -> dict:
    """イベントリストから summary dict を作成。
    Returns: {'buyback': bool, 'dividend': bool, 'upward': bool, 'downward': bool, 'ma': bool,
              'days_since_buyback': int|None, 'days_since_upward': int|None}
    """
    result = {t: False for t in ["buyback", "dividend", "upward", "downward", "ma",
                                  "tob", "alliance", "order", "newbiz", "restructure",
                                  "parent", "mgmt", "holding"]}
    result["days_since_buyback"] = None
    result["days_since_upward"]  = None
    today = date.today()
    for ev in events:
        if ev.get("announce_date", "") == today_str_or_placeholder(ev):
            continue
        etype = ev.get("event_type", "other")
        if etype in result:
            result[etype] = True
        try:
            ann = date.fromisoformat(ev["announce_date"])
            days_ago = (today - ann).days
            if etype == "buyback" and (result["days_since_buyback"] is None or days_ago < result["days_since_buyback"]):
                result["days_since_buyback"] = days_ago
            if etype == "upward" and (result["days_since_upward"] is None or days_ago < result["days_since_upward"]):
                result["days_since_upward"] = days_ago
        except (ValueError, KeyError):
            pass
    return result


def today_str_or_placeholder(ev: dict) -> str:
    return "_check"


# ── 3. 空売り残高 (TSE/JPX 週次公開データ) ──────────────────────────────

_SHORT_INDEX_URL = "https://www.jpx.co.jp/markets/statistics-equities/short-selling/"
_SHORT_ATT_BASE  = "https://www.jpx.co.jp"

_short_cache: dict = {}     # {code_str: {'week_date': str, 'short_balance': float, 'short_amount': float}}
_short_cache_date: str = "" # 最終ロード日


def _load_tse_short_data() -> bool:
    """TSE から最新の週次空売り残高Excelをダウンロードして _short_cache を更新。"""
    global _short_cache, _short_cache_date
    today_str = date.today().isoformat()
    if _short_cache_date == today_str and _short_cache:
        return True
    try:
        import pandas as pd
        # インデックスページから最新ファイルのリンクを取得
        resp = requests.get(_SHORT_INDEX_URL, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return False
        # xlsx or xls ファイルへのリンクを探す
        links = re.findall(r'href="([^"]+short[^"]*\.xls[x]?)"', resp.text, re.IGNORECASE)
        if not links:
            links = re.findall(r'href="(/[^"]+tvdivq[^"]+\.xls[x]?)"', resp.text, re.IGNORECASE)
        if not links:
            return False
        file_url = _SHORT_ATT_BASE + links[0] if links[0].startswith("/") else links[0]
        resp2 = requests.get(file_url, headers=_HEADERS, timeout=30)
        if resp2.status_code != 200:
            return False
        df = pd.read_excel(io.BytesIO(resp2.content), dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        # 列名の揺れに対応
        code_col   = next((c for c in df.columns if "コード" in c or "code" in c.lower()), None)
        bal_col    = next((c for c in df.columns if "残高株数" in c or "株数" in c), None)
        amount_col = next((c for c in df.columns if "残高金額" in c or "金額" in c), None)
        if code_col is None or bal_col is None:
            return False
        # ファイル名から週の日付を推定
        week_m = re.search(r'(\d{8})', file_url)
        week_date = (datetime.strptime(week_m.group(1), "%Y%m%d").date().isoformat()
                     if week_m else today_str)
        rows_to_save = []
        _short_cache = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip().zfill(4)
            if not re.match(r'^\d{4}$', code):
                continue
            try:
                bal = float(str(row[bal_col]).replace(",", "").replace("，", ""))
            except (ValueError, TypeError):
                continue
            amt = None
            if amount_col:
                try:
                    amt = float(str(row[amount_col]).replace(",", "").replace("，", ""))
                except (ValueError, TypeError):
                    pass
            _short_cache[code] = {"week_date": week_date, "short_balance": bal, "short_amount": amt}
            rows_to_save.append((code, week_date, bal, amt))
        if rows_to_save:
            bulk_upsert_short_interest(rows_to_save)
        _short_cache_date = today_str
        return bool(_short_cache)
    except Exception:
        return False


def get_short_balance(code: str) -> "dict | None":
    """空売り残高データを返す（単位: 株）。
    Returns: {'week_date': str, 'short_balance': float, 'short_amount': float} or None
    """
    global _short_cache
    code = str(code).zfill(4)
    # メモリキャッシュ確認
    if _short_cache.get(code):
        return _short_cache[code]
    # DBキャッシュ確認
    cached = get_short_interest_latest(code)
    if cached:
        return cached
    # 全件ロード（初回のみ時間かかる）
    if _load_tse_short_data():
        return _short_cache.get(code)
    return None


# ── 4. Google トレンド ───────────────────────────────────────────────────

try:
    from pytrends.request import TrendReq as _TrendReq
    _HAS_PYTRENDS = True
except ImportError:
    _HAS_PYTRENDS = False

_trend_cache: dict = {}  # {query: (score, fetched_date)}


def get_google_trend_score(query: str) -> float:
    """Googleトレンドスコアを返す（-0.5〜+2.0）。
    正の値 = 最近の検索が増加（小売投資家の注目度上昇）
    負の値 = 検索が減少（関心低下）
    0.0    = 変化なし or データなし or pytrends 未インストール

    参考: Da, Engelberg, Gao (2011) — 検索出来高が将来リターンを予測
    """
    global _trend_cache
    if not _HAS_PYTRENDS:
        return 0.0
    today_str = date.today().isoformat()
    cached = _trend_cache.get(query)
    if cached and cached[1] == today_str:
        return cached[0]
    try:
        pt = _TrendReq(hl="ja-JP", tz=540, timeout=(5, 15))
        pt.build_payload([query], cat=0, timeframe="today 3-m", geo="JP", gprop="")
        df = pt.interest_over_time()
        if df.empty or query not in df.columns:
            _trend_cache[query] = (0.0, today_str)
            return 0.0
        vals = df[query].values.astype(float)
        if len(vals) < 8:
            _trend_cache[query] = (0.0, today_str)
            return 0.0
        recent   = float(vals[-4:].mean())
        baseline = float(vals[:-4].mean())
        score = float((recent / (baseline + 1e-5)) - 1.0) if baseline > 0 else 0.0
        score = max(-0.5, min(2.0, score))
        _trend_cache[query] = (score, today_str)
        return score
    except Exception:
        _trend_cache[query] = (0.0, today_str)
        return 0.0


# ── 銘柄ごとのオルタナティブデータまとめ取得 ─────────────────────────

def get_alt_signals(code: str, name: str = "") -> dict:
    """銘柄コードに対するオルタナティブシグナルを一括取得して dict で返す。

    Returns:
        {
          'margin_ratio': float | None,       # 信用倍率
          'margin_buy': float | None,          # 信用買残（株）
          'margin_sell': float | None,         # 信用売残（株）
          'short_balance': float | None,       # 空売り残高（株）
          'tdnet_events': list,                # 適時開示イベント
          'has_buyback': bool,                 # 30日以内に自社株買い発表
          'has_upward': bool,                  # 30日以内に上方修正発表
          'has_downward': bool,                # 30日以内に下方修正発表
          'days_since_buyback': int | None,    # 自社株買い発表からの日数
          'days_since_upward': int | None,     # 上方修正からの日数
          'trend_score': float,                # Googleトレンドスコア
        }
    """
    result = {
        "margin_ratio": None, "margin_buy": None, "margin_sell": None,
        "short_balance": None,
        "tdnet_events": [],
        "has_buyback": False, "has_upward": False, "has_downward": False,
        "days_since_buyback": None, "days_since_upward": None,
        "growth_catalysts": [], "has_growth_catalyst": False,
        "trend_score": 0.0,
    }

    # 信用倍率
    margin = get_margin_data(code)
    if margin:
        result["margin_ratio"] = margin.get("ratio")
        result["margin_buy"]   = margin.get("buy")
        result["margin_sell"]  = margin.get("sell")

    # 空売り残高
    short = get_short_balance(code)
    if short:
        result["short_balance"] = short.get("short_balance")

    # 適時開示
    events = get_tdnet_events(code, days=45)
    if events:
        result["tdnet_events"] = events
        summary = summarize_tdnet_events(events)
        result["has_buyback"]         = summary.get("buyback", False)
        result["has_upward"]          = summary.get("upward", False)
        result["has_downward"]        = summary.get("downward", False)
        result["days_since_buyback"]  = summary.get("days_since_buyback")
        result["days_since_upward"]   = summary.get("days_since_upward")
        # ③⑤ 将来成長カタリスト（GARP定性シグナル・フォワードテスト用）
        # buyback/dividendは資本還元、tob/parentはプレミアム、alliance/order/newbizは事業拡大
        _growth_types = ["buyback", "dividend", "tob", "alliance", "order", "newbiz", "parent"]
        fired = [t for t in _growth_types if summary.get(t)]
        result["growth_catalysts"]     = fired
        result["has_growth_catalyst"]  = len(fired) > 0

    # Googleトレンド（銘柄名で検索）
    if name:
        result["trend_score"] = get_google_trend_score(name)

    return result

"""
lib/alt_data.py
オルタナティブデータ取得モジュール（全て無料・公開データ）

1. Googleトレンド       (pytrends — 要pip install pytrends)

設計方針:
- DBキャッシュ優先（当日1回のみ外部アクセス）
- 失敗時は常に None / [] / 0.0 を返す（例外は伝播しない）
- rank_stocks.py の後処理で使用。モデル特徴量は次回再学習で追加予定。
"""
from datetime import date


# ── 1. Google トレンド ───────────────────────────────────────────────────

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
          'trend_score': float,                # Googleトレンドスコア
        }
    """
    result = {
        "trend_score": 0.0,
    }

    # Googleトレンド（銘柄名で検索）
    if name:
        result["trend_score"] = get_google_trend_score(name)

    return result

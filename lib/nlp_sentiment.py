"""
lib/nlp_sentiment.py
kabutan ニュース見出しを Claude Haiku で感情分析し、決算センチメントスコアを返す。

score: -1.0 (強い悲観) → 0.0 (中立) → +1.0 (強い楽観)
キャッシュ: 当日1回のみ（毎日更新）
"""
import re
import os
import requests
from datetime import datetime

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "ja,en;q=0.9",
}


def _scrape_kabutan_headlines(code: str, n: int = 5) -> list:
    """kabutan ニュースページから最新 n 件の見出しを取得する。"""
    try:
        resp = requests.get(
            f"https://kabutan.jp/news/?code={code}",
            headers=_HEADERS, timeout=8
        )
        if resp.status_code != 200:
            return []
        text = resp.text
        # kabutan ニュース見出し: <a ...>TEXT</a> パターン
        titles = re.findall(r'class="[^"]*news[^"]*"[^>]*>\s*<a[^>]*>([^<]{5,120})</a>', text)
        if not titles:
            # フォールバック: h2/h3 タグの本文
            titles = re.findall(r'<(?:h2|h3)[^>]*>([^<]{10,120})</(?:h2|h3)>', text[:8000])
        # 決算・業績・増配・自社株買いなど投資判断に有用なキーワードを含む見出しを優先
        keywords = ["決算", "業績", "増益", "減益", "黒字", "赤字", "増配", "減配",
                    "自社株買い", "上方修正", "下方修正", "純利益", "営業利益", "売上"]
        keyword_titles = [t for t in titles if any(k in t for k in keywords)]
        other_titles = [t for t in titles if t not in keyword_titles]
        merged = keyword_titles + other_titles
        return [t.strip() for t in merged[:n] if t.strip()]
    except Exception:
        return []


def _score_with_claude(headlines: list, code: str) -> float:
    """Claude Haiku でニュース見出しの感情スコアを判定 (-1.0〜+1.0)。"""
    import anthropic
    if not headlines:
        return 0.0
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return 0.0
    client = anthropic.Anthropic(api_key=api_key)
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""以下は日本株（コード: {code}）の最新ニュース見出しです。
投資家の視点から、この株の業績・決算・財務状況に対するセンチメントを評価してください。

見出し:
{headlines_text}

回答: -1.0（強い悲観）から+1.0（強い楽観）の間の数値を1つだけ答えてください。
例: 0.3"""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        m = re.search(r'-?\d+\.?\d*', text)
        if m:
            return max(-1.0, min(1.0, float(m.group())))
    except Exception:
        pass
    return 0.0


def get_earnings_sentiment(code: str) -> float:
    """決算センチメントスコアを返す（-1〜+1）。
    当日キャッシュがあれば即返却、なければkabutan+Claude Haikuで分析してキャッシュ。
    APIキーなし / スクレイプ失敗時は 0.0（中立）を返す。
    """
    from lib.db import get_earnings_sentiment as _get, set_earnings_sentiment as _set
    today_str = datetime.now().strftime("%Y-%m-%d")
    cached = _get(str(code), today_str)
    if cached is not None:
        return cached
    headlines = _scrape_kabutan_headlines(str(code))
    score = _score_with_claude(headlines, str(code)) if headlines else 0.0
    _set(str(code), today_str, score)
    return score

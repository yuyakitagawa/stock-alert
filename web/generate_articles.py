"""
S買いシグナルが出た銘柄のSEO記事を生成して Supabase articles テーブルへ保存する。
send_user_alerts.py の後に実行する（Step 7）。
"""
import os, sys, json, sqlite3, requests
from datetime import date
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

SUPABASE_URL        = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, ANTHROPIC_API_KEY]):
    print("[generate_articles] 環境変数未設定。スキップ。")
    sys.exit(0)

import anthropic
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

from lib.db import DB_PATH
TODAY = date.today().isoformat()


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def get_buy_signals():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT code, name, net, rise_prob, drop_prob, vol
        FROM daily_ranking
        WHERE date = ? AND recommend LIKE '%S買い%'
        ORDER BY net DESC
    """, (TODAY,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"code": r[0], "name": r[1], "net": r[2],
         "rise_prob": r[3], "drop_prob": r[4], "vol": r[5]}
        for r in rows
    ]


def article_exists(slug: str) -> bool:
    url  = f"{SUPABASE_URL}/rest/v1/articles?slug=eq.{slug}&limit=1"
    resp = requests.get(url, headers=_sb_headers(), timeout=10)
    return resp.ok and len(resp.json()) > 0


def generate_article(stock: dict) -> str | None:
    code      = stock["code"]
    name      = stock["name"]
    net       = stock["net"]
    rise_prob = stock["rise_prob"]
    drop_prob = stock["drop_prob"]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
            messages=[{"role": "user", "content": f"""
{name}（証券コード: {code}）の個人投資家向けSEO記事を日本語で書いてください。

【モデルシグナル（{TODAY}）】
- ネットスコア: {net:.1f}%（上昇確率 − 下落確率）
- 上昇確率: {rise_prob:.1f}%、下落確率: {drop_prob:.1f}%
- 63日先予測モデルでS買いシグナル発令

{name} の最新ニュース・株価動向をweb検索してから、以下の構成でMarkdown記事を書いてください。
SEOを意識して、見出しにキーワードを入れてください。

# {name}（{code}）S買いシグナル発令 — {TODAY}

## この銘柄が注目される理由
（リード文: 2〜3文で端的に）

## 最新の株価・ニュース動向
（web検索結果をもとに200〜300文字）

## AIが強気シグナルを出した根拠
（ネットスコアや予測確率の意味を投資家向けにわかりやすく）

## チェックすべきリスク
（注意点を箇条書きで2〜3点）

---
*本記事はAIが自動生成したものです。投資は自己責任でお願いします。*
"""}],
        )
        texts = [b.text for b in response.content if hasattr(b, 'text') and b.text]
        return '\n'.join(texts) if texts else None
    except Exception as e:
        print(f"  [generate_articles] 記事生成エラー: {e}")
        return None


def save_article(stock: dict, body: str) -> None:
    slug  = f"{stock['code']}-{TODAY}"
    title = f"{stock['name']}（{stock['code']}）S買いシグナル発令 — {TODAY}"
    row   = {
        "slug":        slug,
        "code":        stock["code"],
        "name":        stock["name"],
        "title":       title,
        "body":        body,
        "signal_date": TODAY,
        "net_score":   stock["net"],
        "published_at": f"{TODAY}T09:00:00+09:00",
    }
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/articles",
        headers=_sb_headers(), json=[row], timeout=30,
    )
    if resp.ok:
        print(f"  保存完了: {slug}")
    else:
        print(f"  保存失敗: {resp.status_code} {resp.text[:200]}")


def main():
    signals = get_buy_signals()
    if not signals:
        print(f"[generate_articles] {TODAY}: S買いシグナルなし")
        return

    print(f"[generate_articles] {len(signals)}銘柄のS買いシグナルを処理")
    for stock in signals:
        slug = f"{stock['code']}-{TODAY}"
        if article_exists(slug):
            print(f"  スキップ（既存）: {slug}")
            continue
        print(f"  生成中: {stock['name']}（{stock['code']}）")
        body = generate_article(stock)
        if body:
            save_article(stock, body)


if __name__ == "__main__":
    main()

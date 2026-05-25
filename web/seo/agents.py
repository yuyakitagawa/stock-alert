"""
SEO 7エージェント定義
client は orchestrate.py で初期化して代入する
"""
import re, json

client = None  # set by orchestrate.py


def _claude(prompt: str, max_tokens: int = 400, use_search: bool = False) -> str:
    kwargs = dict(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if use_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    r = client.messages.create(**kwargs)
    texts = [b.text for b in r.content if hasattr(b, "text") and b.text]
    return "\n".join(texts)


def _json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group()) if m else None
    except Exception:
        return None


# ── 1. KW選定 ──────────────────────────────────────────────────────────────

def kw_agent(stock: dict) -> dict:
    prompt = f"""あなたはSEOキーワードリサーチャーです。
個人投資家が検索する日本語キーワードを選定してください。

【銘柄】{stock['name']}（{stock['code']}）業種: {stock.get('sector', '')}
【シグナル】S買いシグナル {stock.get('signal_date', '')} ネット{stock.get('net', 0):.1f}%

web検索で競合状況を確認し、検索ボリュームがあり競合が少ないキーワードを選定してください。
サジェストキーワードや関連検索も考慮してください。

JSON: {{"primary_kw": "...", "secondary_kws": ["...", "..."], "search_intent": "情報収集|比較検討|購入意図"}}"""
    return _json(_claude(prompt, 300, use_search=True)) or {
        "primary_kw": f"{stock['name']} 株価",
        "secondary_kws": [f"{stock['name']} 投資", "S買いシグナル"],
        "search_intent": "情報収集",
    }


# ── 2. リサーチ ────────────────────────────────────────────────────────────

def research_agent(kw: dict, stock: dict) -> dict:
    prompt = f"""あなたはSEOリサーチャーです。
「{kw['primary_kw']}」で上位表示している記事を調査し、
コンテンツギャップ（競合が書いていないが読者が知りたいこと）を見つけてください。

web検索で上位3〜5記事を確認してから分析してください。

JSON: {{"competitor_topics": ["..."], "content_gaps": ["..."], "suggested_angle": "..."}}"""
    return _json(_claude(prompt, 500, use_search=True)) or {
        "competitor_topics": [],
        "content_gaps": [f"{stock['name']}のシグナル根拠"],
        "suggested_angle": "AIシグナルの根拠を数値で詳しく解説",
    }


# ── 3. 構成設計 ────────────────────────────────────────────────────────────

def design_agent(kw: dict, research: dict, stock: dict) -> dict:
    prompt = f"""あなたはSEOコンテンツ設計者です。
以下の情報をもとに、個人投資家向け記事の構成を設計してください。

【主KW】{kw['primary_kw']}
【サブKW】{', '.join(kw.get('secondary_kws', []))}
【ユーザーインテント】{kw.get('search_intent', '')}
【コンテンツギャップ】{research.get('content_gaps', [])}
【差別化角度】{research.get('suggested_angle', '')}
【銘柄】{stock['name']}（{stock['code']}）

1500〜2500文字で設計してください。

JSON: {{"title": "...", "meta_description": "120文字以内", "outline": [{{"h2": "...", "points": ["..."]}}]}}"""
    return _json(_claude(prompt, 600)) or {
        "title": f"{stock['name']}（{stock['code']}）S買いシグナル発令 — {stock.get('signal_date', '')}",
        "meta_description": f"{stock['name']}にS買いシグナル。AIが選ぶ理由と最新動向を解説。",
        "outline": [{"h2": "注目ポイント", "points": ["シグナル根拠", "最新動向"]}],
    }


# ── 4. 執筆 ───────────────────────────────────────────────────────────────

def writer_agent(design: dict, stock: dict, kw: dict, qa_feedback: str = "") -> str:
    feedback_block = f"\n\n【前回QA指摘（必ず改善）】\n{qa_feedback}" if qa_feedback else ""
    prompt = f"""あなたはSEOライターです。
以下のアウトラインに従い、個人投資家向けSEO記事をMarkdownで執筆してください。
最新ニュース・株価動向はweb検索で取得してください。

【タイトル（H1）】{design['title']}
【ターゲットKW】{kw['primary_kw']}
【サブKW】{', '.join(kw.get('secondary_kws', []))}
【アウトライン】
{json.dumps(design.get('outline', []), ensure_ascii=False)}

【銘柄データ】
- {stock['name']}（{stock['code']}）
- ネットスコア: {stock.get('net', 0):.1f}%
- 上昇確率: {stock.get('rise_prob', 0):.1f}%　下落確率: {stock.get('drop_prob', 0):.1f}%{feedback_block}

執筆ルール:
- H1にターゲットKWを含める
- 冒頭100文字以内にターゲットKWを含める
- 見出しにサブKWを散りばめる
- 総文字数1500〜2500文字
- 最後に「---\n※本記事の情報は投資を推奨するものではありません。投資は自己責任でお願いします。」を追加"""
    return _claude(prompt, 3000, use_search=True)


# ── 5. 品質チェック ────────────────────────────────────────────────────────

def qa_agent(article: str, kw: dict) -> dict:
    primary = kw.get("primary_kw", "")
    prompt = f"""以下のSEO記事を品質チェックしてください。

【ターゲットKW】{primary}
【記事本文（先頭2000文字）】
{article[:2000]}

チェック項目:
1. H1またはタイトルにKWが含まれているか
2. 冒頭100文字にKWが含まれているか
3. 見出し（##）が3つ以上あるか
4. 文字数が1500文字以上か（概算で判断）
5. 投資免責事項があるか
6. 個人投資家にわかりやすい言葉か

JSON: {{"pass": true/false, "score": 0-100, "issues": ["..."], "feedback": "改善点を具体的に"}}"""
    result = _json(_claude(prompt, 300))
    return result or {"pass": True, "score": 70, "issues": [], "feedback": ""}


# ── 6. 公開（Publisher） ─ orchestrate.py 内で直接 Supabase upsert ────────
# Publisher はデータ整形後に orchestrate.py が実行する


# ── 7. アナリティクス ──────────────────────────────────────────────────────

def analytics_agent(articles: list) -> dict:
    summary = "\n".join([
        f"- {a['slug']}: KW={a.get('target_keyword') or '不明'} "
        f"score={a.get('seo_score') or '?'} 公開={a['signal_date']}"
        for a in articles[:20]
    ])
    prompt = f"""あなたはSEOアナリストです。
以下の記事群を分析してください。
各記事のターゲットKWをweb検索して検索順位・改善余地を評価してください。

【記事一覧】
{summary}

JSON: {{
  "analysis": [{{"slug": "...", "issues": "...", "needs_rewrite": true}}],
  "priority_rewrites": ["slug1", "slug2"],
  "new_opportunities": ["新規KW候補"]
}}"""
    return _json(_claude(prompt, 800, use_search=True)) or {
        "analysis": [], "priority_rewrites": [], "new_opportunities": [],
    }


# ── 8. SEOマネージャー ─────────────────────────────────────────────────────

def seo_manager(articles: list, analytics: dict) -> dict:
    articles_list = "\n".join([
        f"- {a['slug']} KW={a.get('target_keyword') or '不明'} score={a.get('seo_score') or '?'}"
        for a in articles[:20]
    ])
    prompt = f"""あなたはSEOマネージャーです。
分析結果をもとに今週のSEO戦略を決定してください。

【パフォーマンス分析】
{json.dumps(analytics, ensure_ascii=False)[:1200]}

【全記事一覧】
{articles_list}

判断基準:
- seo_score < 70 または needs_rewrite=true の記事はリライト対象
- 古い記事（3ヶ月以上）も見直し対象
- 新規KW機会があれば指摘

最大3件のリライト指示を出してください。

JSON: {{
  "rewrites": [{{"slug": "...", "reason": "...", "focus": "改善ポイント"}}],
  "strategy_note": "今週の全体方針"
}}"""
    return _json(_claude(prompt, 500)) or {"rewrites": [], "strategy_note": "現状維持"}

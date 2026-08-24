"""
lib/writing_style.py

自動生成する文章（ブログ記事・動画ナレーション・英訳）が「AIが書いた」と読者に
見抜かれないための共通文体ルール。プロンプトに埋め込む指示文と、生成後に
AI常套句・単調な文末を検出するチェッカーを1箇所に集約する。

読者に機械っぽさを感じさせる主因は (1)決まり文句（「注目が集まっています」
「〜と言えるでしょう」等）、(2)同じ文末の連続による単調な律動、(3)接続詞
（「また、」「さらに、」）で文をつなぐ癖、の3つ。指示だけでは守られないことが
あるため（本文字数指示が守られなかった実績: publish_blog_articles.MIN_BODY_CHARS
のコメント参照）、find_ai_tells() で生成後に実測し、検出時は呼び出し側が再生成する。
"""
import re

_TAG_RE = re.compile(r"<[^>]+>")

# ブログ記事（です・ます調の解説記事）向け。プロンプトにそのまま埋め込む。
JA_STYLE_RULES = """文体の規則（機械的な文章のパターンを避ける）:
- 次の言い回しは使わない: 「注目が集まっています」「注目されています」「注目に値します」\
「〜と言えるでしょう」「〜が示唆されます」「〜が期待されます」「重要なポイントです」\
「大きな意味を持ちます」「目が離せません」「見逃せません」「非常に」「まさに」「〜ではないでしょうか」
- 全体をです・ます調で統一する。である調（「〜している。」「〜だ。」）と混ぜない
（本文が長くなると混在しやすい。文末の変化はです・ます調の中だけでつける）。
- 同じ文末を3文以上続けない（「〜です。〜です。〜です。」を避け、「〜でした」「〜しています」\
「〜になります」「〜が分かります」などを織り交ぜて文末の律動を変える）。
- 短い文と長い文を混ぜる。すべての文を同じ長さ・同じ構造にしない。
- 「また、」「さらに、」「一方、」「なお、」で始まる文は全体で2回まで。接続詞ではなく事実の順序で文をつなぐ。
- 段落の締めは抽象的な感想ではなく、具体的な数字か事実にする。"""

# 英語本文（bodyEn・英訳）向け。
EN_STYLE_RULES = """English style rules (avoid patterns that read as machine-written):
- Never use: "It is worth noting", "notably", "underscores", "highlights", "signals a", \
"landscape", "testament to", "pivotal", "poised to", "delve", "furthermore", "moreover".
- No em dashes.
- Vary sentence length and structure. Do not open consecutive sentences with the same word."""

# 動画ナレーション（話し言葉）向け。
NARRATION_STYLE_RULES = """- 「注目です」「要チェック」「〜と言えるでしょう」「気になりますね」\
などの決まり文句は使わない。事実と数字だけで話を進める。
- 全シーンで同じ文末（「〜です。」等）を繰り返さない。「〜でした。」「〜しています。」を織り交ぜる。"""

# 生成後の検出用。プロンプトで禁止しても混入することがある常套句（部分一致）。
# 「今後の動向」は各プロンプトが個別に禁止している定型結びの中核語なのでここでも拾う。
JA_AI_TELL_PHRASES = [
    "注目が集まって", "注目されてい", "注目を集めて", "注目に値し",
    "と言えるでしょう", "と言えそうです",
    "が示唆され", "が期待され",
    "重要なポイント", "大きな意味を持", "重要な意味を持",
    "今後の動向", "目が離せません", "見逃せません",
    "いかがでしたか", "ではないでしょうか",
    "非常に", "まさに",
]

# 同じ文末（句点前の2文字）が何連続したら「単調」とみなすか
MONOTONE_ENDING_RUN = 4


def find_ai_tells(text: str) -> list:
    """AI生成文の常套句・単調な文末を検出し、該当項目のリストを返す（無ければ空リスト）。
    HTML本文はタグを除いた可視テキストで判定する。"""
    plain = _TAG_RE.sub("", text or "")
    hits = [p for p in JA_AI_TELL_PHRASES if p in plain]
    endings = [s.strip()[-2:] for s in plain.split("。") if len(s.strip()) >= 2]
    run = 1
    for prev, cur in zip(endings, endings[1:]):
        run = run + 1 if cur == prev else 1
        if run >= MONOTONE_ENDING_RUN:
            hits.append(f"文末単調（「{cur}。」が{MONOTONE_ENDING_RUN}連続）")
            break
    return hits

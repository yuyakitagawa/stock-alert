"""
lib/article_text.py
既存ブログ記事の本文を扱う共通ヘルパー。

tools/export_article_fact_cards.py（事実カードの書き出し）と
tools/apply_rewritten_articles.py（書き直した本文の反映）が共有する。
元は tools/rewrite_thin_blog_articles.py に置いていたが、Anthropic APIに
本文を書かせる同ツールを廃止した（2026-08-29）ため、APIを使わない側だけが
残るようここへ移した。
"""
import re

import lib.supabase_client as sb

# 本文目標(1,300〜1,700字)を大きく下回る水準を「薄い」とみなす閾値。
# HTMLタグと株価チャートの<figure>を除いた可視文字数で判定する。
THIN_TEXT_THRESHOLD = 1000

FIGURE_RE = re.compile(r"<figure>.*?</figure>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def visible_text_len(body_html: str) -> int:
    text = FIGURE_RE.sub("", body_html or "")
    text = _TAG_RE.sub("", text)
    return len(text)


def find_filer_names(code: str, disc_date: str) -> list:
    """記事の銘柄コード＋開示日から、その日の提出者候補をEDINET開示から逆引きする。"""
    return sb.select(
        "edinet_large_holdings",
        f"issuer_code=eq.{code}&disc_date=eq.{disc_date}"
        "&select=filer_name,doc_type_code,doc_description,holding_ratio",
    )

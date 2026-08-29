"""下落モデル言及の削除（tools/strip_drop_model_mentions）のユニットテスト。
ネットワークには触れず、本文の書き換えロジックだけを検証する。

実行: python3 tests/test_strip_drop_model_mentions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.strip_drop_model_mentions import (
    JA_PATTERN, is_subset, rewrite_ja_sentence, split_ja,
    strip_html_block, text_of,
)


def test_keeps_close_price_when_sentence_starts_with_it():
    """株価は開示原本と突き合わせられる事実なので、モデルの節だけ落として株価は残す。"""
    s = "報告書時点での株価は70円と、弊社モデルでは中程度の下落リスク水準に位置する局面での取得となった。"
    assert rewrite_ja_sentence(s, polite=False) == "報告書時点での株価は70円だった。"
    assert rewrite_ja_sentence(s, polite=True) == "報告書時点での株価は70円でした。"


def test_deletes_sentence_when_price_cannot_be_preserved():
    """株価が文頭に無い場合は書き足さずに文ごと消す。"""
    assert rewrite_ja_sentence("弊社モデルによる下落リスク水準は低いという評価でした。", polite=True) is None
    assert rewrite_ja_sentence("当該銘柄の下落リスク水準はやや低い局面での取得となっています。", polite=True) is None


def test_does_not_keep_lead_that_still_mentions_the_model():
    """残そうとしている前半にモデルの話が混ざっている場合は残さない。"""
    s = "弊社モデルで評価した株価は700円で、下落リスク水準は中程度だった。"
    assert rewrite_ja_sentence(s, polite=False) is None


def test_strips_only_the_matching_sentence_from_paragraph():
    html = ("<p>A社が5%を取得した。弊社モデルによる下落リスク水準は低いという評価でした。"
            "推定取得金額は12億円だった。</p>")
    out, removed, skipped = strip_html_block(html, JA_PATTERN, split_ja, rewrite_ja_sentence)
    assert removed == 1 and skipped == 0
    assert "下落リスク水準" not in out
    assert "A社が5%を取得した。" in out and "推定取得金額は12億円だった。" in out


def test_drops_paragraph_that_becomes_empty():
    html = "<p>本文。</p><p>弊社モデルによる下落リスク水準は低いという評価でした。</p>"
    out, removed, _ = strip_html_block(html, JA_PATTERN, split_ja, rewrite_ja_sentence)
    assert removed == 1
    assert out == "<p>本文。</p>"


def test_leaves_sentences_containing_tags_untouched():
    """タグを跨ぐ文を削るとタグの対応が壊れるため触らず、件数だけ報告する。"""
    html = "<p>前段。<strong>弊社モデルの下落リスク水準は中程度</strong>だった。</p>"
    out, removed, skipped = strip_html_block(html, JA_PATTERN, split_ja, rewrite_ja_sentence)
    assert (removed, skipped) == (0, 1)
    assert out == html


def test_text_of_separates_paragraphs():
    """タグを空文字で潰すと段落の末尾と次の段落の先頭が繋がり、元の本文に無い文が
    でっち上がって部分集合の検証が誤判定する。"""
    assert text_of("<p>前段。</p><p>*Speculation: X.</p>") == "前段。 *Speculation: X."


def test_is_subset_rejects_invented_text():
    original = "<p>A。B。</p>"
    assert is_subset(original, "<p>A。</p>", split_ja)
    assert not is_subset(original, "<p>A。まったく新しい文。</p>", split_ja)


if __name__ == "__main__":
    test_keeps_close_price_when_sentence_starts_with_it()
    test_deletes_sentence_when_price_cannot_be_preserved()
    test_does_not_keep_lead_that_still_mentions_the_model()
    test_strips_only_the_matching_sentence_from_paragraph()
    test_drops_paragraph_that_becomes_empty()
    test_leaves_sentences_containing_tags_untouched()
    test_text_of_separates_paragraphs()
    test_is_subset_rejects_invented_text()
    print("全テスト成功 (10件)")

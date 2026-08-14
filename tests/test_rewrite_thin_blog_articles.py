"""tools/rewrite_thin_blog_articles.py のロジックのユニットテスト。

実行: python3 tests/test_rewrite_thin_blog_articles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rewrite_thin_blog_articles import visible_text_len, build_patch_payload


def test_visible_text_len_strips_tags():
    assert visible_text_len("<p>あいう</p><p>えお</p>") == 5


def test_visible_text_len_excludes_figure_block():
    """本文末尾の株価チャート<figure>は文字数としてカウントしない
    （altテキストが本文の文字数に混入して閾値判定を狂わせないようにするため）。"""
    body = '<p>あいう</p><figure><img src="x" alt="銘柄名 株価推移" width="1000" height="500"></figure>'
    assert visible_text_len(body) == 3


def test_visible_text_len_empty_body_is_zero():
    assert visible_text_len("") == 0
    assert visible_text_len(None) == 0


def test_build_patch_payload_contains_only_body():
    """PATCHは差分更新のため、他のフィールドを送らず本文だけを渡す。"""
    payload = build_patch_payload("<p>新しい本文</p>")
    assert payload == {"body": "<p>新しい本文</p>"}


if __name__ == "__main__":
    test_visible_text_len_strips_tags()
    test_visible_text_len_excludes_figure_block()
    test_visible_text_len_empty_body_is_zero()
    test_build_patch_payload_contains_only_body()
    print("OK: test_rewrite_thin_blog_articles (4 tests)")

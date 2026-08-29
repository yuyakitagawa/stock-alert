"""既存記事の本文ヘルパー（lib/article_text.py）のユニットテスト。

実行: python3 -m pytest tests/test_article_text.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.article_text import THIN_TEXT_THRESHOLD, restore_figures, visible_text_len

CHART = '<figure><img src="c.png" alt="A社（1234）株価推移（直近3ヶ月）"><figcaption>A社（1234）の株価推移（直近3ヶ月・終値ベース）</figcaption></figure>'
HISTORY = '<figure><img src="h.png" alt="Xによる保有比率の推移"><figcaption>保有比率推移</figcaption></figure>'
HOLDERS = '<figure><img src="s.png" alt="株主の保有比率比較"><figcaption>株主の保有比率</figcaption></figure>'


def test_visible_text_len_excludes_tags_and_figures():
    body = "<p>あいうえお</p>" + CHART + "<p>かきく</p>"
    assert visible_text_len(body) == 8


def test_visible_text_len_empty():
    assert visible_text_len("") == 0
    assert visible_text_len(None) == 0


def test_thin_threshold_is_below_body_target():
    """薄いと判定する閾値は本文目標（1,300〜1,700字）より下にある。"""
    assert THIN_TEXT_THRESHOLD < 1300


def test_restore_figures_keeps_chart_last_and_others_inline():
    """解説図は本文中に戻し、株価チャートだけを末尾に置く。"""
    old = "<p>旧1</p>" + HISTORY + "<p>旧2</p>" + HOLDERS + "<p>旧3</p>" + CHART
    new = "<p>新1</p><p>新2</p><p>新3</p><p>新4</p>"
    body = restore_figures(new, old)
    assert body.endswith(CHART)
    assert HISTORY in body and HOLDERS in body
    # 解説図は末尾のチャートより前（＝本文中）に入る
    assert body.index(HISTORY) < body.index(CHART)
    assert body.index(HOLDERS) < body.index(CHART)


def test_restore_figures_without_figures_returns_new_body():
    assert restore_figures("<p>新</p>", "<p>旧</p>") == "<p>新</p>"


def test_restore_figures_keeps_all_figures_of_a_three_figure_article():
    """図が3枚ある記事で1枚も落とさない（旧実装は最初の1枚だけを末尾に付けていた）。"""
    old = "<p>旧1</p>" + HISTORY + "<p>旧2</p>" + HOLDERS + "<p>旧3</p>" + CHART
    body = restore_figures("<p>新1</p><p>新2</p><p>新3</p><p>新4</p>", old)
    assert body.count("<figure>") == 3

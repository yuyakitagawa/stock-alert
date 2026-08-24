"""AI常套句・単調文末の検出（lib/writing_style.py）のユニットテスト。

実行: python3 tests/test_writing_style.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.writing_style import (
    EN_STYLE_RULES,
    JA_STYLE_RULES,
    NARRATION_STYLE_RULES,
    find_ai_tells,
)


def test_find_ai_tells_detects_stock_phrases():
    text = "<p>今回の取得には注目が集まっています。非常に大きな動きと言えるでしょう。</p>"
    hits = find_ai_tells(text)
    assert "注目が集まって" in hits
    assert "非常に" in hits
    assert "と言えるでしょう" in hits


def test_find_ai_tells_clean_article_returns_empty():
    text = (
        "<p>エフィッシモが任天堂の株式を5.12%取得しました。取得額は約3,400億円です。"
        "同社の開示は今回で12件目でした。任天堂の時価総額は約9兆円で、"
        "5%の保有は大株主上位に入る規模になります。</p>"
    )
    assert find_ai_tells(text) == []


def test_find_ai_tells_detects_monotone_endings():
    # 同じ文末（「ます。」）が4連続 → 単調として検出
    text = "株式を取得します。比率が上がります。開示が続きます。規模も増えます。"
    hits = find_ai_tells(text)
    assert any("文末単調" in h for h in hits)


def test_find_ai_tells_allows_varied_endings():
    text = "株式を取得しました。比率は5.1%です。開示は12件目になります。規模も大きい水準でした。"
    assert find_ai_tells(text) == []


def test_find_ai_tells_strips_html_before_matching():
    # タグ属性はテキストではないので判定対象にしない
    text = '<p class="非常に-wide">保有比率は5.1%でした。</p>'
    assert find_ai_tells(text) == []


def test_find_ai_tells_empty_input():
    assert find_ai_tells("") == []
    assert find_ai_tells(None) == []


def test_style_rules_are_nonempty_prompt_blocks():
    for block in (JA_STYLE_RULES, EN_STYLE_RULES, NARRATION_STYLE_RULES):
        assert isinstance(block, str) and len(block) > 50
    # プロンプトに埋め込むため、f-string を壊す波括弧を含まないこと
    for block in (JA_STYLE_RULES, EN_STYLE_RULES, NARRATION_STYLE_RULES):
        assert "{" not in block and "}" not in block


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

"""EDINET大量保有スキャナー（tools/scan_large_holdings）の判定ロジックのユニットテスト。

実行: python3 tests/test_scan_large_holdings.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.scan_large_holdings import is_sell_disclosure, is_individual_filer, is_noise_match


def test_sell_keywords_detected():
    assert is_sell_disclosure("変更報告書（短期大量譲渡）")
    assert is_sell_disclosure("大量保有報告書の一部売却について")
    assert not is_sell_disclosure("変更報告書")


def test_individual_filer_with_fullwidth_space():
    """実データの提出者名は姓名間に全角スペースが入る（EDINET標準フォーマット）。"""
    assert is_individual_filer("清原　達郎")
    assert is_individual_filer("大江　繭子")


def test_institution_filer_not_individual():
    assert not is_individual_filer("シンプレクス・アセット・マネジメント株式会社")
    assert not is_individual_filer("グロースパートナーズ株式会社")


def test_fullwidth_latin_institution_name_normalized():
    """全角英数のファンド名もNFKC正規化して法人キーワードにマッチする。"""
    assert not is_individual_filer("ＳＨ　Ｉｎｖｅｓｔｍｅｎｔ，　Ｉｎｃ")


def test_empty_filer_not_individual():
    assert not is_individual_filer("")
    assert not is_individual_filer(None)


def test_noise_match_still_detects_sell_and_self_filing():
    assert is_noise_match("清原達郎", "ＡＣＣＥＳＳ", "変更報告書（短期大量譲渡）") == "sell"
    assert is_noise_match("三菱商事株式会社", "三菱商事", "大量保有報告書") == "self_filing"
    assert is_noise_match("○○ファンド", "三菱商事", "大量保有報告書") is None


def test_noise_match_detects_majority_holding():
    """51%以上はスクイーズアウト対象になりうる水準で上値が見込めないため除外する。"""
    assert is_noise_match("メディパルホールディングス株式会社", "ＰＡＬＴＡＣ", "変更報告書", 97.79) == "majority"
    assert is_noise_match("ワールドホールディングス株式会社", "ｎｍｓホールディングス", "変更報告書", 83.96) == "majority"
    assert is_noise_match("○○ファンド", "三菱商事", "大量保有報告書", 42.12) is None
    assert is_noise_match("○○ファンド", "三菱商事", "大量保有報告書", 50.9) is None


if __name__ == "__main__":
    test_sell_keywords_detected()
    test_individual_filer_with_fullwidth_space()
    test_institution_filer_not_individual()
    test_fullwidth_latin_institution_name_normalized()
    test_empty_filer_not_individual()
    test_noise_match_still_detects_sell_and_self_filing()
    test_noise_match_detects_majority_holding()
    print("OK: test_scan_large_holdings (7 tests)")

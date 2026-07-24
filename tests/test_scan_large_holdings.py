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


def test_sell_detected_by_ratio_decrease_even_without_keyword():
    """概要が「変更報告書」とだけ書かれ売買方向のキーワードが無くても、
    直前保有割合との比較で保有比率が減っていれば売りと判定する
    （実例: 東芝がキオクシアHD株を一部売却し16.10%→15.10%になった開示は
    概要に「譲渡」等の語が無く、従来ロジックでは📈買いに誤表示されていた）。"""
    assert is_sell_disclosure("変更報告書", 15.10, 16.10) is True
    assert is_sell_disclosure("変更報告書", 14.44, 16.93) is True
    assert is_sell_disclosure("変更報告書", 16.10, 15.10) is False
    # 片方でも欠けていればテキストのキーワードにフォールバック
    assert is_sell_disclosure("変更報告書", 15.10, None) is False
    assert is_sell_disclosure("変更報告書（一部譲渡）", 15.10, None) is True


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


def test_noise_match_detects_sell_by_ratio_decrease():
    assert is_noise_match("株式会社東芝", "キオクシアホールディングス", "変更報告書", 15.10, 16.10) == "sell"
    assert is_noise_match("○○ファンド", "三菱商事", "変更報告書", 20.0, 15.0) is None


def test_noise_match_detects_majority_holding():
    """51%以上はスクイーズアウト対象になりうる水準で上値が見込めないため除外する。"""
    assert is_noise_match("メディパルホールディングス株式会社", "ＰＡＬＴＡＣ", "変更報告書", 97.79) == "majority"
    assert is_noise_match("ワールドホールディングス株式会社", "ｎｍｓホールディングス", "変更報告書", 83.96) == "majority"
    assert is_noise_match("○○ファンド", "三菱商事", "大量保有報告書", 42.12) is None
    assert is_noise_match("○○ファンド", "三菱商事", "大量保有報告書", 50.9) is None


if __name__ == "__main__":
    test_sell_keywords_detected()
    test_sell_detected_by_ratio_decrease_even_without_keyword()
    test_individual_filer_with_fullwidth_space()
    test_institution_filer_not_individual()
    test_fullwidth_latin_institution_name_normalized()
    test_empty_filer_not_individual()
    test_noise_match_still_detects_sell_and_self_filing()
    test_noise_match_detects_majority_holding()
    test_noise_match_detects_sell_by_ratio_decrease()
    print("OK: test_scan_large_holdings (9 tests)")

"""tools/backfill_article_filer_name.py のユニットテスト。
ネットワーク（microCMS/Supabase）は呼ばず、提出者の突合ロジックのみ検証する。

実行: python3 tests/test_backfill_article_filer_name.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.backfill_article_filer_name import normalize, pick_filer


def test_normalize_drops_legal_suffix_and_width():
    # 「株式会社ストラテジックキャピタル」と記事タイトルの「ストラテジックキャピタル」を揃える
    assert normalize("株式会社ストラテジックキャピタル") == normalize("ストラテジックキャピタル")
    # 全角英数・全角スペース・中黒はNFKC正規化と記号除去で吸収する
    assert normalize("Ｅｖｏ　Ｆｕｎｄ") == "evofund"
    assert normalize("ひびき・パース・アドバイザーズ") == "ひびきパースアドバイザーズ"


def test_pick_filer_single_candidate():
    assert pick_filer("どんなタイトルでも", ["Ｅｖｏ　Ｆｕｎｄ"]) == ("Ｅｖｏ　Ｆｕｎｄ", "候補1件")


def test_pick_filer_no_candidate():
    assert pick_filer("タイトル", [])[0] is None
    assert pick_filer("タイトル", [None, ""])[0] is None


def test_pick_filer_disambiguates_by_title():
    filer, reason = pick_filer(
        "グローバル・ブレイン、アクセルスペースの保有株を売却",
        ["グローバル・ブレイン株式会社", "株式会社ＭＣ"],
    )
    assert filer == "グローバル・ブレイン株式会社"
    assert "タイトル一致で特定" in reason


def test_pick_filer_skips_when_ambiguous():
    # 複数候補でタイトルにどれも出てこない場合は書き込まない（誤った提出者を入れる方が害が大きい）
    assert pick_filer("ある銘柄の大量保有報告書", ["甲野　太郎", "乙野　次郎"])[0] is None
    # 複数候補が両方ヒットしても一意にならないのでスキップ
    assert pick_filer("甲野商事と甲野商事ホールディングスが共同保有", ["甲野商事", "甲野商事ホールディングス"])[0] is None


def test_pick_filer_ignores_too_short_names():
    # 2文字の社名は無関係なタイトルに紛れ込むためタイトル一致の根拠にしない
    assert pick_filer("ＭＣ社の動向を含む記事", ["株式会社ＭＣ", "別会社"])[0] is None


if __name__ == "__main__":
    test_normalize_drops_legal_suffix_and_width()
    test_pick_filer_single_candidate()
    test_pick_filer_no_candidate()
    test_pick_filer_disambiguates_by_title()
    test_pick_filer_skips_when_ambiguous()
    test_pick_filer_ignores_too_short_names()
    print("OK")

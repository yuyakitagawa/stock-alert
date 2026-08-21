"""tools/backfill_article_filer_name.py のユニットテスト。
ネットワーク（microCMS/Supabase）は呼ばず、提出者の突合ロジックのみ検証する。

実行: python3 tests/test_backfill_article_filer_name.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.backfill_article_filer_name import normalize, pick_filer, strip_html


def _article(title: str, body: str = "") -> dict:
    """突合に使うのはタイトルと本文だけ。テストからは読みやすさのためこの関数で組む。"""
    return {"title": title, "body": body}


def test_normalize_drops_legal_suffix_and_width():
    # 「株式会社ストラテジックキャピタル」と記事タイトルの「ストラテジックキャピタル」を揃える
    assert normalize("株式会社ストラテジックキャピタル") == normalize("ストラテジックキャピタル")
    # 全角英数・全角スペース・中黒はNFKC正規化と記号除去で吸収する
    assert normalize("Ｅｖｏ　Ｆｕｎｄ") == "evofund"
    assert normalize("ひびき・パース・アドバイザーズ") == "ひびきパースアドバイザーズ"


def test_pick_filer_single_candidate():
    assert pick_filer(_article("どんなタイトルでも"), ["Ｅｖｏ　Ｆｕｎｄ"]) == ("Ｅｖｏ　Ｆｕｎｄ", "候補1件")


def test_pick_filer_no_candidate():
    assert pick_filer(_article("タイトル"), [])[0] is None
    assert pick_filer(_article("タイトル"), [None, ""])[0] is None


def test_pick_filer_disambiguates_by_title():
    filer, reason = pick_filer(
        _article("グローバル・ブレイン、アクセルスペースの保有株を売却"),
        ["グローバル・ブレイン株式会社", "株式会社ＭＣ"],
    )
    assert filer == "グローバル・ブレイン株式会社"
    assert "タイトル一致で特定" in reason


def test_pick_filer_skips_when_ambiguous():
    # 複数候補でタイトルにどれも出てこない場合は書き込まない（誤った提出者を入れる方が害が大きい）
    assert pick_filer(_article("ある銘柄の大量保有報告書"), ["甲野　太郎", "乙野　次郎"])[0] is None
    # 複数候補が両方ヒットしても一意にならないのでスキップ
    assert pick_filer(_article("甲野商事と甲野商事ホールディングスが共同保有"), ["甲野商事", "甲野商事ホールディングス"])[0] is None


def test_pick_filer_ignores_too_short_names():
    # 2文字の社名は無関係なタイトルに紛れ込むためタイトル一致の根拠にしない
    assert pick_filer(_article("ＭＣ社の動向を含む記事"), ["株式会社ＭＣ", "別会社"])[0] is None


def test_pick_filer_falls_back_to_body_when_title_omits_the_name():
    """タイトルが「個人投資家が」としか書かない回でも、本文には名前が出ている。
    本文まで見ることで、2026-08-21時点の未設定13件のうち10件が埋まる。"""
    filer, reason = pick_filer(
        _article("個人投資家が3.5億円規模を売却、パンチ工業の保有比率が2.44%に",
                 "<p>個人投資家の<b>森久保哲司</b>氏が保有比率を低下させる売却を実施しました。</p>"),
        ["森久保　哲司", "牧　寛之"],
    )
    assert filer == "森久保　哲司"
    assert "本文一致で特定" in reason


def test_pick_filer_skips_when_body_names_several_candidates():
    """本文に複数の候補が出てくる回は特定できない。総称のまま残す方が害が小さい。"""
    assert pick_filer(
        _article("サンクゼールの大量保有報告書",
                 "<p>久世良太氏と公益財団法人サンクゼール財団がともに保有しています。</p>"),
        ["久世　良太", "公益財団法人サンクゼール財団"],
    )[0] is None


def test_pick_filer_prefers_title_over_body():
    """タイトルで決まるならそれを使う（見出しに出る名前の方が記事の主語として確実）。"""
    filer, reason = pick_filer(
        _article("グローバル・ブレインが売却", "<p>株式会社ＭＣも保有しています。</p>"),
        ["グローバル・ブレイン株式会社", "株式会社ＭＣ"],
    )
    assert filer == "グローバル・ブレイン株式会社"
    assert "タイトル一致で特定" in reason


def test_strip_html_removes_tags_only():
    assert strip_html("<p>個人投資家の<b>森久保哲司</b>氏</p>") == "個人投資家の森久保哲司氏"


if __name__ == "__main__":
    test_normalize_drops_legal_suffix_and_width()
    test_pick_filer_single_candidate()
    test_pick_filer_no_candidate()
    test_pick_filer_disambiguates_by_title()
    test_pick_filer_skips_when_ambiguous()
    test_pick_filer_ignores_too_short_names()
    test_pick_filer_falls_back_to_body_when_title_omits_the_name()
    test_pick_filer_skips_when_body_names_several_candidates()
    test_pick_filer_prefers_title_over_body()
    test_strip_html_removes_tags_only()
    print("OK")

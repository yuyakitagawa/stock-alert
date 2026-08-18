"""週次「今週のアクティビストの動き」X投稿（web/x_weekly_activists）のユニットテスト。
ネットワーク（Supabase / X API）は全てモックし、集計と本文組み立てのみ検証する。

実行: python3 tests/test_x_weekly_activists.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_weekly_activists as m

ACTIVISTS = {"Ａｃｔｉｖｉｓｔ　Ｃａｐｉｔａｌ　Ｌｔｄ．", "他のアクティビスト"}


def _row(filer, code, name, disc_date, ratio, prior, desc="変更報告書", doc_type="350",
         submit=None):
    return {
        "filer_name": filer, "issuer_code": code, "issuer_name": name,
        "disc_date": disc_date, "submit_date": submit or f"{disc_date} 10:00",
        "holding_ratio": ratio, "holding_ratio_prior": prior,
        "doc_description": desc, "doc_type_code": doc_type,
    }


def test_build_moves_computes_delta_and_skips_non_activists():
    rows = [
        _row("Ａｃｔｉｖｉｓｔ　Ｃａｐｉｔａｌ　Ｌｔｄ．", "1111", "テスト工業株式会社",
             "2026-08-17", 8.5, 5.2),
        _row("分類外のファンド", "2222", "対象外株式会社", "2026-08-17", 9.0, 5.0),
    ]
    moves = m.build_moves(rows, ACTIVISTS)
    assert len(moves) == 1
    move = moves[0]
    # 全角英字は半角へ・法人格は除去（x_post_format.clean_name）
    assert move["filer"] == "Activist Capital"
    assert move["stock"] == "テスト工業"
    assert round(move["delta"], 2) == 3.3
    assert move["is_new"] is False


def test_build_moves_treats_new_filing_as_new_position():
    rows = [_row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 5.5, None,
                 desc="大量保有報告書")]
    move = m.build_moves(rows, ACTIVISTS)[0]
    assert move["is_new"] is True and move["start"] == 0.0 and move["delta"] == 5.5


def test_build_moves_skips_change_filing_without_prior_ratio():
    """変更報告書で直前保有割合が欠けている行は、週初の水準が不明なので載せない
    （「新規」と誤表示しないため）。"""
    rows = [_row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 7.0, None)]
    assert m.build_moves(rows, ACTIVISTS) == []


def test_build_moves_excludes_corrections():
    rows = [_row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 9.0, 5.0,
                 desc="訂正報告書", doc_type="360")]
    assert m.build_moves(rows, ACTIVISTS) == []


def test_build_moves_merges_multiple_filings_in_week():
    """同じ提出者×銘柄で週内に複数開示がある場合、最初の開示の直前比率と
    最後の開示の比率で「週を通した正味の変化」1行にまとめる。"""
    rows = [
        _row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 7.0, 5.0, submit="2026-08-17 09:00"),
        _row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 9.5, 7.0, submit="2026-08-17 15:00"),
    ]
    moves = m.build_moves(rows, ACTIVISTS)
    assert len(moves) == 1
    assert moves[0]["start"] == 5.0 and moves[0]["end"] == 9.5


def test_build_moves_skips_tiny_changes():
    rows = [_row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 5.2, 5.0)]
    assert m.build_moves(rows, ACTIVISTS) == []


def test_build_activist_text_shows_buys_sells_and_summary():
    moves = [
        {"filer": "Oasis Management", "stock": "電通総研", "code": "4812",
         "start": 0.0, "end": 5.0, "delta": 5.0, "is_new": True},
        {"filer": "3D Investment", "stock": "カシオ計算機", "code": "6952",
         "start": 5.0, "end": 4.0, "delta": -1.0, "is_new": False},
    ]
    text = m.build_activist_text(moves)
    assert "🦈 今週のアクティビストの動き" in text
    assert "・電通総研(4812) 新規5.0% Oasis Management" in text
    assert "・カシオ計算機(6952) 5.0→4.0% 3D Investment" in text
    assert "今週の動き2件・2ファンド" in text
    assert f"{m.SITE_URL}/activists?utm_source=x" in text
    assert text.endswith("#アクティビスト")


def test_build_activist_text_none_when_no_moves():
    assert m.build_activist_text([]) is None


def test_build_activist_text_fits_weighted_limit_with_long_names():
    long_stock = "とても長い正式名称のホールディングス"
    long_fund = "とても長い名前のアセットマネジメント"
    moves = [
        {"filer": f"{long_fund}{i}", "stock": f"{long_stock}{i}", "code": f"{i}000",
         "start": 5.0, "end": 12.0, "delta": 7.0 - i * 0.1, "is_new": False}
        for i in range(1, 4)
    ] + [
        {"filer": f"{long_fund}{i}", "stock": f"{long_stock}{i}", "code": f"{i}100",
         "start": 12.0, "end": 5.0, "delta": -7.0 + i * 0.1, "is_new": False}
        for i in range(1, 4)
    ]
    text = m.build_activist_text(moves)
    url = f"{m.SITE_URL}/activists?utm_source=x&utm_medium=social&utm_campaign=weekly_activists"
    from web.x_post_format import POST_MAX_WEIGHTED, URL_WEIGHTED_UNITS, weighted_len
    assert weighted_len(text.replace(url, "")) + URL_WEIGHTED_UNITS <= POST_MAX_WEIGHTED


def test_run_posts_and_skips_appropriately():
    rows = [_row("他のアクティビスト", "1111", "テスト工業", "2026-08-17", 9.0, 5.0)]
    posted = []
    with mock.patch.object(m, "fetch_activist_filers", return_value=ACTIVISTS), \
         mock.patch.object(m, "fetch_holdings", return_value=rows), \
         mock.patch.object(m, "build_moves", return_value=[
             {"filer": "F", "stock": "テスト工業", "code": "1111",
              "start": 5.0, "end": 9.0, "delta": 4.0, "is_new": False}]), \
         mock.patch.object(m, "post_tweet", side_effect=lambda t: posted.append(t) or True):
        assert m.run(dry_run=False) == 0
    assert len(posted) == 1 and "テスト工業(1111)" in posted[0]


def test_run_skips_when_no_activist_master():
    with mock.patch.object(m, "fetch_activist_filers", return_value=set()), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.run(dry_run=False) == 0
        tweet_mock.assert_not_called()


def test_run_dry_run_does_not_post():
    with mock.patch.object(m, "fetch_activist_filers", return_value=ACTIVISTS), \
         mock.patch.object(m, "fetch_holdings", return_value=[]), \
         mock.patch.object(m, "build_moves", return_value=[
             {"filer": "F", "stock": "S", "code": "1111",
              "start": 5.0, "end": 9.0, "delta": 4.0, "is_new": False}]), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.run(dry_run=True) == 0
        tweet_mock.assert_not_called()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

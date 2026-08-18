"""週次「大口投資家の取引急増ランキング」X投稿（web/x_weekly_trending）のロジックのユニットテスト。
集計はkujira-watch側 /trending（src/lib/trendingStats.ts）のPython移植なので、
同じ入力に対して同じ順位・deltaになることを検証する。ネットワークは全てモック。

実行: python3 tests/test_x_weekly_trending.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_weekly_trending as m

CURRENT_FROM = "2026-08-01"


def _row(code, name, filer, disc_date):
    return {"issuer_code": code, "issuer_name": name, "filer_name": filer, "disc_date": disc_date}


def test_build_trending_counts_delta_and_sorts():
    rows = [
        # 銘柄A: 直近2件・前期間0件 → delta+2
        _row("1111", "A社", "F1", "2026-08-05"),
        _row("1111", "A社", "F2", "2026-08-06"),
        # 銘柄B: 直近3件・前期間2件 → delta+1
        _row("2222", "B社", "F1", "2026-07-01"),
        _row("2222", "B社", "F1", "2026-07-02"),
        _row("2222", "B社", "F1", "2026-08-01"),
        _row("2222", "B社", "F1", "2026-08-02"),
        _row("2222", "B社", "F1", "2026-08-03"),
        # 銘柄C: 直近0件・前期間1件 → delta負のため除外
        _row("3333", "C社", "F3", "2026-07-10"),
    ]
    result = m.build_trending(
        rows, CURRENT_FROM,
        key_of=lambda r: r["issuer_code"],
        label_of=lambda r: f"{r['issuer_name']}（{r['issuer_code']}）",
        limit=10,
    )
    assert [e["key"] for e in result] == ["1111", "2222"]
    assert result[0]["delta"] == 2 and result[0]["prev_count"] == 0
    assert result[1]["delta"] == 1 and result[1]["count"] == 3


def test_build_trending_same_delta_prefers_more_recent_count():
    rows = [
        # 銘柄A: 直近1件（delta+1）
        _row("1111", "A社", "F1", "2026-08-05"),
        # 銘柄B: 直近3件・前期間2件（delta+1だが直近件数が多い）→ Aより上
        _row("2222", "B社", "F1", "2026-07-01"),
        _row("2222", "B社", "F1", "2026-07-02"),
        _row("2222", "B社", "F1", "2026-08-01"),
        _row("2222", "B社", "F1", "2026-08-02"),
        _row("2222", "B社", "F1", "2026-08-03"),
    ]
    result = m.build_trending(
        rows, CURRENT_FROM,
        key_of=lambda r: r["issuer_code"], label_of=lambda r: r["issuer_name"], limit=10,
    )
    assert [e["key"] for e in result] == ["2222", "1111"]


def test_build_weekly_trending_text_includes_entries_url_and_tags():
    issuers = [
        {"key": "1111", "label": "玉井商船（1111）", "delta": 5},
        {"key": "2222", "label": "B社（2222）", "delta": 3},
    ]
    filers = [{"key": "F1", "label": "テスト投資組合", "delta": 4}]
    text = m.build_weekly_trending_text(issuers, filers)
    assert "🐋 大口投資家の取引急増ランキング（前週比）" in text
    assert "1. 玉井商船（1111） +5件" in text
    assert "1. テスト投資組合 +4件" in text
    assert f"{m.SITE_URL}/trending?utm_source=x" in text
    # ハッシュタグは銘柄名のみ（証券コードは付けない）
    assert "#大量保有報告書 #玉井商船" in text
    assert "#玉井商船1111" not in text


def test_build_weekly_trending_text_none_when_no_issuers():
    assert m.build_weekly_trending_text([], []) is None


def test_clean_name_normalizes_fullwidth_and_strips_corporate_suffix():
    # 全角英字は半角へ（Xのカウント2単位→1単位）、法人格は除去、空白は1つに畳む
    assert m.clean_name("ＮＩＰＰＯＮ　ＡＣＴＩＶＥ　ＶＡＬＵＥ") == "NIPPON ACTIVE VALUE"
    assert m.clean_name("玉井商船　株式会社") == "玉井商船"
    assert m.clean_name("シンプレクス・アセット・マネジメント株式会社") == "シンプレクス・アセット・マネジメント"
    assert m.clean_name("（株）テスト") == "テスト"
    # 英文の法人格は末尾のものだけ落とし、社名の一部（Capital/Management）は残す
    assert m.clean_name("Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ　Ｃｏｍｐａｎｙ　Ｌｔｄ．") == "Oasis Management"
    assert m.clean_name("Sand Grove Capital Management LLP") == "Sand Grove Capital Management"


def test_build_weekly_trending_text_truncates_long_labels():
    issuers = [{"key": "1111", "label": "とても長い正式名称のホールディングス株式会社（1111）", "delta": 2}]
    text = m.build_weekly_trending_text(issuers, [])
    assert "1. とても長い正式名称のホール… +2件" in text


def test_build_weekly_trending_text_fits_weighted_limit_with_long_names():
    long = "とても長い正式名称のホールディングス株式会社"
    issuers = [{"key": str(i), "label": f"{long}（{i}000）", "delta": 10 - i} for i in range(1, 4)]
    filers = [{"key": f"F{i}", "label": long, "delta": 9 - i} for i in range(1, 3)]
    text = m.build_weekly_trending_text(issuers, filers)
    url = f"{m.SITE_URL}/trending?utm_source=x&utm_medium=social&utm_campaign=weekly_trending"
    assert m.weighted_len(text.replace(url, "")) + m.URL_WEIGHTED_UNITS <= m.POST_MAX_WEIGHTED


def test_run_posts_text_built_from_fetched_rows():
    from datetime import date, timedelta

    # 実行日に依存しないよう、直近窓に必ず入る日付（今日と昨日）で組み立てる。
    recent = [date.today().isoformat(), (date.today() - timedelta(days=1)).isoformat()]
    rows = [_row("1111", "A社", "F1", recent[0]), _row("1111", "A社", "F2", recent[1])]
    posted = []
    with mock.patch.object(m, "fetch_holdings", return_value=rows), \
         mock.patch.object(m, "post_tweet", side_effect=lambda t: posted.append(t) or True):
        assert m.run(dry_run=False) == 0
    assert len(posted) == 1
    assert "A社（1111）" in posted[0]


def test_run_skips_when_no_rows():
    with mock.patch.object(m, "fetch_holdings", return_value=[]), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.run(dry_run=False) == 0
        tweet_mock.assert_not_called()


def test_run_dry_run_does_not_post():
    rows = [_row("1111", "A社", "F1", "2026-08-05"), _row("1111", "A社", "F2", "2026-08-06")]
    with mock.patch.object(m, "fetch_holdings", return_value=rows), \
         mock.patch.object(m, "post_tweet") as tweet_mock:
        assert m.run(dry_run=True) == 0
        tweet_mock.assert_not_called()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

"""web/x_filer_record.py のユニットテスト。SupabaseもX APIも呼ばない。

実行: python3 tests/test_x_filer_record.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import x_filer_record as m


def _series(pairs):
    return [(d, float(c)) for d, c in pairs]


def test_excludes_self_name_filers_including_old_kanji():
    """証券会社・銀行の自己名義は投資判断ではないので実績に混ぜない。
    野村證券は旧字体の「證券」で登記されており、「証券」だけの判定では漏れる。"""
    assert m._has_excluded_keyword("野村證券株式会社") is True
    assert m._has_excluded_keyword("株式会社ＳＢＩ証券") is True
    assert m._has_excluded_keyword("三井住友信託銀行株式会社") is True
    assert m._has_excluded_keyword("Ｏａｓｉｓ　Ｍａｎａｇｅｍｅｎｔ　Ｃｏｍｐａｎｙ　Ｌｔｄ．") is False
    assert m._has_excluded_keyword("光通信株式会社") is False


def test_close_lookup_allows_holidays_but_not_long_gaps():
    """休場は数日またいでよいが、上場廃止等で価格が飛ぶ場合は使わない。"""
    s = _series([("2026-01-05", 100), ("2026-01-20", 120)])
    assert m._close_on_or_after(s, "2026-01-01") == 100      # 4日先の終値は許容
    assert m._close_on_or_after(s, "2026-01-06") is None     # 14日先は許容外


def test_split_guard_rejects_discontinuous_series():
    """株式分割で終値が半分になった銘柄は、リターンが実態とかけ離れるので除外する。"""
    assert m._is_discontinuous(_series([("2026-01-05", 100), ("2026-01-06", 49)])) is True
    assert m._is_discontinuous(_series([("2026-01-05", 100), ("2026-01-06", 110)])) is False


def test_compute_returns_drops_events_without_both_prices():
    """開示日と3ヶ月後の両方の終値が揃った開示だけを使う。"""
    events = [
        {"filer_name": "F", "issuer_code": "1111", "disc_date": "2026-01-05"},
        {"filer_name": "F", "issuer_code": "2222", "disc_date": "2026-01-05"},
    ]
    prices = {
        "1111": _series([("2026-01-05", 100), ("2026-04-06", 130)]),
        "2222": _series([("2026-01-05", 100)]),          # 3ヶ月後が無い
    }
    rows = m.compute_returns(events, prices)
    assert [r["issuer_code"] for r in rows] == ["1111"]
    assert round(rows[0]["ret"], 1) == 30.0


def test_rank_filers_needs_minimum_events():
    """件数が少ない提出者は、数件の当たり外れで平均が動くので出さない。"""
    rows = ([{"filer_name": "A", "issuer_code": "1", "issuer_name": "あ", "ret": 10.0}] * m.MIN_EVENTS
            + [{"filer_name": "B", "issuer_code": "2", "issuer_name": "い", "ret": 99.0}] * (m.MIN_EVENTS - 1))
    names = [f["filer_name"] for f in m.rank_filers(rows)]
    assert names == ["A"]


def test_rank_filers_drops_those_that_match_the_market():
    """全開示平均と差が無い提出者は投稿する内容が無いので落とす。
    ここを外すと「平均+3.6%・全開示平均+3.6%」で上下を断定してしまう。"""
    rows = ([{"filer_name": "EDGE", "issuer_code": "1", "issuer_name": "あ", "ret": 30.0}] * m.MIN_EVENTS
            + [{"filer_name": "SAME", "issuer_code": "2", "issuer_name": "い", "ret": 10.0}] * m.MIN_EVENTS)
    overall = m.summarize(rows)                      # 平均 +20.0%
    names = [f["filer_name"] for f in m.rank_filers(rows, overall)]
    assert names == ["EDGE", "SAME"] or names == ["SAME", "EDGE"]   # 両方とも±10ptで残る
    tight = ([{"filer_name": "SAME", "issuer_code": "2", "issuer_name": "い", "ret": 10.0}] * m.MIN_EVENTS)
    assert m.rank_filers(tight, m.summarize(tight)) == []           # 差0なら残らない


def test_pick_weekly_rotates_through_filers():
    """毎週1位だけを出すと同じ投稿が並ぶため、週ごとに別の提出者を選ぶ。"""
    ranked = [{"filer_name": n} for n in ("A", "B", "C")]
    picked = {m.pick_weekly(ranked, date(2026, 1, 5 + 7 * w))["filer_name"] for w in range(3)}
    assert picked == {"A", "B", "C"}
    assert m.pick_weekly([], date(2026, 1, 5)) is None


def test_build_text_always_shows_the_overall_average():
    """地合いを実力と誤認させないため、全開示平均を必ず併記する。"""
    rec = {"filer_name": "テストファンド", "n": 20, "mean": 18.1, "win_rate": 85.0,
           "best": {"issuer_name": "あ社", "issuer_code": "1234", "ret": 55.5}}
    text = m.build_text(rec, {"mean": 5.5})
    assert "20銘柄" in text and "+18.1%" in text and "勝率85%" in text
    assert "+5.5%" in text and "明確に上" in text
    assert "http" not in text          # URLは入れない（リンク投稿は$0.20課金）


def test_build_text_says_below_when_the_filer_trails_the_market():
    """負けている投資家を「上」と書かないこと。"""
    rec = {"filer_name": "テストファンド", "n": 25, "mean": -11.2, "win_rate": 28.0,
           "best": {"issuer_name": "あ社", "issuer_code": "1234", "ret": 5.0}}
    assert "明確に下" in m.build_text(rec, {"mean": 6.7})


def test_build_text_refuses_when_there_is_no_gap_to_the_market():
    """表示上おなじ数字なのに上下を断定しないこと（CI実測で出た不具合）。"""
    rec = {"filer_name": "テストファンド", "n": 28, "mean": 3.62, "win_rate": 43.0,
           "best": {"issuer_name": "あ社", "issuer_code": "1234", "ret": 60.6}}
    assert m.build_text(rec, {"mean": 3.64}) is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

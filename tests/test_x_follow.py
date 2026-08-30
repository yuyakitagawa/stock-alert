"""tools/x_follow.py のロジックのユニットテスト。X APIは呼ばない。

実行: python3 tests/test_x_follow.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.x_follow as m


def _author(username, followers, hits=1, protected=False, description="日本株"):
    return {
        "id": username, "username": username, "name": username,
        "protected": protected, "description": description,
        "public_metrics": {"followers_count": followers, "tweet_count": 100},
        "hits": hits, "queries": {"大量保有報告書"},
    }


def test_build_candidates_drops_protected_and_out_of_range():
    """鍵アカと、フォロワー数が範囲外のアカウントは候補から外す。"""
    authors = [
        _author("ok", 1000),
        _author("locked", 1000, protected=True),
        _author("too_small", 10),
        _author("too_big", 500_000),
    ]
    got = [c["username"] for c in m.build_candidates(authors, 300, 200_000)]
    assert got == ["ok"]


def test_build_candidates_sorts_by_hits_then_followers():
    """そのテーマでの発言回数が多い順、同数ならフォロワーが多い順。"""
    authors = [
        _author("a", 1000, hits=1),
        _author("b", 500, hits=3),
        _author("c", 900, hits=3),
    ]
    got = [c["username"] for c in m.build_candidates(authors, 300, 200_000)]
    assert got == ["c", "b", "a"]


def test_build_candidates_flattens_description_to_one_line():
    """一覧を1行1件で読めるように、改行を潰して80字で切る。"""
    a = _author("x", 1000, description="日本株\n大量保有" + "あ" * 100)
    c = m.build_candidates([a], 300, 200_000)[0]
    assert "\n" not in c["description"]
    assert len(c["description"]) == 80


def test_follow_refuses_more_than_the_per_run_cap():
    """1回の上限を超える指定は、無差別な大量フォローになる前に止める。"""
    names = [f"user{i}" for i in range(m.MAX_FOLLOWS_PER_RUN + 1)]
    with mock.patch.object(m, "resolve_usernames") as resolve:
        assert m.follow(names, execute=True) == 1
        resolve.assert_not_called()


def test_follow_without_execute_does_not_post():
    """--execute が無いときは解決だけして、フォローのPOSTは投げない。"""
    users = [{"id": "1", "username": "a", "name": "a", "public_metrics": {"followers_count": 10}}]
    with mock.patch.object(m, "resolve_usernames", return_value=(users, "")), \
         mock.patch.object(m.requests, "post") as post:
        assert m.follow(["a"], execute=False) == 0
        post.assert_not_called()


def test_follow_stops_on_permission_error():
    """403は権限かAPIプランの問題で、続けても全件同じ結果になるので即やめる。"""
    users = [{"id": str(i), "username": f"u{i}", "name": "n",
              "public_metrics": {"followers_count": 10}} for i in range(3)]
    resp = mock.Mock(status_code=403, text="not permitted")
    with mock.patch.object(m, "resolve_usernames", return_value=(users, "")), \
         mock.patch.object(m, "me_id", return_value=("me", "kujira")), \
         mock.patch.object(m, "_auth", return_value=None), \
         mock.patch.object(m, "time"), \
         mock.patch.object(m.requests, "post", return_value=resp) as post:
        assert m.follow(["u0", "u1", "u2"], execute=True) == 1
        assert post.call_count == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

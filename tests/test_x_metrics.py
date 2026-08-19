"""X投稿メトリクス収集（web/x_metrics）のユニットテスト。
X APIとSupabaseは全てモックし、パースと集計のロジックのみ検証する。

実行: python3 tests/test_x_metrics.py
"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_metrics as m


def _resp(status=200, payload=None):
    return mock.Mock(status_code=status, json=lambda: payload or {}, text="")


# ------------------------------------------------- 投稿メトリクスのパース

def test_fetch_metrics_parses_profile_clicks():
    payload = {"data": [{
        "id": "111",
        "public_metrics": {"like_count": 3, "retweet_count": 1, "reply_count": 0,
                           "quote_count": 0, "bookmark_count": 2},
        "non_public_metrics": {"impression_count": 900, "url_link_clicks": 12,
                               "user_profile_clicks": 7},
    }]}
    with mock.patch.object(m, "_auth", return_value=object()), \
         mock.patch.object(m.requests, "get", return_value=_resp(payload=payload)):
        out = m.fetch_metrics(["111"])
    assert out["111"]["impressions"] == 900, out
    assert out["111"]["url_link_clicks"] == 12, out
    assert out["111"]["user_profile_clicks"] == 7, out


def test_fetch_metrics_without_non_public_metrics():
    """non_public_metricsが権限で取れない場合でもpublic_metricsだけで成立する。"""
    payload = {"data": [{"id": "111", "public_metrics": {"like_count": 3}}]}
    with mock.patch.object(m, "_auth", return_value=object()), \
         mock.patch.object(m.requests, "get", return_value=_resp(payload=payload)):
        out = m.fetch_metrics(["111"])
    assert out["111"]["user_profile_clicks"] is None, out
    assert out["111"]["likes"] == 3, out


# ------------------------------------------------- フォロワー数

def test_fetch_followers():
    payload = {"data": {"public_metrics": {"followers_count": 120, "following_count": 30,
                                           "tweet_count": 400, "listed_count": 2}}}
    with mock.patch.object(m, "_auth", return_value=object()), \
         mock.patch.object(m.requests, "get", return_value=_resp(payload=payload)):
        snap = m.fetch_followers()
    assert snap == {"followers": 120, "following": 30, "tweets": 400, "listed": 2}, snap


def test_fetch_followers_returns_empty_without_auth():
    with mock.patch.object(m, "_auth", return_value=None):
        assert m.fetch_followers() == {}


def test_fetch_followers_returns_empty_on_error():
    with mock.patch.object(m, "_auth", return_value=object()), \
         mock.patch.object(m.requests, "get", return_value=_resp(status=403)):
        assert m.fetch_followers() == {}


def test_follower_report_compares_with_nearest_older_record():
    """記録が飛んでいる日があっても、7日前以前で最も新しい記録と比べる。"""
    rows = [{"measured_on": "2026-08-19", "followers": 150},
            {"measured_on": "2026-08-10", "followers": 130},   # 7日前(8/12)より古い最新
            {"measured_on": "2026-07-15", "followers": 100}]   # 30日前(7/20)より古い最新
    buf = io.StringIO()
    with mock.patch.object(m.sb, "select", return_value=rows), redirect_stdout(buf):
        m.follower_report()
    out = buf.getvalue()
    assert "フォロワー 150人" in out, out
    assert "7日前比: +20人（2026-08-10 130人）" in out, out
    assert "30日前比: +50人（2026-07-15 100人）" in out, out


def test_follower_report_without_records():
    buf = io.StringIO()
    with mock.patch.object(m.sb, "select", return_value=[]), redirect_stdout(buf):
        m.follower_report()
    assert "記録がまだありません" in buf.getvalue()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'FAILED' if fails else 'PASSED'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)

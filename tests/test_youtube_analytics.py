"""video/youtube_analytics.py のユニットテスト（Analytics APIは全てモック）。"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video import youtube_analytics as a  # noqa: E402


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status == 200
        self.text = str(payload)

    def json(self):
        return self._payload


def test_video_stats_maps_rows_to_video_ids():
    """行は video, views, 視聴率, 視聴秒, 登録者獲得 の順。viewsはソート用に要るだけで捨てる。"""
    rows = {"rows": [["vid1", 1333, 41.6, 18, 2], ["vid2", 520, 25.0, 11, 0]]}
    with mock.patch.object(a.requests, "get", return_value=_Resp(rows)):
        out = a.video_stats("tok", "2026-08-30")
    assert out["vid1"] == {"avg_view_pct": 41.6, "avg_view_sec": 18, "subscribers_gained": 2}
    assert out["vid2"]["subscribers_gained"] == 0


def test_video_stats_keeps_percentages_above_100():
    """Shortsはループ再生ぶんが積み上がり視聴率が100%を超える（実測169.6%）。頭打ちにしない。"""
    rows = {"rows": [["vid1", 1197, 169.58, 76, 1]]}
    with mock.patch.object(a.requests, "get", return_value=_Resp(rows)):
        assert a.video_stats("tok", "2026-08-30")["vid1"]["avg_view_pct"] == 169.6


def test_video_stats_sorts_by_views_because_the_api_requires_it():
    """video次元はviews降順でないと400 The query is not supportedになる。"""
    captured = {}

    def fake_get(url, params=None, **kw):
        captured.update(params or {})
        return _Resp({"rows": []})

    with mock.patch.object(a.requests, "get", side_effect=fake_get):
        a.video_stats("tok", "2026-08-30")
    assert captured["sort"] == "-views"
    assert captured["metrics"].startswith("views,")


def test_video_stats_returns_empty_on_insufficient_scope():
    """トークンが古い（upload だけ）と403になる。ここで例外を投げると再生数の記録まで落ちる。"""
    with mock.patch.object(a.requests, "get", return_value=_Resp({"error": "scope"}, status=403)):
        assert a.video_stats("tok", "2026-08-30") == {}


def test_retention_curve_is_sorted_by_elapsed_ratio():
    rows = {"rows": [[0.5, 0.4], [0.0, 1.0], [1.0, 0.2]]}
    with mock.patch.object(a.requests, "get", return_value=_Resp(rows)):
        curve = a.retention_curve("tok", "vid1", "2026-08-30")
    assert curve == [(0.0, 1.0), (0.5, 0.4), (1.0, 0.2)]


def test_survival_at_converts_seconds_to_elapsed_ratio():
    """カーブは尺に対する割合で返るので、本ごとに「3秒時点」を比べるには尺で割り戻す。"""
    curve = [(i / 100, 1.0 - i / 200) for i in range(101)]
    # 45秒の動画の3秒 = 経過6.7% → 直近の実測点は7%
    assert a.survival_at(curve, 45, 3) == 0.965
    assert a.survival_at(curve, 45, 0) == 1.0


def test_survival_at_normalizes_by_the_head_of_the_curve():
    """ループ再生で冒頭が1.0を超える回がある。生値で並べるとhookの出来を比べられない。"""
    loopy = [(0.01, 1.40), (0.07, 1.30), (1.0, 0.39)]
    plain = [(0.01, 1.05), (0.07, 0.95), (1.0, 0.30)]
    # 生値だと1.30 vs 0.95で前者が圧勝に見えるが、冒頭比では92.9%と90.5%でほぼ互角
    assert a.survival_at(loopy, 45, 3) == 0.929
    assert a.survival_at(plain, 45, 3) == 0.905


def test_survival_at_returns_none_when_unmeasurable():
    assert a.survival_at([], 45, 3) is None
    assert a.survival_at([(0.0, 1.0)], 0, 3) is None
    # 尺より後ろの時点は存在しない
    assert a.survival_at([(0.0, 1.0)], 45, 60) is None

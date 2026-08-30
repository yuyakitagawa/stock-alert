"""video/youtube_metrics.py のユニットテスト（YouTube API・Supabaseは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video import youtube_metrics as y  # noqa: E402


def _video(vid, dur, views=0, published="2026-08-20T00:00:00Z"):
    return {"id": vid, "snippet": {"publishedAt": published, "title": f"t{vid}"},
            "contentDetails": {"duration": dur},
            "statistics": {"viewCount": str(views), "likeCount": "1", "commentCount": "0"}}


def test_parse_duration_handles_minutes_and_seconds():
    assert y.parse_duration("PT1M29S") == 89
    assert y.parse_duration("PT39S") == 39
    assert y.parse_duration("PT1H2M3S") == 3723
    assert y.parse_duration("PT2M") == 120


def test_parse_duration_returns_zero_for_unparsable():
    """尺が読めない動画で例外を投げると、その日の記録が丸ごと落ちる。0にして残す。"""
    assert y.parse_duration("") == 0
    assert y.parse_duration("garbage") == 0


def test_summarize_splits_by_duration():
    videos = [{"duration_sec": 40, "views": 1000}, {"duration_sec": 45, "views": 1400},
              {"duration_sec": 89, "views": 500}]
    s = y.summarize(videos)
    assert s["short_n"] == 2 and s["short_avg_views"] == 1200
    assert s["long_n"] == 1 and s["long_avg_views"] == 500


def test_summarize_handles_empty_without_dividing_by_zero():
    s = y.summarize([])
    assert s == {"short_n": 0, "short_avg_views": 0.0, "long_n": 0, "long_avg_views": 0.0}


def test_summarize_ignores_unknown_duration():
    """尺0（読めなかった動画）を短尺に数えると、短尺の平均が実態からずれる。"""
    assert y.summarize([{"duration_sec": 0, "views": 999}])["short_n"] == 0


def test_fetch_channel_raises_when_handle_not_found():
    with mock.patch.object(y, "_get", return_value={"items": []}):
        try:
            y.fetch_channel("tok")
            raise AssertionError("例外が出るべき")
        except RuntimeError as e:
            assert y.CHANNEL_HANDLE in str(e)


def test_fetch_channel_reads_statistics():
    payload = {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU1"}},
                          "statistics": {"subscriberCount": "3", "viewCount": "4747",
                                         "videoCount": "7"}}]}
    with mock.patch.object(y, "_get", return_value=payload):
        assert y.fetch_channel("tok") == {"uploads": "UU1", "subscribers": 3,
                                          "total_views": 4747, "video_count": 7}


def test_fetch_videos_follows_pagination_and_sorts_oldest_first():
    pages = [
        {"items": [{"contentDetails": {"videoId": "a"}}], "nextPageToken": "p2"},
        {"items": [{"contentDetails": {"videoId": "b"}}]},
        {"items": [_video("a", "PT1M29S", 100, "2026-08-20T00:00:00Z"),
                   _video("b", "PT39S", 200, "2026-08-10T00:00:00Z")]},
    ]
    with mock.patch.object(y, "_get", side_effect=pages):
        videos = y.fetch_videos("tok", "UU1")
    assert [v["video_id"] for v in videos] == ["b", "a"]      # 古い順
    assert videos[0]["duration_sec"] == 39 and videos[1]["views"] == 100


def test_record_upload_writes_row_without_clobbering_metrics():
    """投稿直後の記録。収集済みの再生数を潰さないよう insert_ignore で書く。"""
    with mock.patch.object(y.sb, "is_configured", return_value=True), \
            mock.patch.object(y.sb, "insert_ignore") as ins:
        assert y.record_upload("vid1", "【日立建機】…") is True
    table, rows = ins.call_args.args[0], ins.call_args.args[1]
    assert table == "youtube_videos" and rows[0]["video_id"] == "vid1"
    assert rows[0]["published_at"] and rows[0]["title"] == "【日立建機】…"
    assert ins.call_args.kwargs["on_conflict"] == "video_id"


def test_record_upload_skips_without_video_id_or_supabase():
    with mock.patch.object(y.sb, "is_configured", return_value=True), \
            mock.patch.object(y.sb, "insert_ignore") as ins:
        assert y.record_upload("", "t") is False
    with mock.patch.object(y.sb, "is_configured", return_value=False):
        assert y.record_upload("vid1", "t") is False
    ins.assert_not_called()


def test_save_skips_when_supabase_not_configured():
    with mock.patch.object(y.sb, "is_configured", return_value=False), \
            mock.patch.object(y.sb, "upsert") as up, redirect_stdout(io.StringIO()):
        y.save([{"video_id": "a"}], {"subscribers": 1, "total_views": 2, "video_count": 3})
    up.assert_not_called()


def test_save_writes_latest_and_daily_snapshot():
    """最新値だけだと伸びている最中か頭打ちかが分からないため、日次スナップショットも要る。"""
    v = {"video_id": "a", "published_at": "2026-08-20T00:00:00Z", "title": "t",
         "duration_sec": 40, "views": 10, "likes": 1, "comments": 0}
    with mock.patch.object(y.sb, "is_configured", return_value=True), \
            mock.patch.object(y.sb, "upsert") as up:
        y.save([v], {"subscribers": 3, "total_views": 4747, "video_count": 7})
    tables = [c.args[0] for c in up.call_args_list]
    assert tables == ["youtube_videos", "youtube_video_metrics", "youtube_channel_stats"]


def test_main_returns_1_without_credentials():
    with mock.patch.object(y, "access_token", return_value=None), \
            mock.patch.object(sys, "argv", ["x"]), redirect_stdout(io.StringIO()) as buf:
        assert y.main() == 1
    assert "サービスアカウント鍵" in buf.getvalue()


def test_main_returns_1_when_no_videos():
    """1本も取れないのは公開失敗か認証の異常。緑で流すと何日でも気付けない。"""
    with mock.patch.object(y, "access_token", return_value="tok"), \
            mock.patch.object(y, "fetch_channel", return_value={"uploads": "UU1", "subscribers": 0,
                                                                "total_views": 0, "video_count": 0}), \
            mock.patch.object(y, "fetch_videos", return_value=[]), \
            mock.patch.object(y, "save"), \
            mock.patch.object(sys, "argv", ["x"]), redirect_stdout(io.StringIO()):
        assert y.main() == 1


def test_main_returns_1_when_api_fails():
    with mock.patch.object(y, "access_token", return_value="tok"), \
            mock.patch.object(y, "fetch_channel", side_effect=RuntimeError("HTTP 403")), \
            mock.patch.object(sys, "argv", ["x"]), redirect_stdout(io.StringIO()) as buf:
        assert y.main() == 1
    assert "403" in buf.getvalue()


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


def test_access_token_falls_back_to_oauth_without_service_account_key():
    """gcp_key.json は .gitignore されていてCIには無い。日次収集をActionsで回すため、
    鍵が無ければ投稿用のOAuthトークン（scopeにforce-sslを含む）へ落とす。"""
    with mock.patch.object(y, "credentials_path", return_value="/nowhere/gcp_key.json"), \
         mock.patch.object(y.youtube_client, "access_token", return_value="oauth-tok"):
        assert y.access_token() == "oauth-tok"

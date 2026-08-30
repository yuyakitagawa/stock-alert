"""tools/x_benchmark.py のロジックのユニットテスト。X APIは呼ばない。

実行: python3 tests/test_x_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.x_benchmark as m


def _tweet(**over):
    tw = {
        "text": "1行目\n2行目 #日本株",
        "created_at": "2026-08-29T12:30:00.000Z",
        "public_metrics": {"like_count": 5, "retweet_count": 2, "reply_count": 1,
                           "quote_count": 1, "impression_count": 1000},
        "entities": {"hashtags": [{"tag": "日本株"}], "urls": [{"url": "https://x"}]},
        "attachments": {"media_keys": ["m1"]},
    }
    tw.update(over)
    return tw


def test_summarize_tweet_extracts_the_format_features():
    """型の比較に使うのは本文そのものではなく、字数・行数・画像/URL/タグの有無。"""
    t = m.summarize_tweet(_tweet(), "someone")
    assert t["username"] == "someone"
    assert t["first_line"] == "1行目"
    assert t["lines"] == 2
    assert t["hashtags"] == 1
    assert t["has_url"] is True
    assert t["has_media"] is True
    assert t["engagement"] == 5 + 2 + 1 + 1
    assert t["impressions"] == 1000


def test_summarize_tweet_converts_the_hour_to_jst():
    """投稿時間帯はJSTで見る（12:30 UTC = 21時JST）。"""
    assert m.summarize_tweet(_tweet(), "u")["hour_jst"] == 21


def test_summarize_tweet_survives_missing_fields():
    """画像なし・URLなし・メトリクス欠けの投稿でも落ちない。"""
    t = m.summarize_tweet({"text": "本文だけ"}, "u")
    assert t["has_media"] is False and t["has_url"] is False
    assert t["engagement"] == 0 and t["hour_jst"] is None


def test_bucket_report_hides_groups_with_too_few_posts(capsys):
    """3件未満のグループは中央値が1本の当たり外れで動くので表示しない。"""
    many = {"engagement": 10, "impressions": 100}
    groups = {
        "十分": [dict(many) for _ in range(3)],
        "少ない": [dict(many) for _ in range(2)],
    }
    m._bucket_report("テスト", groups)
    out = capsys.readouterr().out
    assert "十分" in out
    assert "少ない" not in out


def test_chars_bucket_boundaries():
    """文字数帯の境界。X本文の実質上限140字に合わせて切る。"""
    assert m._chars_bucket({"chars": 59}) == "〜59字"
    assert m._chars_bucket({"chars": 60}) == "60〜99字"
    assert m._chars_bucket({"chars": 139}) == "100〜139字"
    assert m._chars_bucket({"chars": 140}) == "140字〜"


def test_hour_bucket_covers_every_hour():
    """どの時刻でも必ずどれかの帯に入る（穴があると集計から投稿が消える）。"""
    labels = {m._hour_bucket({"hour_jst": h}) for h in range(24)}
    assert len(labels) == 5
    assert m._hour_bucket({"hour_jst": None}) == "不明"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))

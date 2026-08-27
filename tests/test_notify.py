"""lib/notify.py と tools/output_heartbeat.py のユニットテスト。

実行: python3 tests/test_notify.py
"""
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import api_budget  # noqa: E402
from lib import notify  # noqa: E402
from tools import output_heartbeat as hb  # noqa: E402

CREDS = {"LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_USER_ID": "u"}


class NotifyTest(unittest.TestCase):
    def test_skips_without_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("lib.notify.requests.post") as post:
                self.assertFalse(notify.push("テスト"))
                post.assert_not_called()

    def test_pushes_with_credentials(self):
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.notify.requests.post") as post:
                post.return_value = mock.Mock(ok=True)
                self.assertTrue(notify.error("ブログ生成", "止まりました", detail="400 error"))
        body = post.call_args.kwargs["json"]["messages"][0]["text"]
        self.assertIn("🚨 ブログ生成", body)
        self.assertIn("止まりました", body)
        self.assertIn("400 error", body)

    def test_network_failure_does_not_raise(self):
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.notify.requests.post", side_effect=OSError("boom")):
                self.assertFalse(notify.push("テスト"))

    def test_long_text_is_truncated(self):
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.notify.requests.post") as post:
                post.return_value = mock.Mock(ok=True)
                notify.push("あ" * (notify.MAX_CHARS + 500))
        self.assertEqual(len(post.call_args.kwargs["json"]["messages"][0]["text"]),
                         notify.MAX_CHARS)


class ApiBudgetNotifyTest(unittest.TestCase):
    """2026-08-24の無言停止: 利用上限に当たった瞬間にLINEへ出す（1プロセス1回だけ）。"""

    def setUp(self):
        api_budget.reset()

    def tearDown(self):
        api_budget.reset()

    def test_notifies_once_on_usage_limit(self):
        exc = Exception("You have reached your specified API usage limits.")
        with mock.patch("lib.notify.error", return_value=True) as err:
            self.assertTrue(api_budget.note(exc))
            self.assertTrue(api_budget.note(exc))  # 2件目以降は通知しない
        self.assertEqual(err.call_count, 1)

    def test_does_not_notify_on_transient_error(self):
        with mock.patch("lib.notify.error", return_value=True) as err:
            self.assertFalse(api_budget.note(Exception("Error code: 529 - overloaded_error")))
        err.assert_not_called()


class HeartbeatTest(unittest.TestCase):
    BASE = {"date": "2026-08-24", "holdings": 39, "buybacks": 2, "edinet_api": 39,
            "articles": 18, "x_posts": 2, "videos": 1}

    def test_healthy_day_has_no_problem(self):
        self.assertEqual(hb.judge(self.BASE), [])

    def test_flags_zero_articles_when_material_exists(self):
        problems = hb.judge({**self.BASE, "articles": 0, "videos": 0})
        self.assertTrue(any("ブログ記事が0件" in p for p in problems))

    def test_quiet_day_without_material_is_not_flagged(self):
        """開示が無い日（祝日等）は記事0件でも異常ではない。
        「静かな日」はEDINET側も0件であることまで確認する（DBだけ0件なら保存の故障）。"""
        problems = hb.judge({**self.BASE, "edinet_api": 0, "holdings": 0, "buybacks": 0,
                             "articles": 0, "videos": 0})
        self.assertFalse(any("ブログ記事" in p for p in problems))

    def test_flags_missing_video_when_articles_exist(self):
        """2026-08-24の実データ: 記事は出たが動画は0本。"""
        problems = hb.judge({**self.BASE, "videos": 0})
        self.assertTrue(any("動画" in p for p in problems))

    def test_flags_zero_x_posts(self):
        self.assertTrue(any("X投稿が0件" in p for p in hb.judge({**self.BASE, "x_posts": 0})))

    def test_unknown_counts_are_not_flagged(self):
        """Supabase/microCMSが引けなかった項目(-1)で誤報を出さない。"""
        self.assertEqual(hb.judge({**self.BASE, "articles": -1, "x_posts": -1, "videos": -1}), [])

    def test_message_marks_problem_and_lists_counts(self):
        msg = hb.build_message(self.BASE, hb.judge({**self.BASE, "videos": 0}))
        self.assertTrue(msg.startswith("🚨"))
        self.assertIn("ブログ18件", msg)

    def test_message_is_ok_when_healthy(self):
        self.assertTrue(hb.build_message(self.BASE, []).startswith("✅"))

    def test_day_start_utc_is_jst_midnight(self):
        self.assertEqual(hb._day_start_utc("2026-08-24"), "2026-08-23T15:00:00.000Z")

    def test_day_end_utc_is_next_jst_midnight(self):
        self.assertEqual(hb._day_end_utc("2026-08-24"), "2026-08-24T15:00:00.000Z")

    def test_target_date_is_today_for_the_scheduled_evening_run(self):
        run_at = datetime(2026, 8, 27, 22, 0, tzinfo=hb.JST)  # 予定どおり22:00 JST
        self.assertEqual(hb.target_date(now=run_at), "2026-08-27")

    def test_delayed_morning_run_judges_the_previous_day(self):
        """13:00 UTCの便が翌朝に遅れて起動した実例（8/28 07:40 JST）。
        始まったばかりの当日を見ると「X0件・動画0本」で毎朝誤報になる。"""
        run_at = datetime(2026, 8, 28, 7, 40, tzinfo=hb.JST)
        self.assertEqual(hb.target_date(now=run_at), "2026-08-27")

    def test_explicit_date_wins(self):
        self.assertEqual(hb.target_date("2026-08-24",
                                        now=datetime(2026, 8, 28, 7, 40, tzinfo=hb.JST)),
                         "2026-08-24")

    def test_counts_are_bounded_to_the_target_day(self):
        """上限が無いと、前日を見る便が当日ぶん（backfillの記事等）まで数えてしまう。
        動画はXのクロス投稿ではなくYouTubeの公開実績から数える。"""
        with mock.patch.object(hb, "count_edinet_disclosures", return_value=0), \
                mock.patch.object(hb, "count_blog_articles", return_value=3), \
                mock.patch.object(hb.sb, "select", return_value=[]) as sel:
            hb.collect("2026-08-27")
        queries = {c.args[0]: c.args[1] for c in sel.call_args_list}
        self.assertIn("youtube_videos", queries)
        self.assertNotIn("kind=eq.video", "".join(queries.values()))
        for table in ("x_posts", "youtube_videos"):
            self.assertIn("2026-08-26T15:00:00.000Z", queries[table])
            self.assertIn("2026-08-27T15:00:00.000Z", queries[table])

    def test_flags_db_zero_while_edinet_has_disclosures(self):
        """保存が壊れてDBだけ0件になった日を名指しで検知する（2026-08-26〜27の実例）。"""
        problems = hb.judge({**self.BASE, "holdings": 0, "articles": 0, "videos": 0})
        self.assertTrue(any("DBは0件" in p for p in problems), problems)

    def test_material_is_counted_from_edinet_not_db(self):
        """DBが0件でもEDINETに開示があれば「記事0件」を異常として拾う。
        素材をDBから数えていたため、保存が壊れた日が静かな日と区別できていなかった。"""
        problems = hb.judge({**self.BASE, "holdings": 0, "buybacks": 0,
                             "articles": 0, "videos": 0})
        self.assertTrue(any("ブログ記事が0件" in p for p in problems), problems)

    def test_edinet_unknown_falls_back_to_db_count(self):
        """EDINETを引けなかった(-1)ときは従来どおりDBの件数で判定し、誤報も出さない。"""
        self.assertEqual(hb.judge({**self.BASE, "edinet_api": -1}), [])
        problems = hb.judge({**self.BASE, "edinet_api": -1, "holdings": 0,
                             "buybacks": 0, "articles": 0, "videos": 0})
        self.assertEqual(problems, [])

    def test_message_shows_both_db_and_edinet_counts(self):
        msg = hb.build_message({**self.BASE, "holdings": 0}, [])
        self.assertIn("大量保有0件（EDINET 39件）", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)

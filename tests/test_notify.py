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

    def test_once_skips_when_already_sent(self):
        """同じキーの警告を毎時ジョブが何十通も送らないこと。"""
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.supabase_client.is_configured", return_value=True), \
                    mock.patch("lib.supabase_client.select_one",
                               return_value={"dedupe_key": "k"}), \
                    mock.patch("lib.supabase_client.upsert") as up, \
                    mock.patch("lib.notify.push") as push:
                self.assertFalse(notify.once("k", "本文"))
                push.assert_not_called()
        # 抑制した回数も数える（何回ぶん黙ったかを後から追えるように）
        self.assertEqual(up.call_args.args[1][0]["sent_count"], 1)

    def test_once_sends_and_records_the_first_time(self):
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.supabase_client.is_configured", return_value=True), \
                    mock.patch("lib.supabase_client.select_one", return_value=None), \
                    mock.patch("lib.supabase_client.upsert") as up, \
                    mock.patch("lib.notify.push", return_value=True):
                self.assertTrue(notify.once("k", "本文"))
        table, rows = up.call_args.args
        self.assertEqual(table, "notify_log")
        self.assertEqual(rows[0]["dedupe_key"], "k")

    def test_once_sends_when_the_dedup_lookup_fails(self):
        """送信済みか分からないときは鳴らす側に倒す（沈黙のほうが危険）。"""
        with mock.patch.dict(os.environ, CREDS, clear=True):
            with mock.patch("lib.supabase_client.is_configured", return_value=True), \
                    mock.patch("lib.supabase_client.select_one",
                               side_effect=Exception("boom")), \
                    mock.patch("lib.supabase_client.upsert"), \
                    mock.patch("lib.notify.push", return_value=True) as push:
                self.assertTrue(notify.once("k", "本文"))
                push.assert_called_once()

    def test_once_does_not_record_when_the_push_failed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("lib.supabase_client.is_configured", return_value=False), \
                    mock.patch("lib.supabase_client.upsert") as up:
                self.assertFalse(notify.once("k", "本文"))
                up.assert_not_called()

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
            "articles": 18}

    def test_healthy_day_has_no_problem(self):
        self.assertEqual(hb.judge(self.BASE), [])

    def test_flags_zero_articles_when_material_exists(self):
        problems = hb.judge({**self.BASE, "articles": 0})
        self.assertTrue(any("ブログ記事が0件" in p for p in problems))

    def test_quiet_day_without_material_is_not_flagged(self):
        """開示が無い日（祝日等）は記事0件でも異常ではない。
        「静かな日」はEDINET側も0件であることまで確認する（DBだけ0件なら保存の故障）。"""
        problems = hb.judge({**self.BASE, "edinet_api": 0, "holdings": 0, "buybacks": 0,
                             "articles": 0})
        self.assertFalse(any("ブログ記事" in p for p in problems))

    def test_unknown_counts_are_not_flagged(self):
        """Supabase/microCMSが引けなかった項目(-1)で誤報を出さない。"""
        self.assertEqual(hb.judge({**self.BASE, "articles": -1}), [])

    def test_message_marks_problem_and_lists_counts(self):
        msg = hb.build_message(self.BASE, hb.judge({**self.BASE, "articles": 0}))
        self.assertTrue(msg.startswith("🚨"))
        self.assertIn("ブログ18件", hb.build_message(self.BASE, []))

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
        始まったばかりの当日を見ると「記事0件」で毎朝誤報になる。"""
        run_at = datetime(2026, 8, 28, 7, 40, tzinfo=hb.JST)
        self.assertEqual(hb.target_date(now=run_at), "2026-08-27")

    def test_explicit_date_wins(self):
        self.assertEqual(hb.target_date("2026-08-24",
                                        now=datetime(2026, 8, 28, 7, 40, tzinfo=hb.JST)),
                         "2026-08-24")

    def test_counts_only_blog_articles_and_material(self):
        """X投稿・動画は2026-08-30に定期実行を止めたので数えない（毎日0件で誤報になる）。"""
        with mock.patch.object(hb, "count_edinet_disclosures", return_value=0), \
                mock.patch.object(hb, "count_blog_articles", return_value=3), \
                mock.patch.object(hb.sb, "select", return_value=[]) as sel:
            counts = hb.collect("2026-08-27")
        tables = {c.args[0] for c in sel.call_args_list}
        self.assertEqual(tables, {"edinet_large_holdings", "tdnet_buybacks"})
        self.assertNotIn("x_posts", counts)
        self.assertNotIn("videos", counts)

    def test_flags_db_zero_while_edinet_has_disclosures(self):
        """保存が壊れてDBだけ0件になった日を名指しで検知する（2026-08-26〜27の実例）。"""
        problems = hb.judge({**self.BASE, "holdings": 0, "articles": 0})
        self.assertTrue(any("DBは0件" in p for p in problems), problems)

    def test_material_is_counted_from_edinet_not_db(self):
        """DBが0件でもEDINETに開示があれば「記事0件」を異常として拾う。
        素材をDBから数えていたため、保存が壊れた日が静かな日と区別できていなかった。"""
        problems = hb.judge({**self.BASE, "holdings": 0, "buybacks": 0, "articles": 0})
        self.assertTrue(any("ブログ記事が0件" in p for p in problems), problems)

    def test_edinet_unknown_falls_back_to_db_count(self):
        """EDINETを引けなかった(-1)ときは従来どおりDBの件数で判定し、誤報も出さない。"""
        self.assertEqual(hb.judge({**self.BASE, "edinet_api": -1}), [])
        problems = hb.judge({**self.BASE, "edinet_api": -1, "holdings": 0,
                             "buybacks": 0, "articles": 0})
        self.assertEqual(problems, [])

    def test_message_shows_both_db_and_edinet_counts(self):
        msg = hb.build_message({**self.BASE, "holdings": 0}, [])
        self.assertIn("大量保有0件（EDINET 39件）", msg)


class PushOnceTest(unittest.TestCase):
    """同じ原因の通知を連投しない（P1）。毎時13便が同じ理由で落ちても1日1通に収める。"""

    def setUp(self):
        self.now = notify.datetime.now(notify.timezone.utc)

    def _row(self, hours_ago: float, count: int = 1) -> dict:
        sent = self.now - notify.timedelta(hours=hours_ago)
        return {"dedupe_key": "k", "last_sent_at": sent.isoformat(), "sent_count": count}

    def test_suppresses_within_window(self):
        with mock.patch.dict(os.environ, CREDS), \
             mock.patch("lib.supabase_client.select_one", return_value=self._row(2)), \
             mock.patch("lib.supabase_client.upsert") as up, \
             mock.patch("lib.notify.push") as push:
            self.assertFalse(notify.push_once("k", "本文", window_hours=20))
            push.assert_not_called()
        # 抑制した回数も数える（何便ぶん黙ったかを後から追えるように）
        self.assertEqual(up.call_args[0][1][0]["sent_count"], 2)

    def test_sends_after_window_expires(self):
        with mock.patch.dict(os.environ, CREDS), \
             mock.patch("lib.supabase_client.select_one", return_value=self._row(21)), \
             mock.patch("lib.supabase_client.upsert"), \
             mock.patch("lib.notify.push", return_value=True) as push:
            self.assertTrue(notify.push_once("k", "本文", window_hours=20))
            push.assert_called_once()

    def test_sends_when_never_sent(self):
        with mock.patch.dict(os.environ, CREDS), \
             mock.patch("lib.supabase_client.select_one", return_value=None), \
             mock.patch("lib.supabase_client.upsert"), \
             mock.patch("lib.notify.push", return_value=True) as push:
            self.assertTrue(notify.push_once("k", "本文"))
            push.assert_called_once()

    def test_sends_when_dedupe_lookup_fails(self):
        """重複判定ができないときは送る側に倒す（通知は多いより来ないほうが致命的）。"""
        with mock.patch.dict(os.environ, CREDS), \
             mock.patch("lib.supabase_client.select_one", side_effect=RuntimeError("db down")), \
             mock.patch("lib.supabase_client.upsert"), \
             mock.patch("lib.notify.push", return_value=True) as push:
            self.assertTrue(notify.push_once("k", "本文"))
            push.assert_called_once()

    def test_error_without_dedupe_key_always_pushes(self):
        """既存の呼び出し（dedupe_key無し）は従来どおり毎回送る。"""
        with mock.patch.dict(os.environ, CREDS), \
             mock.patch("lib.notify.push", return_value=True) as push, \
             mock.patch("lib.notify.push_once") as once:
            notify.error("どこか", "何か")
            push.assert_called_once()
            once.assert_not_called()


class UsageLimitMessageTest(unittest.TestCase):
    """通知本文に原因と復旧条件を載せる（P2）。"""

    ERR = ("Error code: 400 - {'message': 'You have reached your specified API usage limits. "
           "You will regain access on 2026-09-01 at 00:00 UTC.'}")

    def test_extracts_regain_access_datetime(self):
        self.assertEqual(api_budget.regain_access_at(Exception(self.ERR)),
                         "2026-09-01 00:00 UTC")
        self.assertEqual(api_budget.regain_access_at(Exception("usage limit")), "")

    def test_message_tells_how_to_fix_and_that_credits_do_not_help(self):
        msg = api_budget._build_message(Exception(self.ERR))
        # チャージでの復旧を試みて空振りした実例（2026-08-24）への対処
        self.assertIn("クレジットの追加では解除されません", msg)
        self.assertIn("使用上限", msg)
        self.assertIn("console.anthropic.com/settings/limits", msg)
        self.assertIn("2026-09-01 00:00 UTC", msg)

    def test_message_omits_recovery_line_when_date_unknown(self):
        msg = api_budget._build_message(Exception("api usage limits reached"))
        self.assertNotIn("自動復旧", msg)

    def test_notifies_with_dedupe_key(self):
        api_budget.reset()
        with mock.patch("lib.notify.error", return_value=True) as err:
            api_budget.note(Exception(self.ERR))
        self.assertEqual(err.call_args.kwargs["dedupe_key"], api_budget.DEDUPE_KEY)
        api_budget.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)

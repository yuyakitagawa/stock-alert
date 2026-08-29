"""lib/api_usage.py のユニットテスト。

実行: python3 tests/test_api_usage.py
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import api_budget  # noqa: E402
from lib import api_usage  # noqa: E402


def _resp(model="claude-haiku-4-5-20251001", **usage):
    counts = dict(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0,
                  cache_read_input_tokens=0)
    searches = usage.pop("web_search_requests", 0)
    counts.update(usage)
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            server_tool_use=SimpleNamespace(web_search_requests=searches), **counts),
    )


class ApiUsageTest(unittest.TestCase):
    def setUp(self):
        api_usage.reset()
        api_budget.reset()
        # 記録の単体テストでは日次上限を切る。切らないと record() の中の判定が
        # Supabaseを見に行き、閾値を跨いだテストだけバッファをflushして落ちる。
        self._env = mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "0"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        api_usage.reset()
        api_budget.reset()

    def test_records_tokens_and_cost(self):
        api_usage.record(_resp(input_tokens=1_000_000, output_tokens=200_000),
                         task="blog_body")
        (row,) = api_usage.pending()
        self.assertEqual(row["task"], "blog_body")
        self.assertEqual(row["calls"], 1)
        self.assertEqual(row["input_tokens"], 1_000_000)
        # Haiku 4.5: 入力 $1.00/1M + 出力 $5.00/1M × 0.2M = $2.00
        self.assertAlmostEqual(row["cost_usd"], 2.00, places=6)

    def test_web_search_is_billed_per_request(self):
        """Web検索は $10/1,000検索。トークンと別建てで数えないと過少報告になる。"""
        api_usage.record(_resp(input_tokens=12_000, web_search_requests=2), task="desc")
        (row,) = api_usage.pending()
        self.assertEqual(row["web_search_requests"], 2)
        self.assertAlmostEqual(row["cost_usd"], 0.02 + 12_000 / 1_000_000, places=6)

    def test_cache_tokens_use_write_and_read_rates(self):
        api_usage.record(_resp(cache_creation_input_tokens=1_000_000,
                               cache_read_input_tokens=1_000_000), task="x")
        (row,) = api_usage.pending()
        self.assertAlmostEqual(row["cost_usd"], 1.25 + 0.10, places=6)

    def test_same_task_and_model_are_aggregated(self):
        for _ in range(3):
            api_usage.record(_resp(input_tokens=1000, output_tokens=100), task="blog_body")
        (row,) = api_usage.pending()
        self.assertEqual(row["calls"], 3)
        self.assertEqual(row["input_tokens"], 3000)

    def test_different_tasks_are_separate_rows(self):
        api_usage.record(_resp(input_tokens=10), task="a")
        api_usage.record(_resp(input_tokens=10), task="b")
        self.assertEqual({r["task"] for r in api_usage.pending()}, {"a", "b"})

    def test_dated_model_id_matches_price_table(self):
        """"claude-haiku-4-5-20251001" でも単価が引けること（引けないと$0で記録される）。"""
        self.assertGreater(
            api_usage.estimate_cost("claude-haiku-4-5-20251001", input_tokens=1_000_000), 0)

    def test_unknown_model_still_counts_web_search(self):
        cost = api_usage.estimate_cost("claude-future-9", input_tokens=1_000_000,
                                       web_search_requests=1)
        self.assertAlmostEqual(cost, 0.01, places=6)

    def test_mock_response_is_ignored(self):
        """テストのMockクライアントを渡しても落ちず、ゴミ行も作らないこと。"""
        api_usage.record(mock.Mock(), task="mocked")
        api_usage.record(SimpleNamespace(model="m"), task="no_usage")
        self.assertEqual(api_usage.pending(), [])

    def test_flush_writes_and_clears_buffer(self):
        api_usage.record(_resp(input_tokens=1000), task="blog_body")
        with mock.patch("lib.supabase_client.upsert", return_value=True) as up, \
                mock.patch("lib.api_usage.check_budget"):
            self.assertTrue(api_usage.flush())
        table, rows = up.call_args.args
        self.assertEqual(table, "api_usage")
        self.assertEqual(rows[0]["task"], "blog_body")
        self.assertEqual(api_usage.pending(), [])

    def test_flush_failure_does_not_raise(self):
        api_usage.record(_resp(input_tokens=1000), task="blog_body")
        with mock.patch("lib.supabase_client.upsert", side_effect=Exception("boom")), \
                mock.patch("lib.api_usage.check_budget"):
            self.assertFalse(api_usage.flush())
        self.assertEqual(api_usage.pending(), [])

    def test_flush_with_nothing_buffered_is_a_noop(self):
        with mock.patch("lib.supabase_client.upsert") as up:
            self.assertTrue(api_usage.flush())
        up.assert_not_called()


class DailyCapTest(unittest.TestCase):
    """日次上限。月次上限に当たって1ヶ月止まる前に、1日ぶんで打ち切る。"""

    def setUp(self):
        api_usage.reset()
        api_budget.reset()

    def tearDown(self):
        api_usage.reset()
        api_budget.reset()

    def test_env_var_overrides_default_daily_budget(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "3"}):
            self.assertEqual(api_usage.daily_budget_usd(), 3.0)
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "x"}):
            self.assertEqual(api_usage.daily_budget_usd(), api_usage.DEFAULT_DAILY_BUDGET_USD)

    def test_zero_budget_never_stops(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "0"}), \
                mock.patch("lib.api_usage.day_usage", return_value=99.0):
            self.assertFalse(api_usage.check_daily_cap())
        self.assertFalse(api_budget.reached())

    def test_under_budget_keeps_going(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "1.2"}), \
                mock.patch("lib.api_usage.day_usage", return_value=0.5):
            api_usage.record(_resp(input_tokens=1000), task="blog_body")
            self.assertFalse(api_usage.check_daily_cap())
        self.assertFalse(api_budget.reached())

    def test_over_budget_stops_and_notifies_once(self):
        """当日の記録済み＋未送信の合計で判定し、超えたら以降の呼び出しを止める。"""
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "1.2"}), \
                mock.patch("lib.api_usage.day_usage", return_value=1.0), \
                mock.patch("lib.supabase_client.upsert", return_value=True), \
                mock.patch("lib.api_usage.check_budget"), \
                mock.patch("lib.notify.error", return_value=True) as err:
            # 入力0.3M トークン = $0.30。記録済み$1.00と合わせて$1.30で上限超え。
            api_usage.record(_resp(input_tokens=300_000), task="blog_body")
            self.assertTrue(api_budget.reached())
            # 2件目以降は呼ぶ前に弾かれる想定なので、通知は増えない
            self.assertFalse(api_usage.check_daily_cap())
        self.assertEqual(err.call_count, 1)
        self.assertEqual(err.call_args.kwargs["dedupe_key"], api_budget.DAILY_DEDUPE_KEY)

    def test_stop_saves_what_it_measured(self):
        """打ち切り時にflushしないと、次の便が「まだ予算内」と誤認して走ってしまう。"""
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "1.2"}), \
                mock.patch("lib.api_usage.day_usage", return_value=1.0), \
                mock.patch("lib.supabase_client.upsert", return_value=True) as up, \
                mock.patch("lib.api_usage.check_budget"), \
                mock.patch("lib.notify.error", return_value=True):
            api_usage.record(_resp(input_tokens=300_000), task="blog_body")
        up.assert_called_once()
        self.assertEqual(api_usage.pending(), [])

    def test_cap_check_survives_a_db_failure(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_DAILY_BUDGET_USD": "1.2"}), \
                mock.patch("lib.api_usage.day_usage", side_effect=Exception("boom")):
            self.assertFalse(api_usage.check_daily_cap())
        self.assertFalse(api_budget.reached())


class BudgetAlertTest(unittest.TestCase):
    """残枠監視。上限に「到達してから」止める api_budget.py の手前で鳴らす。"""

    def setUp(self):
        api_usage.reset()
        api_budget.reset()

    def tearDown(self):
        api_usage.reset()
        api_budget.reset()

    def test_alert_level_returns_highest_crossed_threshold(self):
        self.assertEqual(api_usage.alert_level(7.4, 15.0), 0)
        self.assertEqual(api_usage.alert_level(7.5, 15.0), 50)
        self.assertEqual(api_usage.alert_level(12.0, 15.0), 80)
        self.assertEqual(api_usage.alert_level(15.0, 15.0), 100)
        self.assertEqual(api_usage.alert_level(99.0, 15.0), 100)

    def test_budget_of_zero_disables_monitoring(self):
        """上限を知らないまま鳴らさない。0は「監視しない」の意思表示。"""
        self.assertEqual(api_usage.alert_level(999.0, 0), 0)
        with mock.patch.dict(os.environ, {"ANTHROPIC_MONTHLY_BUDGET_USD": "0"}):
            with mock.patch("lib.notify.once") as once:
                self.assertEqual(api_usage.check_budget(), 0)
            once.assert_not_called()

    def test_env_var_overrides_default_budget(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_MONTHLY_BUDGET_USD": "40"}):
            self.assertEqual(api_usage.monthly_budget_usd(), 40.0)
        with mock.patch.dict(os.environ, {"ANTHROPIC_MONTHLY_BUDGET_USD": "not-a-number"}):
            self.assertEqual(api_usage.monthly_budget_usd(),
                             api_usage.DEFAULT_MONTHLY_BUDGET_USD)

    def test_month_usage_ignores_other_months(self):
        rows = [
            {"usage_date": "2026-08-29", "task": "blog_body", "cost_usd": 1.5},
            {"usage_date": "2026-08-30", "task": "blog_body", "cost_usd": 0.5},
            {"usage_date": "2026-09-01", "task": "blog_body", "cost_usd": 9.9},
        ]
        with mock.patch("lib.supabase_client.select", return_value=rows):
            total, by_task = api_usage.month_usage("2026-08")
        self.assertAlmostEqual(total, 2.0)
        self.assertEqual(by_task, {"blog_body": 2.0})

    def test_check_budget_notifies_once_with_top_tasks(self):
        month = (12.5, {"company_description": 9.0, "blog_body": 3.5})
        with mock.patch("lib.api_usage.month_usage", return_value=month), \
                mock.patch("lib.notify.once", return_value=True) as once:
            level = api_usage.check_budget()
        self.assertEqual(level, 80)
        key, text = once.call_args.args
        self.assertTrue(key.endswith("_80"))
        self.assertIn("company_description", text)

    def test_check_budget_survives_a_db_failure(self):
        """監視が落ちても本処理（flush直後）を止めない。"""
        with mock.patch("lib.api_usage.month_usage", side_effect=Exception("boom")):
            self.assertEqual(api_usage.check_budget(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=1)

"""lib/api_usage.py のユニットテスト。

実行: python3 tests/test_api_usage.py
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    def tearDown(self):
        api_usage.reset()

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
        with mock.patch("lib.supabase_client.upsert", return_value=True) as up:
            self.assertTrue(api_usage.flush())
        table, rows = up.call_args.args
        self.assertEqual(table, "api_usage")
        self.assertEqual(rows[0]["task"], "blog_body")
        self.assertEqual(api_usage.pending(), [])

    def test_flush_failure_does_not_raise(self):
        api_usage.record(_resp(input_tokens=1000), task="blog_body")
        with mock.patch("lib.supabase_client.upsert", side_effect=Exception("boom")):
            self.assertFalse(api_usage.flush())
        self.assertEqual(api_usage.pending(), [])

    def test_flush_with_nothing_buffered_is_a_noop(self):
        with mock.patch("lib.supabase_client.upsert") as up:
            self.assertTrue(api_usage.flush())
        up.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=1)

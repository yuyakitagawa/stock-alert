"""lib/api_budget.py のユニットテスト。

実行: python3 tests/test_api_budget.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import api_budget  # noqa: E402

# 2026-08-24 の EDINET Blog Hourly が実際に受け取った文言。
REAL_LIMIT_ERROR = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'You have reached your specified API usage limits. "
    "You will regain access on 2026-09-01 at 00:00 UTC.'}, "
    "'request_id': 'req_011CeMZpFthxYoU5TMjVrXpW'}"
)


class ApiBudgetTest(unittest.TestCase):
    def setUp(self):
        api_budget.reset()

    def tearDown(self):
        api_budget.reset()

    def test_detects_real_usage_limit_error(self):
        self.assertTrue(api_budget.is_usage_limit_error(Exception(REAL_LIMIT_ERROR)))

    def test_detects_low_credit_balance(self):
        self.assertTrue(
            api_budget.is_usage_limit_error(Exception("Your credit balance is too low"))
        )

    def test_ignores_transient_failures(self):
        for msg in ("Error code: 529 - overloaded_error",
                    "Connection error",
                    "Error code: 429 - rate_limit_error",
                    "Error code: 500 - api_error"):
            self.assertFalse(api_budget.is_usage_limit_error(Exception(msg)), msg)

    def test_rate_limit_is_not_a_usage_limit(self):
        """429は待てば回復する。ここで打ち切るとSDKのリトライを殺してしまう。"""
        api_budget.note(Exception("Error code: 429 - rate_limit_error"))
        self.assertFalse(api_budget.reached())

    def test_note_latches_the_flag(self):
        self.assertFalse(api_budget.reached())
        self.assertTrue(api_budget.note(Exception(REAL_LIMIT_ERROR)))
        self.assertTrue(api_budget.reached())
        # 後続で通常の失敗が起きてもフラグは戻らない
        api_budget.note(Exception("Connection error"))
        self.assertTrue(api_budget.reached())

    def test_reset_clears_the_flag(self):
        api_budget.note(Exception(REAL_LIMIT_ERROR))
        api_budget.reset()
        self.assertFalse(api_budget.reached())


if __name__ == "__main__":
    unittest.main(verbosity=1)

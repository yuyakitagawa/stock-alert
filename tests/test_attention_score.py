import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from lib.attention_score import compute_attention_score, _bin_score, _percentile_score, _stars


class TestBinScore(unittest.TestCase):
    def test_picks_correct_bin(self):
        edges = [-float("inf"), 1.99, 5.06, float("inf")]
        means = [10.0, 20.0, 30.0]
        self.assertEqual(_bin_score(0.5, edges, means), 10.0)
        self.assertEqual(_bin_score(1.99, edges, means), 10.0)
        self.assertEqual(_bin_score(3.0, edges, means), 20.0)
        self.assertEqual(_bin_score(100.0, edges, means), 30.0)


class TestPercentileScore(unittest.TestCase):
    def test_clips_to_0_100(self):
        self.assertEqual(_percentile_score(-1.0), 0)
        self.assertEqual(_percentile_score(1.0), 100)

    def test_monotonic_increasing(self):
        prev = -1
        for x in [-0.05, -0.01, 0.0, 0.02, 0.05, 0.08, 0.12]:
            score = _percentile_score(x)
            self.assertGreaterEqual(score, prev)
            prev = score


class TestStars(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(_stars(100), 5)
        self.assertEqual(_stars(90), 5)
        self.assertEqual(_stars(89), 4)
        self.assertEqual(_stars(55), 3)
        self.assertEqual(_stars(35), 2)
        self.assertEqual(_stars(0), 1)


class TestComputeAttentionScore(unittest.TestCase):
    def test_returns_expected_shape(self):
        result = compute_attention_score(2.5, 0.5, 45.0, "プライムブローカー")
        self.assertIn("score", result)
        self.assertIn("stars", result)
        self.assertIn("expected_return_pct", result)
        self.assertIn("reasons", result)
        self.assertTrue(0 <= result["score"] <= 100)
        self.assertTrue(1 <= result["stars"] <= 5)
        self.assertIsInstance(result["reasons"], list)

    def test_favorable_profile_scores_high(self):
        # 低保有比率・小幅増加・大口取引・上位カテゴリ = 実績データ上リターンが良かった組み合わせ
        favorable = compute_attention_score(2.5, 0.5, 45.0, "プライムブローカー")
        # 高保有比率・急増・小口・下位カテゴリ
        unfavorable = compute_attention_score(25.0, 8.0, 2.0, "個人")
        self.assertGreater(favorable["score"], unfavorable["score"])

    def test_unknown_category_falls_back_to_overall_mean(self):
        # 未知のカテゴリ名でも例外にならず、実行できること
        result = compute_attention_score(5.0, 1.0, 10.0, "未知のカテゴリ")
        self.assertTrue(0 <= result["score"] <= 100)


if __name__ == "__main__":
    unittest.main()

"""tools/daily_log_review.py のユニットテスト（ネットワーク・ghコマンド不要）。"""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import daily_log_review as dlr  # noqa: E402


def _log_line(step: str, msg: str, ts: str = "2026-08-22T07:00:00.0000000Z") -> str:
    return f"run-pipeline\t{step}\t{ts} {msg}"


class CondenseLogTest(unittest.TestCase):
    def test_keeps_errors_and_titles_drops_noise(self):
        lines = [_log_line("Step 3", f"progress {i}") for i in range(200)]
        lines.insert(100, _log_line("Step 3", "Traceback (most recent call last):"))
        lines.insert(101, _log_line("Step 3", "  ValueError: boom"))
        lines.insert(120, _log_line("Step 5c", "タイトル: トヨタ（7203）、X社が保有比率を5.1%に引き上げ｜大量保有報告書"))
        out = dlr.condense_log("\n".join(lines))
        self.assertIn("Traceback", out)
        self.assertIn("ValueError: boom", out)
        self.assertIn("大量保有報告書", out)
        self.assertIn("行省略", out)
        self.assertNotIn("progress 50", out)       # 中間の平常行は落ちる
        self.assertIn("progress 0", out)           # 先頭は残る
        self.assertIn("progress 199", out)         # 末尾は残る
        self.assertNotIn("2026-08-22T07:00:00", out)  # タイムスタンプは除去

    def test_full_step_is_not_condensed(self):
        lines = [_log_line("Step 5b - マーケットタイミング＆dpウォッチアラート", f"本文 {i}") for i in range(200)]
        out = dlr.condense_log("\n".join(lines))
        self.assertIn("全文", out)
        self.assertIn("本文 100", out)
        self.assertNotIn("行省略", out)

    def test_truncates_total_size_keeping_tail(self):
        raw = "\n".join(_log_line("S", f"ERROR line {i}") for i in range(5000))
        out = dlr.condense_log(raw, max_chars=3000)
        self.assertLessEqual(len(out), 3200)
        self.assertIn("ERROR line 4999", out)
        self.assertIn("文字省略", out)


class FilterRunsTest(unittest.TestCase):
    def test_filters_by_time_and_completion(self):
        now = datetime(2026, 8, 22, 15, 30, tzinfo=timezone.utc)
        since = now - timedelta(hours=24)
        runs = [
            {"createdAt": "2026-08-22T07:05:00Z", "status": "completed"},
            {"createdAt": "2026-08-20T07:05:00Z", "status": "completed"},   # 古い
            {"createdAt": "2026-08-22T15:00:00Z", "status": "in_progress"},  # 未完了
        ]
        self.assertEqual(len(dlr.filter_runs(runs, since)), 1)

    def test_group_runs_by_path_excludes_ci_and_self(self):
        runs = [
            {"path": ".github/workflows/ops.yml", "displayTitle": "Keepalive"},
            {"path": ".github/workflows/ops.yml", "displayTitle": "Watchdog"},
            {"path": ".github/workflows/x_post.yml"},
            {"path": ".github/workflows/ci.yml", "conclusion": "success"},
            {"path": ".github/workflows/ci.yml", "conclusion": "failure"},
            {"path": ".github/workflows/daily_log_review.yml"},
        ]
        grouped = dlr.group_runs(runs)
        self.assertEqual(sorted(grouped), ["ci.yml", "ops.yml", "x_post.yml"])
        self.assertEqual(len(grouped["ops.yml"]), 2)
        self.assertEqual(len(grouped["ci.yml"]), 1)  # CIは失敗分だけ

    def test_summarize_snapshot_human_filter_and_sections(self):
        now = datetime(2026, 8, 22, 15, 30, tzinfo=timezone.utc)
        recent, old = "2026-08-22T10:00:00Z", "2026-08-16T10:00:00Z"
        pv = [{"occurred_at": recent, "path": "/articles/a", "visitor_id": "v1", "ip_address": "1.1.1.1"}] * 3
        pv += [{"occurred_at": recent, "path": "/stocks/7203", "visitor_id": "bot", "ip_address": "9.9.9.9"}] * 150
        pv += [{"occurred_at": old, "path": "/", "visitor_id": "v2", "ip_address": "2.2.2.2"}] * 14
        rows = {
            "x_posts_24h": [{"posted_at": "2026-08-22T09:00:00Z", "kind": "trending", "variant": "A",
                             "body": "🟢 今週の急増\n詳細", "impressions": 0, "likes": 0,
                             "url_link_clicks": 0, "has_media": True}],
            "x_followers": [{"measured_on": "2026-08-22", "followers": 120}, {"measured_on": "2026-08-21", "followers": 118}],
            "pv": pv,
            "rankings": [{"recommend": "💎 買い"}, {"recommend": "💎 買い"}, {"recommend": "⚪ 様子見"}],
        }
        out = dlr.summarize_snapshot(rows, now)
        self.assertIn("直近24h: 3 PV / ユニーク 1（前7日平均 2 PV/日）", out)   # 9.9.9.9(150PV)は機械として除外
        self.assertIn("生PV24h: 153", out)
        self.assertIn("/articles/*=3", out)
        self.assertIn("08-21=118, 08-22=120", out)
        self.assertIn("💎=2", out)
        self.assertIn("今週の急増 / 詳細", out)
        self.assertEqual(dlr.summarize_snapshot({}, now), "（成果物スナップショット: 取得不可）")


class FormatAndSummaryTest(unittest.TestCase):
    def test_format_jobs_marks_and_duration(self):
        jobs = [{"name": "run-pipeline", "conclusion": "success", "steps": [
            {"name": "Step 3", "conclusion": "success",
             "startedAt": "2026-08-22T07:00:00Z", "completedAt": "2026-08-22T07:01:30Z"},
            {"name": "Step 6", "conclusion": "failure"},
        ]}]
        out = dlr.format_jobs(jobs)
        self.assertIn("✅ Step 3 90s", out)
        self.assertIn("❌ Step 6", out)

    def test_extract_line_summary(self):
        md = ("## 健全性サマリー\n✅ all\n\n## 改善提案\n- [高] x\n\n"
              "## LINE要約\n（注釈）\n🚨 daily_alertのStep 6が失敗\n💡 提案: リトライ追加\n")
        self.assertEqual(dlr.extract_line_summary(md), "🚨 daily_alertのStep 6が失敗\n💡 提案: リトライ追加")

    def test_extract_line_summary_fallback_and_cap(self):
        self.assertEqual(dlr.extract_line_summary("no heading"), "no heading")
        long = "## LINE要約\n" + "あ" * 2000
        self.assertEqual(len(dlr.extract_line_summary(long)), dlr._LINE_MAX_CHARS)

    def test_build_line_message(self):
        msg = dlr.build_line_message("2026-08-22", "ok", "https://github.com/x/y/issues/1")
        self.assertTrue(msg.startswith("🧑‍💻 2026-08-22 日次ログレビュー"))
        self.assertIn("全文: https://github.com/x/y/issues/1", msg)

    def test_build_review_input_handles_empty(self):
        text = dlr.build_review_input({"ops.yml": []}, "2026-08-22")
        self.assertIn("直近の実行なし", text)


if __name__ == "__main__":
    unittest.main()

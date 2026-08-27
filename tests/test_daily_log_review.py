"""tools/daily_log_review.py のユニットテスト（ネットワーク・ghコマンド不要）。"""
import os
import sys
import unittest
from unittest import mock
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


class SignalOnlyCondenseTest(unittest.TestCase):
    """同じワークフローの2本目以降に使う圧縮。opus-5への入力の8割が毎時実行の
    同一ログの繰り返しだったため（2026-08-24計測: 248,678文字中198,042文字）、
    重要行だけに落とす。"""

    def test_keeps_only_signal_lines(self):
        lines = [_log_line("Step 3", f"progress {i}") for i in range(200)]
        lines.insert(120, _log_line("Step 3", "タイトル: トヨタ（7203）｜大量保有報告書"))
        out = dlr.condense_log("\n".join(lines), signal_only=True)
        self.assertIn("大量保有報告書", out)
        self.assertNotIn("progress 0", out)    # 先頭の定型行も落とす
        self.assertNotIn("progress 199", out)  # 末尾の定型行も落とす

    def test_returns_empty_when_nothing_notable(self):
        raw = "\n".join(_log_line("Step 3", f"progress {i}") for i in range(200))
        self.assertEqual(dlr.condense_log(raw, signal_only=True), "")

    def test_full_step_pattern_is_also_condensed(self):
        """LINE本文の全文保持は最新1本で足りる。2本目以降まで全文だと効果が消える。"""
        lines = [_log_line("Step 5b - マーケットタイミング", f"本文 {i}") for i in range(200)]
        out = dlr.condense_log("\n".join(lines), signal_only=True)
        self.assertNotIn("全文", out)

    def test_signal_only_is_much_smaller_than_default(self):
        lines = [_log_line("Step 3", f"progress {i}") for i in range(500)]
        lines.insert(10, _log_line("Step 3", "⚠ 記事生成失敗"))
        raw = "\n".join(lines)
        self.assertLess(
            len(dlr.condense_log(raw, signal_only=True)),
            len(dlr.condense_log(raw)) / 5,
        )


class PerRunBudgetTest(unittest.TestCase):
    def test_failure_budget_is_larger_than_success(self):
        """失敗runは原因究明に全文が要るので絞らない。"""
        self.assertGreater(dlr._MAX_CHARS_PER_RUN, dlr._MAX_CHARS_PER_SUCCESS_RUN)
        self.assertGreater(dlr._MAX_CHARS_PER_SUCCESS_RUN, dlr._MAX_CHARS_PER_REPEAT_RUN)


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


def _unknown_step_line(msg: str, ts: str = "2026-08-26T07:00:00.0000000Z") -> str:
    """ステップ名を解決できなかったときの gh run view --log の実際の形式。"""
    return f"run-pipeline\tUNKNOWN STEP\t{ts} {msg}"


class StepSegmentationTest(unittest.TestCase):
    """gh がステップ名を解決できない run（実測: daily_alert の996行すべて）の切り直し。"""

    def _log(self) -> str:
        lines = [_unknown_step_line("##[group]Run actions/checkout@v4")]
        lines += [_unknown_step_line(f"checkout {i}") for i in range(120)]
        lines.append(_unknown_step_line("##[group]Run cd ~/stock-alert && python core/rank_stocks.py"))
        lines += [_unknown_step_line(f"rank {i}") for i in range(120)]
        lines.append(_unknown_step_line("##[group]Run cd ~/stock-alert && python web/market_timing_alert.py"))
        lines.append(_unknown_step_line("--- U249f138... ---"))
        lines.append(_unknown_step_line("🐋 本日のクジラ｜8/26の大量保有報告書"))
        lines.append(_unknown_step_line("[market_timing] LINE送信完了"))
        lines.append(_unknown_step_line("##[group]Run actions/upload-artifact@v4"))
        lines += [_unknown_step_line(f"upload {i}") for i in range(120)]
        return "\n".join(lines)

    def test_splits_by_group_run_marker(self):
        out = dlr.condense_log(self._log())
        self.assertEqual(out.count("### step:"), 4)
        self.assertIn("core/rank_stocks.py", out)

    def test_keeps_line_body_in_the_middle_of_the_run(self):
        # 切り直し前は1 run＝1ステップに潰れ、中盤のLINE本文が必ず落ちていた
        out = dlr.condense_log(self._log())
        self.assertIn("🐋 本日のクジラ", out)
        self.assertIn("・全文)", out)  # market_timing のステップは全文保持される

    def test_full_step_survives_tight_budget(self):
        out = dlr.condense_log(self._log(), max_chars=1500)
        self.assertIn("🐋 本日のクジラ", out)
        self.assertLessEqual(len(out), 1600)

    def test_strip_prefix_removes_bom(self):
        step, msg = dlr._strip_prefix("job\tUNKNOWN STEP\t\ufeff2026-08-26T07:00:00.0000000Z hello")
        self.assertEqual(msg, "hello")


class ReferenceRunTest(unittest.TestCase):
    def test_review_input_marks_out_of_window_run(self):
        runs = {"perf_check.yml": [{
            "databaseId": 1, "event": "schedule", "conclusion": "success",
            "createdAt": "2026-08-24T00:30:43Z", "updatedAt": "2026-08-24T00:31:13Z",
            "url": "https://example.test/1", "_age_days": 3,
            "_jobs_text": "job: perf-check → success", "_log_text": "| `/` | 0.23s |",
        }]}
        text = dlr.build_review_input(runs, "2026-08-27")
        self.assertIn("24時間の対象期間外", text)
        self.assertIn("3日前", text)
        self.assertIn("当日の出来事としては扱わない", text)

    def test_review_input_has_no_note_for_normal_runs(self):
        runs = {"ops.yml": [{
            "databaseId": 2, "event": "schedule", "conclusion": "success",
            "createdAt": "2026-08-27T00:00:00Z", "updatedAt": "2026-08-27T00:01:00Z",
            "url": "https://example.test/2", "_jobs_text": "job: ops → success", "_log_text": "ok",
        }]}
        self.assertNotIn("対象期間外", dlr.build_review_input(runs, "2026-08-27"))

    def test_perf_check_is_a_reference_workflow(self):
        self.assertIn("perf_check.yml", dlr._REFERENCE_WORKFLOWS)


_PREV_REVIEW = """# 2026-08-26 日次ログレビュー

## 健全性サマリー
| ops.yml | ✅ | 正常 |

## PdM所見（KPIと優先順位）
**今週やる3件**
1. x_posts の kind を埋める
2. YouTubeトークン再発行

**やらない事**
- Node 20 対応

## 改善提案
- **[高] 提案: kind を必ず埋める / 対象: x_post.yml / 観点: BE**
- **[中] 提案: リンクを必須化 / 対象: web/x_post_format.py / 観点: UX**

## LINE要約
🚨 まずい
"""


class PrevProposalsTest(unittest.TestCase):
    def test_extracts_todo_and_proposals_only(self):
        out = dlr.extract_prev_proposals(_PREV_REVIEW)
        self.assertIn("2026-08-26", out)
        self.assertIn("x_posts の kind を埋める", out)
        self.assertIn("リンクを必須化", out)
        self.assertNotIn("やらない事", out)       # 対象外の節は持ち込まない
        self.assertNotIn("健全性サマリー", out)
        self.assertNotIn("🚨 まずい", out)        # LINE要約も持ち込まない

    def test_returns_empty_when_no_sections(self):
        self.assertEqual(dlr.extract_prev_proposals("# 2026-08-26\n\n## 健全性サマリー\n✅\n"), "")

    def test_caps_length(self):
        md = _PREV_REVIEW.replace("- **[中] 提案: リンクを必須化", "- " + "あ" * 9000)
        out = dlr.extract_prev_proposals(md)
        self.assertEqual(len(out), dlr._PREV_REVIEW_MAX_CHARS)

    def test_system_prompt_asks_for_disposal_section(self):
        self.assertIn("## 前回提案の消化状況", dlr.SYSTEM_PROMPT)
        self.assertIn("[再掲]", dlr.SYSTEM_PROMPT)


def _ga4_metrics():
    """summarize_ga4に渡す最小の実測形（2026-08-21〜27の実データを縮めたもの）。"""
    return {
        "days": 7, "start": "2026-08-21", "end": "2026-08-27",
        "now": {
            "pv": 651.0, "entrances": 225.0, "internal_per_session": 1.89,
            "groups": {
                "データ/一覧ページ": {"pv": 271.0, "entrances": 22.0, "users": 30.0,
                                "engagement": 1980.0, "bounced": 13.0, "clicks": 257.0},
                "カテゴリページ": {"pv": 4.0, "entrances": 2.0, "users": 2.0,
                             "engagement": 0.0, "bounced": 2.0, "clicks": 0.0},
            },
            "pages": {"/trending": [70.0]}, "clicks": {"/trending": [93.0]},
        },
        "prev": {"pv": 1280.0, "entrances": 216.0, "internal_per_session": 4.94,
                 "groups": {}, "pages": {}, "clicks": {}},
        "labels": {},
    }


class SummarizeGa4Test(unittest.TestCase):
    def test_reports_internal_moves_with_week_over_week(self):
        """回遊の中心指標。前週比が無いと「増えたのか減ったのか」を判定できない。"""
        md = dlr.summarize_ga4(_ga4_metrics())
        self.assertIn("1セッションあたりの内部移動 1.89回", md)
        self.assertIn("-62%", md)

    def test_lists_groups_with_entrance_and_internal_split(self):
        md = dlr.summarize_ga4(_ga4_metrics())
        self.assertIn("| データ/一覧ページ | 271 | 22 | 249 |", md)
        # 直帰100%＝行き止まりのページ種別が表から読めること
        self.assertIn("| カテゴリページ | 4 | 2 | 2 | 100.0%", md)

    def test_says_labels_not_ready_instead_of_empty_section(self):
        """(not set)しか無い時期に空欄を出すと「押されていない」と誤読される。"""
        md = dlr.summarize_ga4(_ga4_metrics())
        self.assertIn("まだ集計できていない", md)

    def test_shows_labels_when_available(self):
        m = _ga4_metrics()
        m["labels"] = {"この日の開示をすべて見る": [12.0], "(not set)": [999.0]}
        md = dlr.summarize_ga4(m)
        self.assertIn("この日の開示をすべて見る", md)
        self.assertNotIn("(not set)", md)

    def test_missing_metrics_explains_what_is_unset(self):
        """取得できない日に黙って節ごと消すと、レビュー側が欠落に気付けない。"""
        md = dlr.summarize_ga4({})
        self.assertIn("取得不可", md)
        self.assertIn("GCP_SERVICE_ACCOUNT_JSON", md)


class FetchGa4MetricsTest(unittest.TestCase):
    def test_returns_empty_without_property_id(self):
        with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": ""}):
            self.assertEqual(dlr.fetch_ga4_metrics(), {})


class Ga4PromptTest(unittest.TestCase):
    def test_system_prompt_requires_a_navigation_section(self):
        self.assertIn("## 回遊所見", dlr.SYSTEM_PROMPT)
        self.assertIn("内部移動回数", dlr.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()

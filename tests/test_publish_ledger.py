"""候補>0・公開0を「正常な見送り」と「異常」に切り分ける台帳（lib/publish_ledger）のテスト。

実行: python3 tests/test_publish_ledger.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib import publish_ledger as pl
from lib.publish_ledger import PublishLedger


def test_all_expected_skips_is_not_an_anomaly():
    """候補が全部「基準未満」「既報」で0件公開なのは仕様どおり。鳴らさない。

    edinet_blog.yml は平日13便回り、ほとんどの便がこの状態になる。ここで通知すると
    毎日鳴って誰も見なくなり、本当に壊れた日の通知が埋もれる。
    """
    led = PublishLedger("test")
    led.start(3)
    led.skip(pl.SKIP_BELOW_THRESHOLD, "A(1111)")
    led.skip(pl.SKIP_ALREADY_PUBLISHED, "B(2222)")
    led.skip(pl.SKIP_NO_RATIO_CHANGE, "C(3333)")
    assert led.has_anomaly() is False
    with mock.patch("lib.publish_ledger.notify.error") as err:
        assert led.finish() == 0
    err.assert_not_called()


def test_generation_failure_is_an_anomaly():
    """本文生成に失敗した候補があれば、公開0件でなくても異常として鳴らす。

    2026-08-24のAnthropic API月次上限はこの形（生成が全滅しても run は success）。
    """
    led = PublishLedger("test")
    led.start(2)
    led.skip(pl.FAIL_GENERATION, "A(1111)")
    led.publish("B(2222)")
    assert led.has_anomaly() is True
    with mock.patch("lib.publish_ledger.notify.error") as err:
        assert led.finish() == pl.EXIT_ANOMALY
    err.assert_called_once()
    assert "記事生成に失敗" in err.call_args[0][1]
    assert "A(1111)" in err.call_args[0][1]


def test_publish_failure_is_an_anomaly():
    """microCMSがcontent_idを返さなかった候補も異常（記事が出ていないため）。"""
    led = PublishLedger("test")
    led.start(1)
    led.skip(pl.FAIL_PUBLISH, "A(1111)")
    assert led.anomaly_counts() == {pl.FAIL_PUBLISH: 1}
    with mock.patch("lib.publish_ledger.notify.error"):
        assert led.finish() == pl.EXIT_ANOMALY


def test_candidate_dropped_without_a_reason_is_an_anomaly():
    """理由を記録しないまま脱落した候補は異常。

    この監視自体が腐らないための保険。将来 build_and_publish に `continue` を足して
    ledger.skip() を書き忘れると、その分がここに落ちて通知される。
    """
    led = PublishLedger("test")
    led.start(5)
    led.publish("A(1111)")
    led.skip(pl.SKIP_BELOW_THRESHOLD, "B(2222)")
    # 残り3件はどこにも記録されないまま消えた
    assert led.unclassified == 3
    assert led.has_anomaly() is True
    assert "理由が記録されないまま脱落" in led.summary()


def test_stopping_early_does_not_count_the_rest_as_unclassified():
    """max_articles で打ち切った残りは「評価していない」だけで脱落ではない。"""
    led = PublishLedger("test")
    led.start(10)
    led.publish("A(1111)")
    led.stop_early(pl.SKIP_MAX_ARTICLES)
    assert led.unclassified == 0
    assert led.has_anomaly() is False


def test_permission_error_stop_is_an_anomaly():
    """microCMSの権限エラーで打ち切ったときは、残りを数えなくても異常のまま。"""
    led = PublishLedger("test")
    led.start(10)
    led.stop_early(pl.FAIL_PERMISSION, "A(1111)")
    assert led.unclassified == 0
    assert led.has_anomaly() is True


def test_summary_shows_the_breakdown():
    """公開0件でも「なぜ0件か」がログ1行で分かる（日次ログレビューが読む）。"""
    led = PublishLedger("publish_blog_articles")
    led.start(4)
    led.skip(pl.SKIP_BELOW_THRESHOLD, "A(1111)")
    led.skip(pl.SKIP_BELOW_THRESHOLD, "B(2222)")
    led.skip(pl.SKIP_NO_RATIO_CHANGE, "C(3333)")
    led.skip(pl.SKIP_ALREADY_PUBLISHED, "D(4444)")
    s = led.summary()
    assert "候補4件 → 公開0件" in s
    assert "基準未満2" in s
    assert "比率変化なし1" in s
    assert "既報1" in s


def test_unknown_reason_is_treated_as_an_anomaly():
    """辞書に無い理由が来たら、安全側（異常）に倒す。"""
    led = PublishLedger("test")
    led.start(1)
    led.skip("something_new", "A(1111)")
    assert led.has_anomaly() is True


def test_notification_is_deduped_by_cause():
    """同じ原因で毎便鳴らさない（毎時13便＝13通を防ぐ）。原因が変われば別キーで鳴らし直す。"""
    led = PublishLedger("publish_blog_articles")
    led.start(1)
    led.skip(pl.FAIL_GENERATION, "A(1111)")
    with mock.patch("lib.publish_ledger.notify.error") as err:
        led.finish()
    key = err.call_args.kwargs["dedupe_key"]
    assert key == "publish_ledger:publish_blog_articles:generation_failed"

    # 同じ原因なら件数が違ってもキーは同じ（窓の内側なら notify 側が抑える）
    again = PublishLedger("publish_blog_articles")
    again.start(2)
    again.skip(pl.FAIL_GENERATION, "B(2222)")
    again.skip(pl.FAIL_GENERATION, "C(3333)")
    assert again.dedupe_key() == key

    # 別の原因が乗ったら鳴らし直す
    other = PublishLedger("publish_blog_articles")
    other.start(1)
    other.skip(pl.FAIL_PUBLISH, "D(4444)")
    assert other.dedupe_key() != key


if __name__ == "__main__":
    test_all_expected_skips_is_not_an_anomaly()
    test_generation_failure_is_an_anomaly()
    test_publish_failure_is_an_anomaly()
    test_candidate_dropped_without_a_reason_is_an_anomaly()
    test_stopping_early_does_not_count_the_rest_as_unclassified()
    test_permission_error_stop_is_an_anomaly()
    test_summary_shows_the_breakdown()
    test_unknown_reason_is_treated_as_an_anomaly()
    test_notification_is_deduped_by_cause()
    print("OK: test_publish_ledger (9 tests)")

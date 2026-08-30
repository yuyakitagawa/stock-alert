"""Supabase REST APIクライアント（lib/supabase_client）のリトライ挙動のユニットテスト。

実行: python3 tests/test_supabase_client.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import lib.supabase_client as sb


class _FakeResponse:
    ok = True

    def json(self):
        return []


def test_request_retries_on_timeout_then_succeeds():
    """一時的なタイムアウトはバックオフして再試行し、最終的に成功すればそれを返す。"""
    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ReadTimeout("boom")
        return _FakeResponse()

    with mock.patch("lib.supabase_client.requests.request", side_effect=fake_request), \
         mock.patch("lib.supabase_client.time.sleep") as fake_sleep:
        resp = sb._request("GET", "https://example.test/x")
    assert isinstance(resp, _FakeResponse)
    assert calls["n"] == 3
    assert fake_sleep.call_count == 2


def test_request_raises_after_max_retries():
    """一時的な失敗が続く場合は_MAX_RETRIES回試したうえで最終的に例外を送出する。"""
    def always_fail(method, url, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    with mock.patch("lib.supabase_client.requests.request", side_effect=always_fail), \
         mock.patch("lib.supabase_client.time.sleep"):
        try:
            sb._request("GET", "https://example.test/x")
            assert False, "例外が発生するはず"
        except requests.exceptions.ConnectionError:
            pass


def test_insert_ignore_does_not_crash_on_persistent_network_failure():
    """DB書き込みが一時的な障害で最終的にも失敗しても、呼び出し元は例外で落ちない
    （数時間かかるバックテスト/日次パイプライン全体を1回のタイムアウトで壊さないため）。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    try:
        with mock.patch("lib.supabase_client.requests.request",
                         side_effect=requests.exceptions.ReadTimeout("boom")), \
             mock.patch("lib.supabase_client.time.sleep"):
            sb.insert_ignore("some_table", [{"a": 1}], on_conflict="a")
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""


def test_upsert_splits_rows_with_different_keys():
    """キー構成が違う行が混ざっても、リクエストごとにキーを揃えて送る。
    PostgRESTは1リクエスト内のキーが不一致だと PGRST102 で400を返し、
    バッチ丸ごと保存されない（2026-08-26〜27にEDINET大量保有の全件が消えた実例）。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    bodies = []

    def capture(method, url, **kwargs):
        bodies.append(kwargs["json"])
        return _FakeResponse()

    try:
        with mock.patch("lib.supabase_client.requests.request", side_effect=capture):
            ok = sb.upsert("some_table", [
                {"doc_id": "a", "ratio": 1.0},
                {"doc_id": "b", "ratio": 2.0, "transfers": [{"price": 100}]},
                {"doc_id": "c", "ratio": 3.0},
            ], on_conflict="doc_id")
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""

    assert ok is True
    assert len(bodies) == 2, bodies
    for body in bodies:
        assert len({tuple(sorted(row.keys())) for row in body}) == 1
    assert sorted(r["doc_id"] for body in bodies for r in body) == ["a", "b", "c"]


def test_insert_ignore_splits_rows_with_different_keys():
    """insert_ignoreも同様にキー構成ごとに分割して送る。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    bodies = []

    def capture(method, url, **kwargs):
        bodies.append(kwargs["json"])
        return _FakeResponse()

    try:
        with mock.patch("lib.supabase_client.requests.request", side_effect=capture):
            sb.insert_ignore("some_table",
                             [{"a": 1}, {"a": 2, "b": 3}], on_conflict="a")
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""

    assert len(bodies) == 2, bodies


def test_write_failure_is_recorded_and_notified_once():
    """書き込み失敗はテーブル別に記録し、LINEは1プロセス1回だけ送る。
    ワークフローの各ステップは continue-on-error で緑のまま進むため、失敗した
    その場から鳴らさないと誰も気づけない（2026-08-26〜27の実例）。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    sb._write_failures.clear()
    sb._notified_tables.clear()

    class _Failing:
        ok = False
        status_code = 400
        text = '{"code":"PGRST102"}'

    try:
        with mock.patch("lib.supabase_client.requests.request", return_value=_Failing()), \
             mock.patch("lib.notify.error") as err:
            ok1 = sb.upsert("some_table", [{"a": 1}], on_conflict="a")
            ok2 = sb.upsert("some_table", [{"a": 2}], on_conflict="a")
            assert ok1 is False and ok2 is False
            assert err.call_count == 1, err.call_count
        assert sb.write_failures() == {"some_table": 2}, sb.write_failures()
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""
        sb._write_failures.clear()
        sb._notified_tables.clear()


def test_successful_write_records_no_failure():
    """成功した書き込みは失敗として残らず、通知も出ない。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    sb._write_failures.clear()
    sb._notified_tables.clear()
    try:
        with mock.patch("lib.supabase_client.requests.request", return_value=_FakeResponse()), \
             mock.patch("lib.notify.error") as err:
            assert sb.upsert("some_table", [{"a": 1}], on_conflict="a") is True
            err.assert_not_called()
        assert sb.write_failures() == {}
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""
        sb._write_failures.clear()
        sb._notified_tables.clear()


def test_production_writes_are_blocked_during_tests():
    """テスト実行中は本番プロジェクトへ書かない（読み取りは通す）。

    tests/test_api_usage.py の atexit flush が本番 api_usage へ合成行を書き
    （2026-08-29、job=local / task="x" / $1.35）、その1行で当日合計が日次予算を超えて
    翌営業日の記事生成が全便打ち切られる状態になった。同種の事故の再発検知。"""
    sb.SUPABASE_URL = "https://prod.example"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    prod = sb._ENV_SUPABASE_URL
    sb._ENV_SUPABASE_URL = "https://prod.example"
    try:
        with mock.patch("lib.supabase_client.requests.request",
                        return_value=_FakeResponse()) as req:
            assert sb.upsert("api_usage", [{"a": 1}]) is False
            assert sb.update("api_usage", "id=eq.1", {"a": 1}) is False
            sb.insert_ignore("api_usage", [{"a": 1}])
            sb.delete("api_usage", "id=eq.1")
            assert req.call_count == 0, req.call_args_list
            # 読み取りは止めない（本番データを読むテストは事故ではない）
            sb.select("api_usage", "select=cost_usd")
            assert req.call_count == 1
    finally:
        sb._ENV_SUPABASE_URL = prod
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""


def test_writes_to_a_non_production_url_still_go_through():
    """URLを差し替えているテストの書き込みまで止めない（ガードが広すぎないことの確認）。"""
    sb.SUPABASE_URL = "https://example.test"
    sb.SUPABASE_SERVICE_KEY = "dummy"
    try:
        with mock.patch("lib.supabase_client.requests.request",
                        return_value=_FakeResponse()) as req:
            assert sb.upsert("some_table", [{"a": 1}], on_conflict="a") is True
        assert req.call_count == 1
    finally:
        sb.SUPABASE_URL = ""
        sb.SUPABASE_SERVICE_KEY = ""


if __name__ == "__main__":
    test_request_retries_on_timeout_then_succeeds()
    test_request_raises_after_max_retries()
    test_insert_ignore_does_not_crash_on_persistent_network_failure()
    test_upsert_splits_rows_with_different_keys()
    test_insert_ignore_splits_rows_with_different_keys()
    test_write_failure_is_recorded_and_notified_once()
    test_successful_write_records_no_failure()
    test_production_writes_are_blocked_during_tests()
    test_writes_to_a_non_production_url_still_go_through()
    print("OK: test_supabase_client (9 tests)")

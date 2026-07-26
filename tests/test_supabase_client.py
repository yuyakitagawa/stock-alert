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


if __name__ == "__main__":
    test_request_retries_on_timeout_then_succeeds()
    test_request_raises_after_max_retries()
    test_insert_ignore_does_not_crash_on_persistent_network_failure()
    print("OK: test_supabase_client (3 tests)")

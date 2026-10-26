"""TDnet 適時開示の取得（lib/tdnet.py）のユニットテスト。ネットワークは使わない。

実行: python3 tests/test_tdnet.py
"""
import os
import sys
from datetime import date
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.tdnet as m


def test_fetch_days_requests_each_day_separately():
    """期間をまとめず1日ずつ取る（まとめると件数の上限で古い日が落ちる）。"""
    with mock.patch.object(m, "fetch_disclosures", side_effect=lambda cond, limit: [{"cond": cond}]) as f:
        out = m.fetch_days(date(2026, 8, 5), date(2026, 8, 7), sleep_sec=0)
    assert [c.args[0] for c in f.call_args_list] == ["20260805-20260805", "20260806-20260806", "20260807-20260807"]
    assert all(c.kwargs["limit"] == m.DAY_LIMIT for c in f.call_args_list)
    assert len(out) == 3


def test_scan_disclosures_uses_per_day_fetch():
    with mock.patch.object(m, "fetch_days", return_value=[{"category": "決算"}, {"category": None}]) as f:
        out = m.scan_disclosures(days_back=3, persist=False, sleep_sec=0, only_categorized=True)
    start, end = f.call_args.args
    assert (end - start).days == 3 and out == [{"category": "決算"}]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")

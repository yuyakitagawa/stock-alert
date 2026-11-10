"""lib/price_store.py の差分同期のユニットテスト。

実行: python3 tests/test_price_store.py  /  pytest tests/test_price_store.py
"""
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.price_store as ps


def test_resync_since():
    assert ps.resync_since("2026-11-09") == "2026-11-02"


def test_sync_picks_up_rows_filled_later():
    """当日分が一部だけ入った日の残りが、後の便で埋まったら手元にも入る（2026-11-06: 1,073→3,746銘柄）。"""
    d = np.array(["2026-11-05", "2026-11-06"], dtype="datetime64[D]")
    store = {"built_at": date.today().isoformat(), "max_date": "2026-11-06",
             "prices": {"1000": (d, np.array([100.0, 101.0]), np.array([1.0, 1.0])),
                        "2000": (d[:1], np.array([50.0]), np.array([1.0]))}}   # 2000 は11/6が欠けていた
    queries = []

    def fake_fetch(q):
        queries.append(q)
        return [{"code": "2000", "date": "2026-11-05", "close": 50.0, "volume": 1.0},
                {"code": "2000", "date": "2026-11-06", "close": 49.0, "volume": 2.0},
                {"code": "1000", "date": "2026-11-09", "close": 102.0, "volume": 1.0}]

    orig = ps._fetch, ps.rpc_codes, ps._write_disk
    ps._fetch, ps.rpc_codes, ps._write_disk = fake_fetch, (lambda: ["1000", "2000"]), (lambda s: None)
    try:
        out = ps._sync(store)
    finally:
        ps._fetch, ps.rpc_codes, ps._write_disk = orig
    assert queries[0].startswith("date=gte.2026-10-30&")
    dates, closes, _ = out["prices"]["2000"]
    assert [str(x) for x in dates] == ["2026-11-05", "2026-11-06"] and closes[-1] == 49.0
    assert out["max_date"] == "2026-11-09"


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
    print(f"OK: {len(fns)} tests")

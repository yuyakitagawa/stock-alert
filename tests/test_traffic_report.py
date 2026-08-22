"""tools/traffic_report.py のユニットテスト（ネットワークは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import traffic_report as t  # noqa: E402


def _row(ip, path="/", hour_utc=0):
    return {"occurred_at": f"2026-08-20T{hour_utc:02d}:00:00+00:00",
            "path": path, "ip_address": ip}


def test_heavy_ips_uses_threshold():
    rows = [_row("1.1.1.1") for _ in range(5)] + [_row("2.2.2.2")]
    assert t.heavy_ips(rows, max_pv=3) == {"1.1.1.1"}
    assert t.heavy_ips(rows, max_pv=10) == set()


def test_null_ip_is_grouped_not_dropped():
    """ip_addressがNULLの行を落とすと全体PVが合わなくなる。1つのIP扱いでまとめる。"""
    rows = [_row(None) for _ in range(4)] + [_row("2.2.2.2")]
    assert t.heavy_ips(rows, max_pv=3) == {"(null)"}
    human, machine = t.split(rows, max_pv=3)
    assert len(human) + len(machine) == len(rows)
    assert len(machine) == 4


def test_split_keeps_every_row():
    rows = [_row("1.1.1.1") for _ in range(9)] + [_row("3.3.3.3"), _row(None)]
    human, machine = t.split(rows, max_pv=5)
    assert len(human) + len(machine) == len(rows)
    assert {r["ip_address"] for r in machine} == {"1.1.1.1"}


def test_hour_histogram_converts_to_jst():
    """UTC 0時 = JST 9時。時間帯の波を見る指標なので変換を間違えると結論が反転する。"""
    assert t.hour_histogram([_row("1.1.1.1", hour_utc=0)]) == {9: 1}
    assert t.hour_histogram([_row("1.1.1.1", hour_utc=20)]) == {5: 1}


def test_hour_histogram_accepts_trailing_z():
    rows = [{"occurred_at": "2026-08-20T00:00:00Z", "path": "/", "ip_address": "1.1.1.1"}]
    assert t.hour_histogram(rows) == {9: 1}


def test_article_rate():
    rows = [_row("1.1.1.1", "/articles/abc"), _row("1.1.1.1", "/"),
            _row("1.1.1.1", "/weekly"), _row("1.1.1.1", "/articles/def")]
    assert t.article_rate(rows) == 50.0
    assert t.article_rate([]) == 0.0


def test_report_excludes_heavy_ip_from_summary():
    rows = [_row("9.9.9.9", "/") for _ in range(50)] + [
        _row("1.1.1.1", "/articles/a"), _row("2.2.2.2", "/weekly")]
    buf = io.StringIO()
    with mock.patch.object(t.sb, "is_configured", return_value=True), \
         mock.patch.object(t, "fetch_rows", return_value=rows), redirect_stdout(buf):
        rc = t.report(days=14, max_pv=10)
    out = buf.getvalue()
    assert rc == 0, rc
    assert "52PV" in out, out          # 全体
    assert "1IP / 50PV" in out, out    # 除外された大量アクセス
    assert "2IP / 2PV" in out, out     # 残り


def test_report_without_rows():
    buf = io.StringIO()
    with mock.patch.object(t.sb, "is_configured", return_value=True), \
         mock.patch.object(t, "fetch_rows", return_value=[]), redirect_stdout(buf):
        rc = t.report(days=14, max_pv=100)
    assert rc == 0
    assert "アクセスがありません" in buf.getvalue()


def test_report_requires_supabase():
    buf = io.StringIO()
    with mock.patch.object(t.sb, "is_configured", return_value=False), redirect_stdout(buf):
        assert t.report(days=14, max_pv=100) == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'FAILED' if fails else 'PASSED'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)

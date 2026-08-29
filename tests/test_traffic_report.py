"""tools/traffic_report.py のユニットテスト（ネットワークは全てモック）。"""
import io
import os
import re
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import traffic_report as t  # noqa: E402

HUMAN_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/145.0.0.0 Safari/537.36"
META_UA = (HUMAN_UA + " (compatible; meta-externalagent/1.1 "
           "(+https://developers.facebook.com/docs/sharing/webmasters/crawler))")


def _row(ip, path="/", hour_utc=0, ua=HUMAN_UA, vid="v1"):
    return {"occurred_at": f"2026-08-20T{hour_utc:02d}:00:00+00:00",
            "path": path, "ip_address": ip, "user_agent": ua, "visitor_id": vid}


def _norm(text):
    return re.sub(r"\s+", " ", text)


def test_heavy_ips_uses_threshold():
    rows = [_row("1.1.1.1") for _ in range(5)] + [_row("2.2.2.2")]
    assert t.heavy_ips(rows, max_pv=3) == {"1.1.1.1"}
    assert t.heavy_ips(rows, max_pv=10) == set()


def test_null_ip_is_grouped_not_dropped():
    """ip_addressがNULLの行を落とすと全体PVが合わなくなる。1つのIP扱いでまとめる。"""
    rows = [_row(None) for _ in range(4)] + [_row("2.2.2.2")]
    assert t.heavy_ips(rows, max_pv=3) == {"(null)"}
    buckets = t.classify(rows, max_pv=3)
    assert len(buckets["heavy_ip"]) == 4
    assert len(buckets["human"]) == 1


def test_is_bot_ua_catches_undeclared_crawlers():
    """crawlers.tsのBOT_PATTERNSに載っていない新顔をUAの自己申告で拾う。"""
    assert t.is_bot_ua(META_UA)
    assert t.is_bot_ua("Mozilla/5.0 (compatible; Amazonbot/0.1; +https://example.com)")
    assert t.is_bot_ua("Mozilla/5.0 ... (compatible; AdsBot-Google-Mobile)")
    assert not t.is_bot_ua(HUMAN_UA)
    assert not t.is_bot_ua(None)


def test_cookieless_uas_needs_both_ratio_and_volume():
    """visitor_idがPVと1:1でもPVが少ないうちは「1ページで離脱した人間」と区別できない。"""
    solo = [_row("1.1.1.%d" % i, vid=f"v{i}") for i in range(20)]
    assert t.cookieless_uas(solo) == {HUMAN_UA}
    assert t.cookieless_uas(solo[:19]) == set()          # 母数不足なら判定しない
    shared = [_row("1.1.1.%d" % i, vid=f"v{i % 2}") for i in range(20)]
    assert t.cookieless_uas(shared) == set()             # クッキーを持ち回る＝人間側


def test_cookieless_uas_counts_null_visitor_id_as_unique():
    rows = [_row("1.1.1.%d" % i, vid=None) for i in range(20)]
    assert t.cookieless_uas(rows) == {HUMAN_UA}


def test_classify_assigns_every_row_exactly_once():
    rows = ([_row("::1")] + [_row("2.2.2.2", ua=META_UA)] * 3
            + [_row("3.3.3.3") for _ in range(9)]
            + [_row("4.4.4.4", vid="vx")])
    buckets = t.classify(rows, max_pv=5)
    assert sum(len(v) for v in buckets.values()) == len(rows)
    assert len(buckets["self"]) == 1
    assert len(buckets["bot_ua"]) == 3
    assert len(buckets["heavy_ip"]) == 9
    assert len(buckets["human"]) == 1


def test_classify_does_not_let_bot_pv_make_a_shared_ip_heavy():
    """同じIPからbotが大量に来ていても、そのIPの人間の行まで機械にしない。"""
    rows = [_row("5.5.5.5", ua=META_UA) for _ in range(50)] + [_row("5.5.5.5")]
    buckets = t.classify(rows, max_pv=10)
    assert len(buckets["bot_ua"]) == 50
    assert len(buckets["heavy_ip"]) == 0
    assert len(buckets["human"]) == 1


def test_classify_separates_cookieless_ua_group():
    rows = [_row("6.6.6.%d" % i, vid=f"v{i}") for i in range(20)]
    buckets = t.classify(rows, max_pv=100)
    assert len(buckets["cookieless_ua"]) == 20
    assert buckets["human"] == []


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


def test_repeat_visitors():
    rows = [_row("1.1.1.1", vid="a"), _row("1.1.1.1", vid="a"),
            _row("2.2.2.2", vid="b"), _row("3.3.3.3", vid=None)]
    assert t.repeat_visitors(rows) == (1, 2, 2)


def test_report_excludes_machines_from_summary():
    rows = ([_row("9.9.9.9", "/") for _ in range(50)]
            + [_row("8.8.8.8", "/", ua=META_UA)]
            + [_row("1.1.1.1", "/articles/a", vid="a"), _row("2.2.2.2", "/weekly", vid="b")])
    buf = io.StringIO()
    with mock.patch.object(t.sb, "is_configured", return_value=True), \
         mock.patch.object(t, "fetch_rows", return_value=rows), redirect_stdout(buf):
        rc = t.report(days=14, max_pv=10)
    out = _norm(buf.getvalue())
    assert rc == 0, rc
    assert "53PV" in out, out                    # 全体
    assert "1IP / 50PV" in out, out              # 1IPで10PV超
    assert "1IP / 1PV" in out, out               # UA自己申告
    assert "2IP / 2PV" in out, out               # 残り
    assert "meta-externalagent/1.1" in out, out  # 追加すべき新顔を名指しする


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

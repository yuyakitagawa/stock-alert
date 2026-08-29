"""tools/gsc_report.py のユニットテスト（Search Console APIは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import gsc_report as g  # noqa: E402


def _row(key, clicks, impressions, position):
    return {"keys": [key], "clicks": clicks, "impressions": impressions,
            "ctr": clicks / impressions if impressions else 0.0, "position": position}


def test_page_group_splits_hub_pages_from_others():
    assert g.page_group("https://kujira-watch.com/") == "TOP"
    assert g.page_group("https://kujira-watch.com/articles/abc") == "記事"
    assert g.page_group("https://kujira-watch.com/stocks/6976") == "銘柄ページ"
    assert g.page_group("https://kujira-watch.com/ranking/buys") == "ランキング"
    assert g.page_group("https://kujira-watch.com/contact") == "その他"


def test_totals_weights_ctr_and_position_by_impressions():
    """行ごとの単純平均だと表示1回のクエリが上位ページと同じ重みになる。"""
    t = g.totals([_row("a", 10, 100, 3.0), _row("b", 0, 900, 30.0)])
    assert t["clicks"] == 10 and t["impressions"] == 1000
    assert round(t["ctr"], 4) == 0.01
    assert round(t["position"], 2) == 27.3


def test_totals_of_empty_rows_does_not_divide_by_zero():
    assert g.totals([]) == {"clicks": 0, "impressions": 0, "ctr": 0.0, "position": 0.0}


def test_group_totals_buckets_urls_by_page_type():
    got = g.group_totals([_row("https://x/articles/a", 1, 10, 5.0),
                          _row("https://x/articles/b", 2, 10, 5.0),
                          _row("https://x/stocks/1234", 0, 5, 40.0)])
    assert got["記事"]["clicks"] == 3 and got["記事"]["impressions"] == 20
    assert got["銘柄ページ"]["clicks"] == 0


def test_ctr_opportunities_keeps_only_first_page_underperformers():
    rows = [
        _row("拾う: 3位なのにCTR低い", 2, 400, 3.0),
        _row("除外: 表示が少なすぎる", 0, 5, 2.0),
        _row("除外: 11位で1ページ目に居ない", 1, 500, 11.0),
        _row("除外: CTRがサイト平均以上", 50, 200, 4.0),
    ]
    got = g.ctr_opportunities(rows, site_ctr=0.05)
    assert [r["keys"][0] for r in got] == ["拾う: 3位なのにCTR低い"]


def test_ctr_opportunities_sorted_by_lost_clicks():
    """並び順は「サイト平均CTRなら何クリック増えたか」。表示回数だけで並べると
    順位が良くて元々CTRの高いクエリが上に来てしまう。"""
    small_gap = _row("表示は多いが取りこぼしは小", 18, 400, 5.0)   # 0.045 → 差0.005 × 400 = 2
    big_gap = _row("表示は少ないが取りこぼしは大", 0, 100, 5.0)     # 0.000 → 差0.05 × 100 = 5
    got = g.ctr_opportunities([small_gap, big_gap], site_ctr=0.05)
    assert [r["keys"][0] for r in got] == ["表示は少ないが取りこぼしは大", "表示は多いが取りこぼしは小"]


def test_almost_first_page_takes_ranks_11_to_20_only():
    rows = [_row("1ページ目", 5, 100, 9.9), _row("あと一歩", 0, 100, 12.0),
            _row("圏外", 0, 100, 25.0), _row("表示不足", 0, 3, 12.0)]
    assert [r["keys"][0] for r in g.almost_first_page(rows)] == ["あと一歩"]


def test_delta_marks_new_instead_of_infinite_percent():
    assert g.delta(5, 0) == "  (新規)"
    assert g.delta(0, 0) == ""
    assert g.delta(150, 100) == "  (+50%)"


def test_explain_error_service_disabled_points_to_enable_url():
    msg = g.explain_error(403, {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}, "sc-domain:x")
    assert "searchconsole.googleapis.com" in msg and "有効化" in msg


def test_explain_error_403_points_to_user_management_with_service_account():
    with mock.patch.object(g.gcp_auth, "client_email", return_value="bot@example.com"):
        msg = g.explain_error(403, {"error": {"message": "denied"}}, "sc-domain:kujira-watch.com")
    assert "ユーザーと権限" in msg and "bot@example.com" in msg


def test_explain_error_404_points_to_sites_option():
    msg = g.explain_error(404, {"error": {"message": "not found"}}, "sc-domain:x")
    assert "--sites" in msg and "GSC_SITE_URL" in msg


def test_search_analytics_encodes_site_and_caps_row_limit():
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"], captured["url"], captured["body"] = method, url, kwargs.get("json")
        return mock.Mock(status_code=200, json=lambda: {"rows": [_row("q", 1, 2, 3.0)]})

    with mock.patch.object(g.requests, "request", side_effect=fake_request):
        rows, err = g.search_analytics("tok", "sc-domain:kujira-watch.com",
                                       g.date(2026, 8, 1), g.date(2026, 8, 28), ["query"], 999999)
    assert err == "" and len(rows) == 1
    assert "sc-domain%3Akujira-watch.com" in captured["url"]
    assert captured["body"]["rowLimit"] == g.MAX_ROWS
    assert captured["body"]["startDate"] == "2026-08-01"


def test_report_returns_1_and_prints_fix_when_api_fails():
    with mock.patch.object(g.gcp_auth, "access_token", return_value="tok"), \
            mock.patch.object(g, "search_analytics", return_value=([], "権限がありません")):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = g.report(28, 10)
    assert rc == 1 and "権限がありません" in buf.getvalue()


def test_report_prints_all_sections_from_api_rows():
    queries = [_row("大量保有報告書 とは", 30, 500, 4.0), _row("アクティビスト 一覧", 0, 300, 12.0)]
    pages = [_row("https://kujira-watch.com/articles/a", 20, 400, 5.0),
             _row("https://kujira-watch.com/stocks/6976", 10, 400, 8.0)]

    def fake_sa(token, site, start, end, dimensions, limit=1000):
        return (queries if dimensions == ["query"] else pages), ""

    with mock.patch.object(g.gcp_auth, "access_token", return_value="tok"), \
            mock.patch.object(g, "search_analytics", side_effect=fake_sa):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = g.report(28, 10)
    out = buf.getvalue()
    assert rc == 0
    for section in ("■ 全体", "■ ページ種別", "■ 上位クエリ", "■ CTR改善候補", "■ あと一歩", "■ 上位ページ"):
        assert section in out
    assert "アクティビスト 一覧" in out          # あと一歩（12位）に出る
    assert "/stocks/6976" in out                  # 上位ページはパス表示


def test_show_sites_reports_missing_permission():
    with mock.patch.object(g.gcp_auth, "access_token", return_value="tok"), \
            mock.patch.object(g, "list_sites", return_value=([], "")), \
            mock.patch.object(g.gcp_auth, "client_email", return_value="bot@example.com"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = g.show_sites()
    assert rc == 1 and "bot@example.com" in buf.getvalue()


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

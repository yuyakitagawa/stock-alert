"""tools/ga4_clicks.py のユニットテスト（GA4 APIは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import ga4_clicks as g  # noqa: E402


def _rows(pairs):
    """GA4のrunReport応答形式（dimensionValues/metricValues）を作る。"""
    return [{"dimensionValues": [{"value": v} for v in (dims if isinstance(dims, tuple) else (dims,))],
             "metricValues": [{"value": str(m)} for m in metrics]}
            for dims, metrics in pairs]


def test_parse_rows_single_dimension_uses_string_key():
    assert g.parse_rows(_rows([("/weekly", [12])])) == {"/weekly": [12.0]}


def test_parse_rows_multi_dimension_uses_tuple_key():
    parsed = g.parse_rows(_rows([(("line", "push"), [3, 2])]))
    assert parsed == {("line", "push"): [3.0, 2.0]}


def test_delta_marks_new_instead_of_infinite_percent():
    """前期間0からの増加を%にすると∞になり読めない。「新規」と出す。"""
    assert g.delta(5, 0) == "  (新規)"
    assert g.delta(0, 0) == ""
    assert g.delta(150, 100) == "  (+50%)"
    assert g.delta(50, 100) == "  (-50%)"


def test_explain_error_service_disabled_points_to_enable_url():
    msg = g.explain_error(403, {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}, "1")
    assert "analyticsdata.googleapis.com" in msg and "有効化" in msg


def test_explain_error_403_points_to_property_access():
    msg = g.explain_error(403, {"error": {"message": "denied"}}, "999")
    assert "プロパティのアクセス管理" in msg and "999" in msg


def test_explain_error_404_points_to_property_id():
    msg = g.explain_error(404, {"error": {}}, "999")
    assert "GA4_PROPERTY_ID" in msg


def test_click_filter_matches_event_name_exactly():
    """部分一致にすると click_outbound 等まで混ざり、CTRの分子が水増しされる。"""
    f = g._click_filter()["filter"]
    assert f["fieldName"] == "eventName"
    assert f["stringFilter"] == {"matchType": "EXACT", "value": g.CLICK_EVENT}


def test_period_body_carries_dates_and_filter():
    from datetime import date
    body = g._period_body(["pagePath"], ["eventCount"], date(2026, 8, 1), date(2026, 8, 7), 10,
                          g._click_filter())
    assert body["dateRanges"] == [{"startDate": "2026-08-01", "endDate": "2026-08-07"}]
    assert body["dimensions"] == [{"name": "pagePath"}] and body["limit"] == 10
    assert "dimensionFilter" in body


def test_report_requires_property_id():
    with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": ""}), redirect_stdout(io.StringIO()) as buf:
        assert g.report(days=7, limit=5) == 1
    assert "GA4_PROPERTY_ID" in buf.getvalue()


def _report_with(responses):
    """run_reportの戻り値を順に差し替えてreport()を走らせ、標準出力を返す。"""
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": "123"}), \
            mock.patch.object(g, "access_token", return_value="tok"), \
            mock.patch.object(g, "run_report", side_effect=responses), \
            redirect_stdout(buf):
        g.report(days=7, limit=5)
    return buf.getvalue()


def test_report_rate_is_share_of_visitors_not_pageviews():
    """1人が同じページで何度も押すため、クリック数÷PVだと率が100%を超えて読めなくなる。
    分母は「そのページを見た人数」にして必ず100%以下にする。"""
    out = _report_with([
        (_rows([("/weekly", [20, 5])]), ""),    # 当期: 20クリック / 押した人5人
        (_rows([("/weekly", [10])]), ""),       # 前期のクリック
        (_rows([("/weekly", [200, 50])]), ""),  # 当期: 200PV / 閲覧者50人
        ([], "label未登録"),                     # CTA別（未登録）
        ([], "err"),                            # 流入元
    ])
    assert "10.0%" in out and "+100%" in out    # 5人/50人=10.0%、クリックは前期比+100%


def test_report_rate_never_exceeds_100_percent():
    """クリック数がPVを上回るページ（実測で /ranking/sells）でも率は100%以下に収まる。"""
    out = _report_with([
        (_rows([("/ranking/sells", [26, 4])]), ""),
        ([], ""),
        (_rows([("/ranking/sells", [12, 4])]), ""),
        ([], "e"), ([], "e"),
    ])
    assert "100.0%" in out and "216" not in out


def test_report_guides_when_label_dimension_missing():
    out = _report_with([
        (_rows([("/", [1])]), ""), ([], ""), (_rows([("/", [10])]), ""),
        ([], "customEvent:label did not match"),
        ([], "err"),
    ])
    assert "カスタムディメンション" in out and "イベントパラメータ label" in out


def test_report_fails_when_no_clicks_recorded():
    """クリックが0件なのを「異常なし」で流すと、計測が壊れていても気付けない。"""
    responses = [([], ""), ([], ""), ([], ""), ([], "e"), ([], "e")]
    with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": "123"}), \
            mock.patch.object(g, "access_token", return_value="tok"), \
            mock.patch.object(g, "run_report", side_effect=responses), \
            redirect_stdout(io.StringIO()):
        assert g.report(days=7, limit=5) == 1


def test_report_returns_1_when_api_fails():
    with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": "123"}), \
            mock.patch.object(g, "access_token", return_value="tok"), \
            mock.patch.object(g, "run_report", return_value=([], "APIが無効")), \
            redirect_stdout(io.StringIO()) as buf:
        assert g.report(days=7, limit=5) == 1
    assert "APIが無効" in buf.getvalue()


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

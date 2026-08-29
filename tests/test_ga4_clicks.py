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


def test_page_group_buckets_paths():
    """記事127本が個別に並ぶと読めないため、種別に畳んでから回遊を見る。"""
    assert g.page_group("/") == "TOP"
    assert g.page_group("/articles/abc") == "記事"
    assert g.page_group("/stocks/6305") == "銘柄ページ"
    assert g.page_group("/investors/foo") == "投資家ページ"
    assert g.page_group("/category/vc") == "カテゴリページ"
    assert g.page_group("/trending") == "データ/一覧ページ"
    assert g.page_group("/weekly") == "データ/一覧ページ"


def test_collect_pdca_metrics_computes_internal_moves_per_session():
    """回遊指標は (全PV−入口セッション)÷入口セッション。PV/セッションだと入口の1ページ目を
    含むぶん、導線を直した効果が薄まって見える。"""
    totals = [{"dimensionValues": [], "metricValues": [{"value": "40"}, {"value": "25"},
                                                       {"value": "30"}]}]
    views = _rows([("/", [100, 10, 500]), ("/trending", [50, 5, 400])])
    landings = _rows([("/", [40, 0.5])])
    clicks = _rows([("/", [30])])
    # period() は totals/views/landings/clicks の4リクエストを当期・前期で2周し、最後にlabelを引く
    responses = [(totals, ""), (views, ""), (landings, ""), (clicks, "")] * 2 + [([], "")]
    with mock.patch.object(g, "run_report", side_effect=responses):
        m = g.collect_pdca_metrics("tok", "123", 7)
    # PV150 / 入口40 → 内部110 → 1セッションあたり2.75回
    assert m["now"]["internal_per_session"] == 2.75
    assert m["now"]["groups"]["TOP"]["entrances"] == 40
    assert m["now"]["groups"]["データ/一覧ページ"]["pv"] == 50
    # エンゲージ率＝25/40。訪問者数に引きずられない率で、回遊の判定はこちらを主に使う
    assert m["now"]["engagement_rate"] == 0.625


def test_collect_pdca_metrics_survives_zero_entrances():
    """入口0の日にゼロ除算で日次レビュー全体を落とさない。"""
    responses = [([], ""), ([], ""), ([], ""), ([], "")] * 2 + [([], "")]
    with mock.patch.object(g, "run_report", side_effect=responses):
        m = g.collect_pdca_metrics("tok", "123", 7)
    assert m["now"]["internal_per_session"] == 0.0
    assert m["now"]["engagement_rate"] == 0.0


def test_access_token_prefers_env_json_over_file():
    """CIには鍵ファイルを置けない（.gitignore）。SecretのJSON文字列から読めること。"""
    sentinel = object()
    captured = {}

    class _Creds:
        token = "tok"

        def refresh(self, _req):
            return None

    def _from_info(info, scopes):
        captured["info"] = info
        return _Creds()

    with mock.patch.dict(os.environ, {"GCP_SERVICE_ACCOUNT_JSON": '{"type":"service_account"}'}), \
            mock.patch("google.oauth2.service_account.Credentials.from_service_account_info",
                       side_effect=_from_info), \
            mock.patch("google.auth.transport.requests.Request", return_value=sentinel):
        assert g.access_token() == "tok"
    assert captured["info"] == {"type": "service_account"}


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

"""会社HP URL・最新決算（lib/company_info.py）のユニットテスト。ネットワークは使わない。

実行: python3 tests/test_company_info.py
"""
import os
import sys
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib.company_info as m

TOP_HTML = """<table><tbody><tr>
  <th scope='row'>英語社名</th><td>TOYOTA MOTOR CORPORATION</td></tr>
  <tr><th scope='row'>会社サイト</th>
      <td><a href="https://global.toyota/jp/" target="_blank" rel="noopener">https://global.toyota/jp/</a></td></tr>
</tbody></table>"""

FIN_HTML = """
<h3>通期</h3>
<table>
<tr><th>決算期</th><th>売上高</th><th>営業益</th><th>経常益</th><th>最終益</th><th>修正1株益</th><th>修正1株配</th><th>発表日</th></tr>
<tr><td></td></tr>
<tr><th>I　2024.12</th><td>2,939,422</td><td>269,052</td><td>266,990</td><td>192,080</td><td>126.7</td><td>49</td><td>25/02/14</td></tr>
<tr><th>I　2025.12</th><td>2,894,676</td><td>-185,870</td><td>179,281</td><td>121,574</td><td>81.3</td><td>52</td><td>26/07/08</td></tr>
<tr><th>I予2026.12</th><td>3,220,000</td><td>297,000</td><td>274,000</td><td>194,000</td><td>132.6</td><td>57</td><td>26/07/08</td></tr>
<tr><th>前期比</th><td>+11.2</td></tr>
</table>
<table>
<tr><th>決算期</th><th>売上高</th><th>営業益</th><th>経常益</th><th>最終益</th><th>修正1株益</th><th>修正1株配</th><th>発表日</th></tr>
<tr><th>I予26.07-12</th><td>1,756,037</td><td>1</td><td>1</td><td>1</td><td>1</td><td>31</td><td>26/08/14</td></tr>
</table>
<h3>第２四半期累計決算【実績】</h3>
<table>
<tr><th>決算期</th><th>売上高</th><th>営業益</th><th>経常益</th><th>最終益</th><th>修正1株益</th><th>対通期進捗率</th><th>発表日</th></tr>
<tr><th>I　25.01-06</th><td>1,359,551</td><td>92,269</td><td>87,421</td><td>58,725</td><td>39.1</td><td>48.8</td><td>25/08/07</td></tr>
<tr><th>I　26.01-06</th><td>1,463,963</td><td>144,143</td><td>138,816</td><td>99,148</td><td>67.8</td><td>－</td><td>26/08/14</td></tr>
<tr><th>前年同期比</th><td>+7.7</td></tr>
</table>
"""


def test_parse_website():
    assert m.parse_website(TOP_HTML) == "https://global.toyota/jp/"
    assert m.parse_website("<html>会社サイトなし</html>") is None


def test_parse_finance_fy_forecast_and_latest():
    r = m.parse_finance(FIN_HTML)
    assert r["fy_period"] == "2025.12" and r["fy_sales"] == 2894676.0
    assert r["fy_op"] == -185870.0 and r["fy_dps"] == 52.0
    assert r["fy_announced"] == "2026-07-08"
    assert r["fc_period"] == "2026.12" and r["fc_np"] == 194000.0 and r["fc_eps"] == 132.6
    assert r["latest_label"] == "第２四半期累計" and r["latest_period"] == "26.01-06"
    assert r["latest_np"] == 99148.0 and r["latest_announced"] == "2026-08-14"
    assert r["latest_progress_pct"] is None  # 「－」は欠損


def test_parse_finance_no_tables_returns_nulls():
    r = m.parse_finance("<html><body>決算情報はありません</body></html>")
    assert set(r) >= {"fy_period", "fc_period", "latest_period"}
    assert all(v is None for v in r.values())


NOW = datetime(2026, 10, 20, 7, tzinfo=timezone.utc)


def test_plan_priority_and_website_refresh():
    codes = ["1000", "2000", "3000", "4000", "5000", "6000"]
    saved = {
        "1000": {"fetched_at": "2026-10-19T00:00:00+00:00", "website_checked_at": "2026-10-01T00:00:00+00:00"},
        "3000": {"fetched_at": "2026-10-01T00:00:00+00:00", "website_checked_at": "2026-05-01T00:00:00+00:00"},
        "4000": {"fetched_at": "2026-10-05T00:00:00+00:00", "website_checked_at": "2026-10-05T00:00:00+00:00"},
        "5000": {"fetched_at": "2026-10-18T00:00:00+00:00", "website_checked_at": None},
        "6000": {"fetched_at": "2026-10-19T12:00:00+00:00", "website_checked_at": "2026-10-19T12:00:00+00:00"},
    }
    # 1000 は保存後に決算短信 → 先頭。6000 は短信より後に取得済み → 対象外
    announced = {"1000": "2026-10-19T15:00:00+09:00", "6000": "2026-10-19T15:00:00+09:00"}
    todo, website = m.plan(codes, saved, announced, NOW, max_age_days=7, website_max_age_days=90)
    assert todo == ["1000", "2000", "3000", "4000"]  # 決算短信 → 未取得 → 古い順。5000/6000 は新しい
    assert website == {"2000", "3000"}


def test_fetch_one_keeps_website_when_top_page_fails():
    pages = {"https://kabutan.jp/stock/finance/?code=7203": FIN_HTML}
    with mock.patch.object(m, "_get", side_effect=lambda u: pages.get(u)):
        row = m.fetch_one("7203", with_website=True)
    assert row["fy_period"] == "2025.12"
    assert "website_url" not in row and "website_checked_at" not in row


def test_get_treats_redirect_as_not_listed():
    resp = mock.Mock(status_code=302, text="")
    with mock.patch.object(m.requests, "get", return_value=resp):
        assert m._get("https://kabutan.jp/stock/finance/?code=131A") == ""
    resp = mock.Mock(status_code=503, text="busy")
    with mock.patch.object(m.requests, "get", return_value=resp):
        assert m._get("https://kabutan.jp/stock/finance/?code=7203") is None


def test_fetch_one_not_listed_saves_nulls():
    with mock.patch.object(m, "_get", return_value=""):
        row = m.fetch_one("131A", with_website=True)
    assert row["website_url"] is None and row["website_checked_at"]
    assert row["fy_period"] is None and row["latest_period"] is None


def test_fetch_one_returns_none_when_finance_fails():
    with mock.patch.object(m, "_get", return_value=None):
        assert m.fetch_one("7203", with_website=True) is None


def test_fetch_many_stops_on_consecutive_errors():
    with mock.patch.object(m, "fetch_one", return_value=None):
        try:
            m.fetch_many([str(i) for i in range(1000, 1100)], set(), sleep_sec=0)
        except m.RateLimited as e:
            assert e.args[0] == []
        else:
            raise AssertionError("RateLimited が送出されない")


def test_pick_tanshin_latest_and_skips_corrections():
    rd = "https://webapi.yanoshin.jp/rd.php?https://www.release.tdnet.info/inbs/"
    rows = [
        {"code": "7203", "disclosed_at": "2026-08-01T06:00:00+00:00", "title": "2027年3月期 第1四半期決算短信", "doc_url": rd + "a.pdf"},
        {"code": "7203", "disclosed_at": "2026-10-30T06:00:00+00:00", "title": "2027年3月期 第2四半期（中間期）決算短信", "doc_url": rd + "b.pdf"},
        {"code": "7203", "disclosed_at": "2026-11-05T06:00:00+00:00", "title": "（訂正）「2027年3月期 第2四半期決算短信」の一部訂正", "doc_url": rd + "c.pdf"},
        {"code": "6758", "disclosed_at": "2026-10-30T06:00:00+00:00", "title": "2027年3月期 第2四半期決算短信", "doc_url": None},
        {"code": "9984", "disclosed_at": "2026-10-30T06:00:00+00:00", "title": "自己株式の取得状況", "doc_url": rd + "d.pdf"},
        {"code": "3856", "disclosed_at": "2026-06-26T06:00:00+00:00", "title": "2026年３月期決算短信の開示延期に関するお知らせ", "doc_url": rd + "e.pdf"},
        {"code": "4000", "disclosed_at": "2026-10-10T06:00:00+00:00", "title": "2026年８月期決算短信補足説明資料", "doc_url": rd + "f.pdf"},
        {"code": "4000", "disclosed_at": "2026-10-10T06:00:00+00:00", "title": "決算説明資料（2026年８月期 決算短信）", "doc_url": rd + "g.pdf"},
        {"code": "3492", "disclosed_at": "2026-10-23T06:00:00+00:00", "title": "2026年８月期　決算短信（REIT）", "doc_url": rd + "h.pdf"},
    ]
    out = m.pick_tanshin(rows)
    assert sorted(out) == ["3492", "7203"]  # URL無し・お知らせ・補足資料・短信以外は入れない
    assert out["7203"]["tanshin_url"] == "https://www.release.tdnet.info/inbs/b.pdf"  # 訂正より本体、やのしんを外す
    assert out["7203"]["tanshin_disclosed_at"] == "2026-10-30T06:00:00+00:00"
    assert out["7203"]["tanshin_xbrl_url"] is None  # xbrl_url が無い開示は None


def test_tanshin_updates_only_changed():
    latest = {"1000": {"code": "1000", "tanshin_url": "u1"}, "2000": {"code": "2000", "tanshin_url": "u2new"},
              "3000": {"code": "3000", "tanshin_url": "u3"}}
    saved = {"1000": {"tanshin_url": "u1", "tanshin_fc_fetched_at": "2026-10-25T00:00:00+00:00"},
             "2000": {"tanshin_url": "u2old", "tanshin_fc_fetched_at": "2026-10-25T00:00:00+00:00"}}
    assert [r["code"] for r in m.tanshin_updates(saved, latest)] == ["2000", "3000"]
    saved["1000"]["tanshin_fc_fetched_at"] = None  # 予想をまだ読めていない → 読み直す
    assert [r["code"] for r in m.tanshin_updates(saved, latest)] == ["1000", "2000", "3000"]


def _resp(status, content=b""):
    r = mock.Mock()
    r.status_code, r.content = status, content
    return r


def test_tanshin_forecast_units_and_errors():
    """予想は百万円で保存。5xx・通信断は None（次回また読む）、4xx・予想なしは NULL で読んだ時刻を刻む。"""
    f = {"fy_end": "2027-03-31", "fnp": 29_000_000_000.0, "fop": 41_000_000_000.0, "fsales": None}
    with mock.patch.object(m.requests, "get", return_value=_resp(200, b"zip")), \
         mock.patch("lib.earnings_forecast.parse_forecast", return_value=f):
        row = m.tanshin_forecast("https://www.release.tdnet.info/inbs/x.zip")
    assert row["tanshin_fc_np"] == 29000.0 and row["tanshin_fc_op"] == 41000.0 and row["tanshin_fc_sales"] is None
    assert row["tanshin_fc_fy_end"] == "2027-03-31" and row["tanshin_fc_fetched_at"]
    with mock.patch.object(m.requests, "get", return_value=_resp(503)):
        assert m.tanshin_forecast("u") is None
    with mock.patch.object(m.requests, "get", side_effect=OSError("down")):
        assert m.tanshin_forecast("u") is None
    with mock.patch.object(m.requests, "get", return_value=_resp(404)):
        row = m.tanshin_forecast("u")
    assert row["tanshin_fc_np"] is None and row["tanshin_fc_fetched_at"]
    assert m.tanshin_forecast(None)["tanshin_fc_fetched_at"]


def test_with_tanshin_forecasts_reads_newest_up_to_cap():
    rows = [{"code": c, "tanshin_disclosed_at": d, "tanshin_xbrl_url": c}
            for c, d in (("1000", "2026-10-01"), ("2000", "2026-10-20"), ("3000", "2026-10-10"))]
    fc = {"tanshin_fc_fy_end": "2027-03-31", "tanshin_fc_sales": None, "tanshin_fc_op": None,
          "tanshin_fc_np": 1.0, "tanshin_fc_fetched_at": "now"}
    with mock.patch.object(m, "tanshin_forecast", return_value=fc) as tf:
        out = m.with_tanshin_forecasts(rows, max_reads=2, sleep_sec=0)
    assert [c.args[0] for c in tf.call_args_list] == ["2000", "3000"]  # 新しい順に2件だけ
    by = {r["code"]: r for r in out}
    assert by["2000"]["tanshin_fc_np"] == 1.0 and by["1000"]["tanshin_fc_fetched_at"] is None
    assert len({tuple(sorted(r)) for r in out}) == 1  # 一括upsertのため全行のキーをそろえる


def test_plan_treats_rows_without_fetched_at_as_never():
    """従業員数・短信URLだけの行（fetched_at が NULL）は決算未取得として扱う。"""
    saved = {"1000": {"fetched_at": None, "website_checked_at": None}}
    todo, website = m.plan(["1000"], saved, {}, NOW, max_age_days=7, website_max_age_days=90)
    assert todo == ["1000"] and website == {"1000"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")

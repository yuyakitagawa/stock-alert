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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")

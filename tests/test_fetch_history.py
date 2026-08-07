"""株価キャッシュ更新（tools/fetch_history）の銘柄コード収集ロジックのユニットテスト。

実行: python3 tests/test_fetch_history.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.fetch_history as m


def test_get_all_codes_unions_db_and_jpx():
    """既存のyahoo_price_cacheコードだけで打ち切らず、JPX最新リストとの
    和集合を返す（新規上場銘柄がyahoo_price_cacheに永久に入らない事態を防ぐ）。"""
    with mock.patch("lib.db.get_price_cache_codes", return_value=["1301", "1332"]), \
         mock.patch.object(m, "_fetch_jpx_codes", return_value=["1332", "603A"]):
        codes = m.get_all_codes()
    assert codes == ["1301", "1332", "603A"]


def test_get_all_codes_falls_back_to_db_when_jpx_fetch_fails():
    """JPX取得に失敗（空リスト）してもyahoo_price_cache既存コードは維持する。"""
    with mock.patch("lib.db.get_price_cache_codes", return_value=["1301", "1332"]), \
         mock.patch.object(m, "_fetch_jpx_codes", return_value=[]):
        codes = m.get_all_codes()
    assert codes == ["1301", "1332"]


def test_fetch_jpx_codes_returns_empty_on_request_exception():
    with mock.patch.object(m.requests, "get", side_effect=Exception("network down")):
        codes = m._fetch_jpx_codes()
    assert codes == []


def test_fetch_jpx_codes_includes_reit_and_domestic_stock():
    """J-REIT（市場・商品区分にREITを含む行）も内国株式と同様に含める。
    「内国株式」のみだとJ-REITの株価が yahoo_price_cache に恒久的に入らず、
    ブログ記事の金額推定が常にスキップされていたため対象を広げた。
    ETF・ETN等それ以外の商品区分は引き続き除外する。"""
    import pandas as pd
    fake_df = pd.DataFrame({
        "コード": ["7203", "3269", "1301"],
        "市場・商品区分": [
            "プライム（内国株式）",
            "REIT・ベンチャーファンド・カントリーファンド・インフラファンド",
            "ETF・ETN",
        ],
    })
    with mock.patch.object(m.requests, "get", return_value=mock.MagicMock(content=b"")), \
         mock.patch.object(m.pd, "read_excel", return_value=fake_df):
        codes = m._fetch_jpx_codes()
    assert codes == ["7203", "3269"]


if __name__ == "__main__":
    test_get_all_codes_unions_db_and_jpx()
    test_get_all_codes_falls_back_to_db_when_jpx_fetch_fails()
    test_fetch_jpx_codes_returns_empty_on_request_exception()
    test_fetch_jpx_codes_includes_reit_and_domestic_stock()
    print("OK: test_fetch_history (4 tests)")

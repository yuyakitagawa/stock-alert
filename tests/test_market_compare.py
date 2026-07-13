"""日経225 vs S&P500 相対強弱アドバイザー（lib/market_compare）のユニットテスト。

実行: python3 tests/test_market_compare.py  /  pytest tests/test_market_compare.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.market_compare import compare


def test_us_favored_when_nikkei_weak():
    """日経が弱く米国が強い（20日・60日とも差が閾値超）→ 米国株優位。"""
    r = compare(nk5=-1.0, nk20=-4.0, nk60=-8.0, us5=0.5, us20=2.0, us60=3.0)
    assert r["verdict"] == "us_favored"
    assert r["score"] == -2


def test_jp_favored_when_nikkei_strong():
    """日経が米国より明確に強い（20日・60日とも差が閾値超）→ 日本株優位。"""
    r = compare(nk5=1.0, nk20=5.0, nk60=10.0, us5=0.2, us20=1.0, us60=2.0)
    assert r["verdict"] == "jp_favored"
    assert r["score"] == 2


def test_neutral_when_close():
    """リターン差が閾値未満 → 拮抗。"""
    r = compare(nk5=0.5, nk20=1.0, nk60=2.0, us5=0.4, us20=0.5, us60=1.0)
    assert r["verdict"] == "neutral"
    assert r["score"] == 0


def test_neutral_when_missing_data():
    """データ欠損時は判定不能として拮抗扱い。"""
    r = compare(nk5=None, nk20=None, nk60=None, us5=None, us20=None, us60=None)
    assert r["verdict"] == "neutral"
    assert "データ不足" in r["reasons"][0]


if __name__ == "__main__":
    test_us_favored_when_nikkei_weak()
    test_jp_favored_when_nikkei_strong()
    test_neutral_when_close()
    test_neutral_when_missing_data()
    print("OK: test_market_compare (4 tests)")

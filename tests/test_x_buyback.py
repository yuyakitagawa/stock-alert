"""平日「本日の自社株買い決定」X投稿（web/x_buyback）のユニットテスト。
ネットワーク（Supabase / X API / TDnet）は全てモックし、集計と本文組み立てのみ検証する。

実行: python3 tests/test_x_buyback.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.x_buyback as m
from web.x_post_format import POST_MAX_WEIGHTED, weighted_len


def _row(code, amount, ratio=None, cancel=False):
    return {"code": code, "disclosed_at": "2026-08-20T16:00:00+00:00", "title": "自己株式取得に係る事項の決定",
            "max_amount_yen": amount, "ratio_pct": ratio, "method": "市場買付", "will_cancel": cancel}


def test_fetch_decisions_filters_small_frames_and_resolves_names():
    rows = [_row("7966", 30_000_000_000, 10.55), _row("2669", 200_000_000, 0.5), _row("5973", 24_240_000, 0.62)]
    names = [{"code": "7966", "name": "リンテック株式会社"}, {"code": "2669", "name": "カネ美食品株式会社"}]
    with mock.patch.object(m.sb, "select", side_effect=[rows, names]):
        out = m.fetch_decisions("2026-08-20")
    assert [d["code"] for d in out] == ["7966", "2669"]   # 0.24億円は MIN_AMOUNT_YEN 未満で落ちる
    assert out[0]["stock"] == "リンテック"                 # 法人格は落とす
    assert out[0]["amount_yen"] == 30_000_000_000
    assert out[0]["ratio"] == 10.55


def test_fetch_decisions_falls_back_to_code_when_name_missing():
    with mock.patch.object(m.sb, "select", side_effect=[[_row("9999", 500_000_000)], []]):
        out = m.fetch_decisions("2026-08-20")
    assert out[0]["stock"] == "9999"


def test_format_oku():
    assert m.format_oku(30_000_000_000) == "300億円"
    assert m.format_oku(2_601_000_000) == "26億円"
    assert m.format_oku(200_000_000) == "2.0億円"


def test_build_text_none_when_empty():
    assert m.build_buyback_text([], "2026-08-20") is None


def test_build_text_fits_limit_and_orders_by_amount():
    decisions = [
        {"code": f"{1000+i}", "stock": "とても長い社名の株式会社テスト" * 2, "amount_yen": (10 - i) * 10**9,
         "ratio": 5.0 + i, "method": "", "will_cancel": i == 0}
        for i in range(8)
    ]
    text = m.build_buyback_text(decisions, "2026-08-20")
    assert weighted_len(text) <= POST_MAX_WEIGHTED
    assert text.startswith("🏦 本日の自社株買い決定（8/20）")
    assert "1000" in text and "消却" in text          # 最大の枠は必ず載る
    assert "ほか" in text and "計8件" in text
    assert "http" not in text                          # リンク入り投稿は課金対象なのでURLは入れない
    assert "#自社株買い" in text


def test_run_skips_post_when_no_decisions():
    with mock.patch.object(m, "fetch_decisions", return_value=[]), \
         mock.patch.object(m, "post_tweet") as post:
        assert m.run(dry_run=False, date_str="2026-08-20", no_refresh=True) == 0
    post.assert_not_called()


def test_run_posts_with_media_and_kind():
    decisions = [{"code": "7966", "stock": "リンテック", "amount_yen": 30_000_000_000, "ratio": 10.55, "method": "", "will_cancel": False}]
    with mock.patch.object(m, "fetch_decisions", return_value=decisions), \
         mock.patch.object(m, "build_buyback_media", return_value=["m1"]), \
         mock.patch.object(m, "post_tweet", return_value="t1") as post:
        assert m.run(dry_run=False, date_str="2026-08-20", no_refresh=True) == 0
    assert post.call_args.kwargs["kind"] == "buyback_daily"
    assert post.call_args.kwargs["media_ids"] == ["m1"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

"""tools/backfill_blog_eyecatch.py のユニットテスト。
ネットワーク（microCMS/Supabase/Pexels）は呼ばず、候補抽出とバッジ判定のみ検証する。

実行: python3 tests/test_backfill_blog_eyecatch.py
"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.backfill_blog_eyecatch import (
    badge_label_for, build_card, done_ids_from_logs, ratio_from_title, resolve_holding_ratio,
    select_candidates,
)


def _a(id_, deal_date, amount=0.0, change=0.0, title="", tags="EDINET,自動生成"):
    return {"id": id_, "dealDate": f"{deal_date}T00:00:00.000Z", "dealAmount": amount,
            "ratioChangePct": change, "title": title, "tags": tags,
            "stockCode": "7203", "stockName": "トヨタ", "filerName": "Ｘ社"}


def test_badge_sell_from_tags():
    assert badge_label_for(_a("a", "2026-08-01", tags="EDINET,自動生成,売り")) == "📉 売却"


def test_badge_correction_from_tags_or_title():
    assert badge_label_for(_a("a", "2026-08-01", tags="EDINET,訂正")) == "📝 訂正"
    assert badge_label_for(_a("a", "2026-08-01", title="トヨタ、Ｘ社が保有比率を5%に訂正｜訂正報告書")) == "📝 訂正"


def test_badge_new_holding_from_title():
    assert badge_label_for(_a("a", "2026-08-01", title="トヨタ（7203）、Ｘ社が5.12%を新規保有")) == "📈 新規取得"


def test_badge_default_is_add():
    assert badge_label_for(_a("a", "2026-08-01", title="トヨタ（7203）、Ｘ社が保有比率6%に引き上げ")) == "📈 買い増し"


def test_select_candidates_days_index_only_limit_and_order():
    today = date(2026, 8, 23)
    arts = [
        _a("old", "2026-05-01", amount=10),          # 60日より前
        _a("small", "2026-08-10", amount=1, change=0.5),  # index対象外
        _a("big", "2026-08-05", amount=5),
        _a("ratio", "2026-08-20", amount=0.1, change=-1.2),
        _a("mid", "2026-08-15", amount=4),
    ]
    got = select_candidates(arts, days=60, index_only=True, today=today)
    assert [a["id"] for a in got] == ["ratio", "mid", "big"]
    got = select_candidates(arts, days=60, index_only=True, limit=2, today=today)
    assert [a["id"] for a in got] == ["ratio", "mid"]
    assert len(select_candidates(arts, today=today)) == 5


def test_ratio_from_title_and_fallback():
    assert ratio_from_title("トヨタ（7203）、Ｘ社が保有比率7.48%に引き上げ｜大量保有報告書") == 7.48
    assert ratio_from_title("Ｘ社が5%を新規保有") == 5.0
    assert ratio_from_title("比率なし") is None
    a = _a("a", "2026-08-01", title="Ｘ社が保有比率6.1%に引き上げ")
    assert resolve_holding_ratio(a, {("7203", "2026-08-01", "Ｘ社"): 6.15}) == 6.15
    assert resolve_holding_ratio(a, {}) == 6.1


def test_build_card():
    card = build_card(_a("a", "2026-08-01", tags="EDINET,売り"), 6.15)
    assert card == {"filer_name": "Ｘ社", "stock_name": "トヨタ", "holding_ratio": 6.15,
                    "badge_label": "📉 売却", "disc_date": "2026-08-01"}


def test_done_ids_from_logs_picks_ok_lines():
    """中断からの再開用に、実行ログの「→ OK」行から記事IDだけを拾う。
    スキップ・失敗行と、まだ処理していない行は拾わない。"""
    import tempfile
    log = (
        "画像あり記事(差し替え) 964件 → 対象 964件\n"
        "[1/964] 東邦亜鉛(5707) 2026-08-27 xx7gqfbjx → OK https://images.example/a.jpg\n"
        "[2/964] ヤギ(7460) 2026-08-21 h_7m1wxsa → OK https://images.example/b.jpg\n"
        "[3/964] 某社(1234) 2026-08-20 zzz111 → スキップ（保有比率が取れない）\n"
        "[4/964] 某社(1234) 2026-08-20 yyy222 → 失敗（アップロード）\n"
        "完了: 成功 2 / スキップ 1 / 失敗 1\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".log", encoding="utf-8", delete=False) as f:
        f.write(log)
        path = f.name
    try:
        assert done_ids_from_logs([path]) == {"xx7gqfbjx", "h_7m1wxsa"}
    finally:
        os.unlink(path)


def test_done_ids_from_logs_tolerates_missing_file():
    assert done_ids_from_logs(["/nonexistent/path.log"]) == set()


if __name__ == "__main__":
    test_badge_sell_from_tags()
    test_badge_correction_from_tags_or_title()
    test_badge_new_holding_from_title()
    test_badge_default_is_add()
    test_select_candidates_days_index_only_limit_and_order()
    test_ratio_from_title_and_fallback()
    test_build_card()
    test_done_ids_from_logs_picks_ok_lines()
    test_done_ids_from_logs_tolerates_missing_file()
    print("OK")

"""自社株買い記事の自動投稿（web/publish_buyback_articles）のユニットテスト。
ネットワーク（Supabase / microCMS / Claude）は全てモックし、対象選定・タイトル・payload組み立てのみ検証する。

実行: python3 tests/test_publish_buyback_articles.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.publish_buyback_articles as m

FACT = {
    "stock_code": "7966", "stock_name": "リンテック", "disc_date": "2026-08-20", "board_date": "2026-08-20",
    "amount_oku": 300.0, "max_amount_yen": 30_000_000_000, "max_shares": 6_910_000, "ratio": 10.55,
    "period_from": "2026-08-24", "period_to": "2026-08-26", "method": "ToSTNeT-3（立会外買付）",
    "purpose": None, "will_cancel": False, "doc_url": "https://www.release.tdnet.info/inbs/1.pdf",
    "company_description": "", "context_close": 3000.0, "context_dp_level": "中", "implied_price": 4341, "prior": [],
}


def test_is_worth_publishing_thresholds():
    assert m.is_worth_publishing(10.0, None)
    assert m.is_worth_publishing(1.0, 3.0)
    assert not m.is_worth_publishing(9.9, 2.99)
    assert not m.is_worth_publishing(None, None)


def test_fetch_candidates_filters_by_scale():
    rows = [
        {"code": "7966", "disclosed_at": "2026-08-20T16:00:00+00:00", "max_amount_yen": 30_000_000_000, "ratio_pct": "10.55"},
        {"code": "2669", "disclosed_at": "2026-08-20T16:00:00+00:00", "max_amount_yen": 200_000_000, "ratio_pct": "0.5"},
        {"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "max_amount_yen": 1_000_000_000, "ratio_pct": "12.8"},
    ]
    with mock.patch.object(m.sb, "select", return_value=rows):
        out = m.fetch_candidates(7)
    assert [c["code"] for c in out] == ["7966", "6082"]
    assert out[0]["amount_oku"] == 300.0 and out[0]["ratio"] == 10.55


def test_build_titles_contains_search_terms_and_scale():
    t = m.build_titles(FACT, "Lintec")
    assert t["title"] == "リンテック（7966）、上限300億円・発行済の10.55%の自社株買いを決定｜TDnet適時開示"
    assert t["titleEn"].startswith("Lintec (7966) Approves Share Buyback of Up to ¥30.0 billion (10.55% of Shares Outstanding)")


def test_build_titles_without_ratio_or_amount():
    f = {**FACT, "ratio": None}
    assert "発行済" not in m.build_titles(f)["title"]
    f2 = {**FACT, "amount_oku": None, "max_amount_yen": None, "ratio": 5.0}
    assert m.build_titles(f2)["title"].startswith("リンテック（7966）、発行済の5%の自社株買いを決定")
    assert "5% of Shares Outstanding" in m.build_titles(f2)["titleEn"]


def test_build_payload_uses_buyback_deal_type_and_no_filer():
    payload = m.build_payload(FACT, {"body": "<p>本文</p>", "bodyEn": "<p>Body</p>", "stockNameEn": "Lintec"})
    assert payload["dealType"] == ["自社株買い"]
    assert payload["dealAmount"] == 300.0
    assert payload["ratioChangePct"] == 10.55
    assert payload["sourceUrl"] == FACT["doc_url"]
    assert payload["tags"] == "自社株買い"
    assert "filerName" not in payload      # 記事ページのEDINET保有比率ブロックを出させない
    assert payload["titleEn"] and payload["bodyEn"] == "<p>Body</p>"
    assert payload["dealDate"] == "2026-08-20T00:00:00.000Z"


def test_build_payload_adds_cancel_tag():
    payload = m.build_payload({**FACT, "will_cancel": True}, {"body": "<p>x</p>"})
    assert payload["tags"] == "自社株買い,消却"
    assert "titleEn" not in payload


def test_answer_sentence_is_factual_direct_answer():
    s = m._answer_sentence(FACT)
    assert s.startswith("リンテック（7966）が2026-08-20、上限300億円・6,910,000株・発行済株式総数（自己株式を除く）の10.55%を上限とする")
    assert "TDnetの適時開示" in s


def test_build_and_publish_skips_published_and_respects_max():
    rows = [
        {"code": "7966", "disclosed_at": "2026-08-20T16:00:00+00:00", "amount_oku": 300.0, "ratio": 10.55, "max_amount_yen": 30_000_000_000},
        {"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "amount_oku": 10.0, "ratio": 12.8, "max_amount_yen": 1_000_000_000},
    ]
    with mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", side_effect=[True, False]), \
         mock.patch.object(m, "build_fact_sheet", return_value={**FACT, "stock_code": "6082"}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "publish_article") as publish:
        out = m.build_and_publish(days=7, max_articles=1, dry_run=True)
    assert len(out) == 1 and out[0]["stockCode"] == "6082" and out[0]["id"] is None
    publish.assert_not_called()


def test_build_and_publish_counts_generation_attempts_when_all_fail():
    """記事候補が残っているのに生成が全滅した便は attempts>0 / 投稿0件になり、
    exit_code_for_run が異常(1)を返すこと（2026-08-24のAPI上限超過と同じ形）。"""
    rows = [
        {"code": "9706", "disclosed_at": "2026-08-24T16:00:00+00:00", "amount_oku": 150.0, "ratio": 3.8, "max_amount_yen": 15_000_000_000},
        {"code": "6574", "disclosed_at": "2026-08-24T16:00:00+00:00", "amount_oku": 45.0, "ratio": 7.33, "max_amount_yen": 4_500_000_000},
    ]
    stats = {}
    with mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", return_value=FACT), \
         mock.patch.object(m, "generate_body_checked", return_value=None), \
         mock.patch.object(m, "publish_article") as publish:
        out = m.build_and_publish(days=2, dry_run=False, stats=stats)

    assert out == []
    assert stats["generation_attempts"] == 2
    publish.assert_not_called()
    assert m.exit_code_for_run(stats["generation_attempts"], len(out)) == 1


def test_build_and_publish_reports_zero_attempts_when_all_already_published():
    """既出で全部スキップされた便は attempts=0。記事0件でもジョブは緑のままにする。"""
    rows = [
        {"code": "9706", "disclosed_at": "2026-08-24T16:00:00+00:00", "amount_oku": 150.0, "ratio": 3.8, "max_amount_yen": 15_000_000_000},
    ]
    stats = {}
    with mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", return_value=True), \
         mock.patch.object(m, "generate_body_checked") as gen:
        out = m.build_and_publish(days=2, dry_run=False, stats=stats)

    assert out == []
    assert stats["generation_attempts"] == 0
    gen.assert_not_called()
    assert m.exit_code_for_run(stats["generation_attempts"], len(out)) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

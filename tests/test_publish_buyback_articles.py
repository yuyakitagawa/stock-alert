"""自社株買い記事の自動投稿（web/publish_buyback_articles）のユニットテスト。
ネットワーク（Supabase / microCMS / Claude）は全てモックし、対象選定・タイトル・payload組み立てのみ検証する。

実行: python3 tests/test_publish_buyback_articles.py
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.publish_buyback_articles as m

DECISION_TITLE = "自己株式取得に係る事項の決定に関するお知らせ"

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
        {"code": "7966", "disclosed_at": "2026-08-20T16:00:00+00:00", "max_amount_yen": 30_000_000_000, "ratio_pct": "10.55", "title": DECISION_TITLE},
        {"code": "2669", "disclosed_at": "2026-08-20T16:00:00+00:00", "max_amount_yen": 200_000_000, "ratio_pct": "0.5", "title": DECISION_TITLE},
        {"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "max_amount_yen": 1_000_000_000, "ratio_pct": "12.8", "title": DECISION_TITLE},
        # 既存決議の一部変更。閾値を超えても新規決議ではないので候補にしない（8560 2026-08-18の実例）
        {"code": "8560", "disclosed_at": "2026-08-18T16:00:00+00:00", "max_amount_yen": 1_100_000_000, "ratio_pct": "9.12",
         "title": "自己株式取得に係る事項の一部変更に関するお知らせ"},
    ]
    with mock.patch.object(m.sb, "select", return_value=rows):
        out = m.fetch_candidates(7)
    assert [c["code"] for c in out] == ["7966", "6082"]
    assert out[0]["amount_oku"] == 300.0 and out[0]["ratio"] == 10.55


def test_fetch_candidates_excludes_amendment_of_existing_buyback():
    """既存決議の「一部変更」「（訂正）」は閾値を超えても記事にしない。
    tdnet_buybacksに取り込み済みの行は取り込み時の分類のまま残るため、ここでも分類を見る
    （8560 2026-08-18は取締役会決議が2026-02-09なのに上限11億円・9.12%で閾値を超えており、
    記事にすると「2026-08-18に決議した」という誤報になる）。"""
    rows = [
        {"code": "8560", "disclosed_at": "2026-08-18T16:00:00+00:00", "max_amount_yen": 1_100_000_000,
         "ratio_pct": "9.12", "title": "自己株式取得に係る事項の一部変更に関するお知らせ"},
        {"code": "4323", "disclosed_at": "2026-06-30T16:00:00+00:00", "max_amount_yen": 5_000_000_000,
         "ratio_pct": "5.0",
         "title": "（訂正）「自己株式取得に係る事項の決定に関するお知らせ」の一部訂正について"},
    ]
    with mock.patch.object(m.sb, "select", return_value=rows):
        assert m.fetch_candidates(30) == []


def test_build_titles_contains_search_terms_and_scale():
    t = m.build_titles(FACT)
    assert t["title"] == "リンテック（7966）、上限300億円・発行済の10.55%の自社株買いを決定｜TDnet適時開示"
    assert "titleEn" not in t


def test_build_titles_without_ratio_or_amount():
    f = {**FACT, "ratio": None}
    assert "発行済" not in m.build_titles(f)["title"]
    f2 = {**FACT, "amount_oku": None, "max_amount_yen": None, "ratio": 5.0}
    assert m.build_titles(f2)["title"].startswith("リンテック（7966）、発行済の5%の自社株買いを決定")


def test_build_payload_uses_buyback_deal_type_and_no_filer():
    payload = m.build_payload(FACT, {"body": "<p>本文</p>", "bodyEn": "<p>Body</p>"})
    assert payload["dealType"] == ["自社株買い"]
    assert payload["dealAmount"] == 300.0
    assert payload["ratioChangePct"] == 10.55
    assert payload["sourceUrl"] == FACT["doc_url"]
    assert payload["tags"] == "自社株買い"
    assert "filerName" not in payload      # 記事ページのEDINET保有比率ブロックを出させない
    # 英語版（/en）は2026-08-29に廃止。生成が英語を返してもpayloadへは入れない
    assert "titleEn" not in payload and "bodyEn" not in payload
    assert payload["dealDate"] == "2026-08-20T00:00:00.000Z"


def test_build_payload_adds_cancel_tag():
    payload = m.build_payload({**FACT, "will_cancel": True}, {"body": "<p>x</p>"})
    assert payload["tags"] == "自社株買い,消却"
    assert "titleEn" not in payload


def test_answer_sentence_is_factual_direct_answer():
    s = m._answer_sentence(FACT)
    assert s.startswith("リンテック（7966）が2026-08-20、上限300億円・6,910,000株・発行済株式総数（自己株式を除く）の10.55%を上限とする")
    assert "TDnetの適時開示" in s


def test_generate_body_checked_retries_on_ai_tell():
    # 字数が足りていてもAI常套句があれば1回だけ再生成し、常套句の無い方を採用する
    telly = {"body": "<p>株主還元の姿勢に注目が集まっています。" + "あ" * 700 + "</p>"}
    clean = {"body": "<p>" + "い" * 700 + "</p>"}
    with mock.patch.object(m, "generate_body", side_effect=[telly, clean]) as gen:
        assert m.generate_body_checked(FACT) is clean
    assert gen.call_count == 2


def test_published_deal_keys_uses_tags_not_deal_type():
    """既報判定に dealType（選択肢に無い値は空配列で保存され常に0件）を使わない。
    実害: 2026-08-25、コンヴァノ6574の同一開示から13本の重複記事が公開された。"""
    rows = [{"stockCode": "7966", "dealDate": "2026-08-20T00:00:00.000Z"}]
    with mock.patch.object(m, "fetch_published_index", return_value=rows) as idx:
        assert m.published_deal_keys(30) == {("7966", "2026-08-20")}
    assert idx.call_args.kwargs["extra_filter"] == "tags[contains]自社株買い"


def test_published_deal_keys_none_when_index_unavailable():
    with mock.patch.object(m, "fetch_published_index", return_value=None):
        assert m.published_deal_keys(30) is None


def test_build_and_publish_backfill_aborts_when_index_unavailable():
    """既報一覧が引けないまま30日分を走らせると同じ開示を投稿し直すため、backfillごと中止する。"""
    with mock.patch.object(m, "published_deal_keys", return_value=None), \
         mock.patch.object(m, "fetch_candidates") as fetch:
        assert m.build_and_publish(backfill=True) == []
    fetch.assert_not_called()


def test_build_and_publish_backfill_skips_known_and_takes_oldest_first():
    rows = [
        {"code": "7966", "disclosed_at": "2026-08-20T16:00:00+00:00", "amount_oku": 300.0,
         "ratio": 10.55, "max_amount_yen": 30_000_000_000},
        {"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "amount_oku": 10.0,
         "ratio": 12.8, "max_amount_yen": 1_000_000_000},
        {"code": "8560", "disclosed_at": "2026-08-18T16:00:00+00:00", "amount_oku": 11.0,
         "ratio": 9.12, "max_amount_yen": 1_100_000_000},
    ]
    with mock.patch.object(m, "published_deal_keys", return_value={("7966", "2026-08-20")}), \
         mock.patch.object(m, "fetch_candidates", return_value=rows) as fetch, \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", side_effect=lambda r: {**FACT, "stock_code": r["code"]}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}):
        out = m.build_and_publish(dry_run=True, backfill=True)
    assert fetch.call_args.args[0] == m.BACKFILL_DAYS
    # 既報の7966は落とし、残りは古い順（8/14 → 8/18）に消化する
    assert [o["stockCode"] for o in out] == ["6082", "8560"]


def test_backfill_skips_disclosure_already_articled():
    """microCMSに記事が無くても、過去に作ったことがある決定開示は作り直さない
    （重複記事12件を2026-08-25に削除済み。取りこぼしと誤認して復活させないため）。"""
    rows = [
        {"code": "6574", "disclosed_at": "2026-08-24T16:00:00+00:00", "amount_oku": 10.0,
         "ratio": 5.0, "max_amount_yen": 1_000_000_000,
         "article_published_at": "2026-08-24T12:00:00+00:00"},
        {"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "amount_oku": 10.0,
         "ratio": 12.8, "max_amount_yen": 1_000_000_000, "article_published_at": None},
    ]
    with mock.patch.object(m, "published_deal_keys", return_value=set()), \
         mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", side_effect=lambda r: {**FACT, "stock_code": r["code"]}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}):
        out = m.build_and_publish(dry_run=True, backfill=True)
    assert [o["stockCode"] for o in out] == ["6082"]


def test_build_and_publish_caps_articles_per_day():
    """通常運転は当日の残り枠（DAILY_MAX_ARTICLES − 当日の実績）で止まる。"""
    from lib import publish_ledger as pl
    from lib.publish_ledger import PublishLedger
    rows = [
        {"code": f"{6000 + i}", "disclosed_at": f"2026-08-2{i}T16:00:00+00:00",
         "amount_oku": 10.0, "ratio": 12.8, "max_amount_yen": 1_000_000_000}
        for i in range(m.DAILY_MAX_ARTICLES + 2)
    ]
    led = PublishLedger("test")
    with mock.patch.object(m, "daily_quota", return_value=1) as quota, \
         mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", side_effect=lambda r: {**FACT, "stock_code": r["code"]}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}):
        out = m.build_and_publish(days=7, dry_run=True, ledger=led)
    quota.assert_called_once_with("tdnet_buybacks", "code")
    assert len(out) == 1
    assert led.reasons[pl.SKIP_MAX_ARTICLES] == 1
    assert led.has_anomaly() is False, led.summary()


def test_build_and_publish_records_ledger_after_publishing():
    rows = [{"code": "6082", "disclosed_at": "2026-08-14T16:00:00+00:00", "amount_oku": 10.0,
             "ratio": 12.8, "max_amount_yen": 1_000_000_000}]
    with mock.patch.object(m, "daily_quota", return_value=2), \
         mock.patch.object(m, "fetch_candidates", return_value=rows), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", return_value={**FACT, "stock_code": "6082"}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "publish_article", return_value="fakeid"), \
         mock.patch.object(m, "mark_article_published") as mark:
        out = m.build_and_publish(days=7)
    assert len(out) == 1
    assert mark.call_args.args[0]["code"] == "6082"


def test_stock_name_falls_back_to_tdnet_when_master_has_no_name():
    """jpx_stock_listはJPX（東証）の一覧なので福証・名証単独上場が載らない。
    銘柄名が引けないだけで決定開示を永久に取りこぼしていた（実例: 8560 宮崎太陽銀行）。"""
    with mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch("lib.tdnet.fetch_company_name", return_value="宮崎太銀") as tdnet:
        assert m.stock_name_of("8560") == "宮崎太銀"
    tdnet.assert_called_once_with("8560")


def test_stock_name_prefers_master_and_skips_tdnet():
    with mock.patch.object(m.sb, "select_one", return_value={"name": "リンテック"}), \
         mock.patch("lib.tdnet.fetch_company_name") as tdnet:
        assert m.stock_name_of("7966") == "リンテック"
    tdnet.assert_not_called()


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


_TWO_CANDIDATES = [
    {"code": "7966", "disclosed_at": "2026-08-26T16:00:00+00:00", "amount_oku": 300.0,
     "ratio": 10.55, "max_amount_yen": 30_000_000_000},
    {"code": "6082", "disclosed_at": "2026-08-26T16:00:00+00:00", "amount_oku": 10.0,
     "ratio": 12.8, "max_amount_yen": 1_000_000_000},
]


def test_ledger_treats_all_already_published_as_healthy():
    """候補2件が両方とも既報で公開0件になるのは正常（2026-08-26の21時便の実態）。

    ログには「記事候補: 2件」→「0件処理しました。」しか出ておらず、正常な既報スキップと
    生成失敗の区別が付かなかった。台帳はここを「既報2」と記録して鳴らさない。
    """
    from lib.publish_ledger import PublishLedger
    led = PublishLedger("test")
    with mock.patch.object(m, "daily_quota", return_value=2), \
         mock.patch.object(m, "fetch_candidates", return_value=_TWO_CANDIDATES), \
         mock.patch.object(m, "already_published", return_value=True), \
         mock.patch.object(m, "publish_article") as publish:
        out = m.build_and_publish(days=7, dry_run=True, ledger=led)
    assert out == []
    publish.assert_not_called()
    assert led.has_anomaly() is False, led.summary()
    assert "既報2" in led.summary()


def test_ledger_flags_generation_failure_as_anomaly():
    """同じ「候補2件→公開0件」でも、本文生成が失敗しているなら異常として鳴らす。"""
    from lib import publish_ledger as pl
    from lib.publish_ledger import PublishLedger
    led = PublishLedger("test")
    with mock.patch.object(m, "daily_quota", return_value=2), \
         mock.patch.object(m, "fetch_candidates", return_value=_TWO_CANDIDATES), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "build_fact_sheet", return_value={**FACT, "stock_code": "6082"}), \
         mock.patch.object(m, "generate_body_checked", return_value=None), \
         mock.patch.object(m, "publish_article"):
        out = m.build_and_publish(days=7, dry_run=True, ledger=led)
    assert out == []
    assert led.anomaly_counts() == {pl.FAIL_GENERATION: 2}
    with mock.patch("lib.publish_ledger.notify.error"):
        assert led.finish() == pl.EXIT_ANOMALY


def test_daily_budget_stop_is_not_an_anomaly():
    """日次予算に達した状態で残った候補は生成を呼ばずに打ち切り、正常終了する。"""
    from lib import api_budget
    from lib import publish_ledger as pl
    from lib.publish_ledger import PublishLedger
    led = PublishLedger("test")
    with mock.patch("lib.notify.error"):
        api_budget.stop_for_daily_cap(0.16, 0.15)
    try:
        with mock.patch.object(m, "daily_quota", return_value=2), \
             mock.patch.object(m, "fetch_candidates", return_value=_TWO_CANDIDATES), \
             mock.patch.object(m, "already_published", return_value=False), \
             mock.patch.object(m, "build_fact_sheet", return_value={**FACT, "stock_code": "6082"}), \
             mock.patch.object(m, "generate_body_checked", return_value=None) as gen, \
             mock.patch.object(m, "publish_article"):
            out = m.build_and_publish(days=7, dry_run=True, ledger=led)
    finally:
        api_budget.reset()
    assert out == []
    gen.assert_not_called()
    assert led.reasons == {pl.SKIP_DAILY_BUDGET: 1}
    assert led.unclassified == 0
    assert led.has_anomaly() is False, led.summary()


def test_ledger_counts_every_candidate():
    """公開できた候補・見送った候補の合計が候補数と一致する（未分類が残らない）。"""
    from lib.publish_ledger import PublishLedger
    led = PublishLedger("test")
    with mock.patch.object(m, "daily_quota", return_value=2), \
         mock.patch.object(m, "fetch_candidates", return_value=_TWO_CANDIDATES), \
         mock.patch.object(m, "already_published", side_effect=[True, False]), \
         mock.patch.object(m, "build_fact_sheet", return_value={**FACT, "stock_code": "6082"}), \
         mock.patch.object(m, "generate_body_checked", return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "publish_article"):
        m.build_and_publish(days=7, dry_run=True, ledger=led)
    assert led.published_count == 1
    assert led.unclassified == 0
    assert led.has_anomaly() is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"{len(tests)} tests passed")

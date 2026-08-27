"""ブログ記事自動投稿（web/publish_blog_articles）のロジックのユニットテスト。
ネットワーク（microCMS/yfinance/Supabase/Claude）は全てモックし、純粋なロジックのみ検証する。

実行: python3 tests/test_publish_blog_articles.py
"""
import os
import sys
import json
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.publish_blog_articles as m
from lib import api_budget


def _iso_days_ago(days: float) -> str:
    """ネガティブキャッシュ判定用のISO8601タイムスタンプ。"""
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def test_estimate_deal_amount_oku_calculation():
    with mock.patch.object(m, "shares_outstanding", return_value=1_000_000_000), \
         mock.patch.object(m, "get_price_at_date", return_value=2000.0):
        # 10億株 × 2000円 × 5% = 1000億円
        assert m.estimate_deal_amount_oku("7203", 5.0, "2026-07-20") == 1000.0


def test_estimate_deal_amount_oku_none_when_no_change():
    assert m.estimate_deal_amount_oku("7203", 0, "2026-07-20") is None


def test_estimate_deal_amount_oku_none_when_shares_missing():
    with mock.patch.object(m, "shares_outstanding", return_value=None):
        assert m.estimate_deal_amount_oku("7203", 5.0, "2026-07-20") is None


def test_shares_outstanding_retries_then_succeeds():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(
        side_effect=[Exception("rate limited"), Exception("rate limited"), {"sharesOutstanding": 5_000_000}]
    )
    with mock.patch("yfinance.Ticker", return_value=ticker), mock.patch("time.sleep"):
        assert m.shares_outstanding("7203") == 5_000_000.0


def test_shares_outstanding_falls_back_to_implied_shares_outstanding():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(return_value={"impliedSharesOutstanding": 3_000_000})
    with mock.patch("yfinance.Ticker", return_value=ticker):
        assert m.shares_outstanding("3269") == 3_000_000.0


def test_shares_outstanding_returns_none_after_exhausting_retries():
    ticker = mock.MagicMock()
    type(ticker).info = mock.PropertyMock(side_effect=Exception("rate limited"))
    with mock.patch("yfinance.Ticker", return_value=ticker), mock.patch("time.sleep"):
        assert m.shares_outstanding("7203") is None


def test_generate_article_body_parses_plain_json():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"body": "<p>本文</p>", "bodyEn": "<p>body</p>"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"body": "<p>本文</p>", "bodyEn": "<p>body</p>"}


def test_generate_article_body_strips_code_fence():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"body": "<p>本文</p>"})
    fenced = f"```json\n{raw}\n```"
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(fenced)):
        result = m.generate_article_body(fact_sheet)
    assert result == {"body": "<p>本文</p>"}


def test_generate_article_body_none_on_empty_body():
    fact_sheet = _fact_sheet()
    raw = json.dumps({"body": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.generate_article_body(fact_sheet) is None


def test_build_article_titles_buy_and_sell_templates():
    """タイトルは決定的テンプレ: 銘柄名（コード）・提出者・保有比率・「大量保有報告書」を必ず含む。"""
    buy = m.build_article_titles({"stock_name": "テスト自動車", "stock_code": "7203",
                                  "filer_name": "テストファンド", "holding_ratio": 8.5,
                                  "direction": "buy", "ratio_change_pct": 1.2})
    assert buy["title"] == "テスト自動車（7203）、テストファンドが保有比率8.5%に引き上げ｜大量保有報告書"
    sell = m.build_article_titles({"stock_name": "テスト自動車", "stock_code": "7203",
                                   "filer_name": "テストファンド", "holding_ratio": 4.98,
                                   "direction": "sell", "ratio_change_pct": 2.0})
    assert "保有比率4.98%に引き下げ" in sell["title"]
    assert sell["title"].endswith("｜大量保有報告書")


def test_build_article_titles_new_holding_uses_ratio_change_heuristic():
    """変化幅=今回比率（過去開示なし）は「新規保有」表現になる。"""
    new = m.build_article_titles({"stock_name": "テスト商事", "stock_code": "9999",
                                  "filer_name": "テストファンド", "holding_ratio": 6.0,
                                  "direction": "buy", "ratio_change_pct": 6.0})
    assert "テストファンドが6%を新規保有" in new["title"]


def test_build_article_titles_truncates_long_filer_name():
    long_filer = "とても長い名前の資産運用株式会社" * 4
    result = m.build_article_titles({"stock_name": "テスト", "stock_code": "1234",
                                     "filer_name": long_filer, "holding_ratio": 5.0,
                                     "direction": "buy", "ratio_change_pct": 1.0})
    assert len(result["title"]) <= m.MAX_TITLE_LEN
    assert "…" in result["title"]
    assert result["title"].endswith("｜大量保有報告書")


def test_build_article_titles_uses_english_names_when_given():
    fs = {"stock_name": "テスト自動車", "stock_code": "7203", "filer_name": "テストファンド",
          "holding_ratio": 8.5, "direction": "buy", "ratio_change_pct": 1.2}
    result = m.build_article_titles(fs, stock_name_en="Test Motor", filer_name_en="Test Fund")
    assert result["titleEn"] == "Test Fund Raises Stake in Test Motor (7203) to 8.5% | Large Shareholding Report"
    # 英語名が無ければ日本語名のまま
    fallback = m.build_article_titles(fs)
    assert "テスト自動車 (7203)" in fallback["titleEn"]


def test_classify_filer_returns_cached_master_row_without_calling_claude():
    """edinet_filer_classificationに登録済みの提出者はClaudeを呼ばずマスターの値を返す。"""
    cached = {"category": "外資系伝統運用会社", "is_foreign": True, "description": "米大手運用会社"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.classify_filer("Ｆｉｄｅｌｉｔｙ")
    assert result == cached
    assert select_mock.called
    assert not anthropic_mock.called


def test_classify_filer_asks_claude_and_persists_when_not_cached():
    """マスター未登録の提出者はClaudeに判定させ、結果をedinet_filer_classificationへ保存する。"""
    raw = json.dumps({"category": "アクティビスト", "is_foreign": True, "description": "海外の物言う株主"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.classify_filer("新規ファンド")
    assert result == {"category": "アクティビスト", "is_foreign": True, "description": "海外の物言う株主"}
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["filer_name"] == "新規ファンド"
    assert saved_rows[0]["confidence"] == "low"


def test_classify_filer_falls_back_to_sonota_on_invalid_category():
    """Claudeが決められた選択肢以外を返したら「その他」に丸める。"""
    raw = json.dumps({"category": "謎の分類", "is_foreign": False, "description": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.classify_filer("謎の提出者")
    assert result["category"] == "その他"


def test_build_and_publish_includes_sell_and_tags_them():
    """売り方向（概要のキーワード or 保有比率の減少で判定）も除外せず記事化し、
    tagsに"売り"を付与して買いと区別する（買い側はtagsを変えない後方互換）。"""
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
        {"issuer_code": "9999", "name": "テスト商事", "filer_name": "アセットマネジメント株式会社",
         "holding_ratio": 6.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "変更報告書"},
        {"issuer_code": "1234", "name": "売却テスト", "filer_name": "ファンド株式会社",
         "holding_ratio": 4.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "株式の譲渡・売却による変更報告書"},
        {"issuer_code": "6502", "name": "東芝型テスト", "filer_name": "キオクシアファンド",
         "holding_ratio": 15.10, "holding_ratio_prior": 16.10, "disc_date": "2026-07-20",
         "doc_type_code": "360", "doc_description": "変更報告書"},  # 概要はキーワード無しだが比率減少=売り
    ]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d, prior=None, amend=False: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            side_effect=[
                                {"category": "外資系伝統運用会社", "is_foreign": True, "description": ""},
                                {"category": "個人", "is_foreign": False, "description": ""},
                                {"category": "国内アセットマネジメント", "is_foreign": False, "description": ""},
                                {"category": "PE・メザニンファンド", "is_foreign": False, "description": ""},
                            ]), \
         mock.patch.object(m, "generate_article_body_checked",
                            return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=4, dry_run=False)

    assert len(results) == 4  # 売りも除外されない
    # |比率|降順: 6502(15.10) > 7203(8.5) > 9999(6.0) > 1234(4.0)
    assert [r["stockCode"] for r in results] == ["6502", "7203", "9999", "1234"]
    by_code = {r["stockCode"]: r for r in results}
    assert by_code["6502"]["tags"] == "EDINET,自動生成,売り"  # 比率減少による売り判定
    assert by_code["1234"]["tags"] == "EDINET,自動生成,売り"  # キーワードによる売り判定
    assert by_code["7203"]["tags"] == "EDINET,自動生成"  # 買いはtags不変
    assert by_code["9999"]["tags"] == "EDINET,自動生成"
    assert results[0]["dealDate"] == "2026-07-20T00:00:00.000Z"
    assert results[0]["dealAmount"] == 12.3
    # タイトルはテンプレ生成。ratio_change=ratioのモックでも、報告書種別が変更報告書なら
    # 「新規保有」にはならない（大量保有報告書の7203だけが新規表現）。
    assert by_code["7203"]["title"] == "テスト自動車（7203）、個人 太郎が8.5%を新規保有｜大量保有報告書"
    assert "新規保有" not in by_code["9999"]["title"]
    assert "保有比率6%に引き上げ" in by_code["9999"]["title"]
    # ratioChangePct: 買いは正、売りは負で送る
    assert by_code["7203"]["ratioChangePct"] == 8.5
    assert by_code["6502"]["ratioChangePct"] == -15.1


def test_build_and_publish_uses_disclosed_unit_price_over_market_estimate():
    """短期大量譲渡は開示された単価×株数の実額を使い、株価からの概算に上書きさせない
    （実例: 日立製作所→日立建機は開示日終値ベースの概算1,274.9億円に対し実額1,121.8億円）。"""
    holdings = [
        {"issuer_code": "6305", "name": "日立建機", "filer_name": "株式会社日立製作所",
         "holding_ratio": 0.0, "holding_ratio_prior": 9.98, "disc_date": "2026-08-25",
         "doc_type_code": "350", "doc_description": "変更報告書（短期大量譲渡）",
         "short_term_transfers": [
             {"date": "2026-08-19", "security_type": "普通株式", "shares": 21462310,
              "ratio": 9.98, "venue": "市場外", "action": "処分",
              "counterparty": "SMBC日興証券株式会社", "unit_price": 5227.0,
              "unit_price_note": None},
         ]},
    ]
    captured = {}
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=9.98), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=1274.9), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "事業会社", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked",
                            side_effect=lambda fs: captured.update(fs) or {"body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid6305"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)

    assert results[0]["dealAmount"] == 1121.8  # 概算の1274.9ではなく開示単価ベースの実額
    assert captured["deal_amount_label"] == "売却金額"  # 実額なので「推定」を付けない
    assert captured["transfers"]["counterparties"] == ["SMBC日興証券株式会社"]


def test_build_and_publish_keeps_estimate_when_transfer_table_is_inconclusive():
    """取得と処分が混在する60日間の記録からは差引きが復元できないため概算を使う。"""
    holdings = [
        {"issuer_code": "1234", "name": "混在テスト", "filer_name": "ファンド株式会社",
         "holding_ratio": 3.0, "holding_ratio_prior": 8.09, "disc_date": "2026-08-25",
         "doc_type_code": "360", "doc_description": "変更報告書（短期大量譲渡）",
         "short_term_transfers": [
             {"date": "2026-07-15", "shares": 453800, "ratio": 5.09, "venue": "市場外",
              "action": "取得", "counterparty": None, "unit_price": 4730.0,
              "security_type": "株券", "unit_price_note": None},
             {"date": "2026-07-16", "shares": 10000, "ratio": 0.11, "venue": "市場内",
              "action": "処分", "counterparty": "市場内取引のため不明", "unit_price": 4700.0,
              "security_type": "株券", "unit_price_note": None},
         ]},
    ]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=5.09), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=99.9), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "その他", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked", return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid1234"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)

    assert results[0]["dealAmount"] == 99.9


def test_build_and_publish_skips_when_already_published():
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=True):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []


def _already_published_response(contents):
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"contents": contents}
    return resp


def test_already_published_true_when_filer_and_ratio_change_match():
    # 株価キャッシュ更新でdealAmountが大きくズレても、提出者名＋比率変化幅が一致すれば重複と判定する
    # （2026-08-17の17件重複投稿の再発防止）
    contents = [{"id": "a", "dealDate": "2026-08-17T00:00:00.000Z", "dealAmount": 19.6,
                 "filerName": "テスト運用", "ratioChangePct": 2.5}]
    with mock.patch.object(m.requests, "get", return_value=_already_published_response(contents)):
        assert m.already_published("2492", "2026-08-17", 18.2, "テスト運用", 2.5) is True


def test_already_published_false_for_different_filer_same_day():
    contents = [{"id": "a", "dealDate": "2026-08-17T00:00:00.000Z", "dealAmount": 20.8,
                 "filerName": "提出者A", "ratioChangePct": 3.0}]
    with mock.patch.object(m.requests, "get", return_value=_already_published_response(contents)):
        assert m.already_published("6237", "2026-08-17", 20.8, "提出者B", 3.0) is False


def test_already_published_false_for_same_filer_different_ratio_change():
    # 同一提出者が同日に別々の報告書を出すケース（実例: 2936 2025-08-13の橋本舜2件）は別イベント
    contents = [{"id": "a", "dealDate": "2025-08-13T00:00:00.000Z", "dealAmount": 90.9,
                 "filerName": "橋本 舜", "ratioChangePct": 34.62}]
    with mock.patch.object(m.requests, "get", return_value=_already_published_response(contents)):
        assert m.already_published("2936", "2025-08-13", 84.7, "橋本 舜", 32.23) is False


def test_already_published_falls_back_to_deal_amount_for_legacy_articles():
    # filerName未保存の旧記事は従来通りdealAmount±0.05億円で突き合わせる
    contents = [{"id": "a", "dealDate": "2026-07-20T00:00:00.000Z", "dealAmount": 10.0,
                 "filerName": None, "ratioChangePct": None}]
    with mock.patch.object(m.requests, "get", return_value=_already_published_response(contents)):
        assert m.already_published("7203", "2026-07-20", 10.02, "個人 太郎", 5.0) is True
        assert m.already_published("7203", "2026-07-20", 11.5, "個人 太郎", 5.0) is False


def test_build_and_publish_passes_dedup_keys_to_already_published():
    holdings = [{"issuer_code": "6502", "name": "テスト電機", "filer_name": "売却 花子",
                 "holding_ratio": 4.9, "holding_ratio_prior": 20.0, "disc_date": "2026-07-21",
                 "doc_type_code": "350", "doc_description": "変更報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "ratio_change_pct", return_value=15.1), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=30.0), \
         mock.patch.object(m, "already_published", return_value=True) as ap:
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []
    # 売り方向はratioChangePctを負値で保存するため、突き合わせも負値で行う。
    # 末尾のTrueは「その日その提出者の開示が1件だけ」（=比率変化幅が一致しなくても同一開示）。
    ap.assert_called_once_with("6502", "2026-07-21", 30.0, "売却 花子", -15.1, True)


def test_build_and_publish_skips_when_amount_unestimable():
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=None):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []


def test_ratio_change_pct_uses_disclosure_prior_ratio():
    """開示自体が直前保有割合を持つ場合はDB履歴を引かずにそれを使う。"""
    with mock.patch.object(m, "get_edinet_large_holdings_recent") as hist:
        assert m.ratio_change_pct("6976", "F", 4.41, "2026-08-18", 15.22) == 10.81
    hist.assert_not_called()


def test_ratio_change_pct_counts_full_exit_as_full_prior_ratio():
    """全売却（保有比率0%）は「変化なし」ではなく前回比率分の変化として扱う
    （実例: 2026-08-17、三菱商事のＴＯＹＯ ＴＩＲＥ 20%→0%が不投稿になった）。"""
    with mock.patch.object(m, "get_edinet_large_holdings_recent"):
        assert m.ratio_change_pct("5105", "三菱商事株式会社", 0.0, "2026-08-17", 20.0) == 20.0


def test_ratio_change_pct_falls_back_to_history_without_prior():
    history = [
        {"filer_name": "F", "disc_date": "2026-07-01", "holding_ratio": 5.0},
        {"filer_name": "F", "disc_date": "2026-07-10", "holding_ratio": 6.5},
        {"filer_name": "他社", "disc_date": "2026-07-15", "holding_ratio": 30.0},
    ]
    with mock.patch.object(m, "get_edinet_large_holdings_recent", return_value=history):
        assert m.ratio_change_pct("7203", "F", 8.0, "2026-07-20") == 1.5


def test_is_new_holding_false_for_full_exit():
    """比率0%・変化幅=前回比率の全売却を「0%を新規保有」と誤判定しない。"""
    assert m.is_new_holding({"holding_ratio": 0.0, "ratio_change_pct": 20.0, "prior_ratio": 20.0}) is False
    assert m.is_new_holding({"holding_ratio": 5.05, "ratio_change_pct": 5.05, "prior_ratio": 0.0}) is True
    assert m.is_new_holding({"holding_ratio": 5.05, "ratio_change_pct": 5.05, "prior_ratio": None}) is True


def test_is_new_holding_false_for_amendment_without_prior_ratio():
    """変更報告書は前回比率が取れなくても「新規保有」にしない（提出者は既に5%以上を保有）。

    実例: ＦＭＲ ＬＬＣ等の特例報告は直前保有割合がXBRLに無く、DBに過去開示も無いため、
    従来のヒューリスティックでは「X%を新規保有」＋全量ぶんの推定金額になっていた。"""
    amendment = {"stock_name": "テスト", "holding_ratio": 7.6, "ratio_change_pct": 7.6,
                 "prior_ratio": None, "doc_type_label": "変更報告書"}
    assert m.is_new_holding(amendment) is False
    # 大量保有報告書（新規届出）は従来どおり新規保有と判定する
    first = dict(amendment, doc_type_label="大量保有報告書")
    assert m.is_new_holding(first) is True


def test_ratio_change_pct_returns_none_for_amendment_without_prior_or_history():
    """変更報告書で前回比率も過去開示も無いときは、全量を動いたとみなさずNoneを返す。"""
    with mock.patch.object(m, "get_edinet_large_holdings_recent", return_value=[]):
        assert m.ratio_change_pct("3878", "ＦＭＲ　ＬＬＣ", 7.6, "2026-08-18", None, True) is None
        # 新規の大量保有報告書は従来どおり今回比率をそのまま変化幅として扱う
        assert m.ratio_change_pct("3878", "ＦＭＲ　ＬＬＣ", 7.6, "2026-08-18", None, False) == 7.6


def test_build_article_titles_amendment_without_prior_is_not_new_holding():
    """変更報告書の記事タイトルが「X%を新規保有」にならず、引き上げ/引き下げ表現になる。"""
    fs = {"stock_name": "テスト商事", "stock_code": "9999", "filer_name": "テストファンド",
          "holding_ratio": 7.6, "direction": "buy", "ratio_change_pct": 7.6,
          "prior_ratio": None, "doc_type_label": "変更報告書"}
    title = m.build_article_titles(fs)["title"]
    assert "新規保有" not in title
    assert "テストファンドが保有比率7.6%に引き上げ" in title
    assert "Takes 7.6% Stake" not in m.build_article_titles(fs)["titleEn"]


def test_build_and_publish_skips_amendment_with_unknown_change():
    """前回比率も過去開示も無い変更報告書は、誤った「新規保有」記事にせずスキップする。"""
    # PRIOR_RATIO_WAIT_DAYSの待ちを過ぎてもpriorが入らない開示（特例報告に多い）を想定する
    holdings = [{"issuer_code": "3878", "name": "テスト製紙", "filer_name": "ＦＭＲ　ＬＬＣ",
                 "holding_ratio": 7.6, "disc_date": "2026-07-01", "doc_type_code": "350",
                 "doc_description": "変更報告書（特例対象株券等）"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "get_edinet_large_holdings_recent", return_value=[]), \
         mock.patch.object(m, "estimate_deal_amount_oku") as est, \
         mock.patch.object(m, "publish_article") as pub:
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []
    est.assert_not_called()  # 金額の概算（yfinance）まで到達しない
    pub.assert_not_called()


def test_estimate_deal_amount_falls_back_to_yfinance_price():
    """価格キャッシュ（スクリーニング対象ユニバースのみ）に無い銘柄はyfinanceで株価を補う。"""
    with mock.patch.object(m, "shares_outstanding", return_value=10_000_000), \
         mock.patch.object(m, "get_price_at_date", return_value=None), \
         mock.patch.object(m, "close_price_from_yfinance", return_value=1000.0) as yf:
        # 1000万株 × 1000円 × 5% = 5億円
        assert m.estimate_deal_amount_oku("603A", 5.0, "2026-08-18") == 5.0
    yf.assert_called_once()


def test_build_and_publish_skips_when_ratio_unchanged():
    """比率が動いていない開示は「金額を概算できない」ではなく変化なしとしてスキップする。"""
    holdings = [{"issuer_code": "2371", "name": "カカクコム", "filer_name": "株式会社デジタルガレージ",
                 "holding_ratio": 20.64, "holding_ratio_prior": 20.64, "disc_date": "2026-08-17",
                 "doc_type_code": "360", "doc_description": "変更報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "estimate_deal_amount_oku") as est:
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []
    est.assert_not_called()


def test_should_wait_for_prior_ratio_true_for_fresh_change_report():
    """変更報告書なのに直前保有割合が未取得＝XBRL未反映。その便では記事化しない。"""
    from datetime import date as _date
    assert m.should_wait_for_prior_ratio("変更報告書", None, "2026-08-18",
                                         today=_date(2026, 8, 18)) is True
    assert m.should_wait_for_prior_ratio("変更報告書（特例対象株券等）", None, "2026-08-18",
                                         today=_date(2026, 8, 19)) is True


def test_should_wait_for_prior_ratio_false_after_wait_days():
    """待機日数を過ぎても埋まらない開示はXBRLの書式差とみなし、履歴からの再導出で記事化する。"""
    from datetime import date as _date
    assert m.should_wait_for_prior_ratio("変更報告書", None, "2026-08-18",
                                         today=_date(2026, 8, 20)) is False


def test_should_wait_for_prior_ratio_false_for_new_filing_and_when_prior_exists():
    """大量保有報告書（新規）はそもそも直前保有割合を持たない。前回比率が有る開示も待たない。"""
    from datetime import date as _date
    assert m.should_wait_for_prior_ratio("大量保有報告書", None, "2026-08-18",
                                         today=_date(2026, 8, 18)) is False
    assert m.should_wait_for_prior_ratio("変更報告書", 10.72, "2026-08-18",
                                         today=_date(2026, 8, 18)) is False


def test_build_and_publish_defers_change_report_without_prior_ratio():
    """直前保有割合が未取得の変更報告書は、比率全量を変化幅とみなして記事化してしまうため
    その便では投稿せず次の便へ持ち越す（実例: 2026-08-18のセイコーグループ8050が
    実際は-0.15ptなのに-10.57pt・966.1億円として公開された）。"""
    from datetime import date as _date
    today = _date.today().isoformat()
    holdings = [{"issuer_code": "8050", "name": "セイコーグループ", "filer_name": "三光起業株式会社",
                 "holding_ratio": 10.57, "holding_ratio_prior": None, "disc_date": today,
                 "doc_type_code": "360", "doc_description": "変更報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct") as chg, \
         mock.patch.object(m, "estimate_deal_amount_oku") as est:
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert results == []
    chg.assert_not_called()
    est.assert_not_called()


def test_build_and_publish_publishes_material_correction_without_amount():
    """大幅な訂正報告書は記事化する。ただし売買を伴わないため推定金額は付けず(dealAmount=0)、
    tagsに"訂正"を立ててフロント側で金額の代わりに「訂正」と表示させる。"""
    holdings = [{"issuer_code": "6976", "name": "太陽誘電", "filer_name": "Situational Awareness LP",
                 "holding_ratio": 4.41, "holding_ratio_prior": 15.22, "disc_date": "2026-08-18",
                 "doc_type_code": "360", "doc_description": "訂正報告書（大量保有報告書・変更報告書）"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "estimate_deal_amount_oku") as est, \
         mock.patch.object(m, "get_pit_ranking_snapshot", return_value=None), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "外資系伝統運用会社", "is_foreign": True, "description": ""}), \
         mock.patch.object(m, "get_company_description", return_value=""), \
         mock.patch.object(m, "get_filer_profile", return_value=""), \
         mock.patch.object(m, "generate_article_body_checked", return_value={"body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article", return_value=None), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid999"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert len(results) == 1
    article = results[0]
    est.assert_not_called()  # 訂正に推定売買金額は付けない
    assert article["dealAmount"] == 0.0
    assert article["tags"] == "EDINET,自動生成,訂正,売り"
    assert article["ratioChangePct"] == -10.81
    assert article["title"] == "太陽誘電（6976）、Situational Awareness LPが保有比率を4.41%に訂正｜訂正報告書"


def test_already_published_true_for_unique_filing_even_if_ratio_differs():
    """その日その提出者の開示が1件だけなら、比率変化幅の算出が変わっても再投稿しない。"""
    resp = _already_published_response([
        {"id": "a1", "dealDate": "2026-08-18T00:00:00.000Z", "dealAmount": 10.0,
         "filerName": "F", "ratioChangePct": 5.0},
    ])
    with mock.patch.object(m.requests, "get", return_value=resp):
        assert m.already_published("6976", "2026-08-18", 10.0, "F", -10.81, unique_filing=True) is True
        assert m.already_published("6976", "2026-08-18", 10.0, "F", -10.81, unique_filing=False) is False


def test_get_featured_article_ids_picks_top_deal_amount():
    """kujira-watch側getFeaturedArticles()と同じロジック: プール（microCMS側で
    -dealDate,-dealAmount順に取得済み）の中から推定取引金額が大きい順に先頭count件を採用する。"""
    pool = [
        {"id": "today-big", "dealAmount": 50},
        {"id": "today-small", "dealAmount": 1},
        {"id": "older-huge", "dealAmount": 999},
        {"id": "older-medium", "dealAmount": 30},
    ]
    resp = _FakeResponse(200, "", {"contents": pool})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", return_value=resp):
        ids = m.get_featured_article_ids(pool_size=20, count=2)
    assert ids == {"older-huge", "today-big"}


def test_get_featured_article_ids_returns_empty_set_on_http_error():
    resp = _FakeResponse(500, "server error")
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", return_value=resp):
        assert m.get_featured_article_ids() == set()


def test_get_featured_article_ids_returns_empty_set_on_exception():
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=Exception("timeout")):
        assert m.get_featured_article_ids() == set()


class _FakeResponse:
    def __init__(self, status_code, text, json_data=None, content=b""):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_publish_article_retries_as_array_on_type_mismatch():
    """セレクトフィールドが複数選択(配列)設定の場合、'has unexpected data type' を
    検知してその項目だけ配列に包んで一度だけ再送信する。"""
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(201, "", {"id": "retried-id"}),
    ]
    payload = {"title": "t", "dealType": "個人"}
    with mock.patch.object(m, "_post_once", side_effect=responses) as post_mock:
        content_id = m.publish_article(payload)
    assert content_id == "retried-id"
    assert post_mock.call_count == 2
    retried_payload = post_mock.call_args_list[1].args[0]
    assert retried_payload["dealType"] == ["個人"]


def test_publish_article_drops_non_string_field_on_type_mismatch():
    """eyecatch等のオブジェクト値フィールドは配列化では直せないため、
    そのフィールドを除外して再送信し、記事自体は投稿される。"""
    responses = [
        _FakeResponse(400, '{"message":"\'eyecatch\' has unexpected data type."}'),
        _FakeResponse(201, "", {"id": "no-eyecatch-id"}),
    ]
    payload = {"title": "t", "eyecatch": {"url": "https://example.test/x.png"}}
    with mock.patch.object(m, "_post_once", side_effect=responses) as post_mock:
        content_id = m.publish_article(payload)
    assert content_id == "no-eyecatch-id"
    assert post_mock.call_count == 2
    retried_payload = post_mock.call_args_list[1].args[0]
    assert "eyecatch" not in retried_payload


def test_publish_article_gives_up_when_same_field_fails_twice():
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
    ]
    payload = {"title": "t", "dealType": "個人"}
    with mock.patch.object(m, "_post_once", side_effect=responses):
        content_id = m.publish_article(payload)
    assert content_id is None


def test_update_article_retries_as_array_on_type_mismatch():
    """update_article()（PATCH）もpublish_article()（POST）と同じ型不一致リトライを行う
    （tools/reclassify_blog_articles.py の一括再分類・tools/rewrite_thin_blog_articles.py の
    本文リライトで使う）。"""
    responses = [
        _FakeResponse(400, '{"message":"\'dealType\' has unexpected data type."}'),
        _FakeResponse(200, "", {"id": "content-1"}),
    ]
    payload = {"dealType": "アクティビスト"}
    with mock.patch.object(m, "_patch_once", side_effect=responses) as patch_mock:
        ok = m.update_article("content-1", payload)
    assert ok is True
    assert patch_mock.call_count == 2
    retried_payload = patch_mock.call_args_list[1].args[1]
    assert retried_payload["dealType"] == ["アクティビスト"]


def test_update_article_returns_false_on_failure():
    responses = [_FakeResponse(400, '{"message":"invalid"}')]
    with mock.patch.object(m, "_patch_once", side_effect=responses):
        ok = m.update_article("content-1", {"dealType": "その他"})
    assert ok is False


class _FakeDraw:
    """textbboxの幅を文字数×10pxで返すダミーdraw（実フォント無しで折り返しロジックだけ検証する）。"""

    def textbbox(self, xy, text, font=None):
        return (0, 0, len(text) * 10, 20)


def test_wrap_text_lines_breaks_on_width():
    lines = m._wrap_text_lines(_FakeDraw(), "あいうえおかきくけこ", font=None, max_width=50)
    assert lines == ["あいうえお", "かきくけこ"]


def test_wrap_text_lines_respects_max_lines():
    lines = m._wrap_text_lines(_FakeDraw(), "あ" * 30, font=None, max_width=50, max_lines=2)
    assert len(lines) == 2


def test_search_pexels_photo_returns_none_without_api_key():
    m._PEXELS_CANDIDATE_CACHE.clear()
    with mock.patch.object(m, "PEXELS_API_KEY", ""):
        assert m.search_pexels_photo("finance") is None


def test_search_pexels_photo_returns_bytes_and_photographer_on_success():
    m._PEXELS_CANDIDATE_CACHE.clear()
    search_resp = _FakeResponse(200, "", {"photos": [{
        "src": {"large": "https://example.test/a.jpg"}, "photographer": "Jane Doe",
    }]})
    photo_resp = _FakeResponse(200, "", content=b"fake-image-bytes")
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=[search_resp, photo_resp]):
        result = m.search_pexels_photo("finance")
    assert result == {"bytes": b"fake-image-bytes", "photographer": "Jane Doe"}


def test_search_pexels_photo_defaults_photographer_when_missing():
    m._PEXELS_CANDIDATE_CACHE.clear()
    search_resp = _FakeResponse(200, "", {"photos": [{"src": {"large": "https://example.test/a.jpg"}}]})
    photo_resp = _FakeResponse(200, "", content=b"fake-image-bytes")
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=[search_resp, photo_resp]):
        result = m.search_pexels_photo("finance")
    assert result["photographer"] == "Pexels"


def test_search_pexels_photo_returns_none_when_no_results():
    m._PEXELS_CANDIDATE_CACHE.clear()
    search_resp = _FakeResponse(200, "", {"photos": []})
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", return_value=search_resp):
        assert m.search_pexels_photo("finance") is None


def test_search_pexels_photo_returns_none_on_exception():
    m._PEXELS_CANDIDATE_CACHE.clear()
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=Exception("timeout")):
        assert m.search_pexels_photo("finance") is None


def test_search_pexels_photo_picks_different_photos_for_different_seeds():
    """同じ分類でも記事ごとに違う写真を引く（以前は常にphotos[0]で全記事が同じ写真だった）。"""
    photos = [{"src": {"large": f"https://example.test/{i}.jpg"}, "photographer": f"p{i}"}
              for i in range(20)]
    picked = set()
    for seed in ("A|X|2026-01-01", "B|Y|2026-01-02", "C|Z|2026-01-03"):
        m._PEXELS_CANDIDATE_CACHE.clear()
        with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
             mock.patch("requests.get", side_effect=[
                 _FakeResponse(200, "", {"photos": photos}),
                 _FakeResponse(200, "", content=b"img"),
             ]):
            picked.add(m.search_pexels_photo("finance", seed=seed)["photographer"])
    assert len(picked) > 1


def test_search_pexels_photo_is_stable_for_same_seed():
    """同じ記事を再生成しても写真は入れ替わらない。"""
    photos = [{"src": {"large": f"https://example.test/{i}.jpg"}, "photographer": f"p{i}"}
              for i in range(20)]
    results = []
    for _ in range(2):
        m._PEXELS_CANDIDATE_CACHE.clear()
        with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
             mock.patch("requests.get", side_effect=[
                 _FakeResponse(200, "", {"photos": photos}),
                 _FakeResponse(200, "", content=b"img"),
             ]):
            results.append(m.search_pexels_photo("finance", seed="A|X|2026-01-01")["photographer"])
    assert results[0] == results[1]


def test_search_pexels_photo_caches_candidates_per_query():
    """同じクエリの2記事目は検索APIを叩き直さない（Pexels無料枠200req/時を守るため）。"""
    m._PEXELS_CANDIDATE_CACHE.clear()
    photos = [{"src": {"large": "https://example.test/a.jpg"}, "photographer": "p"}]
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch("requests.get", side_effect=[
             _FakeResponse(200, "", {"photos": photos}),
             _FakeResponse(200, "", content=b"img"),
             _FakeResponse(200, "", content=b"img"),
         ]) as g:
        m.search_pexels_photo("finance", seed="a")
        m.search_pexels_photo("finance", seed="b")
    search_calls = [c for c in g.call_args_list if "api.pexels.com" in c.args[0]]
    assert len(search_calls) == 1


def test_wrap_text_lines_keeps_number_token_whole():
    """「13.41%」が「13.」「41%」に割れて別の数字に読めないようにする。"""
    lines = m._wrap_text_lines(_FakeDraw(), "あいうえお13.41%", font=None, max_width=70)
    assert lines == ["あいうえお", "13.41%"]


def test_wrap_text_lines_breaks_number_longer_than_line():
    """1行に収まらない数字は無限に送れないのでその場で折る。"""
    lines = m._wrap_text_lines(_FakeDraw(), "1234567890", font=None, max_width=50)
    assert lines == ["12345", "67890"]


def test_eyecatch_stock_line_variants():
    """保有比率0%は全売却（売り記事）なら「全株売却」、それ以外は数字を出さない。
    素の「0.00%」はデータ欠損に見えるので焼き込まない。"""
    assert m._stock_line_text({"stock_name": "A", "holding_ratio": 13.41}, "▼ 売却") == "A　13.41%"
    assert m._stock_line_text({"stock_name": "A", "holding_ratio": 0.0}, "▼ 売却") == "A　全株売却"
    assert m._stock_line_text({"stock_name": "A", "holding_ratio": 0.0}, "🏦 自社株買い") == "A"
    assert m._stock_line_text({"stock_name": "A", "holding_ratio": None}, "▲ 買い増し") == "A"


def test_upload_eyecatch_returns_url_on_success():
    resp = _FakeResponse(201, "", {"url": "https://images.microcms-assets.io/assets/x/y.png"})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        url = m.upload_eyecatch(b"png-bytes")
    assert url == "https://images.microcms-assets.io/assets/x/y.png"


def test_upload_eyecatch_returns_none_on_failure():
    resp = _FakeResponse(403, "forbidden")
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        assert m.upload_eyecatch(b"png-bytes") is None


_SAMPLE_EYECATCH_CARD = {
    "filer_name": "Oasis Management",
    "stock_name": "アインHD",
    "holding_ratio": 20.93,
    "badge_label": "📈 買い増し",
    "disc_date": "2026-07-20",
}


def test_build_eyecatch_for_article_none_without_pexels_key():
    with mock.patch.object(m, "PEXELS_API_KEY", ""):
        assert m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD) is None


def test_build_eyecatch_for_article_none_when_generation_fails():
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch.object(m, "generate_eyecatch_image", return_value=None):
        assert m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD) is None


def test_build_eyecatch_for_article_returns_url_string_on_success():
    """microCMSの画像フィールドはPOST時にメディアURL文字列を要求する（{"url": ...}の
    オブジェクトだと 'eyecatch' has unexpected data type で除外される）。"""
    with mock.patch.object(m, "PEXELS_API_KEY", "dummy"), \
         mock.patch.object(m, "generate_eyecatch_image", return_value=b"jpg-bytes"), \
         mock.patch.object(m, "upload_eyecatch", return_value="https://images.microcms-assets.io/x.jpg"):
        result = m.build_eyecatch_for_article("その他", _SAMPLE_EYECATCH_CARD)
    assert result == "https://images.microcms-assets.io/x.jpg"


def test_build_and_publish_includes_eyecatch_when_available():
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
    ]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article",
                            return_value={"url": "https://images.microcms-assets.io/x.png"}) as eyecatch_mock, \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert results[0]["eyecatch"] == {"url": "https://images.microcms-assets.io/x.png"}
    eyecatch_mock.assert_called_once_with("個人", {
        "filer_name": "個人 太郎",
        "stock_name": "テスト自動車",
        "holding_ratio": 8.5,
        "badge_label": "📈 新規取得",
        "disc_date": "2026-07-20",
    })


def test_build_and_publish_skips_eyecatch_on_dry_run():
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
    ]
    with mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_eyecatch_for_article") as eyecatch_mock:
        m.build_and_publish(days=3, max_articles=3, dry_run=True)

    eyecatch_mock.assert_not_called()


def test_build_and_publish_stops_early_on_permission_error():
    """1件目でAPIキーの権限エラーが出たら、2件目以降はClaude呼び出しごと打ち切る
    （無駄なトークン消費を防ぐ）。"""
    holdings = [
        {"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
         "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
         "doc_description": "大量保有報告書"},
        {"issuer_code": "9999", "name": "テスト商事", "filer_name": "アセットマネジメント株式会社",
         "holding_ratio": 6.0, "disc_date": "2026-07-20", "doc_type_code": "360",
         "doc_description": "変更報告書"},
    ]
    generate_calls = []

    def _track_generate(fact_sheet):
        generate_calls.append(fact_sheet)
        return {"title": "テストタイトル", "body": "<p>本文</p>"}

    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d, prior=None, amend=False: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "その他", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked", side_effect=_track_generate), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article",
                            side_effect=m.MicroCMSPermissionError("HTTP 400: forbidden")):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert results == []
    assert len(generate_calls) == 1  # 2件目はClaudeを呼ばずに打ち切られる


def _fact_sheet():
    return {"stock_name": "テスト", "stock_code": "7203", "filer_name": "X",
            "doc_type_label": "大量保有報告書", "holding_ratio": 8.5,
            "disc_date": "2026-07-20", "deal_amount_oku": 12.3}


def test_get_pit_ranking_snapshot_queries_as_of_disc_date():
    """記事公開時点(post-hoc)ではなく、開示日以前で直近のスナップショットを取る
    （先読みバイアス防止、CLAUDE.md PIT規律）。"""
    with mock.patch.object(m.sb, "select_one", return_value={"close": 3000}) as select_mock:
        result = m.get_pit_ranking_snapshot("7203", "2026-07-20")
    assert result == {"close": 3000}
    query = select_mock.call_args.args[1]
    assert "code=eq.7203" in query
    assert "date=lte.2026-07-20" in query


def _capturing_client(text):
    calls = []

    class _Block:
        # web_search併用時はcontentに検索結果ブロックも混ざるため、本番側はtypeで
        # テキストブロックだけを拾う。ダミーもtypeを持たせる。
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Resp(text)

    class _Client:
        def __init__(self):
            self.messages = _Messages()

    return _Client(), calls


def test_generate_article_body_includes_close_price_but_never_the_drop_model():
    """開示日終値は開示原本と突き合わせられる事実なのでプロンプトに渡す。
    一方で下落モデルの水準は渡さない（モデルの説明ページがサイトに無いまま
    検証不能な独自指標をYMYLの判断材料として本文に書かせないため、2026-08-25に廃止）。"""
    fact_sheet = _fact_sheet()
    fact_sheet["context_close"] = 3000.0
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "3,000円" in prompt
    assert "下落リスク水準" not in prompt
    assert "弊社モデル" not in prompt


def test_generate_article_body_omits_context_when_unavailable():
    fact_sheet = _fact_sheet()  # context_close 無し
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "開示日時点の株価" not in prompt


def test_generate_article_body_includes_ratio_increase_when_available():
    """既存開示からの増加分が分かる場合、変化幅(ポイント)をプロンプトに織り込む。"""
    fact_sheet = _fact_sheet()
    fact_sheet["ratio_change_pct"] = 2.48
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "これまでの開示から2.48ポイント増加" in prompt


def test_generate_article_body_describes_new_position_when_change_equals_ratio():
    """過去開示が無く変化幅=保有比率そのものの場合は「新規保有」の文脈で伝える（実際には
    5%未満だった保証は無いため、データで確認できる範囲の表現に留める）。"""
    fact_sheet = _fact_sheet()
    fact_sheet["ratio_change_pct"] = fact_sheet["holding_ratio"]
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "新規保有" in prompt
    assert "ポイント増加" not in prompt


def test_generate_article_body_uses_buy_wording_by_default():
    """directionを指定しない場合は従来通り「取得」「推定取得金額」として扱う（後方互換）。"""
    fact_sheet = _fact_sheet()
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "推定取得金額" in prompt
    assert "推定売却金額" not in prompt


def test_generate_article_body_uses_sell_wording_when_direction_is_sell():
    """direction="sell"なら「売却」「推定売却金額」の文言でプロンプトを構成する。"""
    fact_sheet = _fact_sheet()
    fact_sheet["direction"] = "sell"
    fact_sheet["deal_amount_label"] = "推定売却金額"
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "推定売却金額" in prompt
    assert "推定取得金額" not in prompt
    assert "この売却が今後" in prompt


def test_generate_article_body_includes_company_description_when_available():
    """事業内容の事実があればプロンプトに織り込み、冒頭で触れるよう指示する。"""
    fact_sheet = _fact_sheet()
    fact_sheet["company_description"] = "美容院ブランドのライセンス展開を行う企業"
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "美容院ブランドのライセンス展開を行う企業" in prompt


def test_generate_article_body_always_requests_labelled_speculation():
    """事業内容・下落リスク文脈の有無に関わらず、「※推測:」ラベル付きの1文を必ず要求する。"""
    fact_sheet = _fact_sheet()  # company_description/context 無し
    raw = json.dumps({"title": "タイトル", "body": "<p>本文</p>"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "※推測:" in prompt
    assert "創作しないでください" in prompt


def test_generate_article_body_prompt_requests_english_translation():
    """kujira-watch(/en)向けにbodyEnと英語タイトル用のローマ字名もJSONに含めるよう
    1回の呼び出しでプロンプトに要求する（JA/ENを別々に生成すると事実がズレたり
    API呼び出しが倍になるため）。冒頭アンサー文の指定もプロンプトに含める。"""
    fact_sheet = _fact_sheet()
    raw = json.dumps({"body": "<p>本文</p>", "bodyEn": "<p>Body</p>",
                      "stockNameEn": "Test", "filerNameEn": "X Fund"})
    client, calls = _capturing_client(raw)
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch("anthropic.Anthropic", return_value=client):
        result = m.generate_article_body(fact_sheet)
    prompt = calls[0]["messages"][0]["content"]
    assert "bodyEn" in prompt
    assert "stockNameEn" in prompt
    assert "本文の1文目は、必ず次の文をそのまま使ってください" in prompt
    assert "大量保有報告書（EDINET）で分かりました" in prompt
    assert result["bodyEn"] == "<p>Body</p>"
    assert result["stockNameEn"] == "Test"


def test_build_and_publish_includes_english_fields_when_generated():
    """generate_article_body()がtitleEn/bodyEnを返した場合、publish_article()へのpayloadに含める。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d, prior=None, amend=False: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked",
                            return_value={"body": "<p>本文</p>", "bodyEn": "<p>Body</p>",
                                          "stockNameEn": "Test Motor", "filerNameEn": "Taro Kojin"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)
    assert results[0]["titleEn"] == "Taro Kojin Takes 8.5% Stake in Test Motor (7203) | Large Shareholding Report"
    assert results[0]["bodyEn"] == "<p>Body</p>"


def test_build_and_publish_omits_english_fields_when_not_generated():
    """titleEn/bodyEnが無い（部分失敗・後方互換ケース）場合はpayloadにキー自体を含めない。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", side_effect=lambda code, filer, ratio, d, prior=None, amend=False: ratio), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body_checked",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=1, dry_run=False)
    assert "titleEn" not in results[0]
    assert "bodyEn" not in results[0]


def test_get_company_description_returns_cached_without_calling_claude():
    cached = {"description": "美容院チェーンを展開する企業"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを展開する企業"
    assert select_mock.called
    assert not anthropic_mock.called


def test_get_company_description_asks_claude_and_persists_when_not_cached():
    raw = json.dumps({"description": "美容院チェーンを展開する企業"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを展開する企業"
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["code"] == "9439"
    assert saved_rows[0]["description"] == "美容院チェーンを展開する企業"


def test_get_company_description_uses_web_search_and_parses_trailing_json():
    """web_searchツールを渡し、検索結果ブロックが混ざったcontentからも末尾のJSONを取れること
    （一般知識のみでは中小型株の大半が空文字になるため、web検索での裏取りを必須にしている）。"""
    class _SearchBlock:
        type = "web_search_tool_result"

    class _TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        content = [
            _TextBlock("会社概要を検索します。"),
            _SearchBlock(),
            _TextBlock('調査の結果は以下の通りです。\n{"description": "美容院チェーンを展開する企業"}'),
        ]

    captured = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", _Client):
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを展開する企業"
    assert captured["tools"][0]["name"] == "web_search"


def test_get_company_description_tolerates_raw_newlines_in_json():
    """説明文の途中に生の改行が入ったJSONでも落ちずに1行へ正規化して取れること
    （strict=Falseでないと "Invalid control character" で全滅する。2026-08-18に8件発生）。"""
    raw = '{"description": "美容院チェーンを\n展開する企業"}'
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_company_description("9439", "エム・エイチ・グループ")
    assert result == "美容院チェーンを 展開する企業"


def test_get_company_description_returns_empty_without_api_key_when_not_cached():
    with mock.patch.object(m, "ANTHROPIC_API_KEY", ""), \
         mock.patch.object(m.sb, "select_one", return_value=None):
        assert m.get_company_description("9439", "エム・エイチ・グループ") == ""


def test_get_company_description_skips_claude_when_checked_recently():
    """空振り済みの銘柄はweb_searchを叩き直さない（1社あたり約$0.05かかるため）。"""
    cached = {"description": "", "description_checked_at": _iso_days_ago(1)}
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=cached), \
         mock.patch("anthropic.Anthropic") as client_mock:
        assert m.get_company_description("9439", "エム・エイチ・グループ") == ""
    client_mock.assert_not_called()


def test_get_company_description_retries_after_recheck_window():
    """RECHECK_DAYSを過ぎた銘柄は再挑戦する。"""
    cached = {"description": "", "description_checked_at": _iso_days_ago(m.RECHECK_DAYS + 1)}
    raw = json.dumps({"description": "美容室運営会社。"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=cached), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.get_company_description("9439", "エム・エイチ・グループ") == "美容室運営会社。"


def test_get_company_description_records_checked_at_even_when_blank():
    """空文字でもchecked_atを刻む。これが無いと同じ銘柄に何度でも課金される。"""
    raw = json.dumps({"description": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.get_company_description("9439", "エム・エイチ・グループ") == ""
    saved = upsert_mock.call_args.args[1][0]
    assert "description" not in saved  # 空文字で既存の説明を潰さない
    assert saved["description_checked_at"]


def test_get_company_description_caps_web_search_uses():
    """max_usesは検索料と入力トークンに直結するので、増やすときは意図的に。"""
    raw = json.dumps({"description": "美容室運営会社。"})
    captured = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_client(raw).messages.create(**kwargs)

    class _Client:
        messages = _Messages()

    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_Client()):
        m.get_company_description("9439", "エム・エイチ・グループ")
    assert captured["tools"][0]["max_uses"] == 2


def _usage_limit_error() -> Exception:
    return Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'You have reached your specified API usage limits. "
        "You will regain access on 2026-09-01 at 00:00 UTC.'}}"
    )


def test_usage_limit_stops_further_claude_calls():
    """上限到達を検知したら、同一プロセスの後続呼び出しはAPIを叩かずに諦める。

    2026-08-24の毎時実行では、上限後も候補ごとに叩き続けて1回の実行で十数回失敗していた。
    """
    api_budget.reset()
    try:
        failing = mock.MagicMock()
        failing.messages.create.side_effect = _usage_limit_error()
        with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
             mock.patch.object(m.sb, "select_one", return_value=None), \
             mock.patch.object(m.sb, "upsert"), \
             mock.patch("anthropic.Anthropic", return_value=failing):
            assert m.get_company_description("9439", "エム・エイチ・グループ") == ""
        assert api_budget.reached()

        # 2件目以降はクライアントすら生成しない
        with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
             mock.patch.object(m.sb, "select_one", return_value=None), \
             mock.patch("anthropic.Anthropic") as client_mock:
            assert m.get_company_description("7203", "テスト自動車") == ""
            assert m.get_filer_profile("テストファンド", "個人") == ""
            assert m.classify_filer("テストファンド")["category"] == "その他"
        client_mock.assert_not_called()
    finally:
        api_budget.reset()


def test_usage_limit_does_not_poison_negative_cache():
    """上限エラーは課金されていないので「試行済み」に含めない（次回ちゃんと再挑戦する）。"""
    api_budget.reset()
    try:
        failing = mock.MagicMock()
        failing.messages.create.side_effect = _usage_limit_error()
        with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
             mock.patch.object(m.sb, "select_one", return_value=None), \
             mock.patch.object(m.sb, "upsert") as upsert_mock, \
             mock.patch("anthropic.Anthropic", return_value=failing):
            m.get_company_description("9439", "エム・エイチ・グループ")
        upsert_mock.assert_not_called()
    finally:
        api_budget.reset()


def test_ordinary_failure_is_not_treated_as_usage_limit():
    """通常のAPI障害でパイプライン全体を止めてしまわないこと。"""
    api_budget.reset()
    try:
        failing = mock.MagicMock()
        failing.messages.create.side_effect = Exception("Error code: 529 - overloaded_error")
        with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
             mock.patch.object(m.sb, "select_one", return_value=None), \
             mock.patch.object(m.sb, "upsert"), \
             mock.patch("anthropic.Anthropic", return_value=failing):
            m.get_company_description("9439", "エム・エイチ・グループ")
        assert not api_budget.reached()
    finally:
        api_budget.reset()


def test_checked_recently_handles_missing_and_malformed_values():
    assert m.checked_recently(None) is False
    assert m.checked_recently("") is False
    assert m.checked_recently("not-a-timestamp") is False
    assert m.checked_recently(_iso_days_ago(1)) is True
    assert m.checked_recently(_iso_days_ago(m.RECHECK_DAYS + 1)) is False
    # タイムゾーン無しの値もUTC扱いで判定できること（Supabaseの返し方に依存しない）
    naive = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    assert m.checked_recently(naive) is True


def test_is_worth_publishing_accepts_large_amount():
    # 金額が基準以上なら比率変化が小さくても記事にする
    assert m.is_worth_publishing(3.0, 0.02) is True


def test_is_worth_publishing_accepts_large_ratio_change():
    # 金額が小さくても保有方針が動いた開示は記事にする（売りの負値も絶対値で判定）
    assert m.is_worth_publishing(0.5, -1.2) is True


def test_is_worth_publishing_rejects_trivial_disclosure():
    # 保有比率0.04%・推定0億円のような実質ニュース価値の無い変更報告書は落とす
    assert m.is_worth_publishing(0.0, 0.01) is False


def test_body_char_count_excludes_tags_and_whitespace():
    assert m.body_char_count("<p>大量保有</p>\n<p>報告書</p>") == 7


def test_generate_article_body_checked_retries_when_body_too_short():
    short = {"body": "<p>" + "あ" * 100 + "</p>"}
    long = {"body": "<p>" + "い" * 700 + "</p>"}
    with mock.patch.object(m, "generate_article_body", side_effect=[short, long]) as gen:
        result = m.generate_article_body_checked({})
    assert gen.call_count == 2
    assert result is long


def test_generate_article_body_checked_keeps_longer_of_two_attempts():
    first = {"body": "<p>" + "あ" * 500 + "</p>"}
    second = {"body": "<p>" + "い" * 300 + "</p>"}
    with mock.patch.object(m, "generate_article_body", side_effect=[first, second]):
        assert m.generate_article_body_checked({}) is first


def test_generate_article_body_checked_no_retry_when_long_enough():
    ok = {"body": "<p>" + "あ" * 700 + "</p>"}
    with mock.patch.object(m, "generate_article_body", side_effect=[ok]) as gen:
        assert m.generate_article_body_checked({}) is ok
    assert gen.call_count == 1


def test_generate_article_body_checked_retries_on_ai_tell_and_keeps_clean():
    # 字数は足りていてもAI常套句があれば再生成し、常套句の無い方を採用する
    telly = {"body": "<p>今回の取得には注目が集まっています。" + "あ" * 700 + "</p>"}
    clean = {"body": "<p>" + "い" * 700 + "</p>"}
    with mock.patch.object(m, "generate_article_body", side_effect=[telly, clean]) as gen:
        assert m.generate_article_body_checked({}) is clean
    assert gen.call_count == 2


def test_generate_article_body_checked_length_beats_ai_tell():
    # 字数充足はAI常套句の少なさより優先（短い記事はインデックスされない実害があるため）
    telly_long = {"body": "<p>今回の取得には注目が集まっています。" + "あ" * 700 + "</p>"}
    clean_short = {"body": "<p>" + "い" * 300 + "</p>"}
    with mock.patch.object(m, "generate_article_body", side_effect=[telly_long, clean_short]):
        assert m.generate_article_body_checked({}) is telly_long


def test_build_and_publish_skips_disclosure_below_threshold(capsys=None):
    holdings = [{
        "issuer_code": "1234", "name": "テスト社", "filer_name": "野村證券株式会社",
        "holding_ratio": 0.04, "holding_ratio_prior": 0.03, "disc_date": "2026-08-07",
        "doc_description": "変更報告書", "doc_type_code": "350",
    }]
    with mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "ratio_change_pct", return_value=0.01), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=0.0), \
         mock.patch.object(m, "generate_article_body_checked") as gen:
        result = m.build_and_publish(dry_run=True)
    assert result == []
    gen.assert_not_called()

def test_get_filer_profile_returns_cached_without_calling_claude():
    cached = {"profile": "1990年代設立の国内独立系運用会社。"}
    with mock.patch.object(m.sb, "select_one", return_value=cached) as select_mock, \
         mock.patch("anthropic.Anthropic") as anthropic_mock:
        result = m.get_filer_profile("テストファンド", "独立系ブティックAM")
    assert result == "1990年代設立の国内独立系運用会社。"
    assert select_mock.called
    assert not anthropic_mock.called


def test_get_filer_profile_asks_claude_and_persists_when_not_cached():
    raw = json.dumps({"profile": "1990年代設立の国内独立系運用会社。"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_filer_profile("テストファンド", "独立系ブティックAM")
    assert result == "1990年代設立の国内独立系運用会社。"
    upsert_mock.assert_called_once()
    saved_rows = upsert_mock.call_args.args[1]
    assert saved_rows[0]["filer_name"] == "テストファンド"
    # categoryも含めること: PostgreSQLはON CONFLICT時のUPDATE分岐でも候補行構築時点で
    # NOT NULL制約(category)を評価するため、欠くと既存行の更新のつもりでも失敗する。
    assert saved_rows[0]["category"] == "独立系ブティックAM"
    assert saved_rows[0]["profile"] == "1990年代設立の国内独立系運用会社。"


def test_get_filer_profile_returns_empty_without_api_key_when_not_cached():
    with mock.patch.object(m, "ANTHROPIC_API_KEY", ""), \
         mock.patch.object(m.sb, "select_one", return_value=None):
        assert m.get_filer_profile("テストファンド", "独立系ブティックAM") == ""


def test_get_filer_profile_returns_empty_when_claude_returns_blank():
    """情報が乏しい個人名義等の提出者は空文字のまま(創作させない)。

    空文字をprofileに書き込んでしまうと創作より悪い（不明が確定情報として残る）ので
    profileキーは含めない。一方でprofile_checked_atは刻む（刻まないと同じ提出者を
    記事のたびに叩き直すため）。
    """
    raw = json.dumps({"profile": ""})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=None), \
         mock.patch.object(m.sb, "upsert") as upsert_mock, \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        result = m.get_filer_profile("個人 太郎", "個人")
    assert result == ""
    upsert_mock.assert_called_once()
    saved = upsert_mock.call_args.args[1][0]
    assert "profile" not in saved
    assert saved["profile_checked_at"]


def test_get_filer_profile_skips_claude_when_checked_recently():
    """空振り済みの提出者は再課金しない（ネガティブキャッシュ）。"""
    cached = {"profile": "", "profile_checked_at": _iso_days_ago(1)}
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=cached), \
         mock.patch("anthropic.Anthropic") as client_mock:
        assert m.get_filer_profile("個人 太郎", "個人") == ""
    client_mock.assert_not_called()


def test_get_filer_profile_retries_after_recheck_window():
    """RECHECK_DAYSを過ぎたら再挑戦する（会社が有名になった等の変化を拾う）。"""
    cached = {"profile": "", "profile_checked_at": _iso_days_ago(m.RECHECK_DAYS + 1)}
    raw = json.dumps({"profile": "国内独立系の運用会社。"})
    with mock.patch.object(m, "ANTHROPIC_API_KEY", "dummy"), \
         mock.patch.object(m.sb, "select_one", return_value=cached), \
         mock.patch.object(m.sb, "upsert"), \
         mock.patch("anthropic.Anthropic", return_value=_fake_client(raw)):
        assert m.get_filer_profile("テストファンド", "独立系ブティックAM") == "国内独立系の運用会社。"


def test_upload_price_chart_returns_url_on_success():
    resp = _FakeResponse(201, "", {"url": "https://images.microcms-assets.io/assets/x/chart.png"})
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch("requests.post", return_value=resp):
        url = m.upload_price_chart(b"png-bytes")
    assert url == "https://images.microcms-assets.io/assets/x/chart.png"


def test_build_price_chart_for_article_none_when_generation_fails():
    with mock.patch.object(m, "generate_price_chart_image", return_value=None):
        assert m.build_price_chart_for_article("7203", "テスト自動車") is None


def test_build_price_chart_for_article_returns_url_on_success():
    with mock.patch.object(m, "generate_price_chart_image", return_value=b"png-bytes"), \
         mock.patch.object(m, "upload_price_chart",
                            return_value="https://images.microcms-assets.io/x/chart.png"):
        assert m.build_price_chart_for_article("7203", "テスト自動車") == \
            "https://images.microcms-assets.io/x/chart.png"


def test_build_and_publish_embeds_chart_image_in_body():
    """チャートが生成できた場合、本文HTMLの末尾に<img>タグとして埋め込む。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body",
                            return_value={"title": "テストタイトル", "body": "<p>本文</p>"}), \
         mock.patch.object(m, "build_price_chart_for_article",
                            return_value="https://images.microcms-assets.io/x/chart.png"), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        results = m.build_and_publish(days=3, max_articles=3, dry_run=False)
    assert "https://images.microcms-assets.io/x/chart.png" in results[0]["body"]
    assert "<p>本文</p>" in results[0]["body"]


def test_build_and_publish_includes_pit_context_in_fact_sheet():
    """本文へ渡す株価は金額の概算に使った開示日終値と同じ値（サイトの基準終値と同源）にする。
    下落モデルの水準はfact_sheetに入れない（2026-08-25に記事から廃止）。"""
    holdings = [{"issuer_code": "7203", "name": "テスト自動車", "filer_name": "個人 太郎",
                 "holding_ratio": 8.5, "disc_date": "2026-07-20", "doc_type_code": "350",
                 "doc_description": "大量保有報告書"}]
    captured = {}

    def _fake_generate(fact_sheet):
        captured.update(fact_sheet)
        return {"title": "t", "body": "<p>本文</p>"}

    with mock.patch.object(m, "MICROCMS_DOMAIN", "dummy"), \
         mock.patch.object(m, "MICROCMS_KEY", "dummy"), \
         mock.patch.object(m, "get_recent_large_holdings", return_value=holdings), \
         mock.patch.object(m, "already_published", return_value=False), \
         mock.patch.object(m, "ratio_change_pct", return_value=8.5), \
         mock.patch.object(m, "estimate_deal_amount_oku", return_value=12.3), \
         mock.patch.object(m, "get_pit_ranking_snapshot", return_value={"close": 2900.0}), \
         mock.patch.object(m, "disclosure_close_price", return_value=3000.0), \
         mock.patch.object(m, "classify_filer",
                            return_value={"category": "個人", "is_foreign": False, "description": ""}), \
         mock.patch.object(m, "generate_article_body", side_effect=_fake_generate), \
         mock.patch.object(m, "build_price_chart_for_article", return_value=None), \
         mock.patch.object(m, "attach_figures", return_value=0), \
         mock.patch.object(m, "publish_article", return_value="fakeid123"):
        m.build_and_publish(days=3, max_articles=3, dry_run=False)

    assert captured["context_close"] == 3000.0
    assert "context_dp_level" not in captured
    assert captured["ratio_change_pct"] == 8.5


def _fake_client(text):
    class _Block:
        # web_search併用時はcontentに検索結果ブロックも混ざるため、本番側はtypeで
        # テキストブロックだけを拾う。ダミーもtypeを持たせる。
        type = "text"

        def __init__(self, text):
            self.text = text

    class _Resp:
        def __init__(self, text):
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self, text):
            self._text = text

        def create(self, **kwargs):
            return _Resp(self._text)

    class _Client:
        def __init__(self, text):
            self.messages = _Messages(text)

    return _Client(text)


if __name__ == "__main__":
    test_estimate_deal_amount_oku_calculation()
    test_estimate_deal_amount_oku_none_when_no_change()
    test_estimate_deal_amount_oku_none_when_shares_missing()
    test_shares_outstanding_retries_then_succeeds()
    test_shares_outstanding_falls_back_to_implied_shares_outstanding()
    test_shares_outstanding_returns_none_after_exhausting_retries()
    test_generate_article_body_parses_plain_json()
    test_generate_article_body_strips_code_fence()
    test_generate_article_body_none_on_empty_body()
    test_build_article_titles_buy_and_sell_templates()
    test_build_article_titles_new_holding_uses_ratio_change_heuristic()
    test_build_article_titles_truncates_long_filer_name()
    test_build_article_titles_uses_english_names_when_given()
    test_classify_filer_returns_cached_master_row_without_calling_claude()
    test_classify_filer_asks_claude_and_persists_when_not_cached()
    test_classify_filer_falls_back_to_sonota_on_invalid_category()
    test_get_pit_ranking_snapshot_queries_as_of_disc_date()
    test_generate_article_body_includes_close_price_but_never_the_drop_model()
    test_generate_article_body_omits_context_when_unavailable()
    test_generate_article_body_includes_ratio_increase_when_available()
    test_generate_article_body_describes_new_position_when_change_equals_ratio()
    test_generate_article_body_uses_buy_wording_by_default()
    test_generate_article_body_uses_sell_wording_when_direction_is_sell()
    test_generate_article_body_prompt_requests_english_translation()
    test_build_and_publish_includes_english_fields_when_generated()
    test_build_and_publish_omits_english_fields_when_not_generated()
    test_build_and_publish_includes_pit_context_in_fact_sheet()
    test_build_and_publish_includes_sell_and_tags_them()
    test_build_and_publish_skips_when_already_published()
    test_already_published_true_when_filer_and_ratio_change_match()
    test_already_published_false_for_different_filer_same_day()
    test_already_published_false_for_same_filer_different_ratio_change()
    test_already_published_falls_back_to_deal_amount_for_legacy_articles()
    test_build_and_publish_passes_dedup_keys_to_already_published()
    test_build_and_publish_skips_when_amount_unestimable()
    test_build_and_publish_stops_early_on_permission_error()
    test_publish_article_retries_as_array_on_type_mismatch()
    test_publish_article_drops_non_string_field_on_type_mismatch()
    test_publish_article_gives_up_when_same_field_fails_twice()
    test_update_article_retries_as_array_on_type_mismatch()
    test_update_article_returns_false_on_failure()
    test_wrap_text_lines_breaks_on_width()
    test_wrap_text_lines_respects_max_lines()
    test_search_pexels_photo_returns_none_without_api_key()
    test_search_pexels_photo_returns_bytes_and_photographer_on_success()
    test_search_pexels_photo_defaults_photographer_when_missing()
    test_search_pexels_photo_returns_none_when_no_results()
    test_search_pexels_photo_returns_none_on_exception()
    test_search_pexels_photo_picks_different_photos_for_different_seeds()
    test_search_pexels_photo_is_stable_for_same_seed()
    test_search_pexels_photo_caches_candidates_per_query()
    test_wrap_text_lines_keeps_number_token_whole()
    test_wrap_text_lines_breaks_number_longer_than_line()
    test_eyecatch_stock_line_variants()
    test_upload_eyecatch_returns_url_on_success()
    test_upload_eyecatch_returns_none_on_failure()
    test_build_eyecatch_for_article_none_without_pexels_key()
    test_build_eyecatch_for_article_none_when_generation_fails()
    test_build_eyecatch_for_article_returns_url_string_on_success()
    test_build_and_publish_includes_eyecatch_when_available()
    test_build_and_publish_skips_eyecatch_on_dry_run()
    test_generate_article_body_includes_company_description_when_available()
    test_generate_article_body_always_requests_labelled_speculation()
    test_get_company_description_returns_cached_without_calling_claude()
    test_get_company_description_asks_claude_and_persists_when_not_cached()
    test_get_company_description_uses_web_search_and_parses_trailing_json()
    test_get_company_description_tolerates_raw_newlines_in_json()
    test_get_company_description_returns_empty_without_api_key_when_not_cached()
    test_get_company_description_skips_claude_when_checked_recently()
    test_get_company_description_retries_after_recheck_window()
    test_get_company_description_records_checked_at_even_when_blank()
    test_get_company_description_caps_web_search_uses()
    test_usage_limit_stops_further_claude_calls()
    test_usage_limit_does_not_poison_negative_cache()
    test_ordinary_failure_is_not_treated_as_usage_limit()
    test_checked_recently_handles_missing_and_malformed_values()
    test_upload_price_chart_returns_url_on_success()
    test_build_price_chart_for_article_none_when_generation_fails()
    test_build_price_chart_for_article_returns_url_on_success()
    test_build_and_publish_embeds_chart_image_in_body()
    test_get_featured_article_ids_picks_top_deal_amount()
    test_get_featured_article_ids_returns_empty_set_on_http_error()
    test_get_featured_article_ids_returns_empty_set_on_exception()
    test_get_filer_profile_returns_cached_without_calling_claude()
    test_get_filer_profile_asks_claude_and_persists_when_not_cached()
    test_get_filer_profile_returns_empty_without_api_key_when_not_cached()
    test_get_filer_profile_returns_empty_when_claude_returns_blank()
    test_get_filer_profile_skips_claude_when_checked_recently()
    test_get_filer_profile_retries_after_recheck_window()
    test_is_worth_publishing_accepts_large_amount()
    test_is_worth_publishing_accepts_large_ratio_change()
    test_is_worth_publishing_rejects_trivial_disclosure()
    test_body_char_count_excludes_tags_and_whitespace()
    test_generate_article_body_checked_retries_when_body_too_short()
    test_generate_article_body_checked_keeps_longer_of_two_attempts()
    test_generate_article_body_checked_no_retry_when_long_enough()
    test_generate_article_body_checked_retries_on_ai_tell_and_keeps_clean()
    test_generate_article_body_checked_length_beats_ai_tell()
    test_build_and_publish_skips_disclosure_below_threshold()
    test_ratio_change_pct_uses_disclosure_prior_ratio()
    test_ratio_change_pct_counts_full_exit_as_full_prior_ratio()
    test_ratio_change_pct_falls_back_to_history_without_prior()
    test_is_new_holding_false_for_full_exit()
    test_is_new_holding_false_for_amendment_without_prior_ratio()
    test_ratio_change_pct_returns_none_for_amendment_without_prior_or_history()
    test_build_article_titles_amendment_without_prior_is_not_new_holding()
    test_build_and_publish_skips_amendment_with_unknown_change()
    test_estimate_deal_amount_falls_back_to_yfinance_price()
    test_build_and_publish_skips_when_ratio_unchanged()
    test_should_wait_for_prior_ratio_true_for_fresh_change_report()
    test_should_wait_for_prior_ratio_false_after_wait_days()
    test_should_wait_for_prior_ratio_false_for_new_filing_and_when_prior_exists()
    test_build_and_publish_defers_change_report_without_prior_ratio()
    test_build_and_publish_publishes_material_correction_without_amount()
    test_already_published_true_for_unique_filing_even_if_ratio_differs()
    test_build_and_publish_uses_disclosed_unit_price_over_market_estimate()
    test_build_and_publish_keeps_estimate_when_transfer_table_is_inconclusive()
    print("全テスト成功 (117件)")

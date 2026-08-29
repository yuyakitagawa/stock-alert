"""tools/geo_report.py のユニットテスト（ネットワークは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import geo_report as g  # noqa: E402


def _row(path, bot="OAI-SearchBot"):
    return {"occurred_at": "2026-08-20T00:00:00+00:00", "path": path, "bot_name": bot}


def test_existing_paths_are_ok():
    for path in ["/", "/about", "/weekly", "/ranking/returns", "/llms.txt",
                 "/articles/abc123", "/investors/616", "/category/事業会社",
                 "/date/2026-08-20", "/monthly/2026-08", "/faq/usage"]:
        assert g.path_status(path) == "ok", path


def test_stock_code_with_letter_is_ok():
    """新規上場銘柄の証券コードは数字4桁ではなく英字を含む（603A等）。
    数字だけを実在扱いにすると、正常なページを404候補として毎回並べてしまう。"""
    assert g.path_status("/stocks/4591") == "ok"
    assert g.path_status("/stocks/603A") == "ok"


def test_missing_paths_are_detected():
    """廃止済み・存在しないURLはmissingとして拾う（発見の実例が/articlesと/watchlist）。"""
    assert g.path_status("/articles") == "missing"
    assert g.path_status("/watchlist") == "missing"
    assert g.path_status("/stocks/4591/feed.xml") == "missing"


def test_redirected_paths_are_separated_from_missing():
    """301で受けているURLは404ではないので別枠にする（対応の要否が違う）。"""
    for path in ["/en", "/en/articles/abc", "/ranking", "/ranking/buys", "/disclosures"]:
        assert g.path_status(path) == "redirect", path


def test_query_and_trailing_slash_are_ignored():
    assert g.path_status("/about/") == "ok"
    assert g.path_status("/stocks/4591?utm_source=x") == "ok"


def test_page_group_folds_paths():
    assert g.page_group("/") == "トップ"
    assert g.page_group("/articles/abc") == "記事"
    assert g.page_group("/stocks/4591") == "銘柄"
    assert g.page_group("/ranking/returns") == "ランキング"
    assert g.page_group("/llms.txt") == "機械向け"
    assert g.page_group("/unknown-page") == "その他"


def test_is_ai_source_matches_medium_and_host():
    """GA4の分類（medium=ai-assistant）が付いていない新顔もホスト名で拾う。"""
    assert g.is_ai_source("chatgpt.com", "ai-assistant")
    assert g.is_ai_source("perplexity.ai", "referral")
    assert g.is_ai_source("(not set)", "ai-assistant")
    assert not g.is_ai_source("google", "organic")
    assert not g.is_ai_source("youtube", "social")


def test_crawler_sections_separates_on_demand_bots():
    """引用の代理指標は ChatGPT-User / PerplexityBot だけ。一括クロールを混ぜない。"""
    now = [_row("/articles/a", "ChatGPT-User"), _row("/articles/b", "OAI-SearchBot"),
           _row("/disclosures", "OAI-SearchBot"), _row("/watchlist", "GPTBot")]
    buf = io.StringIO()
    with mock.patch.object(g, "fetch_ai_rows", side_effect=[now, []]), redirect_stdout(buf):
        g.crawler_sections(days=14, limit=10)
    out = buf.getvalue()
    on_demand = out.split("引用の代理指標")[1].split("存在しないURL")[0]
    assert "/articles/a" in on_demand
    assert "/articles/b" not in on_demand
    assert "/watchlist" in out.split("存在しないURL")[1].split("旧URL")[0]
    assert "/disclosures" in out.split("旧URL")[1]


def test_crawler_sections_handles_empty_log():
    buf = io.StringIO()
    with mock.patch.object(g, "fetch_ai_rows", side_effect=[[], []]), redirect_stdout(buf):
        g.crawler_sections(days=14, limit=10)
    assert "記録がありません" in buf.getvalue()


def test_ga4_section_skips_without_credentials():
    """GA4の鍵が無くてもクローラー側のレポートは出す（片方だけで止めない）。"""
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"GA4_PROPERTY_ID": ""}), \
         mock.patch.object(g, "access_token", return_value=None), redirect_stdout(buf):
        g.ga4_section(days=14, limit=10)
    assert "スキップ" in buf.getvalue()


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

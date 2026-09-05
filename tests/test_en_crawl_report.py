"""tools/en_crawl_report.py のユニットテスト（ネットワークは全てモック）。"""
import io
import os
import sys
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import en_crawl_report as e  # noqa: E402


def _row(path, bot="Googlebot", host=e.EN_HOST, at="2026-09-05T01:00:00+00:00"):
    return {"occurred_at": at, "path": path, "host": host, "bot_name": bot}


def test_en_paths_are_ok():
    for path in ["/", "/about", "/privacy", "/robots.txt", "/sitemap-en.xml",
                 "/articles/abc123", "/icon", "/_next/static/x.js", "/about/"]:
        assert e.en_path_status(path) == "ok", path


def test_en_missing_paths_are_detected():
    """日本語版にしか無いページは英語版では404。rewrite漏れやリンク切れの発見に使う。"""
    for path in ["/stocks/4591", "/investors/616", "/sitemap.xml", "/feed.xml", "/category/x"]:
        assert e.en_path_status(path) == "missing", path


def test_en_redirected_paths_are_separated():
    assert e.en_path_status("/en/articles/abc") == "redirect"
    assert e.en_path_status("/articles") == "redirect"


def test_is_en_treats_null_host_as_japanese():
    """host列追加（2026-09-04）より前の行はNULL＝日本語版として扱う。"""
    assert e.is_en(_row("/"))
    assert not e.is_en(_row("/", host=None))
    assert not e.is_en(_row("/", host="kujira-watch.com"))


def test_report_separates_en_from_ja_and_flags_missing():
    now = [_row("/articles/a"), _row("/articles/a", bot="GPTBot"), _row("/stocks/4591"),
           _row("/", bot="Browser"), _row("/articles/b", host="kujira-watch.com"),
           _row("/articles/c", host=None)]
    buf = io.StringIO()
    with mock.patch.object(e, "fetch_rows", side_effect=[now, []]), redirect_stdout(buf):
        e.report(days=14, limit=10)
    out = buf.getvalue()
    bots = out.split("クローラー別ヒット数")[1].split("日本語版ヒット数")[0]
    assert "Googlebot" in bots and "Browser" in bots
    ratio = out.split("英語版の比率")[1].split("日別ヒット数")[0]
    assert "Googlebot" in ratio and "日本語        2" in ratio
    assert "/stocks/4591" in out.split("404の可能性")[1]
    assert "/articles/b" not in out


def test_report_handles_no_en_rows():
    buf = io.StringIO()
    with mock.patch.object(e, "fetch_rows", side_effect=[[_row("/", host=None)], []]), \
         redirect_stdout(buf):
        e.report(days=14, limit=10)
    assert "アクセス記録がありません" in buf.getvalue()


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as ex:
                fails += 1
                print(f"FAIL  {name}: {ex}")
    print(f"\n{'FAILED' if fails else 'PASSED'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)

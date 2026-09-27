"""tools/indexnow_ping.py のユニットテスト（通信なし）。"""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import indexnow_ping as ix  # noqa: E402

KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "kujira-watch", "public", f"{ix.KEY}.txt")


def test_child_sitemaps_only_from_sitemapindex():
    idx = ("<sitemapindex><sitemap><loc>https://x/sitemap/pages.xml</loc></sitemap>"
           "<sitemap><loc>https://x/sitemap/stocks.xml</loc></sitemap></sitemapindex>")
    assert ix.child_sitemaps(idx) == ["https://x/sitemap/pages.xml", "https://x/sitemap/stocks.xml"]
    assert ix.child_sitemaps("<urlset><url><loc>https://x/a</loc></url></urlset>") == []


def test_recent_urls_filters_by_lastmod_and_skips_missing():
    xml = ("<urlset>"
           "<url><loc>https://x/new</loc><lastmod>2026-09-24T10:00:00.000Z</lastmod></url>"
           "<url><loc>https://x/old</loc><lastmod>2026-09-01</lastmod></url>"
           "<url><loc>https://x/nomod</loc></url>"
           "</urlset>")
    assert ix.recent_urls(xml, date(2026, 9, 23)) == ["https://x/new"]


def test_payload_caps_urls_and_points_to_key_file():
    p = ix.payload([f"https://x/{i}" for i in range(ix.MAX_URLS + 5)])
    assert len(p["urlList"]) == ix.MAX_URLS
    assert p["keyLocation"] == f"https://kujira-watch.com/{ix.KEY}.txt"


def test_key_file_is_committed_with_key_as_content():
    """鍵ファイルが公開ディレクトリに無いとIndexNowは403を返す。"""
    with open(KEY_FILE) as f:
        assert f.read().strip() == ix.KEY

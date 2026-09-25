"""
tools/indexnow_ping.py

サイトマップの <lastmod> が直近N日のURLを IndexNow（Bing・Yandex等が共有する更新通知API）に送る。

なぜ作ったか（2026-09-25）:
  ChatGPT検索はBingの索引を使う。Googleと違いBingは巡回を待たずに更新を受け付けるので、
  新規・更新ページを毎日通知すれば索引入りまでの時間を縮められる。サイトマップを入力にするのは、
  「どのページが変わったか」の定義をサイトマップの lastmod 1か所に揃えるため（二重管理しない）。

鍵: IndexNowはサイト直下に置いた鍵ファイルで所有確認する。鍵は公開前提の値なので
  `kujira-watch/public/<KEY>.txt` としてコミットしている（中身は鍵そのもの）。

実行:
  python3 tools/indexnow_ping.py              # 直近2日に更新されたURLを送る
  python3 tools/indexnow_ping.py --days 7
  python3 tools/indexnow_ping.py --dry-run    # 送らずに件数とURLを表示
"""
import argparse
import os
import re
import sys
from datetime import date, timedelta

import requests

SITE_URL = "https://kujira-watch.com"
HOST = "kujira-watch.com"
KEY = os.getenv("INDEXNOW_KEY", "ee4abf7fbf2bf0f3178e209506356c0d")
ENDPOINT = "https://api.indexnow.org/indexnow"
SITEMAP_INDEX = f"{SITE_URL}/sitemap.xml"
DEFAULT_DAYS = 2
# IndexNowの1リクエスト上限。
MAX_URLS = 10000

_LOC = re.compile(r"<loc>\s*([^<]+?)\s*</loc>")
_URL_BLOCK = re.compile(r"<url>(.*?)</url>", re.S)
_LASTMOD = re.compile(r"<lastmod>\s*(\d{4}-\d{2}-\d{2})")


def child_sitemaps(index_xml: str) -> list:
    return _LOC.findall(index_xml) if "<sitemapindex" in index_xml else []


def recent_urls(urlset_xml: str, since: date) -> list:
    """lastmod が since 以降のURL。lastmod が無いURLは「変わったか分からない」ので送らない。"""
    out = []
    for block in _URL_BLOCK.findall(urlset_xml):
        loc, mod = _LOC.search(block), _LASTMOD.search(block)
        if loc and mod and date.fromisoformat(mod.group(1)) >= since:
            out.append(loc.group(1))
    return out


def payload(urls: list) -> dict:
    return {"host": HOST, "key": KEY, "keyLocation": f"{SITE_URL}/{KEY}.txt",
            "urlList": urls[:MAX_URLS]}


def collect(days: int) -> list:
    since = date.today() - timedelta(days=days)
    index = requests.get(SITEMAP_INDEX, timeout=30)
    index.raise_for_status()
    children = child_sitemaps(index.text) or [SITEMAP_INDEX]
    urls = []
    for child in children:
        resp = requests.get(child, timeout=60)
        if resp.status_code != 200:
            print(f"[indexnow] {child}: HTTP {resp.status_code}（スキップ）")
            continue
        urls.extend(recent_urls(resp.text, since))
    return list(dict.fromkeys(urls))


def main():
    p = argparse.ArgumentParser(description="直近更新のURLをIndexNowに送る")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    try:
        urls = collect(a.days)
    except requests.RequestException as e:
        print(f"[indexnow] サイトマップ取得に失敗: {e}")
        sys.exit(1)
    print(f"[indexnow] 直近{a.days}日の更新URL: {len(urls)}件")
    if a.dry_run or not urls:
        for u in urls[:20]:
            print(f"  {u}")
        return
    resp = requests.post(ENDPOINT, json=payload(urls), timeout=60,
                         headers={"Content-Type": "application/json; charset=utf-8"})
    # 200=受理 / 202=受理（鍵の検証待ち）。それ以外は鍵ファイル未配置などの設定ミス。
    print(f"[indexnow] HTTP {resp.status_code} {resp.text[:200]}")
    sys.exit(0 if resp.status_code in (200, 202) else 1)


if __name__ == "__main__":
    main()

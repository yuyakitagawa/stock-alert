#!/usr/bin/env python3
"""本番サイトの表示速度を計測し、閾値を超えたページを報告する。

週1回 GitHub Actions（.github/workflows/perf_check.yml）から実行する。

計測するのは「初期表示までに待たされる量」に直結する3つ。
  1. TTFB: サーバーが最初の1バイトを返すまで。ISR/キャッシュが効いていないページはここが伸びる。
     （実例: /investors が searchParams で dynamic rendering になりキャッシュが無く1.6〜1.9秒）
  2. HTML転送量(gzip後): 一覧ページで全件描画すると数百KB〜1.5MBに膨らむ。
  3. レンダリングブロッキングCSS(gzip後): <head>内の<link rel=stylesheet>の合計。
     ここが大きいと、HTMLが届いても何も描画されない。
     （実例: Noto Sans JPのウェブフォントで@font-faceが496個・gzip 130KBになっていた）

JSは初期表示を直接ブロックしないので警告閾値は緩め、参考値として出す。

閾値超過時は exit 1 する。ワークフロー側がそれを見てIssueを立てる。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict

BASE_URL = "https://kujira-watch.com"

# 計測対象。実ユーザーの動線（TOP→記事、ヘッダーの各一覧）を代表するページを選ぶ。
# パスを増やすとその分ワークフローの実行時間が伸びるので、代表ページに絞る。
TARGET_PATHS = [
    "/",
    "/ranking",
    "/investors",
    "/stocks",
    "/disclosures",
    "/activists",
    "/weekly",
    "/monthly",
    "/trending",
    "/faq",
    "/about",
]

# 閾値。超えたら「調べる価値がある」ライン。
# TTFBはVercelのISRキャッシュヒット時なら0.1秒未満に収まる。0.8秒を超えていたら
# キャッシュが効いていないか、サーバー側で重い処理をしている。
TTFB_WARN_SEC = 0.8
# gzip後のHTML。一覧ページでも通常は50KB以下に収まる。
HTML_WARN_BYTES = 100 * 1024
# gzip後のレンダリングブロッキングCSS。Tailwindのみなら10KB程度。
CSS_WARN_BYTES = 30 * 1024

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36 "
    "(kujira-watch perf_check)"
)

# 計測のばらつき（コールドスタート・ネットワークの揺らぎ）を均すため複数回測って最小値を取る。
# 平均だと1回の外れ値に引きずられるが、最小値は「うまくいったときの実力」を表す。
MEASURE_ATTEMPTS = 3

# 属性値はクォート有り/無しの両方があり得る（HTMLとして両方valid）。
# クォートを必須にすると取りこぼして「CSSが軽い」と誤判定するので両対応にする。
STYLESHEET_RE = re.compile(
    r'<link[^>]+rel=(?:["\']stylesheet["\']|stylesheet(?=[\s>]))[^>]*>', re.IGNORECASE
)
SCRIPT_SRC_RE = re.compile(
    r'<script[^>]+src=(?:["\']([^"\']+)["\']|([^\s>"\']+))', re.IGNORECASE
)
HREF_RE = re.compile(r'href=(?:["\']([^"\']+)["\']|([^\s>"\']+))', re.IGNORECASE)


@dataclass
class PageResult:
    path: str
    ttfb_sec: float = 0.0
    html_bytes: int = 0
    css_bytes: int = 0
    js_bytes: int = 0
    css_files: int = 0
    js_files: int = 0
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def _request(url: str, timeout: int = 30) -> tuple[bytes, float]:
    """URLを取得し、(生バイト列, TTFB秒) を返す。

    gzipを要求し、Content-Encodingがgzipならそのままの転送量として扱う
    （実際に回線を流れるバイト数を測りたいため、展開しない）。
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # 最初の1バイトを読んだ時点をTTFBとする。
        first = response.read(1)
        ttfb = time.perf_counter() - started
        body = first + response.read()
        if response.headers.get("Content-Encoding") != "gzip":
            # サーバーが非圧縮で返した場合は、比較可能にするため自前でgzipして測る。
            body = gzip.compress(body)
    return body, ttfb


def _decode(body: bytes) -> str:
    try:
        return gzip.decompress(body).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        return body.decode("utf-8", errors="replace")


def _absolute(base: str, url: str) -> str:
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return base + url
    return f"{base}/{url}"


def _sum_assets(base: str, urls: list[str]) -> tuple[int, int]:
    """アセットを取得し (合計バイト数, 取得できた個数) を返す。

    1つ落ちても計測全体は続ける（外部CDNの一時的な失敗で週次チェックを落とさない）。
    """
    total = 0
    count = 0
    for url in urls:
        try:
            body, _ = _request(_absolute(base, url), timeout=20)
        except Exception:
            continue
        total += len(body)
        count += 1
    return total, count


def measure(base: str, path: str) -> PageResult:
    result = PageResult(path=path)
    url = base + path

    try:
        # TTFBは複数回測って最小値。HTMLの中身は最後の1回を使う（毎回同じはずなので）。
        ttfbs: list[float] = []
        body = b""
        for _ in range(MEASURE_ATTEMPTS):
            body, ttfb = _request(url)
            ttfbs.append(ttfb)
        result.ttfb_sec = min(ttfbs)
        result.html_bytes = len(body)
    except urllib.error.HTTPError as exc:
        result.error = f"HTTP {exc.code}"
        return result
    except Exception as exc:  # noqa: BLE001 - 計測失敗の理由をそのまま出したい
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    html = _decode(body)

    # <head>内のstylesheetだけがレンダリングブロッキング。bodyに現れるものは対象外。
    head = html.split("</head>", 1)[0]
    css_urls: list[str] = []
    for tag in STYLESHEET_RE.findall(head):
        href = HREF_RE.search(tag)
        if href:
            css_urls.append(href.group(1) or href.group(2))
    result.css_bytes, result.css_files = _sum_assets(base, css_urls)

    # 自ドメインのJSのみ（外部の広告・計測タグは自分たちで削れないため対象外）。
    js_urls = [
        quoted or bare
        for quoted, bare in SCRIPT_SRC_RE.findall(html)
        if (quoted or bare).startswith("/")
    ]
    result.js_bytes, result.js_files = _sum_assets(base, js_urls)

    if result.ttfb_sec > TTFB_WARN_SEC:
        result.warnings.append(
            f"TTFB {result.ttfb_sec:.2f}秒 (>{TTFB_WARN_SEC}秒)。"
            "searchParamsでdynamic renderingになっていてキャッシュが効いていない可能性"
        )
    if result.html_bytes > HTML_WARN_BYTES:
        result.warnings.append(
            f"HTML {result.html_bytes / 1024:.0f}KB (>{HTML_WARN_BYTES // 1024}KB)。"
            "1ページに詰め込みすぎ。ページ送りを検討"
        )
    if result.css_bytes > CSS_WARN_BYTES:
        result.warnings.append(
            f"CSS {result.css_bytes / 1024:.0f}KB (>{CSS_WARN_BYTES // 1024}KB)。"
            "レンダリングブロッキング。ウェブフォントの@font-faceが混ざっていないか確認"
        )

    return result


def to_markdown(results: list[PageResult], base: str) -> str:
    lines = [
        f"## 表示速度チェック: {base}",
        "",
        "| ページ | TTFB | HTML | CSS | JS | 判定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| `{r.path}` | - | - | - | - | ⚠️ 取得失敗: {r.error} |")
            continue
        verdict = "🔴 要改善" if r.warnings else "🟢"
        lines.append(
            f"| `{r.path}` | {r.ttfb_sec:.2f}s | {r.html_bytes / 1024:.0f}KB "
            f"| {r.css_bytes / 1024:.0f}KB ({r.css_files}) "
            f"| {r.js_bytes / 1024:.0f}KB ({r.js_files}) | {verdict} |"
        )

    lines += [
        "",
        f"数値はいずれもgzip後。TTFBは{MEASURE_ATTEMPTS}回計測の最小値。"
        f"括弧内はファイル数。CSSは`<head>`内のレンダリングブロッキング分のみ。",
        "",
        f"閾値: TTFB {TTFB_WARN_SEC}秒 / HTML {HTML_WARN_BYTES // 1024}KB "
        f"/ CSS {CSS_WARN_BYTES // 1024}KB",
    ]

    flagged = [r for r in results if r.warnings or r.error]
    if flagged:
        lines += ["", "### 要改善の詳細", ""]
        for r in flagged:
            lines.append(f"**`{r.path}`**")
            if r.error:
                lines.append(f"- 取得失敗: {r.error}")
            for warning in r.warnings:
                lines.append(f"- {warning}")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--paths", nargs="*", default=TARGET_PATHS)
    parser.add_argument("--json", action="store_true", help="JSONでも出力する")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    results = [measure(base, path) for path in args.paths]

    markdown = to_markdown(results, base)
    print(markdown)

    if args.json:
        print("\n<!--JSON\n" + json.dumps([asdict(r) for r in results], ensure_ascii=False) + "\nJSON-->")

    flagged = [r for r in results if r.warnings or r.error]
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())

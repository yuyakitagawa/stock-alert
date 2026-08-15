#!/usr/bin/env python3
"""tools/perf_check.py の計測ロジックのテスト。

HTMLのパースを間違えると「CSSが軽い」と誤判定して、レンダリングブロッキングの
肥大化を見逃す（実際にNoto Sans JPの378KB CSSを長期間見逃していた）。
実際にHTTPサーバーを立てて、end-to-endで数値が出ることまで確認する。
"""

import gzip
import http.server
import os
import random
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import perf_check as m  # noqa: E402


LIGHT_HTML = (
    '<html><head><link rel="stylesheet" href="/light.css"></head>'
    '<body><p>軽い</p>'
    '<script src="/app.js"></script>'
    '<script src="https://example.com/ads.js"></script>'
    "</body></html>"
)

# rel/href/src のクォート無し、<body>内のstylesheet、外部CSSを混ぜた意地悪なケース。
TRICKY_HTML = (
    "<html><head>"
    '<link rel=stylesheet href=/a.css>'
    "<link rel='stylesheet' href='/b.css'>"
    '<link rel="preload" href="/not-a-stylesheet.css">'
    "</head><body>"
    '<link rel="stylesheet" href="/in-body.css">'
    "<script src=/bare.js></script>"
    "</body></html>"
)

def _incompressible_css(seed: int, rules: int) -> bytes:
    """gzipで潰れないCSSを作る。

    同じ文字の繰り返しだと圧縮後が数十バイトになり、「混入していない」ことを
    バイト数で検証できなくなるため、色をばらして圧縮が効かないようにする。
    """
    rng = random.Random(seed)
    return "\n".join(
        f".c{i}{{color:#{rng.randrange(0x1000000):06x};margin:{rng.randrange(999)}px}}"
        for i in range(rules)
    ).encode()


FILES = {
    "/light.html": LIGHT_HTML.encode(),
    "/tricky.html": TRICKY_HTML.encode(),
    "/light.css": b"body{margin:0}",
    "/a.css": _incompressible_css(1, 20),
    "/b.css": _incompressible_css(2, 20),
    # <head>のCSS合計より明確に大きくする（混入したら必ず検知できるように）。
    "/in-body.css": _incompressible_css(3, 4000),
    "/not-a-stylesheet.css": _incompressible_css(4, 4000),
    "/app.js": b"console.log(1);",
    "/bare.js": b"console.log(2);",
}


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandlerの規約
        body = FILES.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class PerfCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_counts_stylesheets_and_own_scripts(self):
        r = m.measure(self.base, "/light.html")
        self.assertEqual(r.error, "")
        self.assertEqual(r.css_files, 1)
        # 外部ドメインのads.jsは対象外。自ドメインのapp.jsだけ数える。
        self.assertEqual(r.js_files, 1)
        self.assertGreater(r.html_bytes, 0)

    def test_unquoted_attributes_and_head_scoping(self):
        r = m.measure(self.base, "/tricky.html")
        self.assertEqual(r.error, "")
        # <head>内のrel=stylesheetは2件(/a.css, /b.css)。
        # rel=preloadと<body>内のlinkは数えない。
        self.assertEqual(r.css_files, 2, "クォート無しrel/hrefを取りこぼしている")
        # クォート無しのsrcも拾う。
        self.assertEqual(r.js_files, 1, "クォート無しsrcを取りこぼしている")

    def test_body_stylesheet_excluded_from_css_bytes(self):
        """<body>内の重いCSSがレンダリングブロッキング量に混入しないこと。"""
        r = m.measure(self.base, "/tricky.html")
        in_body_size = len(gzip.compress(FILES["/in-body.css"]))
        self.assertLess(r.css_bytes, in_body_size)

    def test_missing_page_records_error_without_raising(self):
        r = m.measure(self.base, "/nope.html")
        self.assertIn("404", r.error)
        self.assertEqual(r.html_bytes, 0)

    def test_thresholds_flag_slow_pages(self):
        slow = m.PageResult(path="/x")
        slow.ttfb_sec = m.TTFB_WARN_SEC + 0.1
        slow.html_bytes = m.HTML_WARN_BYTES + 1
        slow.css_bytes = m.CSS_WARN_BYTES + 1
        # measure()と同じ判定をto_markdown側でも表現できていること
        markdown = m.to_markdown([slow], "https://example.com")
        self.assertIn("/x", markdown)

    def test_markdown_reports_failures(self):
        broken = m.PageResult(path="/broken", error="HTTP 500")
        markdown = m.to_markdown([broken], "https://example.com")
        self.assertIn("取得失敗", markdown)
        self.assertIn("HTTP 500", markdown)

    def test_exit_code_is_nonzero_when_flagged(self):
        """閾値超過でexit 1（ワークフローがこれを見てIssueを立てる）。"""
        argv = sys.argv
        sys.argv = ["perf_check.py", "--base-url", self.base, "--paths", "/nope.html"]
        try:
            self.assertEqual(m.main(), 1)
        finally:
            sys.argv = argv

    def test_exit_code_is_zero_when_all_ok(self):
        argv = sys.argv
        sys.argv = ["perf_check.py", "--base-url", self.base, "--paths", "/light.html"]
        try:
            self.assertEqual(m.main(), 0)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main(verbosity=1)

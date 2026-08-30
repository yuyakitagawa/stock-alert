"""
video/youtube_auth.py

YouTube アップロード用のリフレッシュトークンを取得する**ローカル1回きり**のスクリプト。
GitHub Actions では使わない（CI では YOUTUBE_REFRESH_TOKEN を Secrets から読む）。

事前準備:
  1. Google Cloud Console でプロジェクトを作り、YouTube Data API v3 を有効化する
  2. OAuth 同意画面を作成し、テストユーザーに投稿先チャンネルの Google アカウントを追加する
  3. 認証情報 → OAuth クライアント ID → アプリの種類「デスクトップアプリ」で作成する
  4. 取得した Client ID / Client Secret を環境変数に入れて、このスクリプトを実行する

使い方:
  python video/youtube_auth.py     # Client ID / Secret は .env から読む

ブラウザが開くので投稿先チャンネルのアカウントで許可すると、リフレッシュトークンが
表示される。それを .env と GitHub Secrets の YOUTUBE_REFRESH_TOKEN に登録する。

2026-08-30に scope を3つへ拡張した（upload だけでは視聴維持率もコメント投稿もできない）。
既存のトークンは upload だけなので、維持率の記録と記事URLコメントを使うには
このスクリプトをもう一度実行してトークンを取り直す必要がある。取り直したトークンは
upload も含むので、投稿側の動作は変わらない。
"""
import os
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from dotenv import load_dotenv

# Client ID / Secret は .env に入っている。環境変数への手打ちを求めると、
# 値があるのに「未設定」で弾かれる（2026-08-30にオーナーが実際に踏んだ）。
load_dotenv(os.path.expanduser("~/stock-alert/.env"))

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
# 3つ要る。upload=動画投稿、force-ssl=投稿直後の記事URLコメント、
# yt-analytics.readonly=平均視聴率と維持率カーブ（Data APIやサービスアカウントでは読めない）。
SCOPES = " ".join([
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
])
PORT = 8765
REDIRECT_URI = f"http://localhost:{PORT}"

_code_holder: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _code_holder["code"] = params.get("code", [None])[0]
        _code_holder["error"] = params.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        message = "認証に失敗しました" if _code_holder.get("error") else "認証しました。ターミナルに戻ってください。"
        self.wfile.write(f"<html><body><h2>{message}</h2></body></html>".encode())

    def log_message(self, *args):  # サーバのアクセスログは不要
        pass


def main():
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("YOUTUBE_CLIENT_ID と YOUTUBE_CLIENT_SECRET が .env にも環境変数にもありません")
        sys.exit(2)

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPES,
        # refresh_token は初回同意時にしか返らないため、確実に受け取れるよう毎回同意を求める
        "access_type": "offline",
        "prompt": "consent",
    })
    url = f"{AUTH_URL}?{params}"
    print(f"ブラウザで次のURLを開いて許可してください:\n{url}\n")
    webbrowser.open(url)

    server = HTTPServer(("localhost", PORT), _Handler)
    server.handle_request()

    if _code_holder.get("error") or not _code_holder.get("code"):
        print(f"認証コードを取得できませんでした: {_code_holder.get('error')}")
        sys.exit(1)

    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": _code_holder["code"],
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }, timeout=30)
    if not resp.ok:
        print(f"トークン交換に失敗 HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    refresh_token = resp.json().get("refresh_token")
    if not refresh_token:
        print("refresh_token が返りませんでした。一度アカウントのアプリ連携を解除してから再実行してください。")
        sys.exit(1)

    print("\n=== 以下を GitHub Secrets の YOUTUBE_REFRESH_TOKEN に登録してください ===")
    print(refresh_token)


if __name__ == "__main__":
    main()

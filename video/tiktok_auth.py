"""
video/tiktok_auth.py

TikTok アップロード用のリフレッシュトークンを取得する**ローカル1回きり**のスクリプト。
GitHub Actions では使わない（CI では TIKTOK_REFRESH_TOKEN を Secrets から読む）。

事前準備:
  1. TikTok for Developers でアプリを作成する
  2. Products に「Content Posting API」を追加する
  3. スコープに video.upload（審査通過後に直接公開したい場合は video.publish も）を追加する
  4. Redirect URI に、自分が管理する https の URL を登録する
     （TikTok は localhost / http を受け付けないため、kujira-watch.com 配下の
      適当なパス、例えば https://kujira-watch.com/tiktok-callback を登録すればよい。
      その URL が 404 でも、ブラウザのアドレスバーに code が乗るので手で貼り付けられる）

使い方:
  TIKTOK_CLIENT_KEY=xxx TIKTOK_CLIENT_SECRET=yyy \\
  TIKTOK_REDIRECT_URI=https://kujira-watch.com/tiktok-callback \\
  python video/tiktok_auth.py

表示された URL をブラウザで開いて許可し、リダイレクト先URLの ?code=... の値を貼り付ける。
"""
import os
import sys
import urllib.parse

import requests

AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"


def main():
    client_key = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
    redirect_uri = os.getenv("TIKTOK_REDIRECT_URI")
    if not client_key or not client_secret or not redirect_uri:
        print("TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET / TIKTOK_REDIRECT_URI を環境変数に設定してください")
        sys.exit(2)

    # 審査前は video.upload（下書き送信）だけで動く。審査通過後に video.publish を足すと
    # tiktok_client.py の TIKTOK_DIRECT_POST=1 による直接公開が使えるようになる。
    scope = os.getenv("TIKTOK_SCOPE", "video.upload")
    params = urllib.parse.urlencode({
        "client_key": client_key,
        "scope": scope,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": "kujira",
    })
    print(f"ブラウザで次のURLを開いて許可してください:\n{AUTH_URL}?{params}\n")
    print("許可後、リダイレクト先URLの ?code=... の値（&の手前まで）を貼り付けてください。")
    code = input("code: ").strip()
    if not code:
        print("code が空です")
        sys.exit(1)
    # TikTok は code を URL エンコードして返すことがある（末尾の *1 など）
    code = urllib.parse.unquote(code)

    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if not resp.ok:
        print(f"トークン交換に失敗 HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        print(f"refresh_token が返りませんでした: {resp.text[:300]}")
        sys.exit(1)

    print("\n=== 以下を GitHub Secrets の TIKTOK_REFRESH_TOKEN に登録してください ===")
    print(refresh_token)
    print(f"\n（このリフレッシュトークンの有効期限: {data.get('refresh_expires_in')} 秒）")


if __name__ == "__main__":
    main()

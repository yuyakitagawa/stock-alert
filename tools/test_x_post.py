"""
tools/test_x_post.py
X(Twitter)投稿の認証情報(X_API_KEY等)が正しく機能するか確認するための一回限りの
テストスクリプト。daily_alert.ymlの日次パイプライン全体を経由せず、
web.x_client.post_tweet()だけを直接呼ぶ。

投稿先アカウントが想定と違う（複数アカウントを行き来した等）事態を切り分けられるよう、
GET /2/users/me で「このキーが実際にどのアカウントとして認証されているか」も表示する。

使い方: python3 tools/test_x_post.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import requests

from web.x_client import _auth, post_tweet


def print_authenticated_account():
    auth = _auth()
    if auth is None:
        return
    resp = requests.get("https://api.x.com/2/users/me", auth=auth, timeout=20)
    if resp.ok:
        data = resp.json().get("data", {})
        username = data.get("username")
        print(f"🔑 このAPIキーは @{username} として認証されています → https://x.com/{username}")
    else:
        print(f"⚠ アカウント確認に失敗: HTTP {resp.status_code} {resp.text[:200]}")


def main():
    print_authenticated_account()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = f"【テスト投稿】大口投資家の監視ブログ 🐋 認証確認 ({now})"
    ok = post_tweet(text)
    if ok:
        print("✅ 投稿成功")
    else:
        print("❌ 投稿失敗（認証情報未設定、またはAPIエラー。ログを確認してください）")
        sys.exit(1)


if __name__ == "__main__":
    main()

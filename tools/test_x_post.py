"""
tools/test_x_post.py
X(Twitter)投稿の認証情報(X_API_KEY等)が正しく機能するか確認するための一回限りの
テストスクリプト。daily_alert.ymlの日次パイプライン全体を経由せず、
web.x_client.post_tweet()だけを直接呼ぶ。

使い方: python3 tools/test_x_post.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from web.x_client import post_tweet


def main():
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

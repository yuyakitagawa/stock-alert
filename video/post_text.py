"""
video/post_text.py

YouTube の投稿テキストで使う、サイトの名乗りとハッシュタグの整形。

サイト名を各クライアントに直書きすると、動画側で「クジラウォッチ」という
実在しない名前を名乗ってしまった事故（2026-08-19）と同じことが投稿文でも起きる。
名乗りはここ1箇所に集約し、kujira-watch/src/lib/site.ts の SITE_NAME と対で管理する。
"""
import re

# kujira-watch/src/lib/site.ts の SITE_NAME と同じ。変更したら両方を直すこと。
SITE_NAME = "大口投資家の監視ブログ"
SITE_URL = "https://kujira-watch.com"
# 押せないリンク欄で見せる用の、スキーム無しの短い表記。
SITE_HOST = "kujira-watch.com"

# ハッシュタグに使えない文字。銘柄名には空白や「．」が入ることがあり
# （例: Ｊ．フロント リテイリング）、そのまま「#」に続けるとタグが途中で切れて
# 残りが本文として漏れる。日本語・英数字・長音記号だけを残す。
# ひらがな・カタカナ（「・」U+30FB を含めない）・長音記号・漢字・々・半角/全角英数字のみ許可。
_TAG_NG = re.compile(
    "[^0-9A-Za-z"
    "\u3041-\u3096"          # ひらがな
    "\u30A1-\u30FA\u30FC"    # カタカナ（中黒を除く）＋長音記号
    "\u4E00-\u9FFF\u3005"    # 漢字＋々
    "\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A"  # 全角英数字
    "]"
)


def hashtag(name: str) -> str:
    """銘柄名などからハッシュタグを作る。タグにできない場合は空文字。"""
    cleaned = _TAG_NG.sub("", name or "")
    return f"#{cleaned}" if cleaned else ""


def article_url(article_id: str, source: str) -> str:
    """記事URL（UTM付き）。記事IDが無ければサイトのトップに落とす。
    UTMを付けるのはGA4で動画経由の流入をプラットフォーム別に識別するため。"""
    utm = f"utm_source={source}&utm_medium=social&utm_campaign=auto_video"
    if article_id:
        return f"{SITE_URL}/articles/{article_id}?{utm}"
    return f"{SITE_URL}?{utm}"

"""
web/x_post_format.py

週末のX自動投稿（x_weekly_trending.py / x_weekly_activists.py）で共用する整形ヘルパー。

Xの投稿上限は280「単位」で、全角文字は2単位・半角は1単位・URLは長さに関わらず一律23単位。
本文は日本語の固有名詞（社名・ファンド名）が主役で長さが読めないため、組み立てたあとに
`weighted_len()` で実測し、収まるまで行を落とす方式を両モジュールで共通にする。
"""
import re
import unicodedata

# 280単位に対する安全マージン。絵文字（環境により2単位以上に数えられることがある）と
# 改行の揺らぎを吸収するため10単位残す。
POST_MAX_WEIGHTED = 270
URL_WEIGHTED_UNITS = 23


def weighted_len(text: str) -> int:
    """Xの文字数カウントの近似。ASCII=1単位・それ以外（全角・絵文字）=2単位。
    URLは呼び出し側で `URL_WEIGHTED_UNITS` として別途加算する。"""
    return sum(1 if ord(c) < 128 else 2 for c in text)


def fits(text: str, url: str) -> bool:
    """本文（URLを含む）が投稿上限に収まるか。URLは実長ではなく23単位で数える。"""
    return weighted_len(text.replace(url, "")) + URL_WEIGHTED_UNITS <= POST_MAX_WEIGHTED


def clean_name(name: str) -> str:
    """EDINET正式名称を表示用に短くする。全角英数（ＮＩＰＰＯＮ等）はNFKCで半角へ寄せ
    （Xのカウントも全角2単位→半角1単位になり文字数の節約になる）、法人格の表記を落とし、
    空白は1つに畳む。280単位制限下では法人格の数文字が1行ぶんに相当するため。"""
    text = unicodedata.normalize("NFKC", name or "")
    text = re.sub(r"株式会社|\(株\)", "", text)  # NFKCで（株）→(株)に寄っている
    # 英文の法人格（末尾に付くもののみ。社名の一部である Capital / Management は残す）
    text = re.sub(
        r"[,、]?\s*(?:Pte\.?|Ptd\.?|Co\.?|Company|Corp\.?|Corporation|Inc\.?|"
        r"Ltd\.?|Limited|LLC|LLP|L\.?P\.?|PLC|S\.?A\.?|N\.?V\.?|GmbH)"
        r"(?:\s|\.|,|$)+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()


def label(text: str, max_units: int) -> str:
    """表示用に切り詰める。EDINETの正式名称は長く、行が折り返すと読みにくいため。
    上限は文字数ではなく単位数（weighted_len基準）。同じ行幅でも半角の海外ファンド名は
    全角の社名より多くの文字を置けるため、単位で測ったほうが行の見た目が揃う。"""
    if weighted_len(text) <= max_units:
        return text
    out = ""
    used = 0
    for char in text:
        cost = 1 if ord(char) < 128 else 2
        if used + cost > max_units - 2:  # 末尾の「…」ぶん(2単位)を残す
            break
        out += char
        used += cost
    return out + "…"


def hashtag(name: str) -> str:
    """銘柄名などをハッシュタグにする。Xのハッシュタグは記号・空白で切れてしまうため、
    記号類を取り除いた連続文字列にする。空になったら付けない。"""
    cleaned = re.sub(r"\W+", "", name or "")
    return f"#{cleaned}" if cleaned else ""

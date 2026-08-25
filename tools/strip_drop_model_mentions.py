#!/usr/bin/env python3
"""既存のブログ記事本文から「下落モデルの水準」への言及を削除する。

記事本文の54%（502/921本、2026-08-25実測）が「弊社モデルでは下落リスク水準を◯◯と評価」
の類を書いている一方、モデルの説明ページがサイトに無い（/methodology は404）。YMYL（金融）で
検証不能な独自指標を判断材料として提示している状態で、AdSense/E-E-A-Tの信頼性評価で減点される。
生成側は既に渡すのをやめた（web/publish_blog_articles.py）ので、こちらは既存記事を掃除する。

方針は「文を消すだけ。書き足さない」。LLMに書き直させると事実が混ざるリスクがあるため使わない。
  - 該当文が「〜株価は1,234円で、」で始まる場合のみ、その株価の節を残して文を締め直す
    （株価は開示原本と突き合わせられる事実なので捨てない）。
  - それ以外は文ごと削除する。
  - 削除の結果、中身が空になった<p>は段落ごと落とす。
  - HTMLタグを含む文には触らない（タグの対応が壊れるため）。件数だけ報告する。

出力が入力の部分集合であることを検証してから送信する（新しい文が混ざっていないこと）。

実行:
    python3 tools/strip_drop_model_mentions.py                 # dry-run（変更内容の表示のみ）
    python3 tools/strip_drop_model_mentions.py --limit 5 -v    # 変更前後を並べて確認
    python3 tools/strip_drop_model_mentions.py --apply         # バックアップを取ってから反映
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import update_article

load_dotenv()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 記事本文に実際に現れた言い回しから作ったパターン（2026-08-25に全921記事を走査して収集）。
# 「弊社/当社のモデル」だけでなく、主語を省いた「下落リスク水準は〜」「下落リスク局面」も拾う。
JA_PATTERN = re.compile(
    r"弊社モデル|当社モデル|弊社のリスク評価|当社のリスク評価|弊社リスク評価|当社リスク評価"
    r"|下落リスク水準|下落リスク局面|下落リスクが(?:低|高)い局面|下落リスクを(?:低|中程度|高)"
)
EN_PATTERN = re.compile(
    r"\b(?:our|proprietary)\s+(?:risk\s+)?models?\b|\bdownside\s+risk\b"
    r"|\brisk[- ]assessment\s+perspective\b",
    re.IGNORECASE,
)

# 「（前置き）株価は1,234円で、」の形で始まる文から株価の節だけを残すためのパターン。
JA_PRICE_LEAD = re.compile(r"^(.{0,40}?株価は[\d,]+円)(?:で[、,]|と[、,]|であり[、,]|であった[、,])")

TAG = re.compile(r"<[^>]+>")


def text_of(html: str) -> str:
    """タグを空白に置き換えて本文テキストにする。空文字で潰すと段落の末尾と次の段落の先頭が
    くっつき、元の本文には存在しない文がでっち上がって部分集合の検証が誤判定する。"""
    return re.sub(r"\s+", " ", TAG.sub(" ", html)).strip()


def split_ja(text: str) -> list:
    """句点で文に分割する（区切り文字は各文の末尾に残す）。"""
    return [s for s in re.split(r"(?<=。)", text) if s]


def split_en(text: str) -> list:
    """ピリオドで文に分割する（区切り文字は各文の末尾に残す）。

    英訳本文には「...acquisition price.From a risk assessment...」のようにピリオドの直後に
    空白が無い箇所があるため、空白だけを手掛かりにすると複数の文が1つに繋がったまま扱われ、
    削除対象が必要以上に大きくなる。大文字や「*」が続く場合も区切りとして扱う。"""
    return [s for s in re.split(r"(?<=[.!?])(?=\s*(?:\*|[A-Z]))", text) if s]


def polite_style(text: str) -> bool:
    """本文が「です・ます」調かどうか。残した株価の節を締める語尾を合わせるために見る。"""
    return len(re.findall(r"(?:です|ます|でした|ました)。", text)) >= len(
        re.findall(r"(?:だ|である|った|ている)。", text)
    )


def rewrite_ja_sentence(sentence: str, polite: bool) -> "str | None":
    """該当文を、株価の節だけ残した文に置き換える。残せなければ None（＝文ごと削除）。"""
    m = JA_PRICE_LEAD.match(sentence.strip())
    if not m:
        return None
    lead = m.group(1)
    if JA_PATTERN.search(lead):  # 残す部分にモデルの話が混ざっているなら残さない
        return None
    return f"{lead}{'でした' if polite else 'だった'}。"


def strip_html_block(html: str, pattern, splitter, rewriter=None) -> tuple:
    """<p>単位で該当文を落とす。返り値は (新しいHTML, 削除数, タグを含むため触れなかった数)。"""
    removed = skipped = 0
    polite = polite_style(text_of(html))

    def handle(match):
        nonlocal removed, skipped
        inner = match.group(1)
        out = []
        for sentence in splitter(inner):
            if not pattern.search(TAG.sub("", sentence)):
                out.append(sentence)
                continue
            if TAG.search(sentence):
                # タグを跨ぐ文は対応関係が壊れるので触らない
                skipped += 1
                out.append(sentence)
                continue
            replacement = rewriter(sentence, polite) if rewriter else None
            removed += 1
            if replacement:
                out.append(replacement)
        rebuilt = "".join(out)
        return "" if not text_of(rebuilt) else f"<p>{rebuilt}</p>"

    return re.sub(r"<p>(.*?)</p>", handle, html, flags=re.S), removed, skipped


def is_subset(original: str, updated: str, splitter) -> bool:
    """書き足しが無いことの検証。更新後の各文は、元の本文に含まれていなければならない
    （株価の節だけ残した文は元文の接頭辞なので、前方一致でも許容する）。"""
    src = text_of(original)
    for sentence in splitter(text_of(updated)):
        s = sentence.strip()
        if not s or s in src:
            continue
        if s.endswith(("だった。", "でした。")) and s[: s.rindex("。")].rstrip("だっでした") in src:
            continue
        return False
    return True


def process(article: dict) -> "dict | None":
    """1記事ぶんの差分を返す。変更が無ければ None。"""
    patch, stats = {}, {"removed": 0, "skipped": 0}
    for field, pattern, splitter, rewriter in (
        ("body", JA_PATTERN, split_ja, rewrite_ja_sentence),
        ("bodyEn", EN_PATTERN, split_en, None),
    ):
        html = article.get(field) or ""
        if not html or not pattern.search(text_of(html)):
            continue
        updated, removed, skipped = strip_html_block(html, pattern, splitter, rewriter)
        stats["removed"] += removed
        stats["skipped"] += skipped
        if updated != html:
            if not is_subset(html, updated, splitter):
                print(f"  ⚠ {article['id']}: 部分集合の検証に失敗したためスキップ（{field}）")
                return None
            patch[field] = updated
    return {"id": article["id"], "patch": patch, **stats} if patch else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="実際に反映する（無指定はdry-run）")
    p.add_argument("--limit", type=int, help="処理する記事数の上限")
    p.add_argument("-v", "--verbose", action="store_true", help="変更前後を表示する")
    args = p.parse_args()

    articles = fetch_all_articles()
    targets = [c for c in (process(a) for a in articles) if c]
    if args.limit:
        targets = targets[: args.limit]

    by_id = {a["id"]: a for a in articles}
    removed = sum(t["removed"] for t in targets)
    skipped = sum(t["skipped"] for t in targets)
    print(f"対象 {len(targets)}本 / 全{len(articles)}本、削除する文 {removed}件"
          f"{f'、タグを含むため触れない文 {skipped}件' if skipped else ''}")

    if args.verbose:
        for t in targets[:10]:
            print(f"\n--- {t['id']}")
            for field in t["patch"]:
                before = text_of(by_id[t["id"]][field])
                after = text_of(t["patch"][field])
                print(f"  [{field}] {len(before)}字 → {len(after)}字")
                for sentence in (split_ja if field == "body" else split_en)(before):
                    if (JA_PATTERN if field == "body" else EN_PATTERN).search(sentence):
                        print(f"    − {sentence.strip()[:150]}")

    if not args.apply:
        print("\n--apply で反映します（反映前にlogs/へバックアップを取ります）")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(REPO_ROOT, "logs", f"strip_drop_model_backup_{stamp}.json")
    with open(backup, "w", encoding="utf-8") as f:
        json.dump([{"id": t["id"], **{k: by_id[t["id"]].get(k) for k in t["patch"]}} for t in targets],
                  f, ensure_ascii=False, indent=2)
    print(f"バックアップ: {backup}")

    ok = 0
    for i, t in enumerate(targets, 1):
        if update_article(t["id"], t["patch"]):
            ok += 1
        else:
            print(f"  ⚠ 反映失敗: {t['id']}")
        if i % 50 == 0:
            print(f"  {i}/{len(targets)} 反映済み")
    print(f"反映 {ok}/{len(targets)}本")


if __name__ == "__main__":
    main()

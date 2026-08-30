#!/usr/bin/env python3
"""
tools/annotate_ratio_correction.py
保有比率の合算バグで数値が変わった公開記事に、機械生成の訂正注記を1ブロック足す。

背景（2026-08-30）: lib/edinet.py は共同保有の報告書で保有者ごとのcontextを先勝ちで
拾っていたため、`holding_ratio` が「提出者＋共同保有者の合算」ではなく
「筆頭保有者の1枠」になっていた（実測1,101件中656件＝60%がズレ）。
これを `_aggregate_ratio()` で合算へ直したので、公開済みの記事は誤った比率を
載せたまま残る（実例: 野村證券→トリケミカル研究所 0.46% / 実際は10.41%）。

なぜ本文を書き直さず注記にするか:
  - 本文の再生成は Anthropic API を呼ぶ（1本 $0.0092 × 数百本）。オーナー判断で非課金運用。
  - 数字だけの置換（fix_misreported_blog_articles.py --fix-body-numbers）では足りない。
    公開記事200本の実測で182本（91%）が本文に4種類以上の異なるパーセント値を含み
    （他の提出者の比率・過去の比率・5%ルールの5・業種の数字）、特定の値だけを
    安全に置換できない。さらに今回の誤りは数値の大小ではなく**意味が変わる**
    （「単独で△%を保有」→「共同保有者と合わせて△%」）ため、数字を差し替えても
    文章は誤ったまま残る。
  - 注記なら「何がどう違ったのか」を明示でき、記事も失わない。文面はテンプレートで
    数値はDBから流し込むだけなので LLM を一切呼ばない。

注記が「他の株主との比較や順位の記述は訂正前の数値にもとづく場合がある」と断るのは、
本文が引用している**他の提出者の比率も同じバグの影響を受けている**ため。数字の置換
（--fix-body-numbers）はその記事自身の比率しか直さないので、「筆頭株主はどちらか」
のような比較の結論が古いまま残りうる（実例: 日本製麻3306の記事は自社29.28%と
他社37.55%を比べており、29.28%だけ直しても比較が成立しない）。直せない範囲を
記事の上で開示する。

旧比率の出どころ: 記事側は保有比率をフィールドに持たないためタイトルから読む
（`_title_ratio()`）。ただし fix_misreported_blog_articles.py がタイトルを直すと
旧値が失われるので、**先に --snapshot で全記事のタイトル比率を保存**し、以降は
そのスナップショットを正とする。実行順に依存しないための措置。

冪等: 注記には `data-correction="ratio-aggregate-20260830"` を埋めてあり、
既に入っている記事は再実行しても二重に付かない。

実行:
    python3 tools/annotate_ratio_correction.py --snapshot   # 旧比率の退避（最初に1回）
    python3 tools/annotate_ratio_correction.py              # dry-run（対象の表示のみ）
    python3 tools/annotate_ratio_correction.py --limit 5    # 先頭5件だけ確認
    python3 tools/annotate_ratio_correction.py --apply      # microCMSへPATCH
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

import re

import lib.supabase_client as sb
from tools.fix_misreported_blog_articles import (
    _title_ratio, fetch_disclosures, find_disclosure,
)
from tools.reclassify_blog_articles import fetch_all_articles
from web.publish_blog_articles import MicroCMSPermissionError, update_article

load_dotenv()

# 旧テンプレートのタイトル（「赤松洋介氏が50.37%を保有」「日亜化学がエノモト8.74%を取得」）は
# fix_misreported_blog_articles._title_ratio() の「保有比率X%」「X%を新規保有」に当たらない。
# これらのタイトルに出てくる%は保有比率そのものなので、最初の%を旧比率として拾う
# （実測67件。金額は「13.6億円」のように%を伴わないので取り違えない）。
_ANY_PERCENT_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)%')


def old_ratio_from_title(article: dict) -> "float | None":
    ratio = _title_ratio(article)
    if ratio is not None:
        return ratio
    m = _ANY_PERCENT_RE.search(article.get("title") or "")
    return float(m.group(1)) if m else None

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
SNAPSHOT_PATH = os.path.join(LOG_DIR, "article_title_ratio_snapshot_20260830.json")

# 注記の識別子。冪等の判定にも使うので、文面を変えるときもこの値は変えない。
MARKER = "ratio-aggregate-20260830"
# 比率が「変わった」と数える閾値（%pt）。丸め差で注記を付けないための幅。
# tools/backfill_holding_details.py の RATIO_EPS と揃える。
RATIO_EPS = 0.05


def build_note(old_ratio: float, new_ratio: float) -> str:
    """訂正注記のHTML。数値以外は固定文。記事本文の先頭に差し込む。

    kujira-watch側は本文をそのままHTMLとして描画する（articles/[id]/page.tsx の
    dangerouslySetInnerHTML）。

    クラスは lib/format.ts の frameSpeculation()（「編集部の見立て」枠）と、
    見出しの `text-loss` だけを変えて揃える。**ここで使えるのはソース中に既に
    現れているクラスだけ**で、TailwindのJITはリポジトリのソースを走査して
    CSSを生成するため、microCMSの本文にしか出てこないクラス（例 border-l-loss）は
    生成されず無効になる。枠線の色を変えたくなったら、先にソース側でその
    ユーティリティを使うこと。
    """
    return (
        f'<aside data-correction="{MARKER}" '
        'class="not-prose my-6 rounded-md border border-rule border-l-4 border-l-brand-blue '
        'bg-section-tint px-4 py-3">'
        '<p class="m-0 text-xs font-bold text-loss">【訂正】保有比率について</p>'
        '<p class="m-0 mt-1 text-sm leading-relaxed text-ink-secondary">'
        f'本記事は当初、保有比率を提出者単独の保有分（{old_ratio}%）として記載していました。'
        '大量保有報告書の「株券等保有割合」は共同保有者を含む合算値であり、'
        f'この開示の正しい保有比率は{new_ratio}%です。2026年8月30日に訂正しました。'
        'なお本文中の、他の株主との比較や順位に関する記述は、訂正前の数値にもとづく場合があります。'
        '</p></aside>'
    )


def write_snapshot() -> int:
    """全記事のタイトル比率を退避する。他の是正ツールがタイトルを直すと旧値が
    失われるため、注記を付ける前に必ず1回走らせる。"""
    articles = fetch_all_articles()
    snapshot = {}
    for a in articles:
        ratio = old_ratio_from_title(a)
        if ratio is not None:
            snapshot[a["id"]] = ratio
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)
    print(f"タイトル比率を退避: {len(snapshot)}件 / 全{len(articles)}件 → {SNAPSHOT_PATH}")
    return 0


def load_snapshot() -> dict:
    if not os.path.exists(SNAPSHOT_PATH):
        return {}
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


def fetch_swept_doc_ids() -> set:
    """XBRLを引き直し済みの doc_id。tools/backfill_holding_details.py などのスイープが
    `xbrl_detail_fetched_date` を立てる。未スイープの行は比率が旧ロジックのままなので
    注記の根拠に使わない。"""
    out = set()
    step = 5000
    offset = 0
    while True:
        rows = sb.select(
            "edinet_large_holdings",
            f"select=doc_id&xbrl_detail_fetched_date=not.is.null&order=doc_id&offset={offset}",
            limit=step)
        out.update(r["doc_id"] for r in rows)
        if len(rows) < step:
            return out
        offset += step


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="store_true",
                   help="全記事のタイトル比率を退避して終了（注記を付ける前に1回）")
    p.add_argument("--apply", action="store_true", help="microCMSを実際に更新する")
    p.add_argument("--limit", type=int, default=None, help="先頭N件だけ処理する")
    args = p.parse_args()

    if args.snapshot:
        return write_snapshot()

    snapshot = load_snapshot()
    if not snapshot:
        print(f"⚠ {SNAPSHOT_PATH} がありません。先に --snapshot を実行してください。")
        return 1

    articles = fetch_all_articles()
    codes = sorted({str(a["stockCode"]) for a in articles if a.get("stockCode")})
    by_key = fetch_disclosures(codes)
    swept = fetch_swept_doc_ids()

    targets, skipped_no_row, skipped_no_old, already = [], 0, 0, 0
    skipped_not_swept = 0
    for a in articles:
        if MARKER in (a.get("body") or ""):
            already += 1
            continue
        old_ratio = snapshot.get(a["id"])
        if old_ratio is None:
            old_ratio = old_ratio_from_title(a)
        if old_ratio is None:
            skipped_no_old += 1
            continue
        row = find_disclosure(a, by_key)
        if not row or row.get("holding_ratio") is None:
            skipped_no_row += 1
            continue
        # まだXBRLを引き直していない行は、holding_ratio が旧ロジック（筆頭保有者の1枠）の
        # ままかもしれない。誤った値で「正しくは□□%」と書くほうが無注記より有害なので、
        # スイープ済みの行だけを注記の根拠にする。
        if row["doc_id"] not in swept:
            skipped_not_swept += 1
            continue
        new_ratio = row["holding_ratio"]
        if abs(float(old_ratio) - float(new_ratio)) < RATIO_EPS:
            continue
        targets.append((a, float(old_ratio), float(new_ratio)))

    print(f"記事 {len(articles)}件 / 注記が要る {len(targets)}件 / 既に注記あり {already}件")
    print(f"  対象外: 開示を一意に特定できない {skipped_no_row}件 "
          f"/ タイトルから旧比率を読めない {skipped_no_old}件 "
          f"/ XBRL未スイープで正しい比率が確定していない {skipped_not_swept}件")
    if not targets:
        return 0
    if args.limit is not None:
        targets = targets[: args.limit]

    if args.apply:
        # 本文を書き換えるので、元の本文を必ず退避してから送る。注記は固定ブロックなので
        # 消すのは容易だが、バックアップが無いと「注記を足す前の状態」に戻せない。
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(LOG_DIR, f"ratio_correction_body_backup_{stamp}.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump({a["id"]: a.get("body") or "" for a, _, _ in targets},
                      f, ensure_ascii=False, indent=1)
        print(f"本文を退避: {backup_path}")

    updated, failed = 0, []
    for a, old_ratio, new_ratio in targets:
        code = str(a.get("stockCode") or "")
        name = a.get("stockName") or code
        print(f"  {a['id']}: {name}({code}) {old_ratio}% → {new_ratio}%")
        if not args.apply:
            continue
        body = build_note(old_ratio, new_ratio) + (a.get("body") or "")
        try:
            if update_article(a["id"], {"body": body}):
                updated += 1
            else:
                failed.append(a["id"])
        except MicroCMSPermissionError as e:
            print(f"    ⚠ microCMSの権限エラーで中断: {e}")
            failed.append(a["id"])
            break

    if args.apply:
        print(f"更新 {updated}件 / 失敗 {len(failed)}件")
        if failed:
            print(f"  失敗: {', '.join(failed[:20])}")
            return 1
    else:
        print("dry-run（--apply で実際に更新）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

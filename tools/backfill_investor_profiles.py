"""
tools/backfill_investor_profiles.py
EDINET大量保有報告書(edinet_large_holdings)に登場する提出者のうち、kujira-watch側
/investors/[filer] 表示用のプロフィール(edinet_filer_classification.profile)が
未設定の投資家をまとめて分類＋プロフィール生成する。対象は2種類:
  - 未分類（edinet_filer_classificationに未登録）→ classify_filer()で分類してから生成
  - 分類済みだがprofile未生成 → get_filer_profile()のみ呼ぶ（classify_filerはキャッシュを
    即返却するため、対象を分けず両方に毎回classify_filer()を呼んでも追加コストは無い）

日次パイプライン（web/publish_blog_articles.py）は新規に記事化した提出者についてのみ
都度分類・生成を行うため、記事化されずに大量保有履歴だけ残っている既存提出者
（機能追加以前からのバックログ）をこのスクリプトでまとめて埋める。
tools/backfill_filer_profiles.py（分類済み・profile未生成のみが対象だった旧版）を置き換える。

使い方:
  python3 tools/backfill_investor_profiles.py
  python3 tools/backfill_investor_profiles.py --limit 20   # 件数を絞ってお試し実行
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from dotenv import load_dotenv
load_dotenv()

import lib.supabase_client as sb
from web.publish_blog_articles import classify_filer, get_filer_profile

_parser = argparse.ArgumentParser()
_parser.add_argument("--limit", type=int, default=None, help="処理する件数の上限（未指定なら全件）")
_parser.add_argument("--sleep", type=float, default=1.0, help="Claude呼び出し間隔（秒、レート制限対策）")
args = _parser.parse_args()


def main():
    if not sb.is_configured():
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です")
        return

    holdings = sb.select("edinet_large_holdings", "select=filer_name")
    all_filers = sorted({r["filer_name"] for r in holdings if r.get("filer_name")})

    classification_rows = sb.select("edinet_filer_classification", "select=filer_name,category,profile")
    classification = {r["filer_name"]: r for r in classification_rows}

    targets = [name for name in all_filers if not (classification.get(name) or {}).get("profile")]
    if args.limit:
        targets = targets[: args.limit]

    unclassified_count = sum(1 for name in targets if name not in classification)
    print(f"対象: {len(targets)}件（プロフィール未設定、大量保有履歴に登場する全{len(all_filers)}件中。"
          f"うち未分類{unclassified_count}件）")
    print("=" * 60)

    done = skipped = 0
    for i, filer_name in enumerate(targets, 1):
        existing = classification.get(filer_name)
        category = existing["category"] if existing else classify_filer(filer_name)["category"]
        profile = get_filer_profile(filer_name, category)
        if profile:
            done += 1
            print(f"  [{i}/{len(targets)}] ✅ {filer_name}（{category}）")
        else:
            skipped += 1
            print(f"  [{i}/{len(targets)}] ⏭ {filer_name}（{category}、情報不足等で空欄）")
        if i < len(targets):
            time.sleep(args.sleep)

    print("=" * 60)
    print(f"完了: プロフィール生成={done}  スキップ（空欄）={skipped}")


if __name__ == "__main__":
    main()

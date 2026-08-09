"""
tools/backfill_filer_profiles.py
既存のedinet_filer_classificationマスターのうち、profile（/investors/[filer]表示用の
800〜1000字程度の説明）が未設定の投資家をまとめてバックフィルする。

日次パイプライン（web/publish_blog_articles.py）は新規に遭遇した提出者についてのみ
都度generate_filer_profile()を呼ぶため、既存の登録済み投資家（新機能リリース時点で
未生成）をまとめて埋めるための一回限りのスクリプト。

使い方:
  python3 tools/backfill_filer_profiles.py
  python3 tools/backfill_filer_profiles.py --limit 20   # 件数を絞ってお試し実行
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from dotenv import load_dotenv
load_dotenv()

import lib.supabase_client as sb
from web.publish_blog_articles import get_filer_profile

_parser = argparse.ArgumentParser()
_parser.add_argument("--limit", type=int, default=None, help="処理する件数の上限（未指定なら全件）")
_parser.add_argument("--sleep", type=float, default=1.0, help="Claude呼び出し間隔（秒、レート制限対策）")
args = _parser.parse_args()


def main():
    if not sb.is_configured():
        print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定です")
        return

    rows = sb.select("edinet_filer_classification", "select=filer_name,category,profile")
    targets = [r for r in rows if not r.get("profile")]
    if args.limit:
        targets = targets[: args.limit]

    print(f"対象: {len(targets)}件（プロフィール未設定、全{len(rows)}件中）")
    print("=" * 60)

    done = skipped = 0
    for i, r in enumerate(targets, 1):
        profile = get_filer_profile(r["filer_name"], r["category"])
        if profile:
            done += 1
            print(f"  [{i}/{len(targets)}] ✅ {r['filer_name']}")
        else:
            skipped += 1
            print(f"  [{i}/{len(targets)}] ⏭ {r['filer_name']}（情報不足等で空欄）")
        if i < len(targets):
            time.sleep(args.sleep)

    print("=" * 60)
    print(f"完了: 生成={done}  スキップ（空欄）={skipped}")


if __name__ == "__main__":
    main()

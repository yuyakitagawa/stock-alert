"""
tools/fetch_tdnet.py
TDnet 適時開示（やのしん WEB-API・個人運営）を取得し、
隔離テーブル ext_tdnet_disclosures に保存するバッチ。

⚠️ 個人運営APIのため停止リスクあり。保存先は ext_ 隔離テーブル。
   このバッチが失敗してもコア・パイプラインには影響しない。

使い方:
    python3 tools/fetch_tdnet.py                  # 直近3日分（全銘柄）
    python3 tools/fetch_tdnet.py --days 7         # 直近7日分
    python3 tools/fetch_tdnet.py --watchlist      # ウォッチリスト銘柄のみ
    python3 tools/fetch_tdnet.py --catalyst-only  # カタリスト分類できた開示のみ
    python3 tools/fetch_tdnet.py --dry-run        # DB保存なし
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_watchlist_codes() -> list[str]:
    """dp_watchlist の全ユーザーのウォッチ銘柄コード（重複なし）。"""
    try:
        import lib.supabase_client as sb
        rows = sb.select("dp_watchlist", "select=code")
        return sorted({r["code"] for r in rows})
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser(description="TDnet適時開示取得（やのしん・隔離）")
    p.add_argument("--days", type=int, default=3, help="遡る日数（デフォルト3）")
    p.add_argument("--watchlist", action="store_true", help="ウォッチリスト銘柄のみ取得")
    p.add_argument("--catalyst-only", action="store_true", help="カタリスト分類できた開示のみ保存")
    p.add_argument("--dry-run", action="store_true", help="DB保存しない")
    args = p.parse_args()

    load_env()
    from lib.tdnet import scan_disclosures

    codes = None
    if args.watchlist:
        codes = get_watchlist_codes()
        if not codes:
            print("[tdnet] ウォッチリスト銘柄なし。終了。")
            return
        print(f"[tdnet] ウォッチリスト {len(codes)}銘柄を取得")

    print("=" * 60)
    print("TDnet適時開示取得（やのしんWEB-API・個人運営/隔離）")
    print(f"  対象: {'ウォッチリスト' if codes else f'全銘柄 直近{args.days}日'}")
    print(f"  分類フィルタ: {'カタリストのみ' if args.catalyst_only else 'なし'}")
    print(f"  DB保存: {'OFF (dry-run)' if args.dry_run else 'ON → ext_tdnet_disclosures'}")
    print("=" * 60)

    records = scan_disclosures(
        days_back=args.days,
        codes=codes,
        persist=not args.dry_run,
        only_categorized=args.catalyst_only,
    )

    print(f"\n完了: {len(records)}件")
    cats: dict[str, int] = {}
    for r in records:
        c = r["category"] or "（分類なし）"
        cats[c] = cats.get(c, 0) + 1
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}件")

    print("\n  サンプル（カタリスト系・先頭10件）:")
    shown = 0
    for r in records:
        if r["category"]:
            print(f"    {r['disclosed_at']} {r['code']} [{r['category']}] {r['title'][:40]}")
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()

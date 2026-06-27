"""
tools/fetch_jpx_market.py
JPX公式の無料データ（空売り残高 / 信用取引週末残高）を取得し、
jpx_short_selling / jpx_margin_balance に保存するバッチ。

使い方:
    python3 tools/fetch_jpx_market.py              # 空売り＋信用残を両方取得
    python3 tools/fetch_jpx_market.py --short      # 空売り残高のみ
    python3 tools/fetch_jpx_market.py --margin     # 信用取引残高のみ
    python3 tools/fetch_jpx_market.py --dry-run    # DB保存なし（列確認用）

⚠️ JPXのファイルレイアウトは変わりうる。初回は --dry-run で取得列を確認すること。
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


def main():
    p = argparse.ArgumentParser(description="JPX空売り・信用残取得")
    p.add_argument("--short", action="store_true", help="空売り残高のみ")
    p.add_argument("--margin", action="store_true", help="信用取引残高のみ")
    p.add_argument("--dry-run", action="store_true", help="DB保存しない")
    args = p.parse_args()

    load_env()
    do_short = args.short or not args.margin
    do_margin = args.margin or not args.short

    persist = not args.dry_run
    print("=" * 60)
    print("JPX公式マーケットデータ取得")
    print(f"  空売り残高: {'ON' if do_short else 'OFF'} / 信用取引残高: {'ON' if do_margin else 'OFF'}")
    print(f"  DB保存: {'OFF (dry-run)' if not persist else 'ON'}")
    print("=" * 60)

    if do_short:
        from lib.jpx_market_data import fetch_short_selling
        print("\n[空売り残高報告]")
        recs = fetch_short_selling(persist=persist)
        print(f"  取得: {len(recs)}件")
        for r in recs[:5]:
            print(f"    {r['calc_date']} {r['code']} {r['short_seller'][:20]} "
                  f"割合={r['short_ratio']} 数量={r['short_shares']}")

    if do_margin:
        from lib.jpx_market_data import fetch_margin_balance
        print("\n[信用取引週末残高]")
        recs = fetch_margin_balance(persist=persist)
        print(f"  取得: {len(recs)}件")
        for r in recs[:5]:
            print(f"    {r['record_date']} {r['code']} "
                  f"買残={r['margin_buy']} 売残={r['margin_sell']}")


if __name__ == "__main__":
    main()

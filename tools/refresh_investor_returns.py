"""投資家リターンランキング(/ranking/returns)の集計ビューを再計算する。

Supabaseのマテリアライズドビュー investor_returns_3m は
「EDINET買い開示の3ヶ月(63営業日)後リターン」を投資家別に平均したもので、
元データが株価キャッシュ(yahoo_price_cache)とEDINET開示(edinet_large_holdings)の
両方にまたがるため、株価更新の直後に再計算する必要がある。
定義とレビュー観点は supabase/create_investor_returns_3m.sql を参照。

使い方: python tools/refresh_investor_returns.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.supabase_client import is_configured, rpc  # noqa: E402


def main() -> int:
    if not is_configured():
        print("[investor_returns] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定のためスキップ")
        return 0

    rows = rpc("refresh_investor_returns_3m", {})
    # rpc()は失敗時にNoneを返す。ビューが空(0件)なのも異常なので同じ扱いにする。
    if not isinstance(rows, int) or rows <= 0:
        print(f"[investor_returns] リフレッシュ失敗 (戻り値: {rows!r})")
        return 1

    print(f"[investor_returns] リフレッシュ完了: 投資家{rows}名")
    return 0


if __name__ == "__main__":
    sys.exit(main())

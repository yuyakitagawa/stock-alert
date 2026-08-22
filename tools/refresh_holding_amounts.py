"""銘柄ランキング(/trending)の推定売買金額ビュー edinet_holding_amounts を再計算する。

EDINET大量保有報告書には取引金額が載らないため、開示1件ごとに
「保有比率の変化幅 × 発行済株式数 × 開示日終値」で概算した金額をマテリアライズド
ビューに持たせている。元データが開示(edinet_large_holdings)・株価(yahoo_price_cache)・
発行済株式数(jquants_fin_summary)の3つにまたがるので、開示スキャン後と株価更新後に呼ぶ。
定義とレビュー観点は supabase/create_edinet_holding_amounts.sql を参照。

使い方: python tools/refresh_holding_amounts.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.supabase_client import is_configured, rpc  # noqa: E402


def main() -> int:
    if not is_configured():
        print("[holding_amounts] SUPABASE_URL / SUPABASE_SERVICE_KEY が未設定のためスキップ")
        return 0

    rows = rpc("refresh_edinet_holding_amounts", {})
    # rpc()は失敗時にNoneを返す。ビューが空(0件)なのも異常なので同じ扱いにする。
    if not isinstance(rows, int) or rows <= 0:
        print(f"[holding_amounts] リフレッシュ失敗 (戻り値: {rows!r})")
        return 1

    print(f"[holding_amounts] リフレッシュ完了: 開示{rows}件")
    return 0


if __name__ == "__main__":
    sys.exit(main())

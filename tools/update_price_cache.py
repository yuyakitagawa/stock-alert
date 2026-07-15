"""
tools/update_price_cache.py
J-Quants API (v2) で直近N日分の全銘柄株価四本値を一括取得し、yahoo_price_cache を差分更新する。

daily_alert.yml の Step 0 で毎日実行し、rank_stocks.py が参照する価格キャッシュを
最新に保つ（get_prices()/get_price_df() は yahoo_price_cache の最終行を「直近株価」として使うため、
このスクリプトが動いていないとランキングの価格が古いまま更新されなくなる）。

使い方:
    python3 tools/update_price_cache.py --days 30
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def normalize_code(code_raw: str) -> str:
    code_raw = str(code_raw).strip()
    return code_raw[:4] if len(code_raw) == 5 else code_raw.zfill(4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="遡って取得する日数")
    args = parser.parse_args()

    load_env()

    api_key = os.environ.get("JQUANTS_API_KEY", "")
    if not api_key:
        print("ERROR: JQUANTS_API_KEY が設定されていません")
        sys.exit(1)

    import pandas as pd
    from jquantsapi import ClientV2
    from lib.db import init_db
    from lib.supabase_client import insert_ignore

    init_db()
    cli = ClientV2(api_key=api_key)

    start_dt = date.today() - timedelta(days=args.days)
    end_dt = date.today()
    print(f"取得期間: {start_dt} 〜 {end_dt}")

    df = cli.get_eq_bars_daily_range(start_dt=start_dt.isoformat(), end_dt=end_dt.isoformat())
    if df is None or len(df) == 0:
        print("データ取得なし。終了。")
        return

    close_col = "AdjustmentClose" if "AdjustmentClose" in df.columns else "Close"
    volume_col = "AdjustmentVolume" if "AdjustmentVolume" in df.columns else "Volume"

    rows = []
    for _, r in df.iterrows():
        code = normalize_code(r.get("Code", ""))
        d = r.get("Date")
        d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        close = r.get(close_col)
        if not code or not d_str or close is None or pd.isna(close):
            continue
        volume = r.get(volume_col)
        rows.append({
            "code": code,
            "date": d_str,
            "close": float(close),
            "volume": int(volume) if volume is not None and not pd.isna(volume) else None,
        })

    n_codes = df["Code"].nunique() if "Code" in df.columns else "?"
    print(f"取得件数: {len(rows):,} 行 ({n_codes}銘柄)")
    insert_ignore("yahoo_price_cache", rows, on_conflict="code,date")
    print("yahoo_price_cache 更新完了。")


if __name__ == "__main__":
    main()

"""
tools/update_price_cache.py
J-Quants API (v2) で直近N日分の全銘柄株価四本値を日ごとに取得し、yahoo_price_cache を差分更新する。

daily_alert.yml の Step 0 で毎日実行し、rank_stocks.py が参照する価格キャッシュを
最新に保つ（get_prices()/get_price_df() は yahoo_price_cache の最終行を「直近株価」として使うため、
このスクリプトが動いていないとランキングの価格が古いまま更新されなくなる）。

API制限: 5件/分 → 13秒スリープ（fetch_jquants_fin.py と同じレート）。
1日分の取得が失敗しても中断せず、残りの日付を続行する。

使い方:
    python3 tools/update_price_cache.py --days 30
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

RATE_SLEEP = 13  # 12秒+余裕 (5件/分)


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


def all_business_days(start_dt: date, end_dt: date) -> list[str]:
    days = []
    d = start_dt
    while d <= end_dt:
        if d.weekday() < 5:  # 月〜金のみ（土日はJ-Quantsも0件のためスキップ）
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


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
    target_dates = all_business_days(start_dt, end_dt)
    print(f"取得期間: {start_dt} 〜 {end_dt}  対象営業日: {len(target_dates)}日")
    print(f"予想時間: {len(target_dates) * RATE_SLEEP // 60}分")

    total_rows = 0
    errors = 0
    close_col = volume_col = None

    for i, d_str in enumerate(target_dates):
        try:
            df = cli.get_eq_bars_daily(date_yyyymmdd=d_str.replace("-", ""))
        except Exception as e:
            print(f"[{i+1}/{len(target_dates)}] {d_str}: ERROR {e}")
            errors += 1
            time.sleep(RATE_SLEEP)
            continue

        if df is None or len(df) == 0:
            print(f"[{i+1}/{len(target_dates)}] {d_str}: 0行")
            time.sleep(RATE_SLEEP)
            continue

        if close_col is None:
            close_col = "AdjustmentClose" if "AdjustmentClose" in df.columns else "Close"
            volume_col = "AdjustmentVolume" if "AdjustmentVolume" in df.columns else "Volume"

        rows = []
        for _, r in df.iterrows():
            code = normalize_code(r.get("Code", ""))
            close = r.get(close_col)
            if not code or close is None or pd.isna(close):
                continue
            volume = r.get(volume_col)
            rows.append({
                "code": code,
                "date": d_str,
                "close": float(close),
                "volume": int(volume) if volume is not None and not pd.isna(volume) else None,
            })

        insert_ignore("yahoo_price_cache", rows, on_conflict="code,date")
        total_rows += len(rows)
        print(f"[{i+1}/{len(target_dates)}] {d_str}: {len(rows)}銘柄 (累計: {total_rows:,})")

        if i < len(target_dates) - 1:
            time.sleep(RATE_SLEEP)

    print(f"完了: {total_rows:,}行保存 / {errors}日エラー")


if __name__ == "__main__":
    main()

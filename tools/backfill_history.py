"""
tools/backfill_history.py
2026-01-01 から今日まで、各営業日のランキングを生成して DB & Supabase に保存する。
既に DB に存在する日付はスキップ。

使い方:
  python3 tools/backfill_history.py
  python3 tools/backfill_history.py --start 2026-01-01 --end 2026-05-20
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import argparse
import time
import numpy as np
import joblib
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from lib.utils import get_prices, extract_features, add_cs_rank_features, sell_label
from lib.db import save_daily_ranking
import lib.supabase_client as sb
from config import BASE_DIR
from core.screener import get_tse_stock_list

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")

_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default="2026-01-01")
_parser.add_argument("--end",   default=date.today().isoformat())
_parser.add_argument("--force", action="store_true",
                      help="既存日付もスキップせず再生成・上書き（価格データ修正後の再生成用）")
args, _ = _parser.parse_known_args()

START_DATE = date.fromisoformat(args.start)
END_DATE   = date.fromisoformat(args.end)


def fetch_nikkei_history():
    """日経225の終値を {date_str: close} で返す（DBから取得）"""
    from lib.db import load_market_index_data
    nk_df = load_market_index_data("N225", days=2200)
    if nk_df is None or len(nk_df) == 0:
        return {}
    return {d.strftime("%Y-%m-%d"): float(c) for d, c in zip(nk_df.index, nk_df["Close"])}


def nk_rets_at(nk_hist, trading_dates, target_date_str):
    """target_date_str 時点の nk_rets (fractions) を返す"""
    idx = trading_dates.index(target_date_str) if target_date_str in trading_dates else -1
    if idx < 0:
        return None
    dates_up_to = trading_dates[:idx+1]
    closes = [nk_hist[d] for d in dates_up_to if d in nk_hist]
    if len(closes) < 6:
        return None
    p = closes
    r5  = (p[-1]-p[-6]) /p[-6]  if len(p)>=6  else 0
    r20 = (p[-1]-p[-21])/p[-21] if len(p)>=21 else 0
    r60 = (p[-1]-p[-61])/p[-61] if len(p)>=61 else 0
    return (r5, r20, r60)


def get_existing_dates():
    """DB に既存のランキング日付セットを返す"""
    from lib.db import get_ranking_dates_desc
    return set(get_ranking_dates_desc())


def main():
    print("=" * 60)
    print(f"バックフィル: {START_DATE} 〜 {END_DATE}")
    print("=" * 60)

    # モデル読み込み（下落モデルのみ）
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(drop_path):
        print("ERROR: rf_drop_model.pkl なし"); return
    drop_model = joblib.load(drop_path)
    print(f"モデル読み込み完了")

    # 銘柄リスト
    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗"); return
    codes = stock_list["code"].tolist()
    names = dict(zip(stock_list["code"], stock_list["name"]))
    print(f"銘柄数: {len(codes)}")

    # 日経225 履歴
    print("日経225履歴取得中...")
    nk_hist = fetch_nikkei_history()
    nk_dates = sorted(nk_hist.keys())
    print(f"  日経営業日: {len(nk_dates)} 日")

    # 対象営業日（START_DATE〜END_DATE）
    target_dates = [d for d in nk_dates if START_DATE.isoformat() <= d <= END_DATE.isoformat()]
    print(f"  対象営業日: {len(target_dates)} 日 ({target_dates[0]} 〜 {target_dates[-1]})")

    # 既存日付をスキップ（--force指定時は上書き再生成のためスキップしない）
    existing = get_existing_dates()
    if args.force:
        print(f"  --force指定: 既存 {len(existing)} 日も含め再生成・上書きします")
    else:
        target_dates = [d for d in target_dates if d not in existing]
        if not target_dates:
            print("全日付は既に DB に存在します。Supabase エクスポートのみ実行...")
            export_all_to_supabase(existing, names)
            return
        print(f"  新規処理対象: {len(target_dates)} 日（既存 {len(existing)} 日スキップ）")

    # 全銘柄の価格データを1回取得（並列）
    print(f"\n全銘柄の価格データ取得中（並列 20workers）...")
    price_cache = {}
    lock = threading.Lock()
    done = [0]

    def fetch_one(code):
        prices = get_prices(code, days=500)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  {done[0]}/{len(codes)} 取得済み...")
        if prices is None or len(prices) < 91:
            time.sleep(0.05)
            return
        time.sleep(0.15)
        with lock:
            price_cache[code] = prices

    with ThreadPoolExecutor(max_workers=20) as ex:
        list(as_completed({ex.submit(fetch_one, c): c for c in codes}))

    print(f"価格データ取得完了: {len(price_cache)} 銘柄")

    # 各営業日のランキング生成
    close_history: dict[str, list[float]] = {}
    for day_idx, date_str in enumerate(target_dates):
        print(f"\n[{day_idx+1}/{len(target_dates)}] {date_str} 処理中...")
        nk = nk_rets_at(nk_hist, nk_dates, date_str)

        # その日以前の価格だけにスライス
        raw_data = []
        for code, prices in price_cache.items():
            sliced = prices[prices.index <= date.fromisoformat(date_str)]
            if len(sliced) < 91:
                continue
            closes = sliced["Close"].values
            volumes = sliced["Volume"].tolist() if "Volume" in sliced.columns else None
            feat = extract_features(closes, volumes, nk)
            if feat is None:
                continue
            raw_data.append((code, closes, volumes, feat))

        if not raw_data:
            print(f"  有効銘柄なし、スキップ")
            continue

        # CS ランク特徴量
        feats_matrix = np.array([d[3] for d in raw_data], dtype=float)
        feats_aug = add_cs_rank_features(feats_matrix)

        # スコア計算
        db_rows = []
        for idx, (code, closes, volumes, feat) in enumerate(raw_data):
            feat_aug = feats_aug[idx]
            drop_prob = float(drop_model.predict_proba([feat_aug])[0][1])
            close = float(closes[-1])
            drop_pct = round(drop_prob * 100, 1)
            vol = round(feat[7], 1)

            nk20_pct = round(nk[1] * 100, 2) if nk else None
            recommend = sell_label(
                drop_pct,
                drawdown60=float(feat[10]),
                down_streak_raw=round(feat[12] * 20),
            )

            p = closes
            s20 = (p[-1]-p[-21])/p[-21]*100 if len(p)>=21 else 0
            rel20 = round(s20 - nk20_pct, 2) if nk20_pct is not None else None

            db_rows.append({
                "code":      str(code),
                "name":      names.get(code, ""),
                "close":     round(close, 1),
                "drop_prob": drop_pct,
                "vol":       vol,
                "recommend": recommend,
                "rel20":     rel20,
                "per":       None,
                "pbr":       None,
            })
            close_history.setdefault(str(code), []).append(round(close, 1))

        # 下落確率が低い順にソートして rank 付け
        db_rows.sort(key=lambda r: r["drop_prob"])

        # DB 保存
        save_daily_ranking(date_str, db_rows)
        sell_count = sum(1 for r in db_rows if r["recommend"] == "🔴 売り検討")
        print(f"  DB保存: {len(db_rows)}件 (🔴売り検討:{sell_count})")

    # 価格凍結チェック（今回生成した複数日にまたがりcloseが同一値のまま=更新漏れの疑い）
    if len(target_dates) >= 2:
        from lib.data_sanity import run_price_freshness_gate
        run_price_freshness_gate(close_history, source="backfill_history")

    # Supabase エクスポート（全対象日）
    all_dates = sorted(get_existing_dates())
    target_export = [d for d in all_dates if START_DATE.isoformat() <= d <= END_DATE.isoformat()]
    print(f"\nSupabase エクスポート: {len(target_export)} 日...")
    export_all_to_supabase(target_export, names)
    print("\n完了")


def export_all_to_supabase(dates, names):
    if not SUPABASE_URL:
        print("  SUPABASE_URL 未設定、スキップ")
        return

    from lib.db import get_ranking_by_date

    for date_str in sorted(dates):
        rows = get_ranking_by_date(date_str)
        if not rows:
            continue

        web_rows = []
        for i, r in enumerate(rows, 1):
            web_rows.append({
                "date":      r["date"],
                "code":      r["code"],
                "rank":      i,
                "name":      r["name"],
                "close":     r["close"],
                "rise_prob": r["rise_prob"],
                "drop_prob": r["drop_prob"],
                "net":       r["net"],
                "vol":       r["vol"],
                "recommend": r["recommend"],
                "rel20":     r["rel20"],
                "per":       r["per"],
                "pbr":       r["pbr"],
            })

        sb.upsert("gen_rankings", web_rows)
        print(f"  {date_str}: {len(web_rows)}件 upsert")


if __name__ == "__main__":
    main()

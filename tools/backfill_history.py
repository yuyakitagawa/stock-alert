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

import math
import threading
import argparse
import time
import numpy as np
import joblib
import requests
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from lib.utils import get_prices, extract_features, add_cs_rank_features, HEADERS, recommend_from_scores
from lib.db import save_daily_ranking, DB_PATH, init_db
from config import BASE_DIR
from core.screener import get_tse_stock_list
from core.rank_stocks import passes_buy_filter, MIN_LIQUIDITY_M, SECTOR_TO_ETF, STRONG_EFFECT_ETFS, get_sector_etf, _load_sector_cache, _save_sector_cache
import sqlite3

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

_parser = argparse.ArgumentParser()
_parser.add_argument("--start", default="2026-01-01")
_parser.add_argument("--end",   default=date.today().isoformat())
args, _ = _parser.parse_known_args()

START_DATE = date.fromisoformat(args.start)
END_DATE   = date.fromisoformat(args.end)

EMOJI_MAP = {
    "🥇 S買い":        "S買い",
    "🥈 A買い":        "A買い",
    "⏳ 方向感なし":   "方向感なし",
    "🔴 下降シグナル": "下降シグナル",
    "⚠️ 弱気シグナル": "弱気シグナル",
    "🟡 高値警戒":     "方向感なし",
    "高値警戒":        "方向感なし",
    "🔻 売り検討":     "下降シグナル",
    "売り検討":        "下降シグナル",
}

def clean_recommend(v):
    return EMOJI_MAP.get(v, v)


def sb_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def upsert(table, rows):
    if not rows or not SUPABASE_URL:
        return
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        resp = requests.post(url, headers=sb_headers(), json=batch, timeout=30)
        if not resp.ok:
            print(f"  [upsert] {table} failed: {resp.status_code} {resp.text[:100]}")


def fetch_nikkei_history():
    """日経225の終値を {date_str: close} で返す（約400日分）"""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
           f"?interval=1d&period1={int((datetime.now()-timedelta(days=500)).timestamp())}"
           f"&period2={int(datetime.now().timestamp())}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return {}
        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
        out = {}
        for ts, c in zip(timestamps, closes):
            if c is not None:
                d = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                out[d] = c
        return out
    except Exception as e:
        print(f"  日経取得失敗: {e}")
        return {}


def fetch_etf_history():
    """米国セクターETFの終値履歴を {etf: {date_str: close}} で返す（約500日分）"""
    import yfinance as yf
    etfs = sorted(set(SECTOR_TO_ETF.values()))
    try:
        data = yf.download(etfs, period="600d", auto_adjust=True, progress=False)["Close"]
        out = {e: {} for e in etfs}
        for e in etfs:
            col = data[e] if e in data.columns else None
            if col is None:
                continue
            for dt, val in col.dropna().items():
                out[e][dt.strftime("%Y-%m-%d")] = float(val)
        return out
    except Exception as ex:
        print(f"  ETF履歴取得失敗: {ex}")
        return {}


def etf_prev_ret(etf_hist, etf, date_str):
    """date_str 時点での etf の前日比リターン(%)を返す。データ不足なら None。"""
    closes = etf_hist.get(etf, {})
    sorted_dates = sorted(closes.keys())
    try:
        idx = sorted_dates.index(date_str)
    except ValueError:
        return None
    if idx < 1:
        return None
    prev_close = closes[sorted_dates[idx - 1]]
    curr_close = closes[date_str]
    return (curr_close - prev_close) / prev_close * 100


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
    init_db()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT DISTINCT date FROM daily_ranking").fetchall()
    con.close()
    return {r[0] for r in rows}


def main():
    print("=" * 60)
    print(f"バックフィル: {START_DATE} 〜 {END_DATE}")
    print("=" * 60)

    # モデル読み込み
    rise_path = os.path.join(BASE_DIR, "rf_model.pkl")
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pkl なし"); return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
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

    # 米国セクターETF 履歴
    print("米国セクターETF履歴取得中...")
    etf_hist = fetch_etf_history()
    _load_sector_cache()
    print(f"  ETF取得完了: {len(etf_hist)} セクター")

    # 対象営業日（START_DATE〜END_DATE）
    target_dates = [d for d in nk_dates if START_DATE.isoformat() <= d <= END_DATE.isoformat()]
    print(f"  対象営業日: {len(target_dates)} 日 ({target_dates[0]} 〜 {target_dates[-1]})")

    # 既存日付をスキップ
    existing = get_existing_dates()
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
    for day_idx, date_str in enumerate(target_dates):
        print(f"\n[{day_idx+1}/{len(target_dates)}] {date_str} 処理中...")
        nk = nk_rets_at(nk_hist, nk_dates, date_str)

        # その日以前の価格だけにスライス
        raw_data = []
        for code, prices in price_cache.items():
            sliced = prices[prices.index.strftime("%Y-%m-%d") <= date_str]
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
            rise_prob = float(rise_model.predict_proba([feat_aug])[0][1])
            drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) if drop_model else None
            close = float(closes[-1])
            rise_pct = round(rise_prob * 100, 1)
            drop_pct = round(drop_prob * 100, 1) if drop_prob is not None else None
            net = round(rise_pct - drop_pct, 1) if drop_pct is not None else rise_pct
            vol = round(feat[7], 1)

            nk20_pct = round(nk[1] * 100, 2) if nk else None
            buy_ok = passes_buy_filter(feat, close, volumes or [], nk20=nk20_pct)
            recommend = recommend_from_scores(net, drop_pct, allow_buy=buy_ok, vol=vol)

            stop_loss = round(close * (1 - 1.5 * vol / 100 * math.sqrt(20 / 252)), 0)
            p = closes
            s20 = (p[-1]-p[-21])/p[-21]*100 if len(p)>=21 else 0
            rel20 = round(s20 - nk20_pct, 2) if nk20_pct is not None else None

            db_rows.append({
                "code":      str(code),
                "name":      names.get(code, ""),
                "close":     round(close, 1),
                "rise_prob": rise_pct,
                "drop_prob": drop_pct,
                "net":       net,
                "vol":       vol,
                "recommend": recommend,
                "rel20":     rel20,
                "stop_loss": stop_loss,
                "per":       None,
                "pbr":       None,
            })

        # ネットスコア順にソートして rank 付け
        db_rows.sort(key=lambda r: r["net"], reverse=True)

        # A買い1日最大3件（net降順で4件目以降は方向感なしに降格）
        abuy_rows = sorted([r for r in db_rows if "A買い" in r["recommend"]], key=lambda r: r["net"], reverse=True)
        if len(abuy_rows) > 3:
            for r in abuy_rows[3:]:
                r["recommend"] = "⏳ 方向感なし"

        # 米国セクターETFリードラグフィルター（強相関セクターでの前日マイナスを降格）
        # S買い→A買い、A買い→方向感なし
        buy_now = [r for r in db_rows if "S買い" in r["recommend"] or "A買い" in r["recommend"]]
        for r in buy_now:
            etf = get_sector_etf(str(r["code"]))
            if etf not in STRONG_EFFECT_ETFS:
                continue
            ret = etf_prev_ret(etf_hist, etf, date_str)
            if ret is not None and ret < 0:
                if "S買い" in r["recommend"]:
                    r["recommend"] = "🥈 A買い"
                else:
                    r["recommend"] = "⏳ 方向感なし"
        _save_sector_cache()

        # DB 保存
        save_daily_ranking(date_str, db_rows)
        s_buy = sum(1 for r in db_rows if "S買い" in r["recommend"])
        a_buy = sum(1 for r in db_rows if "A買い" in r["recommend"])
        print(f"  DB保存: {len(db_rows)}件 (S買い:{s_buy} A買い:{a_buy})")

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

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    for date_str in sorted(dates):
        rows = con.execute(
            "SELECT * FROM daily_ranking WHERE date=? ORDER BY net DESC",
            (date_str,)
        ).fetchall()
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
                "recommend": clean_recommend(r["recommend"]),
                "rel20":     r["rel20"],
                "stop_loss": r["stop_loss"],
                "per":       r["per"],
                "pbr":       r["pbr"],
            })

        upsert("web_rankings", web_rows)
        s_buy = sum(1 for r in web_rows if r["recommend"] == "S買い")
        print(f"  {date_str}: {len(web_rows)}件 upsert (S買い:{s_buy})")

    con.close()


if __name__ == "__main__":
    main()

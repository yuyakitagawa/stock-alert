import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import threading
import pandas as pd
import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from lib.utils import get_prices, get_nikkei_returns, calc_rsi, extract_features, add_cs_rank_features, get_fundamentals, IsotonicCalibrated, HEADERS, SEQ_DAYS, recommend_from_net, recommend_from_scores
from config import BASE_DIR, BEAR_MARKET_THRESHOLD, FORECAST, RISE_THRESHOLD
from core.screener import get_tse_stock_list

TOP_SHOW = 10

MIN_LIQUIDITY_M = 50.0  # 20日平均売買代金(百万円)


def passes_buy_filter(feat, close, volumes, nk20=None):
    """S買い・A買いラベルを付与できる品質フィルター（元のスクリーナー基準）"""
    if feat[12] > 0.15:           return False  # down_streak > 3日
    if feat[10] < -0.15:          return False  # drawdown60 < -15%
    if feat[2] * 100 < 8.0:       return False  # 3ヶ月モメンタム < 8%
    if feat[1] * 100 < 0.0:       return False  # 20日モメンタム < 0%
    if feat[6] < 45.0:            return False  # RSI < 45
    if feat[6] > 70.0:            return False  # RSI > 70
    if feat[7] < 22.0:            return False  # ボラ < 22%
    if feat[7] > 50.0:            return False  # ボラ > 50%
    if feat[4] <= 0:              return False  # slope_up: MA5 ≤ MA25
    if feat[16] < 1.0:            return False  # vr2060 < 1.0
    if nk20 is not None and nk20 < 2.0: return False  # 日経20日リターン < 2%（下降/横ばい相場）
    if volumes and len(volumes) >= 20:
        valid = [v for v in volumes[-20:] if v is not None and not np.isnan(v)]
        if valid:
            va20 = np.mean(valid)
            if va20 * close / 1e6 < MIN_LIQUIDITY_M:
                return False
    return True









def main():
    print("=" * 55)
    print("スクリーナー × RF ランキング  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"スクリーナー通過銘柄に上昇確率スコアをつけてランキング")
    print("=" * 55)

    # モデル読み込み（上昇・下落）
    rise_path = os.path.join(BASE_DIR, "rf_model.pkl")
    drop_path = os.path.join(BASE_DIR, "rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pkl が見つかりません。先に rf_train_v3.py を実行してください")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    print(f"\n上昇モデル読み込み: {rise_path}")
    if drop_model:
        print(f"下落モデル読み込み: {drop_path}")

    # 日経225リターン取得
    print("\n日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    is_bear = nk20 is not None and nk20 < BEAR_MARKET_THRESHOLD
    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
        if is_bear:
            print(f"  ⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")
    else:
        print("  日経225: 取得失敗（相対リターンはN/A）")
        is_bear = False

    # 全TSE銘柄リスト取得（JPX直読み）
    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return
    codes = stock_list["code"].tolist()
    names = dict(zip(stock_list["code"], stock_list["name"]))
    print(f"全銘柄スキャン: {len(codes)} 銘柄")
    print(f"\n確率スコア計算中（並列処理）...")

    # フェーズ1: 全銘柄の特徴量を収集（並列）
    nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
    raw_data = []
    lock = threading.Lock()
    done_count = [0]
    total = len(codes)

    def fetch_one(code):
        prices = get_prices(code, days=400)
        with lock:
            done_count[0] += 1
            if done_count[0] % 500 == 0:
                print(f"  {done_count[0]}/{total} 取得済み... (有効: {len(raw_data)}銘柄)")
        if prices is None or len(prices) < 91:
            time.sleep(0.1)
            return None
        feat = extract_features(
            prices["Close"].values,
            prices["Volume"].tolist() if "Volume" in prices.columns else None,
            nk_rets,
        )
        if feat is None:
            time.sleep(0.1)
            return None
        time.sleep(0.2)
        return (code, prices, feat)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                with lock:
                    raw_data.append(result)

    print(f"有効銘柄: {len(raw_data)} 件")

    # フェーズ2: クロスセクショナルランク特徴量を付加（同日内での相対順位）
    if not raw_data:
        print("ERROR: 有効銘柄なし"); return
    feats_matrix = np.array([d[2] for d in raw_data], dtype=float)
    feats_aug = add_cs_rank_features(feats_matrix)  # 推論時は全銘柄を同一日として扱う

    # フェーズ3: モデルスコア計算
    results = []
    for idx, (code, prices, feat) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise_prob = float(rise_model.predict_proba([feat_aug])[0][1])
        drop_prob = float(drop_model.predict_proba([feat_aug])[0][1]) if drop_model else None
        close = float(prices["Close"].iloc[-1])
        rise_pct = round(rise_prob * 100, 1)
        drop_pct = round(drop_prob * 100, 1) if drop_prob is not None else None
        net = round(rise_pct - drop_pct, 1) if drop_pct is not None else rise_pct

        # ボラティリティ（feat[7] = vol20, 年率換算%）
        vol = round(feat[7], 1)
        if vol < 20:
            vol_label = "🟢低"
        elif vol < 40:
            vol_label = "🟡中"
        elif vol < 60:
            vol_label = "🟠高"
        else:
            vol_label = "🔴超高"

        # ネットスコア判定
        if net >= 15:
            judgment = "🟢強気買い"
        elif net >= 5:
            judgment = "🔵やや強気"
        elif net >= -5:
            judgment = "🟡中立    "
        elif net >= -15:
            judgment = "🟠やや弱気"
        else:
            judgment = "🔴売り検討"

        volumes = prices["Volume"].tolist() if "Volume" in prices.columns else []
        buy_ok = passes_buy_filter(feat, close, volumes, nk20=nk20)
        recommend = recommend_from_scores(net, drop_pct, allow_buy=buy_ok)

        # 損切りライン（1.5 ATR, 20日ボラベース）
        stop_loss = round(close * (1 - 1.5 * vol / 100 * math.sqrt(20 / 252)), 0)
        stop_pct  = round((stop_loss - close) / close * 100, 1)

        # 日経比相対リターン
        p = prices["Close"].values
        s5  = (p[-1] - p[-6])  / p[-6]  * 100 if len(p) >= 6  else 0
        s20 = (p[-1] - p[-21]) / p[-21] * 100 if len(p) >= 21 else 0
        s60 = (p[-1] - p[-61]) / p[-61] * 100 if len(p) >= 61 else 0
        rel5  = round(s5  - nk5,  2) if nk5  is not None else None
        rel20 = round(s20 - nk20, 2) if nk20 is not None else None
        rel60 = round(s60 - nk60, 2) if nk60 is not None else None
        rels = [r for r in [rel5, rel20, rel60] if r is not None]
        rs_score = round(sum(rels) / len(rels), 2) if rels else None

        cs_vol20_rank = round(float(feat_aug[32]) * 100, 0)  # ボラティリティのCS相対ランク(0-100%)

        row = {
            "銘柄コード": code,
            "銘柄名": names.get(code, ""),
            "直近株価(円)": round(close, 1),
            "上昇確率(%)": rise_pct,
            "下落確率(%)": drop_pct if drop_pct is not None else "-",
            "ネット(%)": net,
            "判定": judgment,
            "ボラ(%)": vol,
            "ボラ水準": vol_label,
            "ボラランク(%)": cs_vol20_rank,
            "推奨": recommend,
            "日経比5日(%)": rel5 if rel5 is not None else "-",
            "日経比20日(%)": rel20 if rel20 is not None else "-",
            "日経比60日(%)": rel60 if rel60 is not None else "-",
            "相対強度": rs_score if rs_score is not None else "-",
            "損切り価格(円)": stop_loss,
            "損切り幅(%)": stop_pct,
            "PER": None,
            "PBR": None,
            "ROE(%)": None,
        }
        results.append(row)

    # ランキング（ネットスコア順）
    result_df = pd.DataFrame(results).sort_values("ネット(%)", ascending=False).reset_index(drop=True)
    result_df.index += 1
    result_df.insert(0, "順位", result_df.index)

    # フェーズ4: 上位100銘柄のみ PER/PBR/ROE を取得
    TOP_FUND = 100
    print(f"\n上位{TOP_FUND}銘柄のファンダメンタルズ取得中...")
    for i, (idx, row) in enumerate(result_df.head(TOP_FUND).iterrows()):
        fund = get_fundamentals(str(row["銘柄コード"]))
        if fund:
            result_df.at[idx, "PER"]    = fund.get("PER")
            result_df.at[idx, "PBR"]    = fund.get("PBR")
            result_df.at[idx, "ROE(%)"] = fund.get("ROE")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{TOP_FUND} 完了...")

    # 表示
    print(f"\n{'='*90}")
    print(f"上位{TOP_SHOW}銘柄ランキング（ネットスコア順: 上昇確率-下落確率）")
    if is_bear:
        print(f"⚠️ 下落相場検知（日経20日: {nk20:+.1f}%）: モデルスコアの信頼性低下。買いは慎重に。")
    print(f"{'='*90}")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<18}  {'株価':>8}  {'上昇':>6}  {'下落':>6}  {'ネット':>7}  {'判定':<12}  {'ボラ':>6}  {'損切り':>8}  {'PER':>7}  {'PBR':>5}  {'ROE%':>6}  推奨")
    print("-" * 120)
    for _, row in result_df.head(TOP_SHOW).iterrows():
        drop_str  = f"{row['下落確率(%)']:>5.1f}%" if row['下落確率(%)'] != "-" else "   N/A"
        stop_str  = f"¥{row['損切り価格(円)']:,.0f}({row['損切り幅(%)']:+.1f}%)"
        per_val = row.get("PER"); pbr_val = row.get("PBR"); roe_val = row.get("ROE(%)")
        per_str = f"{per_val:>6.1f}x" if per_val is not None else "    N/A"
        pbr_str = f"{pbr_val:>4.2f}x" if pbr_val is not None else "  N/A"
        roe_str = f"{roe_val:>5.1f}%" if roe_val is not None else "   N/A"
        print(
            f"{int(row['順位']):>4}  {row['銘柄コード']:>6}  "
            f"{str(row['銘柄名']):<18}  "
            f"{row['直近株価(円)']:>8,.0f}円  "
            f"{row['上昇確率(%)']:>5.1f}%  "
            f"{drop_str}  "
            f"{row['ネット(%)']:>+6.1f}%  "
            f"{row['判定']:<12}  "
            f"{row['ボラ(%)']:>5.1f}%  "
            f"{stop_str:<14}  "
            f"{per_str}  "
            f"{pbr_str}  "
            f"{roe_str}  "
            f"{row['推奨']}"
        )

    # CSV保存
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(os.path.join(BASE_DIR, "data", "rankings"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "data", "rankings", f"ranking_{date_str}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")

    # DB保存
    from lib.db import save_daily_ranking
    db_date_str = datetime.now().strftime("%Y-%m-%d")
    db_rows = [
        {
            "code": str(row["銘柄コード"]),
            "name": row["銘柄名"],
            "close": row["直近株価(円)"],
            "rise_prob": row["上昇確率(%)"],
            "drop_prob": row["下落確率(%)"] if row["下落確率(%)"] != "-" else None,
            "net": row["ネット(%)"],
            "vol": row["ボラ(%)"],
            "recommend": row["推奨"],
            "rel20": row["日経比20日(%)"] if row["日経比20日(%)"] != "-" else None,
            "stop_loss": row["損切り価格(円)"],
            "per": row.get("PER"),
            "pbr": row.get("PBR"),
        }
        for _, row in result_df.iterrows()
    ]
    save_daily_ranking(db_date_str, db_rows)
    print(f"DB保存: {len(db_rows)}件 → stock_alert.db")
    print("完了")


if __name__ == "__main__":
    main()

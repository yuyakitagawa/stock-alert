import pandas as pd
import numpy as np
import time
import os
import glob
from datetime import datetime, timedelta
import joblib
from utils import get_prices, get_nikkei_returns, calc_rsi, extract_features, HEADERS, SEQ_DAYS

FORECAST = 63
RISE_THRESHOLD = 15.0
TOP_SHOW = 30










def main():
    print("=" * 55)
    print("スクリーナー × RF ランキング  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"スクリーナー通過銘柄に上昇確率スコアをつけてランキング")
    print("=" * 55)

    # モデル読み込み（上昇・下落）
    rise_path = os.path.expanduser("~/stock-alert/rf_model.pkl")
    drop_path = os.path.expanduser("~/stock-alert/rf_drop_model.pkl")
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pkl が見つかりません。先に rf_predict.py を実行してください")
        return
    rise_model = joblib.load(rise_path)
    drop_model = joblib.load(drop_path) if os.path.exists(drop_path) else None
    print(f"\n上昇モデル読み込み: {rise_path}")
    if drop_model:
        print(f"下落モデル読み込み: {drop_path}")

    # 日経225リターン取得
    print("\n日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    if nk5 is not None:
        print(f"  日経225: 5日{nk5:+.2f}% / 20日{nk20:+.2f}% / 60日{nk60:+.2f}%")
    else:
        print("  日経225: 取得失敗（相対リターンはN/A）")

    # スクリーナーCSV読み込み
    files = glob.glob(os.path.expanduser("~/stock-alert/screener_*.csv"))
    if not files:
        print("ERROR: screener CSVが見つかりません")
        return
    screener_df = pd.read_csv(max(files, key=os.path.getmtime))
    codes = screener_df["銘柄コード"].astype(str).tolist()
    names = dict(zip(screener_df["銘柄コード"].astype(str), screener_df["銘柄名"]))
    print(f"スクリーナー通過銘柄: {len(codes)} 銘柄")
    print(f"\n確率スコア計算中...")

    # 各銘柄の確率を計算
    results = []
    for i, code in enumerate(codes):
        prices = get_prices(code, days=400)
        if prices is None or len(prices) < 91:
            continue
        nk_rets = (nk5/100, nk20/100, nk60/100) if nk5 is not None else None
        feat = extract_features(prices["Close"].values, prices["Volume"].tolist() if "Volume" in prices.columns else None, nk_rets)
        if feat is None:
            continue
        # 連続下落 or 60日高値から15%超下落の銘柄を除外
        if feat[12] > 0.15 or feat[10] < -0.15:
            continue
        rise_prob = float(rise_model.predict_proba([feat])[0][1])
        drop_prob = float(drop_model.predict_proba([feat])[0][1]) if drop_model else None
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

        # 総合推奨（ネット × ボラティリティ）
        if net >= 10 and vol < 40:
            recommend = "✅ 買い可能性あり"
        elif net >= 5 and vol < 40:
            recommend = "🔵 買い可能性あり"
        elif net >= 5 and vol >= 40:
            recommend = "⚡ 買い可能性あり（荒れ注意）"
        elif net < -10 and vol < 40:
            recommend = "🔴 売り可能性あり"
        elif net < -5 and vol < 40:
            recommend = "⚠️ 売り可能性あり"
        elif net < -5 and vol >= 40:
            recommend = "🌀 売り様子見"
        else:
            recommend = "⏳ 様子見"

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
            "推奨": recommend,
            "日経比5日(%)": rel5 if rel5 is not None else "-",
            "日経比20日(%)": rel20 if rel20 is not None else "-",
            "日経比60日(%)": rel60 if rel60 is not None else "-",
            "相対強度": rs_score if rs_score is not None else "-",
        }
        results.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(codes)} 処理済み...")
        time.sleep(0.2)

    # ランキング（ネットスコア順）
    result_df = pd.DataFrame(results).sort_values("ネット(%)", ascending=False).reset_index(drop=True)
    result_df.index += 1
    result_df.insert(0, "順位", result_df.index)

    # 表示
    print(f"\n{'='*90}")
    print(f"上位{TOP_SHOW}銘柄ランキング（ネットスコア順: 上昇確率-下落確率）")
    print(f"{'='*90}")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<18}  {'株価':>8}  {'上昇':>6}  {'下落':>6}  {'ネット':>7}  {'判定':<12}  {'ボラ':>6}  {'水準':<6}  {'配当':>6}  推奨")
    print("-" * 100)
    for _, row in result_df.head(TOP_SHOW).iterrows():
        drop_str = f"{row['下落確率(%)']:>5.1f}%" if row['下落確率(%)'] != "-" else "   N/A"
        div_str = "   N/A"
        print(
            f"{int(row['順位']):>4}  {row['銘柄コード']:>6}  "
            f"{str(row['銘柄名']):<18}  "
            f"{row['直近株価(円)']:>8,.0f}円  "
            f"{row['上昇確率(%)']:>5.1f}%  "
            f"{drop_str}  "
            f"{row['ネット(%)']:>+6.1f}%  "
            f"{row['判定']:<12}  "
            f"{row['ボラ(%)']:>5.1f}%  "
            f"{row['ボラ水準']:<6}  "
            f"{div_str}  "
            f"{row['推奨']}"
        )

    # CSV保存
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = os.path.expanduser(f"~/stock-alert/ranking_{date_str}.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n全結果保存: {out_path}")
    print("完了")


if __name__ == "__main__":
    main()

"""
成績シミュレーション

--horizon 30  : 現在までのリターン（デフォルト、past_month_matches.csv使用）
--horizon 90  : 3ヶ月後リターン（loose_model_simulation.csv使用、Yahoo Financeから取得）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import time
import pandas as pd
from datetime import datetime, timedelta
from config import (BASE_DIR, NEW_CANDIDATE_NET_MIN, CANDIDATE_DROP_PROB_MAX,
                    CANDIDATE_CONFLICT_NET_MIN, CANDIDATE_CONFLICT_DROP_MIN)
from lib.utils import get_prices, get_price_at_date


def classify(net, drop_prob):
    if drop_prob is not None and drop_prob >= CANDIDATE_DROP_PROB_MAX:
        return "除外(高リスク)"
    if net < NEW_CANDIDATE_NET_MIN:
        return "除外(ネット低)"
    if (drop_prob is not None
            and net >= CANDIDATE_CONFLICT_NET_MIN
            and drop_prob >= CANDIDATE_CONFLICT_DROP_MIN):
        return "除外(コンフリクト)"
    return "🥇 S買い"


def __get_current_price(code):
    df = get_prices(code, days=10)
    if df is None or df.empty:
        return None
    closes = df["Close"].dropna()
    return float(closes.iloc[-1]) if len(closes) > 0 else None


def load_30d_data():
    """past_month_matches.csv: 現在までのリターン付き"""
    path = os.path.join(BASE_DIR, "simulations", "past_month_matches.csv")
    if not os.path.exists(path):
        # フォールバック: data/rankings/*.csv
        files = glob.glob(os.path.join(BASE_DIR, "data", "rankings", "ranking_*.csv"))
        if not files:
            return None
        dfs = []
        for f in sorted(files):
            tmp = pd.read_csv(f)
            date_str = os.path.basename(f).replace("ranking_", "").replace(".csv", "")
            tmp["entry_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dfs.append(tmp)
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={
            "銘柄コード": "code", "銘柄名": "name",
            "ネット(%)": "net", "上昇確率(%)": "rise_prob",
            "下落確率(%)": "drop_prob", "直近株価(円)": "entry_price",
        })
        df["current_return"] = None
    else:
        df = pd.read_csv(path)
        df = df.rename(columns={
            "通過日": "entry_date", "コード": "code", "銘柄名": "name",
            "ネット(%)": "net", "上昇(%)": "rise_prob", "下落(%)": "drop_prob",
            "エントリー": "entry_price", "現在まで(%)": "current_return",
        })
    df["code"] = df["code"].astype(str)
    for col in ["net", "drop_prob", "entry_price", "current_return"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_90d_data():
    """loose_model_simulation.csv: drop_prob/net付き、3ヶ月後価格は要取得"""
    path = os.path.join(BASE_DIR, "simulations", "loose_model_simulation.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={
        "通過日": "entry_date", "コード": "code", "銘柄名": "name",
        "ネット(%)": "net", "上昇(%)": "rise_prob", "下落(%)": "drop_prob",
        "エントリー": "entry_price", "6ヶ月リターン(%)": "return_6m",
    })
    df["code"] = df["code"].astype(str)
    for col in ["net", "drop_prob", "entry_price", "return_6m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["current_return"] = None
    return df


def _print_summary(df, horizon_label):
    all_buy = df[df["label"] == "🥇 S買い"].copy()
    ret = all_buy["current_return"].dropna()

    print(f"\n{'='*60}")
    print(f"ラベル別 成績サマリー（{horizon_label}）")
    print(f"{'='*60}")
    print(f"\n【全買い候補】{len(all_buy)}銘柄 (ネット≥8% & 下落確率<10%)")
    if len(ret) > 0:
        print(f"  平均リターン : {ret.mean():+.2f}%")
        print(f"  中央値       : {ret.median():+.2f}%")
        print(f"  勝率         : {(ret > 0).mean()*100:.1f}%")
        print(f"  最大利益     : {ret.max():+.2f}%")
        print(f"  最大損失     : {ret.min():+.2f}%")

    for label in buy_labels:
        sub = df[df["label"] == label].copy()
        r = sub["current_return"].dropna()
        if len(sub) == 0:
            continue
        print(f"\n【{label}】{len(sub)}銘柄")
        if len(r) > 0:
            print(f"  平均リターン : {r.mean():+.2f}%")
            print(f"  勝率         : {(r > 0).mean()*100:.1f}%")
            print(f"  最大利益     : {r.max():+.2f}%")
            print(f"  最大損失     : {r.min():+.2f}%")

    print(f"\n{'='*60}")
    print("個別明細（買い候補のみ、リターン降順）")
    print(f"{'='*60}")
    print(f"{'入日':>10}  {'コード':>6}  {'銘柄名':<18}  {'ラベル':<12}  {'ネット':>6}  {'下落確率':>7}  {'リターン':>8}  {'保有日':>5}")
    print("-" * 93)
    for _, r in all_buy.sort_values("current_return", ascending=False).iterrows():
        ret_v = r["current_return"]
        ret_str  = f"{ret_v:+.1f}%" if pd.notna(ret_v) else "   N/A"
        drop_str = f"{r['drop_prob']:.1f}%" if pd.notna(r.get("drop_prob")) else "  N/A"
        print(
            f"{str(r['entry_date']):>10}  {str(r['code']):>6}  {str(r.get('name','')):<18}  "
            f"{r['label']:<12}  {r['net']:>+5.1f}%  {drop_str:>7}  {ret_str:>8}  {int(r['holding_days']):>4}日"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, choices=[30, 90],
                        help="30=現在まで(past_month) / 90=3ヶ月後(loose_model)")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    today     = datetime.now().date()

    print("=" * 60)
    print(f"成績シミュレーション（{args.horizon}日ホライズン）  {today_str}")
    print("=" * 60)

    if args.horizon == 90:
        df = load_90d_data()
        source = "loose_model_simulation"
    else:
        df = load_30d_data()
        source = "past_month_matches"

    if df is None:
        print("ERROR: ソースデータが見つかりません")
        return
    print(f"データソース: {source} / {len(df)}行")

    df["label"] = df.apply(lambda r: classify(r["net"], r["drop_prob"]), axis=1)

    # 保有日数
    df["holding_days"] = df["entry_date"].apply(
        lambda d: (today - datetime.strptime(str(d), "%Y-%m-%d").date()).days
    )

    # リターン計算
    if args.horizon == 90:
        # 各エントリーの「entry_date + 90日」時点の株価を取得
        print(f"\n3ヶ月後（entry+90日）の株価取得中... ({len(df)}銘柄)")
        for i, (idx, row) in enumerate(df.iterrows()):
            entry_dt = datetime.strptime(str(row["entry_date"]), "%Y-%m-%d")
            target_dt = entry_dt + timedelta(days=90)
            # 未来は現在価格で代用
            if target_dt.date() >= today:
                p = _get_current_price(str(row["code"]))
            else:
                p = get_price_at_date(str(row["code"]), target_dt)
            if p is not None and pd.notna(row["entry_price"]) and row["entry_price"] > 0:
                df.at[idx, "current_return"] = (p - row["entry_price"]) / row["entry_price"] * 100
                df.at[idx, "current_price"] = p
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(df)}...")
            time.sleep(0.3)
    else:
        need_fetch = df["current_return"].isna()
        if need_fetch.any():
            print(f"\n現在価格取得中... ({need_fetch.sum()}銘柄)")
            for i, (idx, row) in enumerate(df[need_fetch].iterrows()):
                p = _get_current_price(str(row["code"]))
                if p is not None and pd.notna(row["entry_price"]) and row["entry_price"] > 0:
                    df.at[idx, "current_return"] = (p - row["entry_price"]) / row["entry_price"] * 100
                    df.at[idx, "current_price"] = p
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{need_fetch.sum()}...")
                time.sleep(0.3)
        if "current_price" not in df.columns:
            df["current_price"] = df["entry_price"] * (1 + df["current_return"] / 100)

    # DB保存
    run_key = f"{today_str}_h{args.horizon}"
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "entry_date": r["entry_date"],
            "code": str(r["code"]),
            "name": r.get("name", ""),
            "label": r["label"],
            "entry_price": r.get("entry_price"),
            "current_price": r.get("current_price"),
            "return_pct": r.get("current_return"),
            "holding_days": r["holding_days"],
            "net_at_entry": r["net"],
            "drop_prob_at_entry": r.get("drop_prob"),
        })
    print(f"\n集計完了: {len(rows)}件")

    horizon_label = "現在まで" if args.horizon == 30 else "3ヶ月後（entry+90日）"
    _print_summary(df, horizon_label)
    print(f"\n完了")


if __name__ == "__main__":
    main()

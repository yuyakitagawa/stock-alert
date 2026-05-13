"""
1ヶ月成績シミュレーション

simulations/past_month_matches.csv（またはdata/rankings/*.csv）を読み込み、
S買い / A買い の実際のリターンを集計してDBに保存する。
"""
import os
import glob
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import BASE_DIR, NEW_CANDIDATE_NET_MIN, CANDIDATE_DROP_PROB_MAX
from lib.db import save_simulation_results, load_simulation_results

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def classify(net, drop_prob):
    if drop_prob is not None and drop_prob >= CANDIDATE_DROP_PROB_MAX:
        return "除外(高リスク)"
    if net < NEW_CANDIDATE_NET_MIN:
        return "除外(ネット低)"
    if drop_prob is not None and drop_prob < 6.0 and net >= 10.0:
        return "🥇 S買い"
    return "🥈 A買い"


def get_current_price(code):
    ticker = f"{code}.T"
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=10)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        closes = (
            result[0].get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose", [])
        )
        closes = [c for c in closes if c is not None]
        return closes[-1] if closes else None
    except Exception:
        return None


def load_source_data():
    """simulations/past_month_matches.csv を優先。なければdata/rankings/*.csvから再構成"""
    sim_path = os.path.join(BASE_DIR, "simulations", "past_month_matches.csv")
    if os.path.exists(sim_path):
        df = pd.read_csv(sim_path)
        df = df.rename(columns={
            "通過日": "entry_date",
            "コード": "code",
            "銘柄名": "name",
            "ネット(%)": "net",
            "上昇(%)": "rise_prob",
            "下落(%)": "drop_prob",
            "エントリー": "entry_price",
            "現在まで(%)": "current_return",
        })
        df["code"] = df["code"].astype(str)
        df["drop_prob"] = pd.to_numeric(df["drop_prob"], errors="coerce")
        df["net"] = pd.to_numeric(df["net"], errors="coerce")
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        df["current_return"] = pd.to_numeric(df["current_return"], errors="coerce")
        return df, "past_month_matches"

    # フォールバック: data/rankings/*.csv
    files = glob.glob(os.path.join(BASE_DIR, "data", "rankings", "ranking_*.csv"))
    if not files:
        return None, None
    dfs = []
    for f in sorted(files):
        tmp = pd.read_csv(f)
        date_str = os.path.basename(f).replace("ranking_", "").replace(".csv", "")
        entry_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        tmp["entry_date"] = entry_date
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={
        "銘柄コード": "code",
        "銘柄名": "name",
        "ネット(%)": "net",
        "上昇確率(%)": "rise_prob",
        "下落確率(%)": "drop_prob",
        "直近株価(円)": "entry_price",
    })
    df["code"] = df["code"].astype(str)
    df["drop_prob"] = pd.to_numeric(df["drop_prob"], errors="coerce")
    df["net"] = pd.to_numeric(df["net"], errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["current_return"] = None
    return df, "rankings_csv"


def main():
    today_str = datetime.now().strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"1ヶ月成績シミュレーション  {today_str}")
    print("=" * 60)

    df, source = load_source_data()
    if df is None:
        print("ERROR: ソースデータが見つかりません")
        return
    print(f"データソース: {source} / {len(df)}行")

    # ラベル付け
    df["label"] = df.apply(lambda r: classify(r["net"], r["drop_prob"]), axis=1)

    # current_returnがない場合は現在価格を取得して計算
    need_fetch = df["current_return"].isna() if "current_return" in df.columns else pd.Series([True] * len(df))
    to_fetch = df[need_fetch].copy()
    if len(to_fetch) > 0:
        print(f"\n現在価格取得中... ({len(to_fetch)}銘柄)")
        prices = {}
        for i, (_, row) in enumerate(to_fetch.iterrows()):
            code = str(row["code"])
            p = get_current_price(code)
            prices[code] = p
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(to_fetch)}...")
            time.sleep(0.3)
        df.loc[need_fetch, "current_price"] = df.loc[need_fetch, "code"].map(prices)
        df.loc[need_fetch, "current_return"] = (
            (df.loc[need_fetch, "current_price"] - df.loc[need_fetch, "entry_price"])
            / df.loc[need_fetch, "entry_price"] * 100
        )
    else:
        # current_returnから逆算
        df["current_price"] = df["entry_price"] * (1 + df["current_return"] / 100)

    # 保有日数
    today = datetime.now().date()
    df["holding_days"] = df["entry_date"].apply(
        lambda d: (today - datetime.strptime(str(d), "%Y-%m-%d").date()).days
    )

    # DB保存
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
    save_simulation_results(today_str, rows)
    print(f"\nDB保存: {len(rows)}件 → simulation_results")

    # 集計レポート
    print(f"\n{'='*60}")
    print("ラベル別 成績サマリー")
    print(f"{'='*60}")

    buy_labels = ["🥇 S買い", "🥈 A買い"]
    all_buy = df[df["label"].isin(buy_labels)].copy()
    ret = all_buy["current_return"].dropna()

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
    print("個別明細 (買い候補のみ、リターン降順)")
    print(f"{'='*60}")
    print(f"{'入日':>10}  {'コード':>6}  {'銘柄名':<18}  {'ラベル':<12}  {'ネット':>6}  {'下落確率':>8}  {'リターン':>8}  {'保有日':>5}")
    print("-" * 95)
    for _, r in all_buy.sort_values("current_return", ascending=False).iterrows():
        ret_v = r["current_return"]
        ret_str = f"{ret_v:+.1f}%" if pd.notna(ret_v) else "  N/A"
        drop_str = f"{r['drop_prob']:.1f}%" if pd.notna(r.get("drop_prob")) else "  N/A"
        print(
            f"{str(r['entry_date']):>10}  {str(r['code']):>6}  {str(r.get('name','')):<18}  "
            f"{r['label']:<12}  {r['net']:>+5.1f}%  {drop_str:>8}  {ret_str:>8}  {int(r['holding_days']):>4}日"
        )

    print(f"\n完了: simulation_results に保存済み（run_date={today_str}）")


if __name__ == "__main__":
    main()

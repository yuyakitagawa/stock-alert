import requests
import pandas as pd
import numpy as np
import time
import io
import os
import argparse
from datetime import datetime, timedelta

# ── v1定数 ───────────────────────────────────
R2_THRESHOLD     = 0.65
MIN_MOMENTUM     = 5.0
MAX_MOMENTUM     = 30.0   # 追加: 急騰後ミーンリバージョン銘柄を除外
MIN_VOLATILITY   = 20.0   # 追加: +15%達成に必要な最低ボラ
MAX_VOLATILITY   = 50.0
MIN_MOMENTUM_20D = -3.0
MIN_PRICE        = 300

# ── v2定数 ───────────────────────────────────
RANK_6M_THRESHOLD = 0.70   # 6ヶ月リターン クロスセクション上位30%
MIN_RETURN_3M_V2  = 0.05   # 3ヶ月リターン下限（+5%）
MAX_RETURN_3M_V2  = 0.30   # 3ヶ月リターン上限（+30%）
MAX_HIGH_DIST_V2  = -0.20  # 52週高値からの乖離（-20%以内）
MIN_VOL_V2        = 20.0   # 年率ボラ下限（%）
MAX_VOL_V2        = 50.0   # 年率ボラ上限（%）
FETCH_DAYS_V2     = 400    # 252営業日≈365暦日+バッファ

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_tse_stock_list():
    """JPXから全上場銘柄を取得（ETF・REIT除外、1000番台含む）"""
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    print("銘柄リストを取得中...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        print(f"   ステータス: {resp.status_code} / サイズ: {len(resp.content)} bytes")
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)
        df.columns = df.columns.str.strip()

        code_col = [c for c in df.columns if "コード" in c][0]
        name_col = [c for c in df.columns if "銘柄名" in c][0]
        result = df[[code_col, name_col]].copy()
        result.columns = ["code", "name"]
        result["code"] = result["code"].str.strip()

        market_col = [c for c in df.columns if "市場" in c or "商品区分" in c]
        if market_col:
            market_filter = df[market_col[0]].str.contains("内国株式", na=False)
            result = result[market_filter]
            result = result[result["code"].str.match(r"^\d{4}$")]
            print(f"   市場区分フィルター適用")
        else:
            result = result[result["code"].str.match(r"^\d{4}$")]
            result = result[~result["name"].str.contains(
                "ETF|REIT|投信|上場投資|ファンド", na=False
            )]
            print(f"   銘柄名フィルター適用（市場区分列なし）")

        print(f"   → ETF除外後: {len(result)} 銘柄")
        return result.reset_index(drop=True)

    except Exception as e:
        print(f"銘柄リスト取得失敗: {e}")
        return None


def get_prices(code, days=180):
    """Yahoo Finance から過去N日の終値を取得"""
    ticker = f"{code}.T"
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        timestamps = result[0].get("timestamp", [])
        closes = (
            result[0].get("indicators", {})
            .get("adjclose", [{}])[0]
            .get("adjclose", [])
        )
        if not timestamps or not closes:
            return None
        idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo")
        df = pd.DataFrame({"Date": idx, "Close": closes}).dropna()
        return df
    except Exception:
        return None


def calc_metrics(df):
    """R²・モメンタム・ボラティリティ・スコアを計算（v1+v2対応）"""
    if df is None or len(df) < 30:
        return None

    prices = df["Close"].values
    n = len(prices)
    x = np.arange(n)

    # R²（線形トレンドの安定度）
    y_mean = prices.mean()
    ss_tot = ((prices - y_mean) ** 2).sum()
    coef = np.polyfit(x, prices, 1)
    ss_res = ((prices - np.polyval(coef, x)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    slope = coef[0]

    # モメンタム（3ヶ月=63営業日、6ヶ月=126営業日、20日）
    n3  = min(63,  n - 1)
    n6  = min(126, n - 1)
    n20 = min(20,  n - 1)
    momentum_20d = (prices[-1] - prices[-n20 - 1]) / prices[-n20 - 1] * 100
    momentum_3m  = (prices[-1] - prices[-n3  - 1]) / prices[-n3  - 1] * 100
    momentum_6m  = (prices[-1] - prices[-n6  - 1]) / prices[-n6  - 1] * 100

    # ボラティリティ（年率換算%）
    vol = (np.diff(prices) / prices[:-1]).std() * np.sqrt(252) * 100

    # 総合スコア（v1用）
    score = r2 * momentum_3m if slope > 0 else 0

    return {
        # v1フィールド
        "r2":           round(r2, 3),
        "momentum":     round(momentum_3m, 2),
        "momentum_20d": round(momentum_20d, 2),
        "vol":          round(vol, 2),
        "score":        round(score, 2),
        "close":        round(float(prices[-1]), 1),
        "slope_up":     bool(slope > 0),
        # v2フィールド
        "return_3m":    momentum_3m / 100,
        "return_6m":    momentum_6m / 100,
        "prices":       prices,
    }


# ── v1スクリーナー（変更禁止） ────────────────
def apply_screener_v1(universe_df):
    """v1スクリーナー"""
    mask = (
        (universe_df["r2"]             >= R2_THRESHOLD)
        & (universe_df["momentum"]     >= MIN_MOMENTUM)
        & (universe_df["momentum"]     <= MAX_MOMENTUM)
        & (universe_df["momentum_20d"] >= MIN_MOMENTUM_20D)
        & (universe_df["vol"]          >= MIN_VOLATILITY)
        & (universe_df["vol"]          <= MAX_VOLATILITY)
        & (universe_df["close"]        >= MIN_PRICE)
        & (universe_df["slope_up"])
    )
    return universe_df[mask].copy()


# ── v2スクリーナー ────────────────────────────
def apply_momentum_v2(universe_df):
    """v2モメンタムフィルター: 6ヶ月クロスセクション上位30% AND 3ヶ月+5%〜+30%

    クロスセクション順位は渡された universe_df 全体を母集団として計算する。
    呼び出し前に他条件でフィルターしないこと（バイアス防止）。
    """
    if universe_df.empty:
        return universe_df.copy()
    df = universe_df.copy()
    df["rank_6m"]  = df["return_6m"].rank(pct=True)
    cond_6m   = df["rank_6m"]   >= RANK_6M_THRESHOLD
    cond_3m_l = df["return_3m"] >= MIN_RETURN_3M_V2
    cond_3m_u = df["return_3m"] <= MAX_RETURN_3M_V2
    return df[cond_6m & cond_3m_l & cond_3m_u]


def apply_high_proximity_v2(df, prices_dict):
    """v2: 52週高値からの乖離が-20%以内（1年データなければ除外）"""
    keep = []
    for _, row in df.iterrows():
        prices = prices_dict.get(row["code"])
        if prices is None or len(prices) < 252:
            continue
        high_52w = max(prices[-252:])
        if high_52w <= 0:
            continue
        if (prices[-1] - high_52w) / high_52w >= MAX_HIGH_DIST_V2:
            keep.append(row)
    return pd.DataFrame(keep, columns=df.columns) if keep else pd.DataFrame(columns=df.columns)


def apply_volatility_band_v2(df, prices_dict):
    """v2: 年率ボラ20%〜50%（60日未満データなければ除外）"""
    keep = []
    for _, row in df.iterrows():
        prices = prices_dict.get(row["code"])
        if prices is None or len(prices) < 60:
            continue
        ann_vol = pd.Series(prices[-60:]).pct_change().dropna().std() * (252 ** 0.5) * 100
        if MIN_VOL_V2 <= ann_vol <= MAX_VOL_V2:
            keep.append(row)
    return pd.DataFrame(keep, columns=df.columns) if keep else pd.DataFrame(columns=df.columns)


def apply_price_filter(df):
    """株価300円以上フィルター（v1・v2共通）"""
    return df[df["close"] >= MIN_PRICE]


def apply_screener_v2(universe_df, prices_dict):
    """v2スクリーナー。順序: モメンタム → 52週高値 → ボラ帯 → 株価"""
    df = apply_momentum_v2(universe_df)
    df = apply_high_proximity_v2(df, prices_dict)
    df = apply_volatility_band_v2(df, prices_dict)
    df = apply_price_filter(df)
    return df


# ── A/B比較レポート ───────────────────────────
def write_compare_report(df_v1, df_v2, today):
    """v1・v2の通過銘柄を比較したMarkdownレポートを出力"""
    v1_codes = set(df_v1["code"].astype(str))
    v2_codes = set(df_v2["code"].astype(str))
    overlap  = v1_codes & v2_codes
    only_v1  = v1_codes - v2_codes
    only_v2  = v2_codes - v1_codes

    report = f"""# スクリーナー A/B比較 {today}

## 通過数
- v1（現状）: {len(v1_codes)} 銘柄
- v2（修正版）: {len(v2_codes)} 銘柄

## 重複
- 両方通過: {len(overlap)} 銘柄
- v1のみ: {len(only_v1)} 銘柄
- v2のみ: {len(only_v2)} 銘柄

## v1のみ通過した銘柄サンプル（最大10件）
{', '.join(list(only_v1)[:10])}

## v2のみ通過した銘柄サンプル（最大10件）
{', '.join(list(only_v2)[:10])}
"""
    path = os.path.expanduser(f"~/stock-alert/screener_compare_{today}.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"[INFO] 比較レポート出力: {path}")


# ── 出力整形ヘルパー ──────────────────────────
def _format_v1_output(df):
    """v1をCSV用DataFrameに変換（既存スキーマ互換）"""
    out = df[["code", "name", "close", "r2", "momentum", "vol", "score"]].rename(columns={
        "code":     "銘柄コード",
        "name":     "銘柄名",
        "close":    "直近株価(円)",
        "r2":       "R²（安定度）",
        "momentum": "3ヶ月モメンタム(%)",
        "vol":      "ボラティリティ(%)",
        "score":    "総合スコア",
    }).sort_values("総合スコア", ascending=False).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def _format_v2_output(df):
    """v2をCSV用DataFrameに変換"""
    out = df[["code", "name", "close", "return_3m", "return_6m", "vol", "score"]].copy()
    out["return_3m"] = (out["return_3m"] * 100).round(2)
    out["return_6m"] = (out["return_6m"] * 100).round(2)
    out = out.rename(columns={
        "code":      "銘柄コード",
        "name":      "銘柄名",
        "close":     "直近株価(円)",
        "return_3m": "3ヶ月リターン(%)",
        "return_6m": "6ヶ月リターン(%)",
        "vol":       "ボラティリティ(%)",
        "score":     "総合スコア",
    }).sort_values("3ヶ月リターン(%)", ascending=False).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def _print_top10(out_df, label, sort_col):
    print(f"\n[{label}] 上位10銘柄:")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<22}  {sort_col:>14}")
    print("-" * 58)
    for _, r in out_df.head(10).iterrows():
        print(f"{int(r['順位']):>4}  {r['銘柄コード']:>6}  {r['銘柄名']:<22}  {r[sort_col]:>+13.1f}%")


# ── メイン ───────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TSE株価スクリーナー")
    parser.add_argument(
        "--mode", choices=["v1", "v2", "both"], default="both",
        help="スクリーニングモード（デフォルト: both）",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="5銘柄のみ処理するテストモード",
    )
    args = parser.parse_args()
    mode      = args.mode
    test_mode = args.test

    # v2は52週高値計算のため252営業日≈400暦日が必要
    fetch_days = FETCH_DAYS_V2 if mode in ["v2", "both"] else 180

    print("=" * 55)
    print("TSE 株価スクリーナー  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    print(f"モード: {mode} | 取得日数: {fetch_days}日")
    if test_mode:
        print("【テストモード: 5銘柄のみ】")
    print("=" * 55)

    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    if test_mode:
        stock_list = stock_list.head(5)

    total       = len(stock_list)
    all_rows    = []
    prices_dict = {}
    errors      = 0

    print(f"\n{total} 銘柄をスクリーニング中...")

    for i, row in stock_list.iterrows():
        code = row["code"]
        name = row["name"]

        df     = get_prices(code, days=fetch_days)
        result = calc_metrics(df)

        if result is None:
            errors += 1
            if test_mode:
                print(f"  ❓ {name}({code}): データ取得失敗")
            continue

        all_rows.append({
            "code":         code,
            "name":         name,
            "r2":           result["r2"],
            "momentum":     result["momentum"],
            "momentum_20d": result["momentum_20d"],
            "vol":          result["vol"],
            "score":        result["score"],
            "close":        result["close"],
            "slope_up":     result["slope_up"],
            "return_3m":    result["return_3m"],
            "return_6m":    result["return_6m"],
        })
        prices_dict[code] = result["prices"]

        if test_mode:
            print(
                f"  {name}({code}): "
                f"R²={result['r2']} / モメンタム={result['momentum']:+.1f}% / "
                f"ボラ={result['vol']:.1f}%"
            )

        if not test_mode and (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} 処理済み... (収集: {len(all_rows)}銘柄)")

        time.sleep(0.1)

    if not all_rows:
        print("\nERROR: データ取得失敗")
        return

    universe_df    = pd.DataFrame(all_rows)
    date_str       = datetime.now().strftime("%Y%m%d")
    df_v1_filtered = None
    df_v2_filtered = None

    # ── v1フィルター ──
    if mode in ["v1", "both"]:
        df_v1_filtered = apply_screener_v1(universe_df)
        if not df_v1_filtered.empty:
            v1_out  = _format_v1_output(df_v1_filtered)
            v1_path = f"screener_v1_{date_str}.csv"
            v1_out.to_csv(v1_path, index=False, encoding="utf-8-sig")
            # 互換維持: 既存スクリプトが参照する screener_YYYYMMDD.csv は v1 のエイリアス
            v1_out.to_csv(f"screener_{date_str}.csv", index=False, encoding="utf-8-sig")
            print(f"\n[v1] {len(df_v1_filtered)} 銘柄通過 → {v1_path}")
            _print_top10(v1_out, "v1", "3ヶ月モメンタム(%)")
        else:
            print("\n[v1] 通過銘柄なし")

    # ── v2フィルター ──
    if mode in ["v2", "both"]:
        df_v2_filtered = apply_screener_v2(universe_df, prices_dict)
        if not df_v2_filtered.empty:
            v2_out  = _format_v2_output(df_v2_filtered)
            v2_path = f"screener_v2_{date_str}.csv"
            v2_out.to_csv(v2_path, index=False, encoding="utf-8-sig")
            print(f"\n[v2] {len(df_v2_filtered)} 銘柄通過 → {v2_path}")
            _print_top10(v2_out, "v2", "3ヶ月リターン(%)")
        else:
            print("\n[v2] 通過銘柄なし")

    # ── A/B比較レポート ──
    if mode == "both" and df_v1_filtered is not None and df_v2_filtered is not None:
        write_compare_report(df_v1_filtered, df_v2_filtered, date_str)

    print(f"\n{'='*55}")
    print(f"完了: 対象 {total} 銘柄 / 取得失敗 {errors} 銘柄")


if __name__ == "__main__":
    main()

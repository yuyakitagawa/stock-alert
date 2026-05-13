import requests
import pandas as pd
import numpy as np
import time
import io
import os
import argparse
from datetime import datetime, timedelta
from lib.utils import calc_rsi, get_sector_cached, get_prices, get_nikkei_returns, HEADERS

MIN_MOMENTUM     = 5.0    # 3ヶ月モメンタム下限（上限は撤廃しモデル判断に委ねる）
MIN_VOLATILITY   = 20.0
MAX_VOLATILITY   = 50.0
MIN_MOMENTUM_20D = -3.0
MIN_PRICE        = 300
MIN_VOL_RATIO    = 1.0
MIN_REL_STRENGTH = 0.0    # 3ヶ月相対強度（通常相場）日経並みでもOK
BEAR_REL_STRENGTH = 0.05  # 3ヶ月相対強度（下落相場：日経20日 < -5%）
MIN_RSI          = 40.0   # 売られすぎ（底割れ）を除外
MAX_RSI          = 70.0   # 買われすぎ（過熱）を除外
BEAR_NKK_20D     = -5.0   # 下落相場判定閾値（日経20日リターン%）
MIN_LIQUIDITY_M  = 50.0   # 20日平均売買代金 ≥ 50百万円（流動性確保）
MAX_SECTOR_COUNT = 2      # 同セクター通過上限（3銘柄以上集まったらバブル兆候とみなしセクター全除外）


def get_tse_stock_list():
    """JPXから全上場銘柄を取得（ETF・REIT除外）"""
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


def calc_metrics(df, nikkei_return_3m=None, nikkei_return_20d=None):
    """モメンタム・ボラティリティ・出来高・相対強度・スコアを計算"""
    if df is None or len(df) < 30:
        return None

    prices = df["Close"].values
    n = len(prices)

    coef  = np.polyfit(np.arange(n), prices, 1)
    slope = coef[0]

    n3  = min(63,  n - 1)
    n20 = min(20,  n - 1)
    momentum_20d = (prices[-1] - prices[-n20 - 1]) / prices[-n20 - 1] * 100
    momentum_3m  = (prices[-1] - prices[-n3  - 1]) / prices[-n3  - 1] * 100

    vol = (np.diff(prices) / prices[:-1]).std() * np.sqrt(252) * 100

    vr2060 = 1.0
    turnover_m = 0.0
    if "Volume" in df.columns:
        vols = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).values
        if len(vols) >= 60:
            vol20 = vols[-20:].mean()
            vol60 = vols[-60:].mean()
            if vol60 > 0:
                vr2060 = vol20 / vol60
        if len(vols) >= 20:
            vol20avg = vols[-20:].mean()
            turnover_m = vol20avg * float(prices[-1]) / 1_000_000

    rsi = calc_rsi(prices)
    rel_strength_3m  = (momentum_3m  / 100 - nikkei_return_3m)  if nikkei_return_3m  is not None else 0.0
    rel_strength_20d = (momentum_20d / 100 - nikkei_return_20d) if nikkei_return_20d is not None else 0.0
    score = momentum_3m * vr2060 if slope > 0 else 0

    return {
        "momentum":        round(momentum_3m, 2),
        "momentum_20d":    round(momentum_20d, 2),
        "vol":             round(vol, 2),
        "vr2060":          round(vr2060, 3),
        "turnover_m":      round(turnover_m, 1),
        "rsi":              round(rsi, 1),
        "rel_strength_3m":  round(rel_strength_3m, 4),
        "rel_strength_20d": round(rel_strength_20d, 4),
        "score":           round(score, 2),
        "close":           round(float(prices[-1]), 1),
        "slope_up":        bool(slope > 0),
    }


def apply_screener_v1(universe_df, rel_strength_min=MIN_REL_STRENGTH):
    mask = (
        (universe_df["momentum"]          >= MIN_MOMENTUM)
        & (universe_df["momentum_20d"]    >= MIN_MOMENTUM_20D)
        & (universe_df["vol"]             >= MIN_VOLATILITY)
        & (universe_df["vol"]             <= MAX_VOLATILITY)
        & (universe_df["close"]           >= MIN_PRICE)
        & (universe_df["slope_up"])
        & (universe_df["vr2060"]          >= MIN_VOL_RATIO)
        & (universe_df["rel_strength_3m"]  >= rel_strength_min)
        & (universe_df["rsi"]              >= MIN_RSI)
        & (universe_df["rsi"]              <= MAX_RSI)
        & (universe_df["turnover_m"]       >= MIN_LIQUIDITY_M)
    )
    return universe_df[mask].copy()


def apply_sector_concentration_filter(df, max_count=MAX_SECTOR_COUNT):
    """同セクター通過数が max_count を超えたら、そのセクター全銘柄を除外。
    バブル兆候の回避（特定セクターが過熱した時の一斉崩壊リスクを抑える）。"""
    if df.empty or "code" not in df.columns:
        return df, []
    sectors = [get_sector_cached(c) for c in df["code"]]
    df = df.copy()
    df["sector"] = sectors
    sec_counts = df["sector"].value_counts()
    over = sec_counts[sec_counts > max_count].index.tolist()
    if not over:
        return df.drop(columns=["sector"]), []
    excluded = df[df["sector"].isin(over)]
    kept = df[~df["sector"].isin(over)]
    return kept.drop(columns=["sector"]), [(s, int(sec_counts[s])) for s in over]


def _format_output(df):
    out = df[["code", "name", "close", "vr2060", "momentum", "vol", "score"]].rename(columns={
        "code":     "銘柄コード",
        "name":     "銘柄名",
        "close":    "直近株価(円)",
        "vr2060":   "出来高比(20d/60d)",
        "momentum": "3ヶ月モメンタム(%)",
        "vol":      "ボラティリティ(%)",
        "score":    "総合スコア",
    }).sort_values("総合スコア", ascending=False).reset_index(drop=True)
    out.insert(0, "順位", range(1, len(out) + 1))
    return out


def _print_top10(out_df):
    print(f"\n上位10銘柄:")
    print(f"{'順位':>4}  {'コード':>6}  {'銘柄名':<22}  {'3ヶ月モメンタム(%)':>14}")
    print("-" * 58)
    for _, r in out_df.head(10).iterrows():
        print(f"{int(r['順位']):>4}  {r['銘柄コード']:>6}  {r['銘柄名']:<22}  {r['3ヶ月モメンタム(%)']:>+13.1f}%")


def main():
    parser = argparse.ArgumentParser(description="TSE株価スクリーナー（v1）")
    parser.add_argument("--test", action="store_true", help="5銘柄のみ処理するテストモード")
    args = parser.parse_args()

    print("=" * 55)
    print("TSE 株価スクリーナー  " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    if args.test:
        print("【テストモード: 5銘柄のみ】")
    print("=" * 55)

    print("日経225リターン取得中...")
    nk5, nk20, nk60 = get_nikkei_returns()
    if nk5 is not None:
        print(f"  日経225: 3ヶ月{nk60:+.1f}% / 20日{nk20:+.1f}%")
    else:
        print("  日経225データ取得失敗 → 相対強度条件をスキップ")
    is_bear = nk20 is not None and nk20 < BEAR_NKK_20D
    if is_bear:
        print(f"  ⚠️ 下落相場（日経20日{nk20:+.1f}%）→ 相対強度閾値を{BEAR_REL_STRENGTH*100:.0f}%に引き上げ")
    # nikkei_return を小数（fraction）に変換して calc_metrics に渡す
    nikkei_return_3m  = nk60 / 100 if nk60 is not None else None
    nikkei_return_20d = nk20 / 100 if nk20 is not None else None

    stock_list = get_tse_stock_list()
    if stock_list is None:
        print("ERROR: 銘柄リスト取得失敗")
        return

    if args.test:
        stock_list = stock_list.head(5)

    total  = len(stock_list)
    rows   = []
    errors = 0

    print(f"\n{total} 銘柄をスクリーニング中...")

    for i, row in stock_list.iterrows():
        code = row["code"]
        name = row["name"]

        df     = get_prices(code, days=180)
        result = calc_metrics(df, nikkei_return_3m, nikkei_return_20d)

        if result is None:
            errors += 1
            if args.test:
                print(f"  ❓ {name}({code}): データ取得失敗")
            continue

        rows.append({
            "code":            code,
            "name":            name,
            "momentum":        result["momentum"],
            "momentum_20d":    result["momentum_20d"],
            "vol":             result["vol"],
            "vr2060":          result["vr2060"],
            "turnover_m":      result["turnover_m"],
            "rsi":              result["rsi"],
            "rel_strength_3m":  result["rel_strength_3m"],
            "rel_strength_20d": result["rel_strength_20d"],
            "score":            result["score"],
            "close":           result["close"],
            "slope_up":        result["slope_up"],
        })

        if args.test:
            print(
                f"  {name}({code}): "
                f"モメンタム={result['momentum']:+.1f}% / "
                f"出来高比={result['vr2060']:.2f} / "
                f"相対強度={result['rel_strength_3m']*100:+.1f}%"
            )

        if not args.test and (i + 1) % 500 == 0:
            print(f"  {i+1}/{total} 処理済み... (収集: {len(rows)}銘柄)")

        time.sleep(0.1)

    if not rows:
        print("\nERROR: データ取得失敗")
        return

    universe_df = pd.DataFrame(rows)
    date_str    = datetime.now().strftime("%Y%m%d")

    rel_strength_min = BEAR_REL_STRENGTH if is_bear else MIN_REL_STRENGTH
    filtered = apply_screener_v1(universe_df, rel_strength_min=rel_strength_min)
    filtered, excluded_sectors = apply_sector_concentration_filter(filtered)
    if excluded_sectors:
        secs_str = " / ".join([f"{s}({n}銘柄)" for s, n in excluded_sectors])
        print(f"\n⚠️ セクター集中除外: {secs_str}")
    if not filtered.empty:
        out      = _format_output(filtered)
        os.makedirs("data/screeners", exist_ok=True)
        out_path = f"data/screeners/screener_{date_str}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n{len(filtered)} 銘柄通過 → {out_path}")
        _print_top10(out)
    else:
        print("\n通過銘柄なし")

    print(f"\n{'='*55}")
    print(f"完了: 対象 {total} 銘柄 / 取得失敗 {errors} 銘柄")


if __name__ == "__main__":
    main()

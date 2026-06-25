"""
tools/optimize_net_weights.py
4モデル（rise/drop/alpha_rise/alpha_drop）の係数を最適化する。

net = w_rise * rise_prob - w_drop * drop_prob + w_alpha_rise * alpha_rise - w_alpha_drop * alpha_drop

💎シミュレーションの平均リターンを最大化する係数を探す。

使い方:
  python3 tools/optimize_net_weights.py
  python3 tools/optimize_net_weights.py --start 2025-01-01
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from datetime import datetime
from lib.utils import extract_features, add_cs_rank_features, HEADERS
from lib.db import load_market_index_data, get_price_raw, get_price_cache_codes
from config import BASE_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--start", default="2025-01-01")
parser.add_argument("--end", default="2026-06-02")
args = parser.parse_args()

START = args.start
END = args.end

# モデル読み込み
rise_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
drop_model = joblib.load(os.path.join(BASE_DIR, "rf_drop_model.pkl"))
alpha_rise_model = joblib.load(os.path.join(BASE_DIR, "rf_alpha_model.pkl"))
alpha_drop_model = joblib.load(os.path.join(BASE_DIR, "rf_alpha_drop_model.pkl"))
print("4モデル読み込み完了")

# 価格データ読み込み（Supabase yahoo_price_cache）
print("価格データをDBから読み込み中...")
codes = get_price_cache_codes()
price_cache = {}
for i, code in enumerate(codes):
    rows = get_price_raw(code)
    if rows:
        price_cache[code] = rows
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(codes)} 銘柄読み込み済...")
print(f"価格キャッシュ（DB）: {len(price_cache)} 銘柄")

# 日経225データ取得（Supabase yahoo_market_index テーブルから）
nk_df = load_market_index_data("N225", days=2200)
if nk_df is not None and len(nk_df) > 0:
    nk_hist = {d.strftime("%Y-%m-%d"): float(c) for d, c in zip(nk_df.index, nk_df["Close"])}
else:
    nk_hist = {}
print(f"日経225（DB）: {len(nk_hist)} 日分")

# 全コードの日付→インデックスマップ構築
all_dates = set()
code_date_map = {}
for code, rows in price_cache.items():
    cdm = {}
    for i, (d, c, v) in enumerate(rows):
        cdm[d] = i
        all_dates.add(d)
    code_date_map[code] = cdm

trading_days = sorted(d for d in all_dates if START <= d <= END)
print(f"対象営業日: {len(trading_days)} ({trading_days[0]} ~ {trading_days[-1]})")

# 営業日を1週間おきにサンプリング
SAMPLE_INTERVAL = 7
sample_days = trading_days[::SAMPLE_INTERVAL]
print(f"サンプル日: {len(sample_days)} 日（{SAMPLE_INTERVAL}日おき）")


def nk_rets_at(date_str):
    nk_dates_sorted = sorted(nk_hist.keys())
    idx = None
    for i, d in enumerate(nk_dates_sorted):
        if d <= date_str:
            idx = i
    if idx is None or idx < 60:
        return None
    closes = [nk_hist[nk_dates_sorted[i]] for i in range(max(0, idx - 60), idx + 1)]
    if len(closes) < 6:
        return None
    p = closes
    r5 = (p[-1] - p[-6]) / p[-6] if len(p) >= 6 else 0
    r20 = (p[-1] - p[-21]) / p[-21] if len(p) >= 21 else 0
    r60 = (p[-1] - p[-61]) / p[-61] if len(p) >= 61 else 0
    return (r5, r20, r60)


def get_future_return(code, date_str, horizon=63):
    """date_strからhorizon営業日後のリターンを取得"""
    cdm = code_date_map.get(code, {})
    idx = cdm.get(date_str)
    if idx is None:
        return None
    rows = price_cache[code]
    future_idx = idx + horizon
    if future_idx >= len(rows):
        return None
    entry_price = rows[idx][1]
    exit_price = rows[future_idx][1]
    if entry_price <= 0:
        return None
    return (exit_price - entry_price) / entry_price * 100


# 各サンプル日の全銘柄について4確率を計算
print("\n4確率の計算中...")
records = []  # [(code, date, rise, drop, alpha_rise, alpha_drop, future_ret)]

for di, day in enumerate(sample_days):
    if di % 5 == 0:
        print(f"  {di+1}/{len(sample_days)} {day}...")
    nk = nk_rets_at(day)

    raw_data = []
    for code, rows in price_cache.items():
        cdm = code_date_map[code]
        idx = cdm.get(day)
        if idx is None:
            continue
        # dayまでのデータをスライス
        sliced = rows[:idx + 1]
        if len(sliced) < 91:
            continue
        closes = np.array([r[1] for r in sliced])
        volumes = [r[2] for r in sliced]
        feat = extract_features(closes, volumes, nk)
        if feat is None:
            continue
        future_ret = get_future_return(code, day)
        if future_ret is None:
            continue
        raw_data.append((code, closes, volumes, feat, future_ret))

    if not raw_data:
        continue

    feats_matrix = np.array([d[3] for d in raw_data], dtype=float)
    feats_aug = add_cs_rank_features(feats_matrix)

    for idx, (code, closes, volumes, feat, future_ret) in enumerate(raw_data):
        feat_aug = feats_aug[idx]
        rise = float(rise_model.predict_proba([feat_aug])[0][1]) * 100
        drop = float(drop_model.predict_proba([feat_aug])[0][1]) * 100
        a_rise = float(alpha_rise_model.predict_proba([feat_aug])[0][1]) * 100
        a_drop = float(alpha_drop_model.predict_proba([feat_aug])[0][1]) * 100
        records.append((code, day, rise, drop, a_rise, a_drop, future_ret))

print(f"\nレコード数: {len(records)}")

# 係数最適化
from scipy.optimize import minimize

records_arr = np.array([(r[2], r[3], r[4], r[5], r[6]) for r in records])
rise_arr = records_arr[:, 0]
drop_arr = records_arr[:, 1]
a_rise_arr = records_arr[:, 2]
a_drop_arr = records_arr[:, 3]
future_arr = records_arr[:, 4]


def simulate_with_weights(weights, return_details=False):
    """💎シミュレーションを指定の係数で実行し、平均リターンを返す"""
    w_rise, w_drop, w_alpha_rise, w_alpha_drop = weights

    # 各レコードのnet scoreを計算
    nets = w_rise * rise_arr - w_drop * drop_arr + w_alpha_rise * a_rise_arr - w_alpha_drop * a_drop_arr

    # 日付別にグルーピング
    day_records = {}
    for i, (code, day, rise, drop, a_rise, a_drop, future_ret) in enumerate(records):
        if day not in day_records:
            day_records[day] = []
        day_records[day].append((i, code, nets[i], drop_arr[i], future_ret))

    # 💎条件でフィルターしてシミュレーション
    # drop < 5%, net >= 20, 上位10銘柄、14日間隔
    HOLD_DAYS = 63
    MAX_POS = 10
    INTERVAL = 14
    DROP_THRESH = 5.0
    NET_THRESH = 20.0

    sorted_days = sorted(day_records.keys())
    positions = {}  # code -> (entry_day_idx, entry_future_ret)
    trades = []
    last_entry_idx = -INTERVAL

    for di, day in enumerate(sorted_days):
        # 既存ポジションのクローズ（HOLD_DAYS経過）は不要 — future_retが63日後リターンなのでそのまま使う

        if di - last_entry_idx < INTERVAL:
            continue

        open_slots = MAX_POS - len(positions)
        if open_slots <= 0:
            continue

        recs = day_records[day]
        # 💎フィルター: drop < 5%, net >= NET_THRESH
        gems = [(idx, code, net, dp, fr) for idx, code, net, dp, fr in recs
                if dp < DROP_THRESH and net >= NET_THRESH and code not in positions]
        gems.sort(key=lambda x: -x[2])  # net降順

        added = 0
        for idx, code, net, dp, fr in gems:
            if added >= open_slots:
                break
            positions[code] = (di, fr)
            trades.append(fr)
            added += 1

        if added > 0:
            last_entry_idx = di

        # 古いポジションをクリア（簡略化: 次のエントリー機会までに全クローズ扱い）
        if di - last_entry_idx >= INTERVAL - 1:
            positions.clear()

    if not trades or len(trades) < 5:
        return (0.0, len(trades), 0.0, 0) if return_details else 999.0

    avg_ret = np.mean(trades)
    win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100

    if return_details:
        return avg_ret, len(trades), win_rate, sum(trades)
    return -avg_ret  # 最小化のため符号反転


# 現行の均等配分
print("\n" + "=" * 60)
current = simulate_with_weights([1, 1, 1, 1], return_details=True)
print(f"現行 (1,1,1,1): 平均={current[0]:+.2f}%, 件数={current[1]}, 勝率={current[2]:.0f}%, 累計={current[3]:+.1f}%")

# グリッドサーチ（粗い）
print("\nグリッドサーチ中...")
best_score = 999
best_w = [1, 1, 1, 1]
candidates = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]

count = 0
total = len(candidates) ** 4
for w1 in candidates:
    for w2 in candidates:
        for w3 in candidates:
            for w4 in candidates:
                if w1 == 0 and w2 == 0 and w3 == 0 and w4 == 0:
                    continue
                score = simulate_with_weights([w1, w2, w3, w4])
                if score < best_score:
                    best_score = score
                    best_w = [w1, w2, w3, w4]
                count += 1

print(f"  探索: {count} 組合せ")
grid_result = simulate_with_weights(best_w, return_details=True)
print(f"  グリッド最良 ({best_w[0]},{best_w[1]},{best_w[2]},{best_w[3]}): 平均={grid_result[0]:+.2f}%, 件数={grid_result[1]}, 勝率={grid_result[2]:.0f}%, 累計={grid_result[3]:+.1f}%")

# scipy最適化（グリッド最良を初期値に）
print("\n微調整最適化中...")
result = minimize(
    simulate_with_weights,
    best_w,
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 0.01, "fatol": 0.01},
)
opt_w = result.x.tolist()
opt_result = simulate_with_weights(opt_w, return_details=True)

print(f"\n{'=' * 60}")
print("最適化結果")
print(f"{'=' * 60}")
print(f"現行 (1.0, 1.0, 1.0, 1.0):")
print(f"  平均リターン={current[0]:+.2f}%, 件数={current[1]}, 勝率={current[2]:.0f}%, 累計={current[3]:+.1f}%")
print(f"\n最適 ({opt_w[0]:.2f}, {opt_w[1]:.2f}, {opt_w[2]:.2f}, {opt_w[3]:.2f}):")
print(f"  平均リターン={opt_result[0]:+.2f}%, 件数={opt_result[1]}, 勝率={opt_result[2]:.0f}%, 累計={opt_result[3]:+.1f}%")
print(f"\n適用式:")
print(f"  net = {opt_w[0]:.2f}*rise - {opt_w[1]:.2f}*drop + {opt_w[2]:.2f}*alpha_rise - {opt_w[3]:.2f}*alpha_drop")

# 感度分析: 各モデルを外した場合
print(f"\n{'=' * 60}")
print("感度分析（各モデルの寄与）")
print(f"{'=' * 60}")
ablations = [
    ("rise のみ",          [1, 0, 0, 0]),
    ("drop のみ",          [0, 1, 0, 0]),
    ("alpha_rise のみ",    [0, 0, 1, 0]),
    ("alpha_drop のみ",    [0, 0, 0, 1]),
    ("abs(rise-drop) のみ",[1, 1, 0, 0]),
    ("alpha のみ",         [0, 0, 1, 1]),
    ("均等 (現行)",        [1, 1, 1, 1]),
    ("最適",               opt_w),
]
for label, w in ablations:
    r = simulate_with_weights(w, return_details=True)
    print(f"  {label:24s} ({w[0]:.1f},{w[1]:.1f},{w[2]:.1f},{w[3]:.1f}): 平均={r[0]:+.2f}% 件数={r[1]:>3} 勝率={r[2]:.0f}%")

"""
30期間マルチバックテスト + フィルタ比較分析（5年間ランダムサンプリング）

実行: python3 multi_backtest.py
  --skip-run   : バックテスト実行をスキップ（既存CSVのみ分析）
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

from config import BASE_DIR

PYTHON = sys.executable

PERIODS = [
    # 既存（2025年）
    ("2025-01-01", "2025-04-01", "2025冬春"),
    ("2025-01-15", "2025-04-15", "2025冬春B"),
    ("2025-02-01", "2025-05-01", "2025冬春C"),
    ("2025-02-14", "2025-05-14", "2025春前"),
    ("2025-03-01", "2025-06-01", "2025春A"),
    ("2025-03-15", "2025-06-15", "2025春B"),
    ("2025-04-01", "2025-07-01", "2025春C"),
    ("2025-04-15", "2025-07-15", "2025初夏"),
    ("2025-05-01", "2025-08-01", "2025夏A"),
    ("2025-05-14", "2025-08-14", "2025夏B"),
    ("2025-06-01", "2025-09-01", "2025夏C"),
    ("2025-07-01", "2025-10-01", "2025夏D"),
    # 既存（2024年暴落）
    ("2024-07-01", "2024-10-01", "2024暴落"),
    # ランダム追加（seed=42, 5年間）
    ("2021-07-07", "2021-10-06", "2021夏A"),
    ("2021-07-19", "2021-10-18", "2021夏B"),
    ("2021-07-21", "2021-10-20", "2021夏C"),
    ("2021-11-11", "2022-02-10", "2021秋冬"),
    ("2021-11-24", "2022-02-23", "2021秋冬B"),
    ("2021-12-13", "2022-03-14", "2021冬A"),
    ("2021-12-31", "2022-04-01", "2021冬B"),
    ("2022-02-28", "2022-05-30", "2022春"),
    ("2022-08-08", "2022-11-07", "2022夏"),
    ("2022-08-17", "2022-11-16", "2022夏B"),
    ("2022-09-30", "2022-12-30", "2022秋"),
    ("2022-12-01", "2023-03-02", "2022冬"),
    ("2023-09-28", "2023-12-28", "2023秋"),
    ("2024-06-06", "2024-09-05", "2024初夏"),
    ("2024-09-09", "2024-12-09", "2024秋"),
    ("2024-12-16", "2025-03-17", "2024冬"),
    ("2025-03-03", "2025-06-02", "2025春D"),
    ("2025-07-03", "2025-10-02", "2025夏E"),
    ("2025-07-11", "2025-10-10", "2025夏F"),
    ("2025-07-14", "2025-10-13", "2025夏G"),
]

CONFIGS = {
    "現行(10/8) ": dict(net_min=10.0, drop_max=8.0,  conflict_net=10.0, conflict_drop=5.0),
    "旧(8/12)  ": dict(net_min=8.0,  drop_max=12.0, conflict_net=None,  conflict_drop=None),
    "緩(8/8)   ": dict(net_min=8.0,  drop_max=8.0,  conflict_net=None,  conflict_drop=None),
    "厳(12/6)  ": dict(net_min=12.0, drop_max=6.0,  conflict_net=None,  conflict_drop=None),
}

# ウォークフォワード: 期間開始日が以下以上のときcutoffモデルを使用
# None = デフォルトモデル (rf_model.pkl, 学習cutoff=2025-01-01)
CUTOFF_SCHEDULE = [
    ("2025-07-01", "2025-07-01"),
    ("2025-05-01", "2025-05-01"),
    ("2025-03-01", "2025-03-01"),
]

def get_model_cutoff(start_date_str):
    """期間開始日に対応するモデルcutoff日を返す。None=デフォルト"""
    for threshold, cutoff in CUTOFF_SCHEDULE:
        if start_date_str >= threshold:
            return cutoff
    return None


def ensure_model(cutoff):
    """cutoff指定モデルが存在しなければ学習して作成。成功でTrue。"""
    if cutoff is None:
        return True
    rise_path = os.path.join(BASE_DIR, f"rf_model_{cutoff}.pkl")
    if os.path.exists(rise_path):
        print(f"  [モデル] cutoff={cutoff} 既存")
        return True
    print(f"\n{'='*60}")
    print(f"  [モデル学習] cutoff={cutoff} 学習開始...")
    print(f"{'='*60}")
    proc = subprocess.run(
        [PYTHON, "rf_train_v3.py", "--cutoff", cutoff],
        cwd=BASE_DIR,
    )
    if proc.returncode != 0:
        print(f"  ERROR: モデル学習失敗 (cutoff={cutoff})")
        return False
    return os.path.exists(rise_path)


def apply_filter(df, cfg):
    mask = (df["ネット(%)"] >= cfg["net_min"]) & (df["下落確率(%)"] < cfg["drop_max"])
    if cfg["conflict_net"] is not None:
        conflict = (df["ネット(%)"] >= cfg["conflict_net"]) & (df["下落確率(%)"] >= cfg["conflict_drop"])
        mask = mask & ~conflict
    return df[mask]


def run_period(start, end, cutoff=None, skip_run=False):
    out_path = os.path.join(BASE_DIR, "simulations", "backtests", f"backtest_{start}_{end}.csv")
    if os.path.exists(out_path):
        print(f"  [{start}→{end}] キャッシュ済み")
        return out_path
    if skip_run:
        print(f"  [{start}→{end}] CSVなし、スキップ")
        return None
    print(f"\n  [{start}→{end}] バックテスト実行中...")
    cmd = [PYTHON, "backtest.py", "--start", start, "--end", end]
    if cutoff:
        cmd += ["--model-cutoff", cutoff]
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    if proc.returncode != 0:
        print(f"  ERROR: {start}→{end} 失敗（終了コード {proc.returncode}）")
        return None
    return out_path if os.path.exists(out_path) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="バックテスト実行をスキップ")
    args = parser.parse_args()

    print("=" * 70)
    print(f"マルチバックテスト + フィルタ比較（{len(PERIODS)}期間）")
    print("=" * 70)

    # ウォークフォワード対象期間の既存CSVを削除（モデルが変わるため再実行）
    if not args.skip_run:
        for start, end, label in PERIODS:
            if get_model_cutoff(start) is not None:
                csv_path = os.path.join(BASE_DIR, "simulations", "backtests", f"backtest_{start}_{end}.csv")
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                    print(f"  [{start}] 旧CSV削除（ウォークフォワード再実行）")

    # ── 1. モデルの事前学習（ウォークフォワード） ──
    needed_cutoffs = set()
    for start, end, label in PERIODS:
        c = get_model_cutoff(start)
        if c:
            needed_cutoffs.add(c)
    for cutoff in sorted(needed_cutoffs):
        ensure_model(cutoff)

    # ── 2. 各期間のバックテスト実行 ──
    csv_map = {}
    for start, end, label in PERIODS:
        cutoff = get_model_cutoff(start)
        path = run_period(start, end, cutoff=cutoff, skip_run=args.skip_run)
        if path:
            csv_map[(start, end)] = (label, path)

    if not csv_map:
        print("ERROR: 有効なCSVがありません")
        return

    # ── 2. フィルタ比較 ──
    period_rows = []
    for (start, end), (label, path) in csv_map.items():
        df = pd.read_csv(path)
        nikkei = df["日経リターン(%)"].iloc[0] if "日経リターン(%)" in df.columns else None

        row = {"label": label, "start": start, "end": end, "nikkei": nikkei, "total_n": len(df)}
        for cfg_name, cfg in CONFIGS.items():
            filtered = apply_filter(df, cfg)
            n = len(filtered)
            if n > 0:
                ret = filtered["実績リターン(%)"]
                avg  = ret.mean()
                beat = (ret > nikkei).mean() * 100 if nikkei is not None else None
            else:
                avg = beat = None
            row[f"{cfg_name}_n"]    = n
            row[f"{cfg_name}_avg"]  = avg
            row[f"{cfg_name}_beat"] = beat
        period_rows.append(row)

    # ── 3. 期間別テーブル ──
    cfg_keys = list(CONFIGS.keys())
    header = f"{'期間':<11}  {'日経':>7}  {'全'}"
    for c in cfg_keys:
        header += f"  {c.strip():>12}(n/avg/beat)"
    print("\n" + header)
    print("-" * 110)

    for r in period_rows:
        nk_s = f"{r['nikkei']:+.1f}%" if r["nikkei"] is not None else "  N/A "
        line = f"{r['label']:<11}  {nk_s:>7}  {r['total_n']:>3}"
        for c in cfg_keys:
            n    = r[f"{c}_n"]
            avg  = r[f"{c}_avg"]
            beat = r[f"{c}_beat"]
            if avg is not None:
                beat_s = f"{beat:.0f}%" if beat is not None else " -"
                line += f"  {n:>2}銘柄 {avg:>+6.1f}% {beat_s:>4}"
            else:
                line += f"  {'候補なし':>18}"
        print(line)

    # ── 4. 集計 ──
    df_res = pd.DataFrame(period_rows)
    nk_avg = df_res["nikkei"].dropna().mean()

    print("\n" + "=" * 70)
    print(f"集計（{len(period_rows)}期間）  日経225平均: {nk_avg:+.1f}%")
    print("=" * 70)
    print(f"{'フィルタ':<16}  {'avg':>7}  {'alpha':>7}  {'日経超え率':>10}  {'平均候補数':>10}  {'有効期間':>8}")
    print("-" * 70)

    summary = {}
    for c in cfg_keys:
        avgs  = df_res[f"{c}_avg"].dropna()
        beats = df_res[f"{c}_beat"].dropna()
        ns    = df_res[f"{c}_n"]
        valid = (df_res[f"{c}_n"] > 0).sum()
        if len(avgs) == 0:
            continue
        avg_v   = avgs.mean()
        alpha   = avg_v - nk_avg
        beat_v  = beats.mean() if len(beats) > 0 else float("nan")
        n_avg   = ns.mean()
        summary[c] = avg_v
        print(f"{c.strip():<16}  {avg_v:>+7.1f}%  {alpha:>+7.1f}%  {beat_v:>10.1f}%  {n_avg:>10.1f}  {valid:>5}/{len(period_rows)}")

    # ── 5. 推奨 ──
    if summary:
        best = max(summary, key=lambda k: summary[k])
        print(f"\n推奨フィルタ: {best.strip()}  (avg {summary[best]:+.1f}%)")

    # ── 6. パラメータ感度（net_min と drop_max の単独効果） ──
    print("\n" + "=" * 70)
    print("パラメータ感度サマリー")
    print("=" * 70)
    base = summary.get('現行(10/8) ', float('nan'))
    print(f"  現行(net10/drop8) ベース:  {base:+.1f}%")
    print(f"  旧(net8/drop12)との差:     {summary.get('旧(8/12)  ', float('nan')) - base:+.1f}%")
    print(f"  緩(net8/drop8)との差:      {summary.get('緩(8/8)   ', float('nan')) - base:+.1f}%")
    print(f"  厳(net12/drop6)との差:     {summary.get('厳(12/6)  ', float('nan')) - base:+.1f}%")


if __name__ == "__main__":
    main()

"""
10期間マルチバックテスト + フィルタ比較分析

実行: python3 multi_backtest.py
  --skip-run   : バックテスト実行をスキップ（既存CSVのみ分析）
  --periods N  : テスト期間数（デフォルト: 10）
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

from config import BASE_DIR

PYTHON = sys.executable

PERIODS = [
    ("2024-04-01", "2024-07-01", "2024春"),
    ("2024-07-01", "2024-10-01", "2024夏(下落)"),
    ("2024-09-01", "2024-12-01", "2024秋"),
    ("2024-10-15", "2025-01-15", "2024秋冬"),
    ("2025-01-01", "2025-04-01", "2025冬春"),
    ("2025-02-14", "2025-05-14", "2025春前"),
    ("2025-03-15", "2025-06-15", "2025春"),
    ("2025-04-15", "2025-07-15", "2025初夏"),
    ("2025-05-14", "2025-08-14", "2025夏前"),
    ("2025-07-01", "2025-10-01", "2025夏"),
]

CONFIGS = {
    "A現行      ": dict(net_min=8.0, drop_max=10.0, conflict_net=None,  conflict_drop=None),
    "B提案      ": dict(net_min=6.0, drop_max=12.0, conflict_net=10.0,  conflict_drop=5.0),
    "Cnet緩和   ": dict(net_min=6.0, drop_max=10.0, conflict_net=None,  conflict_drop=None),
    "Dconflict除": dict(net_min=8.0, drop_max=10.0, conflict_net=10.0,  conflict_drop=5.0),
}


def apply_filter(df, cfg):
    mask = (df["ネット(%)"] >= cfg["net_min"]) & (df["下落確率(%)"] < cfg["drop_max"])
    if cfg["conflict_net"] is not None:
        conflict = (df["ネット(%)"] >= cfg["conflict_net"]) & (df["下落確率(%)"] >= cfg["conflict_drop"])
        mask = mask & ~conflict
    return df[mask]


def run_period(start, end, skip_run=False):
    out_path = os.path.join(BASE_DIR, f"backtest_{start}_{end}.csv")
    if os.path.exists(out_path):
        print(f"  [{start}→{end}] キャッシュ済み")
        return out_path
    if skip_run:
        print(f"  [{start}→{end}] CSVなし、スキップ")
        return None
    print(f"\n  [{start}→{end}] バックテスト実行中...")
    proc = subprocess.run(
        [PYTHON, "backtest.py", "--start", start, "--end", end],
        cwd=BASE_DIR,
    )
    if proc.returncode != 0:
        print(f"  ERROR: {start}→{end} 失敗（終了コード {proc.returncode}）")
        return None
    return out_path if os.path.exists(out_path) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true", help="バックテスト実行をスキップ")
    args = parser.parse_args()

    print("=" * 70)
    print("マルチバックテスト + フィルタ比較（10期間）")
    print("=" * 70)

    # ── 1. 各期間のバックテスト実行 ──
    csv_map = {}
    for start, end, label in PERIODS:
        path = run_period(start, end, skip_run=args.skip_run)
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
    print(f"  net_min 8→6 の効果:  A({summary.get('A現行      ', float('nan')):+.1f}%) → C({summary.get('Cnet緩和   ', float('nan')):+.1f}%)")
    print(f"  conflict除外の効果:  A({summary.get('A現行      ', float('nan')):+.1f}%) → D({summary.get('Dconflict除', float('nan')):+.1f}%)")
    print(f"  両方合わせた効果:    A({summary.get('A現行      ', float('nan')):+.1f}%) → B({summary.get('B提案      ', float('nan')):+.1f}%)")


if __name__ == "__main__":
    main()

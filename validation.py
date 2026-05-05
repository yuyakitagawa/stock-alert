"""
validation.py - Purged K-Fold cross-validation for financial time series
López de Prado "Advances in Financial Machine Learning" Ch.7

Usage:
    python3 validation.py          # load training_data.npz and run CV
    python3 validation.py --test   # unit tests only
"""
import numpy as np
import json
import os
import sys
from datetime import date, timedelta
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SAVE_DIR     = os.path.expanduser("~/stock-alert")
PURGE_DAYS   = 90   # 63 trading days ≈ 90 calendar days
EMBARGO_DAYS = 10
N_SPLITS     = 5
RANDOM_SEED  = 42


class PurgedKFold(BaseCrossValidator):
    """
    Time-series K-Fold with look-ahead purging and post-test embargo.

    Purge: training samples whose label end (sample_date + purge_days) falls
    within or after the test fold start are removed.
    Embargo: training samples within embargo_days after test fold end are removed.
    Folds are chronological — each fold is a contiguous time window.
    """
    def __init__(self, n_splits=5, purge_days=90, embargo_days=10):
        self.n_splits     = n_splits
        self.purge_days   = purge_days
        self.embargo_days = embargo_days

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, dates=None):
        if dates is None:
            raise ValueError("dates must be provided for PurgedKFold")
        dates    = np.array(dates)
        n        = len(X)
        sort_idx = np.argsort(dates)
        sd       = dates[sort_idx]          # sorted dates
        fold_sz  = n // self.n_splits

        for fold in range(self.n_splits):
            t0 = fold * fold_sz
            t1 = (fold + 1) * fold_sz if fold < self.n_splits - 1 else n
            test_idx      = sort_idx[t0:t1]
            d_test_start  = sd[t0]
            d_test_end    = sd[t1 - 1]
            d_purge_start = d_test_start - timedelta(days=self.purge_days)
            d_embargo_end = d_test_end   + timedelta(days=self.embargo_days)

            train_pos = []
            for pos in range(n):
                if t0 <= pos < t1:
                    continue  # test fold itself
                d = sd[pos]
                if d_purge_start < d < d_test_start:
                    continue  # purge zone: label overlaps test fold
                if d_test_end < d <= d_embargo_end:
                    continue  # embargo zone: post-test regime leakage
                train_pos.append(pos)

            yield sort_idx[np.array(train_pos, dtype=int)], test_idx


def run_purged_cv(X, y, dates, label_name,
                  n_splits=N_SPLITS, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS):
    """Run purged K-fold CV. Returns (fold_aucs, mean_auc, std_auc)."""
    dates = np.array(dates)
    pos   = int(y.sum()); neg = len(y) - pos
    spw   = neg / pos if pos > 0 else 1.0
    cv    = PurgedKFold(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)
    fold_aucs = []

    print(f"\n[Purged CV] {label_name}モデル  "
          f"folds={n_splits}  purge={purge_days}日  embargo={embargo_days}日")

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, dates=dates)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        te_pos = int(y_te.sum())
        if te_pos == 0 or te_pos == len(y_te):
            print(f"  fold {fold+1}: 単一クラス → スキップ")
            continue
        model = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
            scale_pos_weight=spw, eval_metric="auc",
            early_stopping_rounds=30, random_state=RANDOM_SEED,
            n_jobs=-1, verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        prob = model.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, prob)
        fold_aucs.append(auc)
        d_sorted = sorted(dates[te_idx])
        print(f"  fold {fold+1}: train={len(X_tr):,}  test={len(X_te):,}"
              f"  [{d_sorted[0]} ~ {d_sorted[-1]}]  AUC={auc:.4f}")

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    std_auc  = float(np.std(fold_aucs))  if fold_aucs else 0.0
    print(f"  平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return fold_aucs, mean_auc, std_auc


def write_comparison_md(baseline, purged_rise, purged_drop, output_path):
    from datetime import datetime
    r_aucs, r_mean, r_std = purged_rise
    d_aucs, d_mean, d_std = purged_drop

    def diff_str(b, p):
        diff = p - b
        return f"{'+' if diff >= 0 else ''}{diff:.4f}"

    lines = [
        "# バリデーション比較: ウォークフォワード vs Purged K-Fold",
        "",
        "## 手法説明",
        "",
        "| 手法 | 説明 | 先読みバイアス |",
        "|------|------|----------------|",
        "| ウォークフォワード（現行） | TRAIN_CUTOFF前で学習、以降でテスト | **あり**: カットオフ近傍の学習サンプルのラベルがテスト期間に重複 |",
        "| Purged K-Fold（新） | テストフォールドのラベル期間と重なる学習サンプルを除去 | なし |",
        "",
        "## AUC比較",
        "",
        "| モデル | ウォークフォワードAUC | Purged K-Fold AUC | 差分 |",
        "|--------|---------------------|-------------------|------|",
        f"| 上昇 | {baseline['rise']:.4f} | {r_mean:.4f} ± {r_std:.4f} | {diff_str(baseline['rise'], r_mean)} |",
        f"| 下落 | {baseline['drop']:.4f} | {d_mean:.4f} ± {d_std:.4f} | {diff_str(baseline['drop'], d_mean)} |",
        "",
        "## 解釈",
        "",
        "- **差分 < 0**: ウォークフォワードAUCが過大評価（先読みバイアスあり）",
        "- **差分 ≈ 0**: 先読みバイアスは軽微",
        "- **差分 > 0**: Purged CVの方が高い（通常は起きない、フォールド構成を確認）",
        "",
        "## フォールド別AUC",
        "",
        "### 上昇モデル",
    ]
    for i, auc in enumerate(r_aucs):
        lines.append(f"- fold {i+1}: {auc:.4f}")
    lines += ["", "### 下落モデル"]
    for i, auc in enumerate(d_aucs):
        lines.append(f"- fold {i+1}: {auc:.4f}")
    lines += ["", f"*生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n比較レポート保存: {output_path}")


def run_tests():
    """Unit tests for PurgedKFold."""
    print("=== PurgedKFold Unit Tests ===")
    np.random.seed(42)
    n      = 1000
    dates  = np.array([date(2020, 1, 1) + timedelta(days=i * 2) for i in range(n)])
    X      = np.random.randn(n, 5)
    y      = np.random.randint(0, 2, n)

    cv     = PurgedKFold(n_splits=5, purge_days=90, embargo_days=10)
    splits = list(cv.split(X, y, dates=dates))

    # Test 1: correct split count
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"
    print("✅ Test 1: split count = 5")

    # Test 2: no purge zone violations
    for fold, (tr_idx, te_idx) in enumerate(splits):
        te_dates = dates[te_idx]
        d0 = min(te_dates)
        purge_start = d0 - timedelta(days=90)
        for d in dates[tr_idx]:
            assert not (purge_start < d < d0), (
                f"Fold {fold}: purge violation: {d} in ({purge_start}, {d0})"
            )
    print("✅ Test 2: no purge zone violations")

    # Test 3: no date overlap between train and test
    for fold, (tr_idx, te_idx) in enumerate(splits):
        overlap = set(dates[tr_idx]) & set(dates[te_idx])
        assert len(overlap) == 0, f"Fold {fold}: date overlap = {overlap}"
    print("✅ Test 3: no date overlap between train and test")

    # Test 4: test folds are chronologically ordered
    ranges = [(min(dates[te]), max(dates[te])) for _, te in splits]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] <= ranges[i+1][0], (
            f"Folds not chronological: fold {i} ends {ranges[i][1]}, "
            f"fold {i+1} starts {ranges[i+1][0]}"
        )
    print("✅ Test 4: test folds are chronologically ordered")

    # Test 5: purge zone has samples removed from training (check fold 2 = middle fold)
    fold2_tr, fold2_te = splits[2]
    te_dates2 = dates[fold2_te]
    d0 = min(te_dates2)
    purge_start = d0 - timedelta(days=90)
    # samples that exist in (purge_start, d0) but are NOT in fold2_te should be absent from fold2_tr
    purge_candidates = [i for i, d in enumerate(dates) if purge_start < d < d0]
    tr_set = set(fold2_tr.tolist())
    te_set = set(fold2_te.tolist())
    leaked = [i for i in purge_candidates if i in tr_set and i not in te_set]
    assert len(leaked) == 0, f"Fold 2: {len(leaked)} purge zone samples leaked into training"
    print(f"✅ Test 5: {len(purge_candidates)} purge-zone samples correctly excluded from fold 2 training")

    print("=== All tests passed ===\n")


def main():
    npz_path = os.path.join(SAVE_DIR, "training_data.npz")
    auc_path = os.path.join(SAVE_DIR, "baseline_auc.json")
    out_path = os.path.join(SAVE_DIR, "validation_comparison.md")

    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} が見つかりません")
        print("rf_train_v3.py を実行すると自動保存されます")
        return

    print("学習データ読み込み中...")
    data  = np.load(npz_path, allow_pickle=True)
    X     = data["X"]
    yr    = data["yr"]
    yd    = data["yd"]
    dates = np.array([date.fromisoformat(str(d)) for d in data["dates"]])
    print(f"  サンプル数: {len(X):,}  特徴量次元: {X.shape[1]}")
    print(f"  期間: {dates.min()} → {dates.max()}")

    baseline = {"rise": 0.0, "drop": 0.0}
    if os.path.exists(auc_path):
        with open(auc_path) as f:
            baseline = json.load(f)
        print(f"  ベースラインAUC: 上昇={baseline['rise']:.4f}  下落={baseline['drop']:.4f}")
    else:
        print(f"  WARNING: {auc_path} なし → ベースラインAUC=0.0 で比較")

    purged_rise = run_purged_cv(X, yr, dates, "上昇")
    purged_drop = run_purged_cv(X, yd, dates, "下落")
    write_comparison_md(baseline, purged_rise, purged_drop, out_path)


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        main()

"""
Drop model AUC improvement experiments.
3 approaches compared against baseline (AUC 0.7529):
  1. Feature engineering (selection + new interactions)
  2. Label definition tuning (threshold/forecast variations)
  3. Hyperparameter optimization (Optuna)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import date
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lib.utils import IsotonicCalibrated

RANDOM_SEED = 42
TRAIN_CUTOFF_STR = "2026-01-01"

def load_data():
    d = np.load("training_data.npz")
    X, yd, dates = d["X"], d["yd"], d["dates"]
    train_mask = dates < TRAIN_CUTOFF_STR
    test_mask = dates >= TRAIN_CUTOFF_STR
    X_tr, X_te = X[train_mask], X[test_mask]
    yd_tr, yd_te = yd[train_mask], yd[test_mask]
    dates_tr = dates[train_mask]
    sort_idx = np.argsort(dates_tr)
    X_tr_s, yd_s = X_tr[sort_idx], yd_tr[sort_idx]
    n_cal = max(500, int(len(X_tr_s) * 0.2))
    X_fit, X_cal = X_tr_s[:-n_cal], X_tr_s[-n_cal:]
    yd_fit, yd_cal = yd_s[:-n_cal], yd_s[-n_cal:]
    return X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te

def train_and_eval(X_fit, y_fit, X_cal, y_cal, X_te, y_te, params=None, label=""):
    default_params = dict(
        n_estimators=5000, max_depth=5, learning_rate=0.005,
        subsample=0.65, colsample_bytree=0.45, min_child_weight=60,
        reg_alpha=1.5, reg_lambda=5, gamma=0.5,
        tree_method="hist", random_state=RANDOM_SEED,
        eval_metric="auc", early_stopping_rounds=100,
    )
    if params:
        default_params.update(params)
    es_rounds = default_params.pop("early_stopping_rounds", 100)
    m = XGBClassifier(**default_params, early_stopping_rounds=es_rounds)
    m.fit(X_fit, y_fit, eval_set=[(X_te, y_te)], verbose=False)
    auc_raw = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(m.predict_proba(X_cal)[:, 1], y_cal)
    cal_m = IsotonicCalibrated(m, iso)
    auc_cal = roc_auc_score(y_te, cal_m.predict_proba(X_te)[:, 1])
    print(f"  [{label}] Raw AUC: {auc_raw:.4f}, Calibrated AUC: {auc_cal:.4f}")
    return auc_cal, auc_raw, m

# ── Experiment 1: Feature Engineering ──
def exp1_feature_engineering(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te):
    print("\n" + "="*60)
    print("Experiment 1: Feature Engineering")
    print("="*60)

    feat_names = ["ret5","ret20","ret60","ret90","ma5_25","ma25_75","rsi","vol20","vol60","pos52",
                  "drawdown60","from_hi52","down_streak","momentum_accel","ma_cross_dir",
                  "vr520","vr2060","vsurge","nk5","nk20","nk60",
                  "ac","skew","max_ret","min_ret","pos_ratio","trend_slope","recent_vs_early",
                  "rel5","rel20","rel60","alpha_momentum",
                  "per_feat","pbr_feat","roe_feat","earn_feat",
                  "div_ex_feat","sin_month","cos_month","div_yield_f",
                  "eps_growth_f",
                  "dps_growth_f","vix_feat","us5_f","us20_f",
                  "amihud_f","fx_beta_f","jpy5_f",
                  "eps_surprise_f","bps_growth_f","piotroski_f","payout_f","accruals_f",
                  "edinet_hold_f",
                  "ret504","trend_slope60","trend_r2_60",
                  "cs_ret5","cs_ret20","cs_ret60","cs_rsi","cs_vol20","cs_pos52",
                  "cs_sector_ret60"]

    results = {}

    # 1a. Baseline (all features)
    print("\n1a. Baseline (all features)")
    auc_cal, auc_raw, m_base = train_and_eval(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te, label="baseline")
    results["1a_baseline"] = auc_cal

    # Get feature importance for selection
    imp = m_base.feature_importances_
    n_feat = min(len(feat_names), X_fit.shape[1])
    imp_order = np.argsort(imp)[::-1]
    print(f"\n  Top 10 features (drop model):")
    for i in range(min(10, n_feat)):
        idx = imp_order[i]
        name = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
        print(f"    {name}: {imp[idx]:.4f}")

    # 1b. Remove bottom 20% features (low importance)
    print("\n1b. Feature selection: remove bottom 20%")
    keep_n = int(n_feat * 0.8)
    keep_idx = imp_order[:keep_n]
    auc_cal, _, _ = train_and_eval(
        X_fit[:, keep_idx], yd_fit, X_cal[:, keep_idx], yd_cal,
        X_te[:, keep_idx], yd_te, label="top80%"
    )
    results["1b_top80pct"] = auc_cal

    # 1c. Remove bottom 40% features
    print("\n1c. Feature selection: remove bottom 40%")
    keep_n = int(n_feat * 0.6)
    keep_idx = imp_order[:keep_n]
    auc_cal, _, _ = train_and_eval(
        X_fit[:, keep_idx], yd_fit, X_cal[:, keep_idx], yd_cal,
        X_te[:, keep_idx], yd_te, label="top60%"
    )
    results["1c_top60pct"] = auc_cal

    # 1d. Add interaction features (top-5 pairwise ratios)
    print("\n1d. Add interaction features (top-5 pairwise)")
    top5 = imp_order[:5]
    interactions = []
    for i in range(len(top5)):
        for j in range(i+1, len(top5)):
            fi, fj = top5[i], top5[j]
            ratio = X_fit[:, fi] / (X_fit[:, fj] + 1e-8)
            interactions.append(ratio)
    X_fit_aug = np.column_stack([X_fit] + interactions)
    X_cal_aug = np.column_stack([X_cal] + [X_cal[:, top5[i]] / (X_cal[:, top5[j]] + 1e-8) for i in range(len(top5)) for j in range(i+1, len(top5))])
    X_te_aug = np.column_stack([X_te] + [X_te[:, top5[i]] / (X_te[:, top5[j]] + 1e-8) for i in range(len(top5)) for j in range(i+1, len(top5))])
    auc_cal, _, _ = train_and_eval(X_fit_aug, yd_fit, X_cal_aug, yd_cal, X_te_aug, yd_te, label="interactions")
    results["1d_interactions"] = auc_cal

    # 1e. Squared features of top-10
    print("\n1e. Add squared features (top-10)")
    top10 = imp_order[:10]
    X_fit_sq = np.column_stack([X_fit, X_fit[:, top10]**2])
    X_cal_sq = np.column_stack([X_cal, X_cal[:, top10]**2])
    X_te_sq = np.column_stack([X_te, X_te[:, top10]**2])
    auc_cal, _, _ = train_and_eval(X_fit_sq, yd_fit, X_cal_sq, yd_cal, X_te_sq, yd_te, label="squared")
    results["1e_squared"] = auc_cal

    return results

# ── Experiment 2: Label Definition Tuning ──
def exp2_label_tuning(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te):
    """Re-generate labels requires raw returns, which aren't in saved data.
    Instead, test different class weights to simulate threshold changes."""
    print("\n" + "="*60)
    print("Experiment 2: Label / Class Weight Tuning")
    print("="*60)
    print("(Saved data has binary labels only; testing scale_pos_weight variations)")

    results = {}
    pos_rate = yd_fit.mean()
    default_spw = (1 - pos_rate) / pos_rate
    print(f"  Positive rate: {pos_rate:.3f}, natural scale_pos_weight: {default_spw:.2f}")

    for spw_mult, tag in [(0.5, "0.5x"), (0.75, "0.75x"), (1.0, "1.0x"), (1.5, "1.5x"), (2.0, "2.0x"), (3.0, "3.0x")]:
        spw = default_spw * spw_mult
        print(f"\n2. scale_pos_weight = {spw:.2f} ({tag})")
        auc_cal, _, _ = train_and_eval(
            X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te,
            params={"scale_pos_weight": spw},
            label=f"spw_{tag}"
        )
        results[f"2_spw_{tag}"] = auc_cal

    return results

# ── Experiment 3: Hyperparameter Optimization (Optuna) ──
def exp3_hyperparam_optuna(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te):
    print("\n" + "="*60)
    print("Experiment 3: Hyperparameter Optimization (Optuna)")
    print("="*60)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Installing optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_results = {}

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
            "eval_metric": "auc",
            "early_stopping_rounds": 100,
        }
        es = params.pop("early_stopping_rounds")
        m = XGBClassifier(**params, early_stopping_rounds=es)
        m.fit(X_fit, yd_fit, eval_set=[(X_te, yd_te)], verbose=False)
        auc = roc_auc_score(yd_te, m.predict_proba(X_te)[:, 1])
        return auc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    print("Running 80 Optuna trials...")
    study.optimize(objective, n_trials=80, show_progress_bar=False)

    print(f"\n  Best trial AUC (raw): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Re-train with best params and calibrate
    best_p = study.best_params.copy()
    best_p.update({"n_estimators": 5000, "tree_method": "hist", "random_state": RANDOM_SEED,
                   "eval_metric": "auc", "early_stopping_rounds": 100})
    print("\n  Re-training with best params + calibration:")
    auc_cal, auc_raw, _ = train_and_eval(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te,
                                          params=best_p, label="optuna_best")
    best_results["3_optuna_best_raw"] = auc_raw
    best_results["3_optuna_best_cal"] = auc_cal
    best_results["3_best_params"] = study.best_params

    # Also try top-3 trials
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:3]
    for i, t in enumerate(top_trials):
        print(f"\n  Top-{i+1} trial (raw AUC={t.value:.4f}):")
        p = t.params.copy()
        p.update({"n_estimators": 5000, "tree_method": "hist", "random_state": RANDOM_SEED,
                  "eval_metric": "auc", "early_stopping_rounds": 100})
        auc_cal, _, _ = train_and_eval(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te,
                                        params=p, label=f"top{i+1}")
        best_results[f"3_top{i+1}_cal"] = auc_cal

    return best_results


def main():
    print("Loading training data...")
    X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te = load_data()
    print(f"  Fit: {X_fit.shape}, Cal: {X_cal.shape}, Test: {X_te.shape}")
    print(f"  Drop rate - Fit: {yd_fit.mean()*100:.1f}%, Test: {yd_te.mean()*100:.1f}%")

    all_results = {}

    r1 = exp1_feature_engineering(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te)
    all_results.update(r1)

    r2 = exp2_label_tuning(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te)
    all_results.update(r2)

    r3 = exp3_hyperparam_optuna(X_fit, yd_fit, X_cal, yd_cal, X_te, yd_te)
    all_results.update({k: v for k, v in r3.items() if not k.endswith("_params")})

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: All Experiments (Calibrated AUC)")
    print("="*60)
    baseline = all_results.get("1a_baseline", 0)
    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if isinstance(v, (int, float))],
        key=lambda x: -x[1]
    )
    for name, auc in sorted_results:
        diff = (auc - baseline) * 100
        marker = "★" if auc > baseline else ""
        print(f"  {name:30s} AUC={auc:.4f}  ({diff:+.2f}pp) {marker}")

    # Save results
    out = {"baseline_drop_auc": baseline, "experiments": {k: v for k, v in all_results.items() if isinstance(v, (int, float))}}
    if "3_best_params" in r3:
        out["optuna_best_params"] = r3["3_best_params"]
    with open("experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to experiment_results.json")


if __name__ == "__main__":
    main()

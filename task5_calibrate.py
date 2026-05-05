"""
Task5: 確率キャリブレーション（Isotonic Regression）
training_data.npz と既存モデルを使い、再学習不要でキャリブレーションを適用する。
"""
import numpy as np, joblib, os, json
from datetime import date
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

SAVE_DIR = os.path.expanduser("~/stock-alert")
TRAIN_CUTOFF = date(2025, 1, 1)
CAL_RATIO = 0.2  # 学習データの最新20%をキャリブレーションに使用

def main():
    npz_path = os.path.join(SAVE_DIR, "training_data.npz")
    rise_path = os.path.join(SAVE_DIR, "rf_model.pkl")
    drop_path = os.path.join(SAVE_DIR, "rf_drop_model.pkl")

    if not os.path.exists(npz_path):
        print("ERROR: training_data.npz が見つかりません。先に rf_train_v3.py を実行してください")
        return
    if not os.path.exists(rise_path):
        print("ERROR: rf_model.pkl が見つかりません")
        return

    print("=== Task5: 確率キャリブレーション ===")
    data = np.load(npz_path, allow_pickle=True)
    X, yr, yd, dates_raw = data["X"], data["yr"], data["yd"], data["dates"]
    dates = np.array([date.fromisoformat(str(d)) for d in dates_raw])

    # 学習/テスト分割
    tr_mask = dates < TRAIN_CUTOFF
    te_mask = ~tr_mask
    X_tr, yr_tr, yd_tr = X[tr_mask], yr[tr_mask], yd[tr_mask]
    X_te, yr_te, yd_te = X[te_mask], yr[te_mask], yd[te_mask]
    print(f"学習: {len(X_tr):,}  テスト: {len(X_te):,}")

    # キャリブレーション分割: 学習データを日付順で最新20%を分離
    tr_dates = dates[tr_mask]
    sort_idx = np.argsort(tr_dates)
    X_tr_s = X_tr[sort_idx]; yr_s = yr_tr[sort_idx]; yd_s = yd_tr[sort_idx]
    n_cal = max(500, int(len(X_tr_s) * CAL_RATIO))
    X_tr_fit, X_cal = X_tr_s[:-n_cal], X_tr_s[-n_cal:]
    yr_fit, yr_cal = yr_s[:-n_cal], yr_s[-n_cal:]
    yd_fit, yd_cal = yd_s[:-n_cal], yd_s[-n_cal:]
    print(f"キャリブレーション用: {len(X_cal):,}サンプル（学習最新{CAL_RATIO*100:.0f}%）")

    for label, model_path, X_fit, y_cal_arr, y_te in [
        ("上昇", rise_path, X_tr_fit, yr_cal, yr_te),
        ("下落", drop_path, X_tr_fit, yd_cal, yd_te),
    ]:
        print(f"\n--- {label}モデル ---")
        m = joblib.load(model_path)

        auc_raw = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
        print(f"  AUC（生）        : {auc_raw:.4f}")

        cal_m = CalibratedClassifierCV(m, cv="prefit", method="isotonic")
        y_cal_labels = yr_cal if label == "上昇" else yd_cal
        cal_m.fit(X_cal, y_cal_labels)

        auc_cal = roc_auc_score(y_te, cal_m.predict_proba(X_te)[:, 1])
        print(f"  AUC（キャリブレーション後）: {auc_cal:.4f}  差分: {auc_cal-auc_raw:+.4f}")

        if auc_cal >= auc_raw - 0.003:
            joblib.dump(cal_m, model_path)
            print(f"  ✅ キャリブレーション済みモデルを保存: {model_path}")
        else:
            print(f"  ⚠️ AUCが0.003以上低下したため元モデルを維持")

    # baseline_auc.json を更新
    rise_m = joblib.load(rise_path)
    drop_m = joblib.load(drop_path)
    rise_auc = roc_auc_score(yr_te, rise_m.predict_proba(X_te)[:, 1])
    drop_auc = roc_auc_score(yd_te, drop_m.predict_proba(X_te)[:, 1])
    with open(os.path.join(SAVE_DIR, "baseline_auc.json"), "w") as f:
        json.dump({"rise": float(rise_auc), "drop": float(drop_auc)}, f)
    print(f"\n最終AUC → 上昇: {rise_auc:.4f} / 下落: {drop_auc:.4f}")
    print("完了 ✅")

if __name__ == "__main__":
    main()

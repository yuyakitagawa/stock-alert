# 改善提案 (2026-05-15)

## 優先度 A: バグ修正（即着手推奨）

### A-1. 特徴量検証の不足 (`rf_train_v3.py`)

**問題**: `feat[:10]` で最初の10次元のみNaN/Infチェック。系列特徴量(dim 21-27)は未検証のまま学習データに混入する恐れがある。

```python
# 現状 (rf_train_v3.py ~137行)
if any(np.isnan(feat[:10])) or any(np.isinf(feat[:10])):
    continue

# 修正後
if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
    continue
```

**影響**: 学習データの汚染防止。AUC改善の可能性あり。修正コスト: 1行。

---

### A-2. `compute_feat()` の重複コード (`rf_train_v3.py`)

**問題**: `rf_train_v3.py::compute_feat()` が `lib/utils.py::extract_features()` とほぼ同一のロジックを持つ。将来的に特徴量定義がずれるリスクがある。

**修正**: `rf_train_v3.py` の `compute_feat()` を削除し、`from lib.utils import extract_features` に統一。

**影響**: 特徴量の一貫性保証。メンテコスト削減。

---

## 優先度 B: パフォーマンス改善

### B-1. 日経データの重複取得排除

**問題**: `screener.py`、`rank_stocks.py`、`alert_email.py` がそれぞれ独立して日経データを取得している。1回の実行で3回以上APIコールが発生。

**修正案**:
```python
# lib/utils.py に追加
import functools

@functools.lru_cache(maxsize=1)
def get_nikkei_cached(date_str: str):
    """当日の日経データを1回だけ取得してキャッシュ"""
    return _fetch_nikkei_raw()
```

`date_str=today` を渡すことでプロセス内で1回のみ取得。推定削減: 2回分のAPIコール/実行。

---

### B-2. 株価データ取得の並列化

**問題**: スクリーナーが4000銘柄を逐次処理(0.3s遅延)。実行時間が約20分超。

**修正案**:
```python
# screener.py
from concurrent.futures import ThreadPoolExecutor
import time

FETCH_WORKERS = 5    # レートリミット配慮
FETCH_DELAY   = 0.1  # ワーカーあたりの遅延(秒)

def _fetch_stock_safe(code):
    time.sleep(FETCH_DELAY)
    return _fetch_one(code)

with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
    results = list(ex.map(_fetch_stock_safe, codes))
```

推定効果: 約4倍高速化(20分→5分)。Yahoo Finance への負荷は変わらない(総リクエスト数同一)。

---

## 優先度 C: モデル品質向上

### C-1. 特徴量重要度の可視化 (`rf_train_v3.py`)

**問題**: どの特徴量が予測に効いているか不明。デバッグ・調整の根拠が薄い。

**修正**: 学習後に SHAP 値または XGBoost 標準の `feature_importances_` を `feature_importance.json` へ保存。

```python
# rf_train_v3.py 学習完了後に追加
import json
importance = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
with open("feature_importance.json", "w") as f:
    json.dump(importance, f, ensure_ascii=False, indent=2)
```

`FEATURE_NAMES` は `utils.py` から定数として公開する。

---

### C-2. 決定閾値の最適化

**問題**: `rise_prob > 0.5`(暗黙のデフォルト)で判定しているが、クラス不均衡(上昇ラベルが少ない)環境ではF1最大化閾値が0.5より低いことが多い。

**修正**: バックテスト時に閾値スイープを実施し、最適閾値を `baseline_auc.json` に追記。

```python
# rf_train_v3.py
from sklearn.metrics import f1_score
thresholds = np.arange(0.3, 0.7, 0.02)
best_t = max(thresholds, key=lambda t: f1_score(y_test, proba >= t))
# baseline_auc.json に optimal_threshold_rise / optimal_threshold_drop を保存
```

`rank_stocks.py` でこの閾値を読み込んで `judgment` 判定に使用。

---

## 優先度 D: 運用性改善

### D-1. データ品質ログの追加

**問題**: ボリュームデータ欠損時に `vr520=vr2060=vsurge=1.0` でサイレント補完。取得失敗が検知されない。

**修正**:
```python
# lib/utils.py
import logging
logger = logging.getLogger(__name__)

if vol is None:
    logger.warning("volume data missing for %s, using defaults", code)
    vr520 = vr2060 = vsurge = 1.0
```

ログレベルを `WARNING` にすることで通常運用では見えるが、CIでは集約可能。

---

### D-2. バックテストのスクリーナーフィルター除外オプション

**問題**: `backtest.py` がスクリーナーフィルターを後付けで適用するため、過去時点での「実際の通過状況」と乖離する可能性がある。

**修正**: `--no-screener` フラグを追加し、モデルスコアのみで評価できるようにする。

```bash
python3 backtest.py bear --no-screener   # スクリーナーなしの純モデル評価
python3 backtest.py bear                 # 従来通り
```

---

## 見送り項目（理由付き）

| 案 | 理由 |
|---|---|
| LightGBM/CatBoost への乗り換え | AUC改善の保証なし。現行XGBoostで十分。 |
| ファーマ・フレンチ因子モデル | データ取得コスト大。現在の相対強度で代替可。 |
| ニューラルネットワーク | 過学習リスク大、解釈性ゼロ。 |
| Isotonic calibration → Platt scaling | 現状で十分に機能している。 |

---

## 実施順序

1. **A-1** → 即修正(1行)
2. **A-2** → 次の再学習日(金曜)前に修正
3. **D-1** → A-2と同コミット
4. **B-1** → 単独コミット(テスト容易)
5. **C-1** → 次回学習時に同梱
6. **B-2** → C-1確認後に実施(副作用注意)
7. **C-2** → バックテストで有意差確認後に採用判断
8. **D-2** → 必要に応じて

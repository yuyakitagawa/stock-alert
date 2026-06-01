# PDCA Log

<!-- 自動記録。手動編集不要 -->
<!-- フォーマット: metrics → FM判定 → analyst提案 → engineer結果 -->

## 2026-05-25
- ERROR: backtest失敗（メトリクス取得不可）[GitHub Actions: rf_model.pkl なし]

## 2026-05-25
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}
- 初回実行: baseline設定
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}

## 2026-05-26
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}
- FM: skip | bear mode -3.48%はベースラインと同値。相対優位性（日経平均比）の実績値がないため、現在の改善判断は時期尚早。次の週次更新で63日間アルファ実績を確認してからconfig調整を検討すべき。
- skip

## 2026-05-26

## 2026-05-26
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}
- FM: improve | 目標未達 (avg=-3.48%<3.0% / win=20.0%<45.0% / big=10.0%<20.0%) → 強制 improve
- analyst: {"param_name": "CANDIDATE_DROP_PROB_MAX", "old_value": 8.0, "new_value": 6.0, "reason": "弱気相場では下落確率の高い銘柄が損失の主因となる。Iwanaga et al.(2024)のPTH成分知見より、bear mode時は下落リスク指標の閾値を厳格化することでドローダウン主因銘柄を排除できる。8.0→6.0(-25%)に絞ることで高リスク帯をさらに除外し、勝率・avg改善を狙う。"}
- metrics: {"avg_return": -3.48, "win_rate": 20.0, "big_win_rate": 10.0}
- signals: なし
- FM: improve | 目標未達 (avg=-3.48%<3.0% / win=20.0%<45.0% / big=10.0%<20.0%) → 強制 improve
- analyst: {"param_name": "CANDIDATE_DROP_PROB_MAX", "old_value": 6.0, "new_value": 4.5, "reason": "bear modeでは下落確率の高い銘柄が損失を拡大させる主因。特徴量重要度でvol60(0.090)・vol20(0.046)が上位を占め、高ボラ局面ではdrop_prob上昇銘柄の下落加速が顕著。閾値を6.0→4.5(-25%)に厳格化することで矛盾シグナル除外と合わせて高リスク帯を二重遮断し、win率改善とavgリターンの底上げを狙う。"}
- engineer: ❌ 改善なし: avg -3.48% (bsl -3.48%)、revert
- engineer: ❌ 改善なし: avg -3.48% (bsl -3.48%)、revert

## 2026-05-28
- ERROR: backtest失敗（メトリクス取得不可）

## 2026-05-28

## 2026-05-28
- metrics: {"avg_return": 1.29, "win_rate": 68.66, "big_win_rate": 0.0}  periods: {"rate_hike_2022": {"avg_return": 0.46, "win_rate": 33.3, "big_win_rate": 0.0, "nk_avg": 0.32, "nk_alpha": 0.13}, "bull_2023": {"avg_return": 2.78, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.26, "nk_alpha": -0.48}, "q1_2024": {"avg_return": 1.91, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.35, "nk_alpha": -1.44}, "bear_2024": {"avg_return": -2.39, "win_rate": 50.0, "big_win_rate": 0.0, "nk_avg": -1.25, "nk_alpha": -1.14}, "q2_2025": {"avg_return": 3.67, "win_rate": 100.0, "big_win_rate": 0.0, "nk_avg": 4.38, "nk_alpha": -0.71}}
- invest_stage: Phase 0
- signals: なし
- FM: improve | avg=1.29%<2.0% / big=0.0%<20.0% / 日経アルファ=-9999%≤0%（日経に負け） → 強制 improve
- analyst: parse error

## 2026-05-29
- metrics: {"avg_return": 1.29, "win_rate": 68.66, "big_win_rate": 0.0, "nk_avg": 2.01, "nk_alpha": -0.72}  periods: {"rate_hike_2022": {"avg_return": 0.47, "win_rate": 33.3, "big_win_rate": 0.0, "nk_avg": 0.32, "nk_alpha": 0.15}, "bull_2023": {"avg_return": 2.78, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.26, "nk_alpha": -0.48}, "q1_2024": {"avg_return": 1.91, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.35, "nk_alpha": -1.44}, "bear_2024": {"avg_return": -2.39, "win_rate": 50.0, "big_win_rate": 0.0, "nk_avg": -1.25, "nk_alpha": -1.14}, "q2_2025": {"avg_return": 3.67, "win_rate": 100.0, "big_win_rate": 0.0, "nk_avg": 4.38, "nk_alpha": -0.71}}
- invest_stage: Phase 0
- signals: なし
- FM: improve | avg=1.29%<2.0% / big=0.0%<20.0% / 日経アルファ=-0.72%≤0%（日経に負け） → 強制 improve
- analyst: {"file": "rf_train_v3.py", "changes": [{"param_name": "RISE_THRESHOLD（目的変数の閾値）", "old_value": 15.0, "new_value": 12.0, "reason": "現在のbig_win=0%は閾値が高すぎて正例が極端に少ないことを示唆。12%に下げることで正例数を増やしモデルの学習効率を向上。年率42%目標には四半期9%=63日で約12%リターンが整合的"}, {"param_name": "DROP_ALPHA_THRESHOLD（下落判定閾値）", "old_value": -10.0, "new_value": -8.0, "reason": "下落AUC=0.6731と高性能。閾値を緩和し正例を増やすことで下落予測の汎化性能を維持しつつリスク回避精度を向上"}, {"param_name": "n_estimators", "old_value": 800, "new_value": 1200, "reason": "early_stopping=50があるため過学習リスクは低い。木の数を増やし複雑なパターン学習を強化"}, {"param_name": "max_depth", "old_value": 5, "new_value": 6, "reason": "特徴量重要度でvol60/nk系が支配的。深さを1増やし特徴量間の交互作用（ボラ×モメンタム等）を捕捉"}, {"param_name": "learning_rate", "old_value": 0.04, "new_value": 0.025, "reason": "n_estimators増加に合わせ学習率を下げ、より細かい勾配更新で汎化性能向上"}, {"param_name": "min_child_weight", "old_value": 8, "new_value": 12, "reason": "ノイズの多い株価データに対し葉ノードの最小サンプル数を増やし過学習を抑制"}, {"param_name": "colsample_bytree", "old_value": 0.7, "new_value": 0.6, "reason": "特徴量のランダムサンプリングを強化し、vol60への過度な依存を軽減。多様な特徴量の活用を促進"}, {"param_name": "reg_alpha（新規追加）", "old_value": 0, "new_value": 0.1, "reason": "L1正則化を追加し重要度の低い特徴量の影響を抑制。スパース性向上でモデル解釈性も改善"}, {"param_name": "reg_lambda（新規追加）", "old_value": 1, "new_value": 2.0, "reason": "L2正則化を強化し過学習を防止。特に長期予測（63日）では汎化性能が重要"}, {"param_name": "gamma（新規追加）", "old_value": 0, "new_value": 0.1, "reason": "分割に必要な最小損失減少量を設定し、ノイズによる無意味な分割を抑制"}]}
- engineer: rf_train_v3.py: すべてのパラメータが見つからず変更失敗
- stagnation: 1サイクル

## 2026-05-29
- metrics: {"avg_return": 1.29, "win_rate": 68.66, "big_win_rate": 0.0, "nk_avg": 2.01, "nk_alpha": -0.72}  periods: {"rate_hike_2022": {"avg_return": 0.47, "win_rate": 33.3, "big_win_rate": 0.0, "nk_avg": 0.32, "nk_alpha": 0.15}, "bull_2023": {"avg_return": 2.78, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.26, "nk_alpha": -0.48}, "q1_2024": {"avg_return": 1.91, "win_rate": 80.0, "big_win_rate": 0.0, "nk_avg": 3.35, "nk_alpha": -1.44}, "bear_2024": {"avg_return": -2.39, "win_rate": 50.0, "big_win_rate": 0.0, "nk_avg": -1.25, "nk_alpha": -1.14}, "q2_2025": {"avg_return": 3.67, "win_rate": 100.0, "big_win_rate": 0.0, "nk_avg": 4.38, "nk_alpha": -0.71}}
- invest_stage: Phase 0
- signals: なし
- FM: improve | avg=1.29%<2.0% / big=0.0%<20.0% / 日経アルファ=-0.72%≤0%（日経に負け） → 強制 improve
- analyst: parse error × 2、スキップ

## 2026-05-29

## 2026-06-01

## 2026-06-01
- metrics: {"avg_return": 2.68, "win_rate": 80.0, "big_win_rate": 10.0, "nk_avg": 1.55, "nk_alpha": 1.13}  periods: {"rate_hike_2022": {"avg_return": 0.13, "win_rate": 50.0, "big_win_rate": 0.0, "nk_avg": -2.41, "nk_alpha": 2.54}, "bull_2023": {"avg_return": 3.96, "win_rate": 75.0, "big_win_rate": 25.0, "nk_avg": 3.47, "nk_alpha": 0.48}, "q1_2024": {"avg_return": 5.93, "win_rate": 75.0, "big_win_rate": 25.0, "nk_avg": 3.66, "nk_alpha": 2.27}, "bear_2024": {"avg_return": 0.44, "win_rate": 100.0, "big_win_rate": 0.0, "nk_avg": -1.34, "nk_alpha": 1.78}, "q2_2025": {"avg_return": 2.96, "win_rate": 100.0, "big_win_rate": 0.0, "nk_avg": 4.38, "nk_alpha": -1.42}}
- invest_stage: Phase 0
- signals: なし
- FM: improve | big=10.0%<20.0% → 強制 improve
- analyst: {"file": "rf_train_v3.py", "changes": [{"param_name": "n_estimators", "old_value": 800, "new_value": 1200, "reason": "63日長期予測では複雑なパターン学習が必要。アンサンブル数増加で汎化性能向上"}, {"param_name": "max_depth", "old_value": 5, "new_value": 4, "reason": "過学習抑制。長期予測ではノイズ回避のため浅い木が有効。最新研究でも正則化強化推奨"}, {"param_name": "learning_rate", "old_value": 0.04, "new_value": 0.025, "reason": "n_estimators増加に伴い学習率を下げて過学習防止。勾配の安定化"}, {"param_name": "min_child_weight", "old_value": 8, "new_value": 15, "reason": "葉ノードの最小サンプル数増加で過学習抑制。63日先の長期予測ではノイズ除去が重要"}, {"param_name": "colsample_bytree", "old_value": 0.7, "new_value": 0.5, "reason": "特徴量サブサンプリング強化。多目的最適化研究の知見を反映し特徴量ノイズを削減"}, {"param_name": "subsample", "old_value": 0.8, "new_value": 0.7, "reason": "行サンプリング率を下げて正則化強化。長期予測での汎化性能向上"}, {"param_name": "reg_alpha", "old_value": 0, "new_value": 0.5, "reason": "L1正則化追加。最新研究のSVM的正則化組み合わせ知見を反映し過学習抑制"}, {"param_name": "reg_lambda", "old_value": 1, "new_value": 3, "reason": "L2正則化強化。特徴量重要度の低い変数の影響を抑制しモデル安定化"}, {"param_name": "gamma", "old_value": 0, "new_value": 0.3, "reason": "分割の最小損失減少量を設定。不要な分割を防ぎ過学習抑制"}]}
- engineer: ✅ 採用 [rf_train_v3.py]: n_estimators 800→1200 / max_depth 5→4 / learning_rate 0.04→0.025 / min_child_weight 8→15 / colsample_bytree 0.7→0.5 / subsample 0.8→0.7 / reg_alpha 0→0.5 / reg_lambda 1→3 / gamma 0→0.3 | avg -3.48%→0.73%  win 20.0%→50.0%  big 10.0%→10.0%
- stagnation: 1サイクル

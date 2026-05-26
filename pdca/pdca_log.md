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

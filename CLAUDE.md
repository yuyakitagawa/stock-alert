# stock-alert (v7-Efficient)

## 0. AI Handling Rules (燃費・安全設計)
- **Think Small**: 大規模なリファクタリングより、バグ修正とパラメータ調整を優先せよ。
- **No Hallucination**: utils.pyの34次元特徴量の定義を勝手に変えないこと。
- **Token Saving**: 解説は最小限にし、実行結果や修正コードを即座に提示せよ。

## 1. File Map (Core Only)
- `utils.py`: 特徴量定義(34次元) & 共通関数。※変更時は要申告。
- `rf_train_v3.py`: XGBoost学習(上昇/下落)。※土曜以外は触らない。
- `screener.py` -> `rank_stocks.py`: 抽出 & ネットスコア計算。
- `alert_email.py` / `generate_post.py`: 出力系。
- `backtest.py`: 検証。`bear`モード（2024/08下落相場）をテスト基準とする。

## 2. Model & Strategy (規律)
- **Target**: 63日(3ヶ月)で±15%変動予測。
- **Logic**: Net Score = 上昇Prob - 下落Prob。
- **Hard Filters (Don't Touch)**:
  - `down_streak > 3日` (0.15換算): 除外
  - `drawdown60 < -15%`: 除外
- **Note**: AUC 0.663 (上昇) / 0.791 (下落)。下落予測の精度を重視せよ。

## 3. Operations (Commands)
- Screening & Rank: `python3 screener.py && python3 rank_stocks.py`
- Test (Daily): `python3 alert_email.py`
- Deep Test: `python3 backtest.py bear` (暴落耐性チェック)

## 4. Context & Workflow
- **CI/CD**: GitHub Actions (Daily 08:00 JST).
- **Env**: GMAIL, GCP_KEY_JSON, SPREADSHEET_ID が必須。
- **Dev Cycle**: 
  1. 修正 2. `backtest.py bear` で性能確認 3. `dev_log.md` 追記 4. Commit

## 5. Skills Check (AIができること)
- [x] 特徴量抽出ロジックの最適化
- [x] XGBoostのハイパーパラメータ調整
- [x] バックテスト結果の統計分析
- [x] SNS投稿テキストの訴求力改善
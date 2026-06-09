# stock-alert (v7-Efficient)

## 0. AI Handling Rules (燃費・安全設計)
- **Think Small**: 大規模なリファクタリングより、バグ修正とパラメータ調整を優先せよ。
- **No Hallucination**: utils.pyの49次元特徴量の定義を勝手に変えないこと。
- **Token Saving**: 解説は最小限にし、実行結果や修正コードを即座に提示せよ。

## 1. File Map (Core Only)
- `lib/utils.py`: 特徴量定義(49次元: 43基本[32テクニカル+11ファンダメンタル]+6クロスセクション) & 共通関数。※変更時は要申告。
- `core/rf_train_v3.py`: XGBoost学習(上昇/下落)。※金曜(再学習日)以外は触らない。
- `core/screener.py` -> `core/rank_stocks.py`: 抽出 & ネットスコア計算。
- `email/alert_email.py`: 出力系（Gmail送信）。
- `web/export_to_web.py` + `web/send_user_alerts.py`: Webアプリ向けエクスポート。
- `tools/backtest.py`: 検証。`bear`モード（2024/08下落相場）をテスト基準とする。

## 2. Model & Strategy (規律)
- **Target**: 63日(3ヶ月)で±15%変動予測。
- **Logic**: Net Score = 上昇Prob - 下落Prob。
- **Hard Filters (Don't Touch)**:
  - `down_streak > 3日` (0.15換算): 除外
  - `drawdown60 < -15%`: 除外
- **Note**: AUC 0.663 (上昇) / 0.791 (下落)。下落予測の精度を重視せよ。

## 3. Operations (Commands)
- Screening & Rank: `python3 core/screener.py && python3 core/rank_stocks.py`
- Test (Daily): `python3 email/alert_email.py`
- Deep Test: `python3 tools/backtest.py bear` (暴落耐性チェック)

## 4. Context & Workflow
- **CI/CD**: GitHub Actions (平日 16:07 JST / 07:07 UTC、金曜に再学習).
- **Env**: GMAIL, GCP_KEY_JSON, SPREADSHEET_ID が必須。
- **Dev Cycle**: 
  1. 修正 2. `backtest.py bear` で性能確認 3. `dev_log.md` 追記 4. Commit
- **改善マージ規律**: 改善案はシミュレーション or バックテストで効果を数値確認してからコミットせよ。平均リターン・勝率・大勝率のいずれも改善しない変更はマージ禁止。

## 5. Skills Check (AIができること)
- [x] 特徴量抽出ロジックの最適化
- [x] XGBoostのハイパーパラメータ調整
- [x] バックテスト結果の統計分析
- [x] SNS投稿テキストの訴求力改善

## 6. 長時間作業の進捗管理
- **進捗ファイルを維持せよ**: セットアップ・移行・インフラ構築など複数ステップにまたがる作業は `docs/progress_<作業名>.md` に進捗を記録しながら進めよ。
- **ステップ完了のたびに即更新**: 各ステップ終了後、ファイルの `[ ]` を `[x]` に変えてからコミットせよ。ターミナルが切れても次のセッションで続きがわかるようにすること。
- **再開時は進捗ファイルを最初に読む**: 会話の冒頭でユーザーが「続きをやろう」と言ったら、まず該当の `docs/progress_*.md` を読んで状態を把握せよ。

## 7. コード規律
- **実験コードは即削除**: 実験が終わったら（採用・不採用どちらでも）その場でファイルを削除せよ。「念のため残す」は禁止。
- **不採用機能はコードからも消す**: A/Bテストで不採用になった条件はコード・定数・分岐ごと削除せよ。コメントアウトも禁止。
- **テストは対象と一緒に消す**: 関数を削除したら、そのテストも同じコミットで削除せよ。
- **READMEは同一コミットで更新**: 機能追加・変更・削除を実装したら、必ず同じコミットで README.md を更新せよ。フィルター値・コマンド・ファイル構成・テスト件数も合わせること。
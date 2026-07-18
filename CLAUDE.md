# stock-alert (v7-Efficient)

## 0. AI Handling Rules (燃費・安全設計)
- **Think Small**: 大規模なリファクタリングより、バグ修正とパラメータ調整を優先せよ。
- **No Hallucination**: utils.pyの64次元特徴量の定義を勝手に変えないこと。
- **Token Saving**: 解説は最小限にし、実行結果や修正コードを即座に提示せよ。
- **SQL Batch & No-Ask**: Supabase SQL は許可不要で勝手に叩いてよい（許可確認は一切しない）。実行前に「何を知りたいか」を洗い出し、必要なデータは**1クエリにまとめて取得**せよ（銘柄リストは配列＝`unnest(ARRAY[...])` / `IN (...)`、複数指標は CTE で横断結合）。同じ調査で何度も往復クエリを投げない。やむを得ず分割する場合も並列でまとめて実行する。
- **SQL PIT規律**: 「上がる前/買う前」のファンダを語るときは、必ず point-in-time で取得せよ。株価は対象日の終値、決算は `disc_date <= 対象日` の最新開示を使う。直近スナップショット（上昇後の株価でPER/PBR等）を「買う前の数字」として提示するのは禁止。

## 1. File Map (Core Only)
- `lib/utils.py`: 特徴量定義(64次元: 57基本[32テクニカル+11ファンダ+4マクロ(VIX/US5/US20/JPY5)+8新規IB(Amihud非流動性/FXβ/JPY5/EPSサプライズ/BPS成長/Piotroski/配当性向/アクルーアル)+1EDINET大量保有+3モメンタム拡張(ret504/trend_slope60/trend_r2_60)]+7CS[6標準+1セクター内相対モメンタム]) & 共通関数。※変更時は要申告。
- `lib/nlp_sentiment.py`: 決算テキスト感情分析（Claude Haiku × kabutan）。ランキング後処理に使用。
- `core/rf_train_v3.py`: XGBoost学習(上昇/下落)。※金曜(再学習日)以外は触らない。
- `core/screener.py` -> `core/rank_stocks.py`: 抽出 & ネットスコア計算。
- `web/export_to_web.py`: Supabaseへランキング・日経 vs S&P500判定をエクスポート（LINE Botが参照）。
- `tools/backtest.py`: 検証。`bear`モード（2024/08下落相場）をテスト基準とする。

## 2. Model & Strategy (規律)
- **Target**: 63日(3ヶ月)で±15%変動予測。
- **Logic**: Net Score = 上昇Prob - 下落Prob。
- **Hard Filters (Don't Touch)**:
  - `down_streak > 3日` (0.15換算): 除外
  - `drawdown60 < -15%`: 除外
- **βフィルター**: 日経強気時(N225>20SMA)はβ≥0.4の銘柄のみ💎対象。低β(ディフェンシブ)は降格。
- **Note**: AUC 0.642 (上昇) / 0.766 (下落)。下落予測の精度を重視せよ。

## 3. Operations (Commands)
- Screening & Rank: `python3 core/screener.py && python3 core/rank_stocks.py`
- Deep Test: `python3 tools/backtest.py bear` (暴落耐性チェック)

## 4. Context & Workflow
- **CI/CD**: GitHub Actions (平日 20:00 JST / 11:00 UTC、金曜に再学習).
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

## 8. マージ規律（PR統合時・絶対厳守）
- **マージ前に必ず最新mainを取り込め**: ブランチのbaseが現在のmainのtipより古い場合、先に最新mainを取り込んで（rebase / update branch）コンフリクトが無いことを確認してからマージする。stale baseのままマージすると、mainの新しいコミットが消える（実例: 2026-07-16、古いブランチのマージで直近5PR分の変更が丸ごと消失）。
- **マージ前にデグレ確認**: 既存テスト（`python3 tests/test_*.py`）を実行して失敗が無いことを確認する。
- **コンフリクト解消・デグレ無しを確認してから初めてマージを実行する**。確認を飛ばしたマージ・force pushでの上書きは禁止。
# 下落モデル一本化 進捗

## 背景
LINE Bot（本番のユーザー接点）は下落確率（drop_prob）のみでウォッチリストの買い時/売り時・ランキングソート・AIチャットの判断根拠を構成しており、上昇モデル・netスコア・QVフィルターに基づく「💎買い」システムはgen_rankingsに保存されるだけでLINE Botのどこからも参照されていないことが判明。
上昇モデル・net計算・recommend_from_scores()を廃止し、下落確率ベースの買い候補判定に一本化する。

**新しい買い候補の定義**: `drop_prob < 8%` + `passes_buy_filter`（株価/流動性/急落除外）。
既存の4防御フィルター（優待権利落ち・米国ETFリードラグ・NLP感情・相場リスク管制官・βフィルター）は買い候補の対象を💎から新ラベルに差し替えて維持する。

**制約**: このサンドボックス環境にはxgboost/scikit-learn/学習済みモデル(.pkl)/DB接続が無いため、`backtest.py bear`での効果検証はここでは実行できない。コード変更後、別環境（GitHub Actions等）での検証が必須。

## ステップ

- [x] 1. `lib/utils.py`: `recommend_from_scores()`を下落確率ベースの新関数に置き換え（net引数を削除）
- [x] 2. `core/rank_stocks.py`: rise_model/alpha_rise_model/alpha_drop_model読み込み・net計算を削除。ソート順をdrop_prob昇順に変更。4防御フィルター(優待/ETF/感情/リスク管制官/β)の対象を新ラベルに差し替え
- [x] 3. `core/rf_train_v3.py`: 上昇モデル(rf_model.pkl)の学習・保存を停止。下落モデルのみ学習。alpha_rise/alpha_drop/label_riseもgenerate_samples()から削除
- [x] 4. `lib/data_sanity.py`: net整合性チェック(net=rise-drop)を廃止し、下落確率のみの検査に変更
- [x] 5. `tools/backtest.py`/`multi_backtest.py`を下落確率ベースに全面書き換え（--net-min→--drop-max、nlargest→nsmallest等）。`optimize_net_weights.py`・`simulate_monthly.py`は目的が消滅したため削除。`export_report_to_sheets.py`はnet/rise_prob列を削除。`catalyst_backtest.py`は元々net/rise非依存のため変更なし。`web/export_to_web.py`のselect/sort順も修正。`core/rank_stocks.py`/`config.py`の関連デッドインポートも削除
- [x] 6. `tests/test_data_sanity.py`を下落確率のみの検査に書き換え（TestNetIntegrity削除、他は移植）。全テスト46件パス
- [x] 7. `README.md` / `CLAUDE.md`の該当箇所を更新（S買い条件・モデル説明・ファイルマップ・予測ラベル・ウォークフォワードAUC表・ランキングロジック・データ永続化・フィルター一覧・設計上の注意点）
- [x] 8. `dev_log.md`に記録
- [ ] 9. 実行可能なテストで確認、コミット・プッシュ・draft PR作成
- [ ] 10. PR説明に「backtest未検証、別環境で要検証」を明記

## 完了後の状態
- rf_model.pkl（上昇モデル）は生成されなくなる
- gen_rankingsの`rise_prob`/`net`列は今後書き込まれなくなる（既存データは残る）
- `recommend`列は新しい下落確率ベースのラベル体系に変わる

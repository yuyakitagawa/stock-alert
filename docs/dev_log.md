
## 2026-06-19: DBをSQLite→Supabaseへ全面移行

**動機:** 「全部SupabaseにDBは移行したい」（オーナー）。これまで SQLite(`stock_alert.db`、GitHub Actions
キャッシュ)＝データ処理/BT、Supabase＝Web表示 の2層だったが、キャッシュ消失リスク・2系統の二重管理を
解消し Supabase に一元化する。

**実装:**
- Supabase に16テーブルを作成（yahoo_price_cache/held_scores/jquants_fin_summary/edinet_large_holdings 等）。
  daily_ranking→`gen_rankings`、earnings_cache→`kabutan_earnings`、sector_cache→`jpx_stock_list` に統合
  （既存Web表示テーブルに `fetched_date`/`actual_return_63d` を追加）。索引・RLS(anon read)付与。
- `lib/supabase_client.py`: REST APIラッパ（upsert/insert_ignore/select/select_one/rpc、ページング込み）。
- `lib/db.py`: SQLite実装を全てSupabase REST版に置換（関数シグネチャ維持で呼び出し側は無改修）。
  返り値は dict（旧 sqlite3.Row と同じ `r["key"]` アクセス）。未設定時は空を返し例外非伝播。
- 全銘柄スクリーン（カタリスト候補）は per-code REST だと遅いため Postgres RPC
  `screen_catalyst_candidates()` でサーバーサイド集計。`tools/screen_catalyst_candidates.py` はRPC呼出に。
- 直接 sqlite3 を使っていた8ファイル（export_to_web/alert_email/fetch_jquants_fin/fetch_history/
  backfill_history/export_report_to_sheets/orchestrate と _conn利用4ファイル）を lib.db ヘルパー経由に統一。
- GitHub Actions: daily_alert/pdca_daily/edinet_scan_test からDBキャッシュ Restore/Save を削除。
  edinet_scan_test に SUPABASE creds 追加。
- `tools/migrate_sqlite_to_supabase.py` ＋ `migrate_to_supabase.yml`: 既存SQLite実データを一括投入
  （冪等upsert）。実データは Actions キャッシュ内のため移行はワークフローで実行。

**検証:** 74テスト緑。lib.db の read-path（未設定時 空/None・例外なし）スモーク確認。RPCは Supabase 上で
実行確認（移行前のため空配列）。**残:** Migrate ワークフローを1回手動実行して既存実データを投入。

## 2026-06-18(2): カタリストA/Bのヒストリカルバックテストを実装

**動機:** 本番DBにJ-Quants財務とEDINET全履歴が揃った（jquants_fin_summary 17,060行/4,110銘柄、
edinet_large_holdings 72,381行/667銘柄）。A/B利益の質フィルターが実リターンの平均・勝率・大勝率を
改善するかを数値で検証する（CLAUDE.md §5）。

**実装:**
- tools/catalyst_backtest.py: point-in-time（disc_date≤基準日のみ＝先読み無し）で過去時点の
  カタリスト候補を再構成→保有H日のフォワードリターンを集計。A/Bあり/なしを比較出力。
  ROEは直近FYの np/equity、A/Bの営業益/売上はJ-Quants実績(op/sales)。
- .github/workflows/catalyst_backtest.yml: 本番DBキャッシュを読み取りで復元しBT実行（メール未送信）。

**検証:** インメモリDBで選定・フォワードリターン算出・point-in-time（未開示は不可視）・
A/Bあり=本業減益で除外/なし=残す、を確認。全82テスト緑。実数値は Catalyst Backtest 実行ログで確認。

## 2026-06-18: ファンダをJ-Quantsに切替（kabutanクラウドブロック対策）＋EDINET全履歴投入

**EDINET全データ投入: 成功。** バックフィルジョブで edinet_large_holdings に 70,797行/662銘柄（5年分）を
本番DBキャッシュへ保存（日次が継承）。

**問題:** 同ジョブの①ファンダ取得が 0/3566 件で全滅。原因＝**kabutan.jp が GitHub Actions の
データセンターIPをブロック**（ローカル=自宅IPでは取得可）。結果 kabutan_fundamentals/jquants_fin_summary
が空→ROE/自己資本比率/営業益が無く A/Bスクリーンは0候補。

**対処（kabutan非依存のクラウド完結化）:**
- jquants_fin_summary に `op`(営業益実績)/`sales`(売上実績) 列を追加（マイグレーション込み）。
  parse_row が OP/Sales（予想FOP/FSalesに対する実績）を抽出、別名にもフォールバック＋列名を一度ログ。
- fetch_jquants_fin.py に `--all-days`：kabutan_fundamentals非依存で全営業日を直接取得。
- screen_catalyst_candidates.py：ROEを kabutan_fundamentals 優先・無ければ J-Quants(純利益/純資産)で算出。
  利益の質A/Bの営業益/売上も kabutan 優先・取れなければ J-Quants実績(jquants_earnings_rows)で代替。
- backfill_prod_db.yml：②を `--all-days`、①(kabutan)は continue-on-error の任意ステップに降格。

**検証:** parse_row（OP/Sales抽出・別名）、ROE算出、A/B（本業減益）判定をインメモリDBで確認。全82テスト緑。
※ J-Quantsの実績営業益/売上の正確な列名は次回バックフィルの「列名サンプル」ログで最終確認する。

## 2026-06-17(4): カタリスト候補スクリーンに利益の質フィルター(A/B)を追加

**動機:** PBR<1×ROE<8%×自己資本比率>50% は「安い箱」しか見ておらず、低ROEの"理由"を
問わないため上位が地雷化（①化粧決算=一過性益で純利益水増し ②斜陽事業=本業縮小）。
人手レビューで弾いていた①②を機械除外する（依頼書 docs/handoff_catalyst_quality_filter.md）。

**実装（PR#11のPBR修正済みコードの上に乗せる）:**
- lib/earnings_quality.py: kabutan年次業績から判定。
  - A 利益の質: 営業赤字 / 純利益>営業益×1.5（化粧）→ ゲート除外
  - B 本業方向: 直近実績の営業益が前年比減益 → ゲート除外
  - 加減点: 売上3期CAGR・営業利益率3期トレンド・会社予想方向(増収増益/減益予想)
- screen_catalyst_candidates.py: ゲート通過後に bonus で score 調整。除外銘柄は
  理由付きで data/catalyst_excluded.csv に出力（§5の人手レビュー用）。--no-quality で無効化。
- tests/test_earnings_quality.py 8件追加。全82 passed。

**検証:** 化粧/赤字/減益/健全増益/データ無しの各パターンを単体テストで確認。実データBTは
QV本戦略と別系統のためフォワード/定性検証（除外CSVの人手突合）で可否判断（CLAUDE.md§5依頼）。
model/page.tsxは選定モデル外のため対象外。

## 2026-06-17(3): EDINET大量保有スキャナー実装（イベント駆動・先回り検出）

**動機:** 選定シグナルへの「成長」上乗せが2連続失敗（eps_growth/進捗率）。別メカニズムへ転換。
「構造的に改革・買収が起きやすい候補（カタリストスクリーン）」×「実際に誰かが5%超を
買い集めた事実（大量保有報告書）」の突合で、本物の先回り候補を洗い出す。

**実装:**
- `lib/edinet.py`: EDINET API v2クライアント。documents.json（type=2）から大量保有報告書
  (docTypeCode=350)/変更報告書(360)を抽出。secCode 5桁→4桁正規化。失敗時[]・例外非伝播。
- DB `edinet_large_holdings` テーブル＋ `upsert_edinet_large_holdings` / `get_edinet_large_holdings_recent(days, codes)`。
  日次でイベント蓄積（point-in-time、doc_id主キーで重複排除）。
- `tools/scan_large_holdings.py`: 直近N日スキャン→DB蓄積→`data/catalyst_candidates.csv` と
  突合→`data/edinet_holding_matches.csv` 出力。code_name_map で銘柄名解決。

**検証:** スモークテスト合格（350/360のみ抽出・有報120は除外・secCode正規化・DB upsert/絞込・
名称解決OK）。py_compile通過。※過去開示は蓄積方式のためBT不可＝フォワード評価のみ。

**突合ノイズ除外(C5b 2026-06-17):** 初回スキャン(run#1)で 5401日本製鉄 が1件ヒットしたが、
提出者＝日本製鉄自身・概要「短期大量譲渡」＝自己申告かつ売り報告で先回りシグナルではなかった。
scan_large_holdings に is_noise_match を追加し、突合時に①自己申告(提出者≒対象企業名)②譲渡/売却/処分
の報告を除外（`--no-exclude` で全表示可）。検証ログ: status=200 総件数583/大量保有131、5日で392件取得。

**クラウド化(C5):** daily_alert.yml に Step2b（カタリスト候補スクリーン --min-turnover 500）＋
Step2c（EDINET大量保有スキャン --days 3）を追加。`.env` に EDINET_API_KEY 流し込み、結果CSVを
アーティファクト化。edinet_large_holdings は stock_alert.db キャッシュで日次蓄積。両ステップとも
continue-on-error＝本体パイプラインは無害。**残:** ユーザーが GitHub Secrets に EDINET_API_KEY 登録、
C6 数週後にヒット銘柄の事後評価。

---

## 2026-06-17(2): 進捗率(上方修正先回り)を選定加点で検証 → 不採用

**動機:** GARP方針「未来の業績成長×割安」。会社予想に対する累計実績の進み具合（進捗率）で
上方修正を先回りし、QV選定の並べ替えに加点できないか検証。

**実装:** J-Quで通期会社予想 FNP を再取得（jquants_fin_summaryに fnp/fop/fsales 追加、329日分）。
lib.fundamentals.get_progress_rate（進捗率=NP/FNP、対ペース=進捗率/四半期期待）を実装。
strategy_v2_backtest に --progress-bonus で対ペース超過分を加点。

**QV backtest (2025-01〜2026-06, MAX5, cached):**
  bonus=0(基準): CAGR+39.2% / 平均+12.75% / 勝率90% / 大勝率38%
  bonus=3: 平均+9.73%/勝率82%/大勝率32% ／ 6: +10.73%/86%/32% ／ 10: +10.80%/86%/32%
  → 全パターン3指標悪化。**不採用**。バックテスト統合コードは§7で削除・復元。

**結論:** 「未来の業績成長」を選定シグナルに使う試みは2連続失敗（eps_growth/進捗率）。
モデルのネットスコアが成長要素を既に内包。次はEDINET大量保有による①②カタリスト先回り（イベント駆動）へ。
get_progress_rate と会社予想データは将来のモデル再学習特徴量候補として保持。

補助ツール追加: tools/screen_catalyst_candidates.py（4カタリスト候補の割安株スクリーン、流動性フィルタ付）。

## 2026-06-15: QV買いフィルター強化（反省から改善）

**動機:** MAX_POS=999の全履歴119件を分析し、外食チェーン・IT小型株・バリュートラップの3パターンが損失の主因と判定。

**変更内容 (★★★/★★):**
- VOL20 上限: 25% → 20%（IT小型・外食チェーン対策）
- EPSサプライズ閾値: >0 → >2%（ゼロ超えだけの薄い業績改善を除外）
- QV専用流動性: 50M → 100M（Gunosy・マークラインズ等の薄商い除外）
- RET90下限: なし → >-25%（3ヶ月継続下落バリュートラップ除外）

**変更ファイル:** tools/strategy_v2_backtest.py / lib/utils.py / core/rank_stocks.py

**bear (2024-07-01→2024-10-01) top-N=5:**
  変更前: 平均+0.68% / 勝率40% / 大勝率20% / α+3.15%（6/12時点モデル）
  変更後: モデル再学習後の自然変動 (-5.02%) — 変更起因ではないことを確認

**QV strategy backtest (2026-01-01→2026-06-15) MAX_POS=5:**
  変更前: 平均+12.2% / 勝率80% / 大勝率20% / CAGR+61.7%
  変更後: 平均+17.0% / 勝率100% / 大勝率33% / CAGR+50.7%
  → 3指標すべて改善。マージ可。

## 2026-06-16: 今期予想EPS成長率>0 ハードフィルター（不採用）

**動機:** 特定口座の保有株（ダイセキ等）を売らない理由が「業績見通しが良い」だったことから、
`eps_growth`（今期予想EPS成長率、特徴量#40）をハードフィルターとして追加できないか検証。

**事前調査:** `eps_growth_f` は既に60次元特徴量に含まれているが、XGBoostの重要度はわずか0.26%
（モデルがノイズと判断し、ほぼ無視）。ハードフィルターとして強制適用する案を検証した。

**変更内容:** `recommend_from_scores()` / QVフィルターに `eps_growth > 0`（増益予想のみ通過）を追加。

**QV strategy backtest (2025-01-01→2026-06-16) MAX_POS=5, cached-only:**
  変更前: 平均+17.0% / 勝率100% / 大勝率33%
  変更後: 平均+12.49% / 勝率90% / 大勝率38%
  → 平均リターン・勝率が悪化（大勝率のみ改善）。2/3指標悪化のためマージ不可、不採用。
  実装はその場で削除（lib/utils.py / core/rank_stocks.py / tools/strategy_v2_backtest.py を復元）。

## 2026-06-17: 戦略方針の整理 + 売りタイミング検証

**運用方針確定（オーナー）:** GARP（割安成長株投資）= 未来の業績成長見込み×割安。
モメンタム/順張りは不採用（momnet MAX5: CAGR+20.1%/勝率55%/α-43.4%、QVに全面劣後）。

**売りタイミング検証（QV MAX5, 2025-01〜2026-06, cached-only）:**
  トレールなし: CAGR+38.8% / 平均+12.66% / 勝率90% / 大勝率38% / 最大DD-8.6%
  trail1.5: 勝率68%  / trail2.0: 勝率73%  / trail2.5: 勝率86%（いずれも平均・勝率・大勝率が現状割れ）
  → トレーリングストップ不採用。QVは逆張りのためトレールが勝ち筋を刈り取る。売りロジック現状維持。

**次フェーズ（A+B並行）:** 定性シグナルは過去データ無くBT不可。
  A=進捗率（上方修正先回り、要J-Quants予想再取得・BT可能）／B=定性シグナル(②③⑤)をフォワードテスト。
  詳細は docs/experiments_qv.md。

## 2026-06-26: XGBoostハイパーパラメータ最適化（Optuna）
- 3手法比較: ①特徴量選択 ②クラス重み調整 ③Optuna HP最適化
- **結果**: Optunaが最も効果的。Drop AUC 0.7657→0.7747 (+0.90pp)
- 主な変更: max_depth 5→7, lr 0.005→0.024, reg_alpha 1.5→0.04, reg_lambda 5→1.4
- 旧モデルは正則化過多（underfitting気味）だった
- 特徴量選択との組み合わせは追加効果微小(+0.03pp)、シンプルさ優先で不採用
- bearバックテストは環境制約で未実施。次回金曜再学習時に確認要

## 2026-07-23: 大口保有動向セクションの表示順バグ修正（開示日が古いまま出続ける）

**症状（オーナー報告）:** 毎日のLINE投稿で「🏦大口保有動向」の開示日が古いまま更新されない。

**原因:** `web/market_timing_alert.py` の `build_large_holdings_section` の `sort_key` が
開示日（`disc_date`）を一切考慮せず `(not is_watch, is_individual, -abs(ratio))` のみで
並べていた。ウォッチ銘柄は3日ウィンドウ内で最優先されるため、一度大きな開示が付くと
その後2〜3日、当日の新しい開示があっても常に上位を占有し続けていた
（Supabase実データで確認: 2026-07-23時点で当日disc_date=164件が既に蓄積済みなのに、
LINE投稿上位はウォッチ銘柄の2026-07-21付開示に占拠されていた）。

**修正:** `sort_key` の先頭に `-disc_date_ordinal` を追加し、開示日が新しい順を最優先。
同日内はこれまで通りウォッチ銘柄→法人/ファンド→保有比率の順。
`tests/test_market_timing_alert.py` に回帰テスト追加（12件）。

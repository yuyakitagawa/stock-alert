# 進捗: GARP定性シグナル取り込み（A+B並行）

運用方針: 未来の業績成長見込み×割安（GARP）。

## 決定事項（確定済み・再検討不要）
- [x] モメンタム戦略: 不採用（CAGR+20%でQVに劣後）
- [x] トレーリングストップ: 不採用（QV逆張りで勝率悪化）
- [x] eps_growth>0 二値フィルター: 不採用（BT悪化）
- [x] 進め方: A（進捗率・BT可）+ B（定性シグナル・フォワードテスト）並行。有料サービス不可。

## 制約
- 定性開示データ（増配/自社株買い/M&A/カタリスト ②③⑤）は kabutan 無料ページが直近1週間しか保持せず、**過去バックテスト不可**。→ フォワードテストのみ。
- EDINET（無料登録）は⑤の歴史的データを持つ唯一の無料源。無料実装でダメなら登録検討。
- J-Quants は .env に JQUANTS_API_KEY あり（無料枠12週遅延=BT用途OK）。

---

## A: 進捗率（上方修正の先回り）— バックテスト可能
- [x] A1. J-Quants 予想カラム確認済み: FNP(通期予想純利益)/FOP/FSales/FEPS、中間=FNP2Q等、来期=NxFNP。進捗率=NP(累計実績)÷FNP。
- [x] A2. スキーマに fnp/fop/fsales 追加（lib/db.py マイグレーション込み）、parse_row更新。スモークテスト合格（明治HD 3Q進捗率73.8%）。
- [~] A2b. フル再取得 実行中（329日・約71分・無料、logs/jquants_forecast_refetch_*.log）。契約範囲外の2024-03-25以前のみエラー（無害）。
- [x] A3. lib/fundamentals.py に get_progress_rate(code, as_of_date) を実装。進捗率=NP/FNP、対ペース=進捗率/四半期期待(1Q0.25/2Q0.5/3Q0.75)。既存データで検証OK（トリドール3Q対ペース2.10等）。点-in-time準拠。
- [x] A4. strategy_v2_backtest.py に --progress-bonus を追加。QV候補のスコアに対ペース超過分(0〜1)×重みを加点。構文・import確認済み。フェッチ完了後にA5で重みをスイープ。
- [x] A5. スイープ完了。bonus=0(基準)+39.2%/勝率90%/大勝率38% vs 3/6/10すべて悪化 → **進捗率加点は不採用**。バックテスト統合コード(A4)は§7で削除・復元済み。get_progress_rate と fnp列はデータ基盤として保持。
- [x] A6. 不採用のため model/page.tsx 更新は不要（戦略ロジック未変更）。

**A結論:** 「未来の業績成長」を選定シグナルに使う試みは2連続失敗（eps_growth/進捗率）。GARPの「成長」要素はモデルのネットスコアが既に内包。次は別メカニズム＝EDINET大量保有による①②カタリスト先回り（イベント駆動）へ。
- [ ] A6. 採用なら model/page.tsx・README更新（CLAUDE.md §7）

## B: 定性シグナル（②③⑤）— フォワードテスト
- 既存: rank_stocks.py が get_alt_signals で 自社株買い/増配/上方修正/M&A を毎日取得・表示中。tdnet_events に日次蓄積。
- [x] B1. alt_data のイベント分類を拡張: tob/parent/mgmt/holding/alliance/order/newbiz/restructure を追加。get_alt_signalsに growth_catalysts/has_growth_catalyst を出力。単体テスト合格。
- [ ] B2. 開示「本文の内容」を Claude Haiku で要約・強度スコア化（ANTHROPIC_API_KEY既存）
- [ ] B3. 定性スコアを QV選定の加点に組み込み（二値ゲートにしない）
- [ ] B4. 日次で「定性シグナル付き選定」をログ保存（フォワードテスト用の記録）
- [ ] B5. 数週間後、定性あり/なしの実績を比較評価

## C: EDINET大量保有スキャナー（①②先回り・イベント駆動）— フォワード
> 結論: 選定シグナルへの「成長」上乗せが2連続失敗（eps/進捗率）したため、別メカニズム＝
> 「構造的に改革が起きやすい候補」×「実際に誰かが5%超を買い集めた事実」の突合に転換。
- [x] C1. `lib/edinet.py`: EDINET API v2クライアント。documents.jsonから大量保有(350)/変更(360)報告書を抽出、secCode 5桁→4桁正規化。失敗時[]・例外非伝播（alt_data流儀）。
- [x] C2. DB `edinet_large_holdings` テーブル＋ `upsert_edinet_large_holdings` / `get_edinet_large_holdings_recent(days, codes)`。日次蓄積（point-in-time）。
- [x] C3. `tools/scan_large_holdings.py`: 直近N日スキャン→DB蓄積→カタリスト候補CSVと突合→`data/edinet_holding_matches.csv`。スモークテスト合格（350/360抽出・コード正規化・突合・名称解決OK）。
- [x] C4. README/.gitignore更新（同コミット）。model/page.tsx は選定ロジック未変更のため対象外。
- [x] C5. クラウド化: daily_alert.yml に Step2b（カタリスト候補スクリーン）＋Step2c（EDINET大量保有スキャン --days 3）を追加。`.env` に EDINET_API_KEY を流し込み、結果CSVをアーティファクト化。stock_alert.db キャッシュで edinet_large_holdings が日次蓄積。**残: ユーザーが GitHub Secrets に `EDINET_API_KEY` を登録**（未登録でもスキャンはスキップされ本体パイプラインは無害）。
- [ ] C6. 数週間フォワードで「突合ヒット銘柄」の事後株価を評価（過去BT不可のため）。

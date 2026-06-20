# SQLite → Supabase 全面移行

## 方針
- パイプライン用テーブル（yahoo_price_cache等16テーブル）をSupabaseに作成
- lib/db.py をSupabase REST API経由に書き換え
- GitHub ActionsのSQLiteキャッシュを廃止
- 既存Web用テーブル（web_rankings等）はそのまま維持（フロント用の読み取りビュー）

## Phase 1: Supabaseテーブル作成 ✓
- [x] yahoo_price_cache / held_scores / simulation_results / kabutan_yutai
- [x] kabutan_fundamentals / kabutan_sentiment / kabutan_jquants_margin / short_interest
- [x] tdnet_events / yahoo_market_index / jquants_fin_summary / edinet_large_holdings / top10_sim
- [x] web_rankings に actual_return_63d 追加（daily_ranking統合）
- [x] web_earnings に fetched_date 追加（earnings_cache統合）
- [x] web_stock_meta に fetched_date 追加（sector_cache統合）
- [x] インデックス・RLSポリシー（anon read）
- [x] RPC `screen_catalyst_candidates()`（全銘柄スクリーンをサーバーサイド集計）

## Phase 2: lib/db.py 書き換え ✓
- [x] lib/supabase_client.py（upsert/insert_ignore/select/select_one/rpc）
- [x] 全関数をSupabase REST版に置換（関数シグネチャは維持）
- [x] 移行ヘルパー追加（get_latest_ranking_date 等）
- [x] 74テスト緑 + read-path スモーク確認（未設定時は空を返す）

## Phase 3: 直接sqlite3呼び出しの更新 ✓
- [x] web/export_to_web.py（web_rankings/web_stock_meta読み込みに変更）
- [x] email/alert_email.py（3箇所）
- [x] tools/screen_catalyst_candidates.py（RPC化）
- [x] tools/fetch_jquants_fin.py / tools/fetch_history.py / tools/backfill_history.py
- [x] tools/export_report_to_sheets.py / pdca/orchestrate.py
- [x] lib/fundamentals.py / lib/alt_data.py / tools/strategy_v2_backtest.py / tools/fetch_fundamentals_history.py（_conn廃止）

## Phase 4: GitHub Actions更新 ✓
- [x] daily_alert.yml からDBキャッシュ Restore/Save 削除
- [x] pdca_daily.yml / edinet_scan_test.yml のDBキャッシュ削除
- [x] edinet_scan_test.yml に SUPABASE creds 追加

## Phase 5: データ移行
- [x] tools/migrate_sqlite_to_supabase.py（SQLite→Supabase一括upsert・冪等）
- [x] .github/workflows/migrate_to_supabase.yml（DBキャッシュ復元→移行）
- [ ] **要手動実行**: Migrate SQLite to Supabase ワークフローを1回実行（既存実データ投入）

## 残作業（ユーザー）
1. GitHub Secrets に `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` 登録済みか確認（既存のはず）
2. **Migrate SQLite to Supabase** ワークフローを手動実行 → 既存の yahoo_price_cache(3.9M)/jquants(17K)/edinet(72K) 等を投入
3. 翌営業日の Daily Stock Alert が Supabase 読み書きで正常完走するか確認

# Supabase を来月 Free に戻す（進捗管理）

目標: 2026-10 の請求サイクルから Free プランに戻す。
Free の壁は 2つ。**DB 500MB** と **egress 5GB/月**。2026-09-07 時点でどちらも超えていた。
Phase 1〜3 は 2026-09-07 に実施済み（DB 544MB → 337MB）。残るは1週間の観察とダウングレード。

## 実測した現状（2026-09-07）

- org `yuyakitagawa` / project `stock-alert`（ref `kxrgyguowxtjqexvmlgx`）は **Pro**
- `pg_database_size` = **544MB**（Free上限 500MB を超過）

| テーブル | 合計 | 行数 | 備考 |
|---|---|---|---|
| yahoo_price_cache | 174MB | 1,610,048 | 3,814銘柄 × 2024-12-03〜2026-09-04 |
| gen_rankings | 162MB | 591,381 | 168営業日 × 約3,520銘柄 |
| blog_crawler_log | 146MB | 432,101 | 2026-08-02〜。**36日で146MB（+4MB/日）** |
| その他 全部 | 62MB | - | edinet_large_holdings 19MB / jpx_short_selling 12MB ほか |

blog_crawler_log の内訳: Browser 228,035行（user_agentだけで37MB）、GoogleOther 109,626、
AhrefsBot 19,670、MJ12bot 18,372、SemrushBot 4,820。

### egress の内訳（コードからの推定。Supabaseの実測値ではない）

| # | 発生源 | 1回 | 頻度 | 月間 |
|---|---|---|---|---|
| 1 | `core/rank_stocks.py` 全銘柄 400日ぶんを1銘柄1リクエストで取得（100.6万行） | 約52MB | 平日毎日 | **約1.1GB** |
| 2 | `tools/fetch_history.py` の `get_price_cache_codes()` が code 列を全行(161万行)取得 | 約27MB | 平日毎日 | **約0.6GB** |
| 3 | `core/rf_train_v3.py` 600日 × 3,814銘柄（約150万行） | 約78MB | 金曜 | 約0.34GB |
| 4 | `tools/backtest.py bear`（全履歴161万行） | 約85MB | 手動 | 実行回数ぶん |
| 5 | サイトのISR再検証（stocks/[code] 3,814ページ ほか） | - | 巡回のたび | 数百MB〜1GB（**要計測**） |

2 は加えて `get_price_cache_coverage()` が銘柄ごとに2リクエスト = 1日 7,628リクエストを投げている。

## やること

### Phase 1: 小さくて効果が大きいもの（2026-09-07 完了）
- [x] C-1 `blog_crawler_log.user_agent` を `blog_crawler_ua` へ正規化（トリガー。62MB→30KB）
      ※当初案の「Browserを記録しない」は取りやめ。`tools/traffic_report.py` が Browser 行の
      user_agent / ip / visitor_id を使って機械と人を切り分けており、消すとこの分析が死ぬため。
- [x] C-2 AIクローラー(GPTBot/ClaudeBot/PerplexityBot 等12種)は400日保持でそのまま残す
      （GEO実験の観測データ。オーナー指示 2026-09-07）
- [x] C-3 robots.txt で AhrefsBot / MJ12bot / SemrushBot を Disallow（4.3万行/月 + ISR再検証も減る）
- [x] C-4 60日超の `blog_crawler_log` を消す日次purge（ops.yml に追加）
- [x] B-1 `get_price_cache_codes()` を `select distinct code` の RPC に置換（27MB → 60KB）
- [x] F-1 `stocks/[code]` の `revalidate` を 86400 → 604800
- [x] F-2 `priceReturns.ts` の yahoo_price_cache クエリに上限を付ける（現在 `.gte` のみで無制限）

### Phase 2: 本丸・価格データをローカルキャッシュに移す（2026-09-07 完了）
- [x] A-1 `lib/price_store.py` を新設。parquet を GitHub Actions cache に保存
- [x] A-2 毎日 `date=gt.<キャッシュ最終日>` の差分だけを1リクエストで取得（3,814行 ≈ 0.2MB/日）
- [x] A-3 `lib/db.get_price_df` / `get_price_cache` / `get_price_cache_coverage` / `get_price_cache_codes` をストア経由に
- [x] A-4 `daily_alert.yml` に actions/cache の restore/save を追加（rf_drop_model.pkl と同じ形）
- [x] A-5 `python3 tools/backtest.py bear` で回帰確認（値が変わらないこと）

### Phase 3: DBを削る（2026-09-07 完了）
- [x] D-1 `gen_rankings` を直近90日保持に（59.1万行 → 22.6万行、162MB → 約62MB）
      ※先に `web/x_insight.py` `web/publish_blog_articles.py` の PIT 参照が90日で足りるか確認する
- [x] E-1 `yahoo_price_cache` を620日保持に（`HISTORY_DAYS=600` が最長。174MB → 約140MB）
- [x] Z-1 `VACUUM FULL` で実サイズを回収（削除だけでは pg_database_size は縮まない）

### Phase 4: 確認とダウングレード（9/8〜10月）※未着手
- [ ] V-1 1週間 Supabase dashboard の usage を観察（egress の実測値をこのファイルに追記）
- [ ] V-2 DB 500MB / egress 5GB を両方下回っていることを確認
- [ ] V-3 次の請求サイクル（10月）で Free にダウングレード

## 結果（2026-09-07 実測）

- DB: **544MB → 337MB**（Free上限500MBに対し163MBの余裕）

| テーブル | 前 | 後 |
|---|---|---|
| yahoo_price_cache | 174MB / 1,610,048行 | 146MB / 1,550,716行（620日保持） |
| blog_crawler_log | 146MB / 432,101行 | 66MB / 374,521行（UA正規化＋保持期間） |
| gen_rankings | 162MB / 591,381行 | 63MB / 366,056行（150日保持） |

- egress: 実測はまだ取れていない（Supabase dashboard の usage は Management API から読めない）。
  コードから消えたぶんの試算は月 約2.05GB:
  - rank_stocks の日次フルフェッチ 約1.1GB → ほぼ0（差分のみ）
  - get_price_cache_codes の全行SELECT 約0.6GB → ほぼ0（RPC）
  - 金曜の再学習 約0.34GB → ほぼ0（同じミラー）
  - backtest の手動実行 1回85MB → 0（同じミラー）
  - 銘柄ページのISR再検証 月約600MB → 約85MB（7日再検証）

## 注意
- Free に戻すと DB 500MB / egress 5GB / Storage 1GB。**再超過すると今回と同じく全RESTが402で止まる**。
- 制限中も MCP の `execute_sql`（Management API）は通る。障害調査はREST経由ではなくSQLで行う。

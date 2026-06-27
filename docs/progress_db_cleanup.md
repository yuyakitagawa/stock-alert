# DB クリーンアップ & インフラ改善 — 進捗メモ

## 完了済み

### 1. gen_rankings NULLバックフィル (407,695行)
- [x] PER: 56,174→7,687 NULL (86%削減) — 残りはeps未取得銘柄
- [x] PBR: 2,585 NULL — bpsデータなし（改善不可）
- [x] piotroski: 327,613→96,285 NULL (71%削減) — 2FY+ta>0が必要
- [x] bps_growth: 330,533→281,151 NULL (15%削減) — 2FY+bps必要
- [x] eps_surprise: 89,791 NULL — 埋められる分は埋め済み
- [x] actual_return_63d: 277,490 NULL — yahoo_price_cacheが2026-06-18で終わり、63営業日先がないため改善不可

### 2. gen_risk_regime バックフィル
- [x] 11→117行に拡充（yahoo_market_indexマクロデータから計算）

### 3. gen_ai_analyses 日付修正
- [x] 3,579行の date を 1970-01-01 → 生成日に修正

### 4. テーブル削除 & シミュレーション機能撤去
- [x] gen_dividend_strategy (5行) DROP済み
- [x] gen_qv_sim DROP済み
- [x] SimulationPanel.tsx, simulation.ts, update_gem_sim.py 削除
- [x] HomeContent.tsx, page.tsx からsim関連コード除去
- [x] daily_alert.yml から Step 4s 削除
- [x] README.md 更新

### 5. jpx_margin_balance クリーンアップ
- [x] margin_buy_chg / margin_sell_chg カラム DROP（100% NULL）
- [x] LINE webhook (route.ts) から該当カラム参照を削除

### 6. jpx_stock_list クリーンアップ
- [x] name=NULL の870行を削除（gen_rankingsに影響なし確認済み）

### 7. Supabase SQL 許可プロンプト対策
- [x] `.claude/settings.json`（プロジェクト）にallow追加 → CCR環境では効かない
- [x] `.claude/settings.local.json`（ローカル）にallow追加 → CCR環境では効かない
- [x] `/root/.claude/settings.json`（ユーザー）にallow追加 → セッション中は反映されず
- [x] `/root/.claude/launcher-settings.json`（ランチャー）にallow追加 → セッション中は反映されず
- [ ] **次セッションで確認**: launcher-settings.jsonへの追加が起動時に読み込まれれば解決するはず。効かない場合はCCR環境固有の制約で、毎回手動許可が必要。

---

## 対応不要と判断

### 1. gen_rankings.pos52
- 前セッションで対応済み（91.3%→5.8% NULL）。残りはyahoo_price_cacheにデータなしで改善不可。

### 2. jquants_fin_summary の高NULL率カラム
- payout_ratio 62.4%、fop/fsales/fnp 54%、div_ann 51.8%、bps 39.4% — すべてデータソース由来。対応不要。

### 3. gen_ai_analyses.verdict — 95.7% NULL
- company-desc-v1 (会社説明) にはverdictがないため正常。対応不要。

### 4. ext_tdnet_disclosures
- xbrl_url: 前セッションで削除済み。
- category: 73.6% NULL。TDnet取得仕様上の限界。コード上「（分類なし）」にフォールバック済みで機能的に問題なし。放置。

### 5. Supabase SQL 許可プロンプト
- .claude/settings.json をgit管理に追加（PR #70でマージ済み）。デスクトップアプリで反映確認待ち。

---

## 全テーブル NULL率サマリー（2026-06-27時点）

### 問題あり
| テーブル | カラム | NULL率 | 対応 |
|----------|--------|--------|------|
| gen_rankings | pos52 | 5.8% | 済（限界） |
| gen_rankings | bps_growth | 69.0% | 済（限界） |
| gen_rankings | actual_return_63d | 68.1% | 済（限界） |
| gen_rankings | piotroski | 23.6% | 済（限界） |
| gen_rankings | eps_surprise | 22.0% | 済（限界） |
| jquants_fin_summary | payout_ratio | 62.4% | データソース由来 |
| jquants_fin_summary | fnp/fop/fsales | 54% | データソース由来 |
| jquants_fin_summary | div_ann | 51.8% | データソース由来 |
| jquants_fin_summary | bps | 39.4% | データソース由来 |
| ext_tdnet_disclosures | xbrl_url | 100% | カラム削除候補 |
| ext_tdnet_disclosures | category | 73.6% | 取得ロジック依存 |
| gen_ai_analyses | verdict | 95.7% | 正常（desc用） |

### 問題なし（0%）
jpx_margin_balance, jpx_short_selling, jpx_stock_list, yahoo_price_cache, yahoo_market_index, dp_watchlist, gen_simulation, gen_risk_regime, app系テーブル

---

## ブランチ・コミット情報
- ブランチ: `claude/inspiring-allen-wp3hnn`
- 未プッシュの変更: なし（すべてコミット・プッシュ済み）
- PRはマージ済み（シミュレーション撤去 + jpxクリーンアップ + gen_rankings backfill）

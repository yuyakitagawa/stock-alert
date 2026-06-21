# 進捗: DB整理（テーブル削除・命名規則・マージ）

## 概要
- 不要テーブル削除、weekly_reviewsシステム削除、kabutan_yutaiマージ、1次/2次命名規則適用

## ステップ
- [x] 1. weekly_reviewsシステム削除（テーブル・コード・ページ・ワークフロー）
- [x] 2. 不要テーブル削除（line_conversations, kabutan_sentiment）
- [x] 3. kabutan_yutai → gen_stock_meta マージ（has_yutai列追加）
- [x] 4. 1次/2次命名規則適用（web_ → gen_ リネーム、web_earnings → kabutan_earnings）
- [ ] 5. README更新
- [ ] 6. tsc / ビルド確認
- [ ] 7. commit & push & PR作成

## 命名規則
| カテゴリ | プレフィックス | 例 |
|---|---|---|
| 1次データ（外部取得） | 取得元名 | yahoo_price_cache, kabutan_earnings, edinet_large_holdings, jquants_fin_summary, tdnet_events |
| 2次生成データ | gen_ | gen_rankings, gen_stock_meta, gen_risk_regime, gen_ai_analyses |
| アプリデータ | app_ | app_bookmarks, app_push_subscriptions |

## 削除したテーブル
- weekly_reviews (2行、レビューシステム非稼働)
- line_conversations (0行、未使用)
- kabutan_sentiment (140行、全スコア0)
- kabutan_yutai (284行 → gen_stock_meta に統合)

## リネームしたテーブル
| 旧名 | 新名 | 理由 |
|---|---|---|
| web_rankings | gen_rankings | 2次生成 |
| web_stock_meta | gen_stock_meta | 2次生成 |
| web_earnings | kabutan_earnings | 1次（kabutan由来） |
| web_risk_regime | gen_risk_regime | 2次生成 |
| web_simulation | gen_simulation | 2次生成 |
| web_dividend_strategy | gen_dividend_strategy | 2次生成 |
| web_qv_sim | gen_qv_sim | 2次生成 |
| claude_ai_analyses | gen_ai_analyses | 2次生成 |
| top10_sim | gen_top10_sim | 2次生成 |
| activity_log | gen_activity_log | 2次生成 |
| web_bookmarks | app_bookmarks | アプリデータ |
| push_subscriptions | app_push_subscriptions | アプリデータ |

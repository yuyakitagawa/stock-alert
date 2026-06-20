# 実装依頼: 銘柄詳細ページ「利益の質」セクション（フェーズ2・データ基盤）

対象セッション: **モデル/DBセッション（DB有・Supabase書込み環境）**
起票: 2026-06-17（株相談セッションより）
関連: フェーズ1（フロントのAI解説拡張）は branch `claude/sharp-faraday-nhtzyi` で実装済み

---

## 1. 背景
Webアプリの銘柄詳細ページに、CMKで手分析したような「利益の質（化粧／本業シュリンク）」を**正確に**表示したい。
しかし現状、Web側データ（Supabase）に**営業利益・ROE・自己資本比率・カタリストスコアが無い**：
- `Ranking`(Supabase) = pbr / per / piotroski / bps_growth / eps_surprise / pos52 / net のみ。
- 四半期決算(`fetchRecentEarnings`, Yahoo由来) = revenue / netIncome のみ（**営業利益なし**）。

→ フロントだけでは「純益>営業益×1.5（化粧）」「営業益YoY≤0（シュリンク）」が計算できない。

## 2. ゴール
詳細ページに、実財務データに基づく**利益の質バッジ＋営業益トレンド**を表示：
- 🟢 健全 / 🟡 注意 / 🔴 化粧の疑い / 🔴 本業シュリンク
- 直近2〜3期の 売上・営業利益・純利益 のミニ推移。

## 3. やること（データ基盤）
データは既にローカルDBにある：
- 営業利益・売上・純利益の年次実績 = `tools/_kabutan_earnings_cache.json`（`revenue/op_profit/net_income/is_forecast`）。
- ROE / BPS = `fundamentals_annual`、自己資本比率・equity = `jquants_fin_summary`。
- カタリスト該当 = `tools/screen_catalyst_candidates.py` の出力（`data/catalyst_candidates.csv`）。

実装ステップ:
1. **判定ロジックを共通関数化**（例 `lib/quality.py`）:
   - `化粧` = 直近実績で `net_income > op_profit*1.5`（op>0）or（op<0 & net>0）
   - `シュリンク` = `op_profit` の前年比 ≤ 0、または `op_profit<=0`
   - `営業益トレンド` = 直近3期の op_profit 配列＋YoY%
2. **Supabaseへエクスポート**（`web/export_to_web.py` に追記）:
   - 新テーブル or 既存ランキング行に列追加: `op_profit_yoy`, `op_margin`, `net_op_ratio`, `quality_flag`(healthy/cosmetic/shrink/caution), `op_trend`(JSON: [{fy, revenue, op, net}]), `catalyst_score`, `roe`, `equity_ratio`。
3. **フロント表示**:
   - `frontend/lib/types.ts` の `Ranking` に上記フィールド追加。
   - `frontend/components/StockInsightPanel.tsx`（フェーズ1で作成済）に「利益の質」ブロックを追加するか、専用 `StockQualityPanel.tsx` を新設。
   - フェーズ1のAI解説は「参考」、本セクションは「実データ」とラベルを分けること。

## 4. 規律（CLAUDE.md）
- §7: ロジック/フィルター変更時は README を同一コミットで更新。
- §8: フロント変更は本番(https://stock-alert-web.vercel.app)の該当ルートが HTTP200/READY で反映されたことを確認するまでが1タスク。
- §0: 60次元特徴量は変更しないこと（本作業は表示用で無関係）。
- 改善マージ規律: 表示の正確性は既知バグ銘柄（東宝/甜菜糖/EIZO 等）で目視照合してから。

## 5. フェーズ1で完了済み（この依頼の前提）
- `app/api/stock/[code]/insight/route.ts`: Claude Haiku で 事業概要(拡張)/主要取引先/カタリスト評価/リスク を生成・Supabase(`ai_analyses`, model_version `company-insight-v1`)キャッシュ。
- `components/StockInsightPanel.tsx`: 上記を「🤖 AI生成・参考」免責付きで表示。取引先はAI推定の注記あり。
- `app/stocks/[code]/page.tsx`: 上記パネルを組込み。
- `lib/types.ts`: `Insight` 型追加。
- ⚠️ フェーズ1は ANTHROPIC_API_KEY / SUPABASE_SERVICE_KEY が本番に設定済み前提。ローカル/当環境では未検証（要 本番 or プレビューでの動作確認）。

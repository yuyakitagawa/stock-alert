# セッション引き継ぎ（GARP定性フェーズ） 2026-06-17

新しいClaude Codeセッションで「続きをやろう」と言うとき、まずこのファイルと
`docs/experiments_qv.md`・`docs/progress_garp_qualitative.md` を読むこと。

---

## 1. 運用方針（最重要・判断軸）
**GARP（割安成長株投資）= 未来の業績成長見込みがあり、かつ割安な株に投資する。**
高値追い（モメンタム/順張り）は方針外。「伸びる株」=割安なまま将来成長が見込める株。

## 2. 現在の本番戦略（QV = Quality × Value）
- 条件: Piotroski≥0.67 / pos52<0.45（52週安値圏）/ (EPSサプライズ>2% or BPS成長>0)
  / VOL20≤20% / RET90>-25% / 流動性≥100M / 90日保有
- 実運用は **MAX_POS=5**（上位5銘柄に集中、ネットスコア＝上昇-下落確率で並べ替え）
- 直近バックテスト成績（2025-01〜2026-06, MAX5）:
  **CAGR+39.2% / 平均+12.75% / 勝率90% / 大勝率38% / 最大DD-8.6%**
- 売りロジック: drop急騰(下落確率≥10%) / net大幅反転(<-5) / 90日時間切れ。**トレールは無し（検証で不採用）**

## 3. このセッションで試して【不採用】になったもの（再提案禁止）
すべて docs/experiments_qv.md に記録済み。
- **モメンタム戦略**（momnet MAX5）: CAGR+20%/勝率55%/対日経-43%。QVに全面劣後。
- **トレーリングストップ**（QVに1.5/2.0/2.5倍）: 全倍率で平均・勝率・大勝率が悪化。QVは逆張りなので勝ち筋を刈り取る。
- **eps_growth>0 二値フィルター**: 平均+勝率悪化。
- **進捗率(対ペース)の選定加点**（bonus=3/6/10）: 全指標悪化。
- 結論: **「未来の業績成長」を選定シグナルに上乗せする試みは2連続失敗**。モデルのネットスコアが
  既に成長要素を内包しており、ファンダ予想を足すと良い銘柄を押しのける。

## 4. このセッションで【保持・採用】したもの
- **会社予想データ**: jquants_fin_summary に fnp/fop/fsales 列を追加し2年分取得済み。将来のモデル再学習の特徴量候補。
- **lib.fundamentals.get_progress_rate(code, as_of_date)**: 進捗率=NP/FNP、対ペース指標。保持（戦略パスからは外した）。
- **lib/alt_data.py イベント分類拡張**: tob/parent(親子解消)/mgmt(経営者交代)/holding(大量保有)/
  alliance/order/newbiz/restructure を判別。get_alt_signals に growth_catalysts/has_growth_catalyst 追加。
- **tools/screen_catalyst_candidates.py**: 4カタリスト候補の割安株スクリーン。
  条件 PBR<1.0 / ROE<8% / 自己資本比率>50% / 売買代金≥指定。流動性500Mで65銘柄該当。
  出力 data/catalyst_candidates.csv。実行: `python3 tools/screen_catalyst_candidates.py --min-turnover 500 --top 30`
- **data/code_name_map.json**: 全4446銘柄の名称マップ（J-Quants get_list由来）。
- いずれもコミット済み（c89fc8c）。

## 5. Webシミュレーション修正（重要な落とし穴）
- 本番Vercelは **/Users/kitagawayuuya/stock-alert/frontend/** からビルドされる。
  （別repo /Users/kitagawayuuya/bussiness/stock-alert-web は本番と繋がっていない＝触っても無意味）
- simulation.ts を gen_qv_sim 読み込み・cache:no-store に修正済み。本番反映確認済み（コミット3728cb7）。

## 6. EDINET（①②カタリスト先回りの次フェーズ）
- EDINET APIキー取得済み・**動作確認OK**（.env の EDINET_API_KEY、大量保有報告書 docTypeCode=350/360 取得可能）。
- 登録の白紙トラブル原因=ポップアップブロック。Chrome＋ポップアップ許可で解決。
- **次の実装**: EDINET大量保有報告書を毎日スキャン →「誰が・どの銘柄を5%超買ったか」検知 →
  カタリスト候補65銘柄と突合 →「構造的候補×実際に買い集めあり＝本物の疑惑」。これはイベント駆動で、
  失敗した「選定への足し算」とは別メカニズム。
- クラウド化するなら EDINET_API_KEY を **GitHub Secrets** に追加（ローカル.envはCI非共有）。

## 7. B: 定性シグナルのフォワードテスト（過去データ無くBT不可）
- 増配/自社株買い/M&A等の開示は kabutan が直近1週間しか保持せず過去検証不可 → フォワードテストのみ。
- alt_data の分類拡張済み（上記4）。次は日次ログ蓄積→数週間後に「定性あり/なし」で実績比較。

## 8. 運用・規律（CLAUDE.md / メモリ）
- Python実行: `/Users/kitagawayuuya/stock-alert/venv/bin/python3` を絶対パスで。
- git push は確認不要。
- 略語は初出時にカッコで日本語説明。
- 改善マージ規律: 平均・勝率・大勝率のいずれも改善しない変更はマージ禁止。
- 実験コードは採用/不採用が決まったら即削除（§7）。
- フィルター/ロジック変更時は frontend/app/model/page.tsx・README・web/メール再出力をセットで（§7,§8）。
- Web変更は本番(https://stock-alert-web.vercel.app)のHTTP200/READY確認までが1タスク。

## 9. インフラ（クラウド/ローカル）
- 毎日のスクリーニング/ランキング/メール/Web出力/金曜再学習 = **GitHub Actions**（平日17:30 JST、daily_alert.yml）。
- DB(stock_alert.db)はgit管理外、GitHub Actionsの**cache**で永続化。APIキーはGitHub Secrets。
- 今日の手動作業（J-Q予想取得・EDINET・カタリストスクリーン）は**ローカルMacのみ**。クラウド未反映。
- Web=Vercel、Webデータ=Supabase(project kxrgyguowxtjqexvmlgx)。

## 10. 次にやる候補（未着手）
- [ ] EDINET大量保有スキャナー実装（①②先回り、イベント駆動）。GitHub Secretsにキー追加。
- [ ] カタリストスクリーンのクラウド化/可視化（Web新ページ or メール、出力先は要相談）。
- [ ] B 定性シグナルの日次ログ蓄積→フォワードテスト評価。
- [ ] 会社予想データを使った金曜モデル再学習での特徴量追加検討。

## 11. 保有株メモ（参考・投資助言ではない）
- NISA口座はそのまま保持の方針。
- 特定口座: KPPグループHDのみ売却済み、他（キヤノンMJ/ダイセキ/F&LC等）は業績見通しを理由に保持。

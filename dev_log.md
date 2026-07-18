# Dev Log

## 2026-07-18 Supabase書き込み/取得ロジックの重複排除リファクタ

```
背景: ユーザーからのリファクタ依頼。挙動を変えずコード量を削減する対象を調査した結果、
      lib/supabase_client.py に既にある upsert()/select()（ページング・バッチ分割・
      ヘッダー生成）と全く同じロジックが web/export_to_web.py・tools/backfill_history.py・
      web/market_timing_alert.py の3ファイルにそれぞれコピペで再実装されていた。
      また推奨ラベルの絵文字正規化マップ（EMOJI_MAP/clean_recommend）も
      export_to_web.py と backfill_history.py に別々に定義されていた。

対応:
  - lib/utils.py に clean_recommend_label() を追加し、両ファイルの重複マップを統合
    （backfill_history.py 側にしか無かった「🥈 A買い」のマッピングが export_to_web.py にも
    適用されるようになった＝表示の統一漏れが直った）
  - web/export_to_web.py・tools/backfill_history.py の自前 _upsert()/upsert()（リクエスト
    ヘッダー生成・500件バッチ分割）を lib.supabase_client.upsert() に置き換え。
    副次効果として、この2ファイルには無かった NaN/inf サニタイズとバッチ内重複キー除去が
    自動的に効くようになった
  - web/market_timing_alert.py の自前 sb_get()（1000件ページング）を
    lib.supabase_client.select()/select_one() に置き換え
  - web/market_timing_alert.py: data/market_timing.json への書き出しが、Webアプリ撤去後
    どこからも読まれていない完全な死んだコードだったため削除（book_watchlist_sectionの
    alerts戻り値もこの出力専用だったため合わせて削除・関数を単純化）
  - 4ファイル合計で 115行削減（+37 / -152）。ロジック変更なし・テスト37件全件パス

判定: モデル・買いフィルター・LINE配信の出力内容への変更なし（純粋なコード整理）。
      backtest対象外。
```

---

## 2026-07-18 Webアプリ（frontend/Vercel）を全面撤去

```
背景: DB整理（app_bookmarks等の使用状況調査）をきっかけに、ユーザーがWebアプリ
      （frontend/・Vercelデプロイ）自体の削除を明示的に指示。運用は既にLINE Bot
      （Supabase Edge Function line-webhook）に一本化されており、Webアプリは
      並行して残っていただけの状態だった。

対応:
  - frontend/ ディレクトリを全削除、.github/workflows/frontend_build.yml を削除
  - web/send_user_alerts.py（Web Push送信）・web/qa_pages.py（全ページQA）・
    web/generate_descriptions.py（会社説明AI生成）・web/sync_descriptions.py
    （会社説明の手動同期）を削除
  - web/export_to_web.py から generate_ai_analyses/export_risk_regime/
    export_simulation_results/qa_site_check とその呼び出しを削除。
    LINE Botが参照する gen_rankings/jpx_stock_list/gen_market_compare の
    エクスポートのみ残した
  - lib/data_sanity.py から check_site/check_pages/run_site_gate/run_pages_gate
    （Webページ・サイト全体QA、いずれもWeb専用）を削除。check_ranking/run_gate
    （行レベルQA、LINE配信前にも使用）は維持
  - tests/test_data_sanity.py から TestCheckSite/TestCheckPages を削除
    （29件→11件）
  - .github/workflows/daily_alert.yml から Step 4a（会社説明生成）・
    Step 4c（全ページQA）・Step 5（Web Push送信）を削除。存在しない
    web/send_catalyst_alerts.py を呼んでいた Step 5c（daily_alert.ymlに
    以前から存在、実装ファイルなしでcontinue-on-errorに握りつぶされ続けていた
    死んだステップ）も削除
  - .claude/skills/web-republish/ を削除（Webアプリ再公開手順のスキルのため）
  - Supabase側: app_bookmarks / app_push_subscriptions / gen_ai_analyses /
    gen_risk_regime / gen_simulation / gen_activity_log / etf_profiles の
    7テーブルをDROP（Web専用または無参照）
  - README.md / CLAUDE.md をWebアプリ言及ゼロの状態に更新（Vercelデプロイ
    確認ルールも削除）
  - Vercelデプロイの停止はAI側にAPI/CLIアクセスが無いため未実施。
    ユーザー側でVercelダッシュボードから手動で行う必要あり

判定: モデル・買いフィルター・LINE配信ロジックへの変更なし。運用中の配信経路
      （LINE Bot・daily_alert.yml）はそのまま維持し、並行して残っていた
      未使用の配信経路を削除しただけのためbacktest対象外。
```

---

## 2026-07-15 yahoo_price_cache 長期停止バグの発見・修正 + 遡及バックフィル基盤追加

```
発見の経緯: daily_alert.yml が2026-07-08以降4営業日連続失敗（別issue: pipキャッシュ設定
            エラー、#147で修正済み）していたのを調査中、その修正確認のため手動実行した
            07-14分のgen_rankingsで銘柄コード7203(トヨタ)のcloseが2844.0円だったが、
            これはyahoo_price_cacheの2026-06-02時点の値と完全一致 → 「直近株価」が
            実際には数週間〜数ヶ月前の価格のまま更新されていなかったことが判明。

根本原因: daily_alert.yml Step 0 が呼ぶ tools/update_price_cache.py が
          リポジトリに一度も存在しておらず（continue-on-error: true で握りつぶされ
          気づかれずにいた）、yahoo_price_cache が全く更新されていなかった。
          実データ確認: 全3747銘柄が2026-06-20より前で停止、最悪ケースは2026-05-01。
          → rank_stocks.py の「直近株価」・全テクニカル特徴量が、この期間の
            全ランキング（Web/メール/LINE配信分含む）で古い価格を基に計算されていた。

対応:
  - tools/update_price_cache.py を新規作成（J-Quants v2 get_eq_bars_daily_range で
    直近N日分を全銘柄一括取得しyahoo_price_cacheへ差分保存。daily_alert.yml Step 0が
    毎日呼ぶことで今後は再発しない）
  - .github/workflows/backfill_rankings.yml を新規作成（手動実行・workflow_dispatch。
    価格キャッシュ更新→tools/backfill_history.py で指定期間のgen_rankingsを再生成。
    アラート再送信はしない設計）

判定: モデル・買いフィルターへの変更なし。データ基盤の欠陥修正のため backtest 対象外。
      本番影響: 2026-06-20頃〜07-14の日次配信は全て古い価格ベースだった可能性が高い。
      07-08/09/10/13は欠損（別issue）、07-14は誤った価格で計算済み → 全て要再生成。
```

---

## 2026-07-13 日経 vs S&P500 相対強弱アドバイザー追加（ユーザーフィードバック対応）

```
背景: 「日経の調子が悪くなってきた、S&P500の方がいいのでは？」という問いに
      システムが答えられない、というフィードバック。マクロ特徴量(us5/us20)は
      既にモデル内部で使われていたが、ユーザー向けの比較表示が無かった。

対応: lib/market_compare.py を新規追加（日経225とS&P500の20日/60日リターン差から
      jp_favored/us_favored/neutral を判定・参考情報のみ、売買シグナルには影響なし）。
      core/rank_stocks.py フェーズ8bで判定・data/market_compare.jsonに保存。
      web/export_to_web.py → gen_market_compare テーブルへexport。
      frontend: MarketCompareBanner をトップページに表示（RiskRegimeBannerと同型）。

判定: モデル・買いフィルターへの変更なし（情報表示機能のため backtest 対象外）。
      unit test 4件追加（tests/test_market_compare.py）。frontend build確認済み。
```

---

## 2026-06-17 カタリスト候補CSVのPBRバグ修正（全65銘柄を実測照合）

```
原因: screen_catalyst_candidates.py の pbr = close / bps で
      close=yahoo_price_cache(分割調整済) と bps=kabutan_fundamentals(旧株ベース) の
      分割調整基準が不整合。株式分割銘柄でPBRが分割比率分だけ過小化。
      ※同じロジックが lib/fundamentals.py:219 にもあり、Web/メール表示PBRも影響。
        ただし bps は表示専用で60次元特徴量には不使用 → モデル精度には無害。

対応: data/catalyst_candidates.csv の全65銘柄PBRを irbank等で実測し正値に置換。
      score=(1-pbr)*equity_ratio を再計算・再ソート。

検出された主な誤り（旧→実測）:
  9602 東宝       0.41 → 2.21  (差1.80・最悪)
  2695 くら寿司    0.98 → 1.91
  9533 東邦瓦斯    0.24 → 0.93
  6592 マブチ     0.58 → 1.19
  5541 大平洋金属  0.64 → 0.88
  6104 芝浦機械    0.77 → 0.98
  1663 K&Oエナジー 0.98 → 1.19

結果: 実PBR>=1.0で除外7銘柄（9602/6592/6201/3765/4078/1663/2695）。
      6201豊田自動織機は2026/06/01 TOB上場廃止済。
      採用65→58銘柄。修正後トップは 6619ダブルスコープ(PBR0.21)。
```

- 今回はCSVデータのみ手修正。**screener本体のbps分割調整は未修正**（DBが当環境では空のため検証不可）。
- 次タスク候補: lib/fundamentals.py / screen_catalyst_candidates.py のBPSを「自己資本÷現発行株数」算出に変更し、分割調整漏れを根絶（要DB環境で再生成・照合）。
- 判定: データ品質修正（バックテスト対象外）。

---

## 2026-06-17 PBR分割調整バグの根本修正（BPSソースをJ-Quantsへ）

```
方針: ユーザー指示「4000銘柄全て正値に」→ Web全件スクレイプは非現実的なため
      根本原因（BPSソース）をコード修正し、DB再計算で全銘柄を一括正値化する。

修正:
  1. tools/screen_catalyst_candidates.py
     - latest_bps_split_safe() を追加。jquants_fin_summary の直近開示BPS(>0)を採用。
       J-QuantsのBPSは開示ごとに分割後株数で再表示され、分割調整漏れが起きない。
     - PBR算出で J-Quants BPS を優先、未取得銘柄のみ株探(kabutan_fundamentals)へフォールバック。
  2. lib/fundamentals.py
     - _jq_split_safe_bps() を追加し get_pit_valuation(表示PER/PBR用)で優先採用。
       → Web/メール表示PBRの分割調整漏れを是正（全銘柄対象）。

未変更（意図的）:
  - pit_fundamental_features() の pbr（60次元特徴量）は学習済みモデルとの分布整合のため
    据え置き。分割調整BPSへの移行は金曜再学習時に申告のうえ実施（CLAUDE.md §0）。

検証状況:
  - 当リモート環境はDBが空のため未検証（compileのみOK）。
  - 次のDB有環境（ローカル/GitHub Actions）で screener 再実行 →
    東宝(9602)/マブチ(6592)/東邦ガス(9533) 等の既知バグ銘柄で実PBRと一致するか照合する。
```

- 判定: 根本修正（要DB環境で再生成・照合）。データ修正版CSVは前コミットで反映済み。

---

## 2026-06-12 モデル再学習 + QV戦略 2026年バックテスト

```
rf_train_v3.py 再学習完了（Jun 12 19:57）

QV戦略 2026-01-01→2026-06-12（162日）
  トレード数: 10 / 期間トータル: +24.2% / CAGR: +63.1%
  平均 +12.19% / 勝率 80% / 大勝率(≥15%) 20%
  最大DD: -6.4% / vs 日経(+31.1%) → アルファ -6.9%

再学習前後比較: 平均 +2.23%→+12.19% / 勝率 60%→80%
```

- モデル再学習で絶対リターン・勝率が大幅改善（2026年前半の相場に追従）
- 日経が+31.1%と強烈な上昇局面のため相対アルファはマイナス
- simulation.ts: 💎 買いシグナルをエントリー条件に変更、since=2026-01-01固定
- 判定: マージ可（再学習・simulation更新）

---

## 2026-06-12 bear バックテスト（💎 買い条件変更後の耐性確認）

```
bear (2024-07-01 → 2024-10-01) top-N=5
  全92銘柄: 平均 -3.48% / 中央 -8.12% / 勝率 32.6%
  上位5:    平均 +0.68% / 勝率 40.0%  / 大勝率 20.0%
  vs 日経225: +0.68% vs -2.47% → アルファ +3.15%
```

- 💎 買い条件を drop<2% × net≥16% × Piotroski≥6/9 × pos52<0.45 × 業績改善 に変更後の確認
- 結果は前回と同値（上位5は net スコアで選出するため変化なし）
- 暴落相場でも日経比 +3.15% アルファを維持 → 下落耐性OK
- 判定: マージ可（buy フィルター強化、シグナル品質向上）

---

## 2026-06-11 bear バックテスト（skillテスト実行）

```
bear (2024-07-01 → 2024-10-01) top-N=5
  全92銘柄: 平均 -3.48% / 中央 -8.12% / 勝率 32.6%
  上位5:    平均 +0.68% / 勝率 40.0%  / 大勝率 20.0%
  vs 日経225: +0.68% vs -2.47% → アルファ +3.15%
```

- 日経がマイナスの暴落相場でも上位5銘柄が微プラスを維持、アルファ +3.15%
- 上位5のうち1銘柄（ゼロ / 9028）が +30.26% と大勝。残り2銘柄は -10%前後の負け
- 下落相場での選別精度は限定的だが、日経比ではアウトパフォーム
- 今回はコード変更なし（bear-backtest skill の動作確認目的）
- 判定: ベースライン確認のみ（マージ評価対象外）

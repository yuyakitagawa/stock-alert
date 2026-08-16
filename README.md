# stock-alert

東証上場株式の機械学習スクリーニング・アラートシステム。毎日自動でランキングを生成し、Gmailでメール通知する。

## システム概要

平日にGitHub Actionsが自動実行される。

```
【16:00 JST】アラートパイプライン（daily_alert.yml）
core/rank_stocks.py（銘柄取得・下落確率ランキング生成を単独で実施。core/screener.pyは2026-08-01に
日次パイプラインから除外済み。詳細は下のファイル構成参照）
→ web/export_to_web.py（Supabase同期）→ web/market_timing_alert.py（LINE通知）
core/rf_train_v3.py（金曜 or モデル未存在時のみ）は配信より後段で実行。
配信のクリティカルパスから切り離すことで、学習が長時間化/タイムアウト
（continue-on-error, timeout-minutes: 180）しても当日のアラート配信は止めない。

【平日9:00-21:00 JST・毎時】EDINETブログパイプライン（edinet_blog.yml）
tools/scan_large_holdings.py（EDINET大量保有スキャン）→ web/publish_blog_articles.py
（ブログ記事自動生成・投稿＋X投稿）。株価更新パイプライン(daily_alert.yml)から完全に独立した
別ワークフローとして毎時実行する（2026-08-15、EDINET記事投稿を日次16:00の1回だけでなく
開示当日のうちに検出・記事化するため分離）。edinet_large_holdings はSupabase経由で
daily_alert.ymlのランキング生成（EDINET大量保有の特徴量）からも参照される。

【平日21:30 JST・1日1回】ショート動画パイプライン（video_post.yml）
video/publish_video.py（microCMSの新着記事×注目枠から1件選定 → Claudeで縦動画の台本生成 →
Remotionで1080x1920/20秒のmp4を書き出し → YouTube Shorts / TikTok へ投稿）。
edinet_blog.ymlがその日の記事を出し切ったあとに走らせる。Remotionのレンダリングは
Chrome Headlessが必要で毎時回すには重いため、記事投稿とは別の1日1回のバッチにしている。
対象記事が無い日は何も投稿しない。

その他ワークフロー: ci.yml（テスト）、
keepalive.yml（Supabase keepalive）、watchdog.yml（daily_alert.yml監視）、
data_backfill.yml（JPX/TDnet/EDINET手動遡及）、backfill_rankings.yml（株価キャッシュ更新+ランキング遡及・手動実行）、
filer_win_rate.yml（投資家別勝率の週次再計算、tools/filer_win_rate.py）
```

ユーザー向けの通知・操作は LINE Messaging API 経由（Supabase Edge Function `supabase/functions/line-webhook`）で提供する。Web/Vercelアプリは廃止済み。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | **手動実行専用ツール**（日次パイプラインからは2026-08-01に除外済み）。`get_tse_stock_list()`（JPX全銘柄取得）のみ`rank_stocks.py`/`backfill_history.py`が再利用。銘柄コード絞り込みは`STOCK_CODE_PATTERN`（`^\d{3}[0-9A-Z]$`）で、旧4桁数字に加えTSEが2024年以降に発行する新形式（末尾1桁が英字。例: 151A）も含める（旧`^\d{4}$`では新形式コードが全銘柄スキャンから恒久的に漏れていた）。`apply_screener_v1`によるスクリーニング自体は現在ほぼ価格・流動性のみで`rank_stocks.py`のハードフィルターと重複しており、出力する`data/screeners/*.csv`はどこからも読まれない（下落確率ランキングは`rank_stocks.py`が全銘柄取得〜フィルターまで単独で実施）。手動での銘柄スクリーニング確認用に残置 |
| `tools/fetch_history.py` | Yahoo Finance で全銘柄株価四本値を取得し `yahoo_price_cache` を差分更新（daily_alert.yml Step 0で毎日 `--years 1` 実行。`rank_stocks.py`の「直近株価」の鮮度に直結。既存(code,date)は insert_ignore で保護されるため初回10年分バックフィルにも日次更新にも使える）。`get_all_codes()`はyahoo_price_cache既存コードだけで打ち切らず、毎回JPX最新銘柄リストとの和集合を対象にする（新規上場銘柄が価格キャッシュに永久に追加されない事態を防止。JPX取得失敗時は既存コードのみにフォールバック）。対象は内国株式に加えJ-REITも含む（`_fetch_jpx_codes()`、ブログ記事の金額推定でJ-REIT銘柄の株価が引けるようにするため。コア銘柄スクリーニングの対象銘柄は`core/screener.py`側で別途REITを除外しており本変更の影響を受けない） |
| `tools/backfill_history.py` | 指定期間の過去営業日ぶんランキングを再生成し`gen_rankings`へupsert（アラート送信はしない。`--start`/`--end`指定可。既存日付は既定でスキップするため、価格データ修正後に再生成したい場合は`--force`で上書き。生成後に`check_price_freshness`で複数日にまたがるclose凍結（更新漏れ）を検査）|
| `core/rf_train_v3.py` | XGBoostの下落モデルを東証全銘柄×5年データで学習（金曜のみ。上昇モデルは廃止済み）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に下落確率をつけてランキング生成・DB保存。フェーズ5(優待権利落ち)→フェーズ7(米国ETFリードラグフィルター)→フェーズ8(相場リスク管制官) |
| `web/export_to_web.py` | Supabaseへランキング・日経 vs S&P500判定をエクスポート（Step 4）|
| `web/market_timing_alert.py` | LINE Messaging APIで日次プッシュ通知（Step 5b）。N225シグナル（平均下落確率→投資/キャッシュ）・🌐日経 vs S&P500相対強弱・🏦直近のEDINET大口保有動向（自己申告・過半数超(51%以上、スクイーズアウト対象で上値が見込めない)は除外、譲渡/売却も📈買い・📉売りを明示して表示。同一提出者の開示が期間内に複数あれば保有比率の変化を「5.2%→10.1%」で表示。開示日が新しい順を最優先し、同日内はウォッチ銘柄→法人/ファンド→保有比率が大きい順に優先し最大3件（通知疲れ防止のためLINEは絞り、残りはmicroCMSブログ「大口投資家の監視ブログ」（`kujira-watch/`、https://kujira-watch.com/ の詳細解説記事）のURLに委ねる。各行の下にはその銘柄の`/stocks/{code}`へのディープリンクを添える（`blog_stock_url()`。トップURLだけだと読者が銘柄を探し直す必要があったため）。流入はGA4で識別できるよう`utm_source=line&utm_medium=push`付き）、個人名の提出者は後回し）・🔍ユーザー別ウォッチ投資家の動き（`filer_watchlist`に登録した提出者名で部分一致照合し、その投資家がどの銘柄を動かしても通知。自己申告・過半数超は除外しない）・ユーザー別ウォッチリストのdp閾値アラート（ランキング本体の推奨が「🔴 売り検討」の銘柄は、個人のdp_sell_threshold設定値に関わらず必ず⚠️売り検討を表示。既定値20%はシステム全体の売り検討基準(drop_prob≥10%等)より緩いため、この上書きが無いと10〜20%の間で警告が沈黙するギャップが生じていた。閾値未達で変化のない銘柄は個別表示せず件数のみ要約し、前日比のdrop_prob変化があれば表示：通知疲れ対策）を配信 |
| `config.py` | 戦略パラメータの一元管理（閾値・フィルター値）|
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_scores 等）|
| `lib/db.py` | Supabase永続化層（gen_rankings / jpx_stock_list / yahoo_price_cache ほか）。`lib/supabase_client.py` のREST API経由（タイムアウト等の一時的なネットワーク失敗は指数バックオフで自動リトライ）|
| `lib/fundamentals.py` | point-in-time（先読みバイアスなし）ファンダメンタル再構成。`rank_stocks.py`/`rf_train_v3.py`/`backtest.py`で共用。`get_pit_fundamentals()`等は`rows`（銘柄のjquants_fin_summary全履歴）を渡すとDB問い合わせせずメモリ上でas_ofフィルタする（`rf_train_v3.py`が銘柄あたり約60サンプル日で呼ぶため、都度クエリだと数時間かかっていたのを銘柄ごと1クエリに削減）|
| `lib/sheets_helper.py` | Googleスプレッドシート連携 |
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（下落確率レンジ・予測多様性等の行レベル、rank_stocks/export_to_webで使用）＋`check_price_freshness`（複数日にまたがるclose凍結=更新漏れ検知、backfill_historyで使用）（alert-only：違反でも更新は止めずメール通知）|
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `lib/market_compare.py` | **日経 vs S&P500 相対強弱アドバイザー**。日経225とS&P500の20日・60日リターン差から「日本株優位／米国株優位／拮抗」を判定(売買シグナルには影響しない参考情報)。rank_stocksのフェーズ8bで判定し `data/market_compare.json` に保存、`gen_market_compare`経由でLINE(`market_timing_alert.py`)に表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。下落確率が低い順に選定。結果は `simulations/backtests/` に保存。`--drop-max`で下落確率上限、`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋下落確率閾値比較分析（ウォークフォワード対応） |
| `tools/screen_catalyst_candidates.py` | カタリスト候補スクリーン（GARP補助）。PBR<1.0・ROE<8%・自己資本比率>50%・流動性の「安い箱」抽出は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計（J-Quants財務データ使用）。通過候補に **利益の質フィルター(A/B)** で化粧決算（営業赤字・純利益>営業益×1.5）と斜陽事業（本業減益）を除外し、売上CAGR・営業利益率・会社予想方向で加減点。`data/catalyst_candidates.csv`（残）＋ `data/catalyst_excluded.csv`（除外理由付き・レビュー用）。`--no-quality` で品質フィルター無効 |
| `tools/catalyst_backtest.py` | カタリスト候補スクリーンのヒストリカルBT（point-in-time・disc_date≤基準日）。A/Bあり/なしで平均・勝率・大勝率を比較。データは J-Quants財務＋yahoo_price_cache |
| `lib/earnings_quality.py` | カタリスト候補の利益の質・本業方向性を判定（年次の営業益/売上/純益から化粧決算/斜陽を機械判定）。データ源は kabutan 優先、取れない環境（クラウドはkabutanがIPブロック）では J-Quants 実績にフォールバック |
| `lib/edinet.py` + `tools/scan_large_holdings.py` | **EDINET大量保有スキャナー**（イベント駆動）。EDINET APIから大量保有報告書(350)/変更報告書(360)を毎時スキャン（edinet_blog.yml、平日9:00-21:00 JST）して `edinet_large_holdings` に蓄積し、カタリスト候補と突合（構造的候補×実際の買い集め＝先回り候補）。突合時に自己申告（提出者≒対象企業）・過半数超(51%以上)・訂正報告書（既存開示の事後修正で実際の持分変動ではない）・譲渡/売却の報告を除外し、外部の買い集めだけ残す（`--no-exclude` で無効化可）。`is_sell_disclosure`/`is_individual_filer` は `market_timing_alert.py` のLINE通知セクションでも再利用（売却を除外せず方向性表示、個人名提出者を優先度で後回し）。買い/売りの方向判定はXBRLの直前保有割合(`holding_ratio_prior`)と現在の保有割合を比較して行い（概要欄の「譲渡/売却」等の文言が無い開示でも保有比率の減少を正しく売りと判定）、取得できない場合のみ概要欄のキーワードにフォールバックし、どちらも取得できない場合は買い/売りを推測せず方向性を表示しない。`EDINET_API_KEY` 必須 |
| `tools/filer_win_rate.py` | **投資家別「乗っかり勝率」バックテスト**（週次、GitHub Actions `filer_win_rate.yml`）。`edinet_large_holdings`全件から`is_noise_match()`（`tools/scan_large_holdings.py`と共用）で自己申告・訂正報告書・過半数超・売り方向を除いた「買い」開示のみを対象に、開示後63営業日(3ヶ月、`--hold`で変更可)保有した場合の株価騰落を投資家別に集計。開示からまだ`--hold`営業日経っていないイベントは結果未確定として除外（point-in-time、先読み無し）。サンプル数が少ない投資家の勝率は信頼性が低いため、投資家分類(13分類)別の勝率を事前分布としたベイズ収縮（仮想サンプル数`--shrink-k`、既定5）で「収縮後勝率」も算出。結果はSupabase `filer_win_rate`テーブルへupsert（kujira-watchの`/ranking`ページが参照）し、`--out`でCSVにも出力可 |
| `tools/reclassify_blog_articles.py` | **既存ブログ記事の投資家分類（dealType）一括再分類**（手動実行専用）。旧dealType体系（インサイダー買い/日系ファンド買い等）で公開済みの記事を、`classify_filer()`が返す新13分類へ移行する。各記事のstockCode+dealDateから`edinet_large_holdings`を逆引きしてfiler_nameを特定（同一銘柄・同一開示日に複数提出者がいて一意特定できない記事はスキップし一覧表示）、記事を全フィールド取得しdealTypeだけ書き換えて`update_article()`でmicroCMSをPATCH更新。`--dry-run`で変更内容の確認のみ可 |
| `tools/rewrite_thin_blog_articles.py` | **既存ブログ記事の本文リライト**（手動実行専用、Google Search Console「クロール済み-インデックス未登録」対策）。可視文字数（HTMLタグ・末尾の株価チャート`<figure>`を除いた文字数）が閾値未満の記事、または`--ids`で明示指定した記事を対象に、stockCode+dealDateから`edinet_large_holdings`を逆引きしてfact_sheetを再構築し、現行の`generate_article_body()`（保有比率の変化幅つき・650〜900字目標）で本文だけを再生成する。既存タイトル（アイキャッチ画像に焼き込み済み）と株価チャート`<figure>`はそのまま維持し、`update_article()`でmicroCMSをPATCH更新。閾値判定だけでは対象記事数（359件中357件）がGSC実測（104件）と大きく乖離するため、`--ids`でSearch Console提示のURL一覧に絞って実行するのが実運用上の想定（閾値判定は目安・動作確認用）。`--dry-run`・`--limit`併用可 |
| `lib/attention_score.py` | **クジラ注目度スコア**算出。「保有比率が高い」「急増」「アクティビスト」は見た目のインパクトはあるが実際の63営業日後リターンとは弱い負の相関〜無相関だった（`tools/filer_win_rate.py`と同じ手法で買い開示4,232件を検証、2026-08-15）ため、直感ではなく実績データで較正した「スコアカード」方式を採用。保有比率・保有比率の変化幅・推定取引金額の3つは5分位ビンの実績平均リターンに、投資家分類(13分類)は縮小推定(shrinkage, k=20)した実績平均リターンにそれぞれ変換し、Ridge回帰の重みで線形結合して「期待リターン」を予測、学習時の予測値分布のパーセンタイルへ変換して0〜100点にスケーリングする（`compute_attention_score()`）。「過去の取得回数」は検証したが有意な関係が見られず不採用。`web/publish_blog_articles.py`が新規記事の投稿時（買い方向のみ）に呼び出し、`tools/backfill_attention_score.py`が既存記事に遡及付与する |
| `tools/backfill_attention_score.py` | **既存ブログ記事へのクジラ注目度スコア一括付与**（手動実行専用）。`lib/attention_score.py`追加前に投稿済みの記事（売り方向を除く）を対象に、stockCode/dealDateから`edinet_large_holdings`を逆引き（同日複数提出者がいる場合は`estimate_deal_amount_oku()`で記事の`dealAmount`と突き合わせて特定）してスコアを計算し、`update_article()`でmicroCMSをPATCH更新。`--dry-run`で計算結果の確認のみ、`--force`で既にスコアがある記事も再計算 |
| `tools/backfill_investor_profiles.py` | **既存投資家の分類・プロフィール一括バックフィル**（手動実行専用）。`edinet_large_holdings`に登場する提出者のうち、kujira-watch `/investors/[filer]`の解説文（`edinet_filer_classification.profile`、800〜1000字程度）が未設定の投資家をまとめて埋める。`edinet_filer_classification`に未登録（未分類）の提出者は`classify_filer()`で分類してから、分類済みだが`profile`未生成の提出者は`get_filer_profile()`のみを呼び出す。日次パイプラインは新規に記事化した提出者のみ都度分類・生成するため、記事化されずに大量保有履歴だけ残っている既存提出者を埋めるためのスクリプト。`--limit`件数上限、`--sleep`秒間隔（デフォルト1秒、レート制限対策） |
| `web/publish_blog_articles.py` | **ブログ記事自動生成・投稿**（edinet_blog.yml、平日9:00-21:00 JST毎時、microCMSブログ「大口投資家の監視ブログ」`kujira-watch/`向け）。株価更新パイプライン(daily_alert.yml)からは独立しており、開示当日のうちに検出・記事化する。`market_timing_alert.get_recent_large_holdings`（自己申告・過半数超・訂正報告書を除外）からネタを取得し、保有比率の増減（取得できない場合のみ概要欄キーワード）で取得(買い)/売却(売り)の方向を判定（`is_sell_disclosure()`）した上で両方向とも記事化し、yfinanceの発行済株式数×株価×保有比率変化で取得・売却金額(億円)を概算（`shares_outstanding()`はyfinance側の一時的なレート制限対策として最大3回リトライし、`sharesOutstanding`が空ならJ-REIT等を想定して`impliedSharesOutstanding`にもフォールバックする。それでも株価・株式数のいずれかが取得できない銘柄はスキップ）。売り方向の記事はmicroCMSのスキーマ変更を避けるため`tags`に"売り"を追加して区別する（買い方向は従来通りtags不変）。プロンプト・見出しラベル（推定取得金額/推定売却金額）・末尾の推測文（「この取得が」/「この売却が」）も方向に応じて分岐させる。`classify_filer()`が提出者の投資家分類（個人/創業家の資産管理会社/公益・一般財団法人/プライムブローカー/アクティビスト/VC/PE・メザニンファンド/独立系ブティックAM/国内アセットマネジメント/外資系伝統運用会社/日系証券銀行/事業会社/その他）をSupabaseの`edinet_filer_classification`マスター（Web検索で確認済みの投資家分類テーブル、バックテスト分析とも共用）から参照し、未登録の提出者のみClaudeの一般知識で判定して結果をマスターへ保存（キーワード一致だけでは日系/外資やスペース無し個人名を判定できないため）。Claude（`ANTHROPIC_API_KEY`）には事実と分類済みdealTypeのみを渡して解説記事本文を生成しmicroCMSへ即時公開。事実の並置だけで終わらず投資家への示唆(so what)を加えられるよう、`gen_rankings`から開示日時点(point-in-time、記事公開時点のpost-hocスナップショットではない)の株価・下落リスク水準(高/やや高/中/やや低/低)を取得できた場合はプロンプトに文脈として渡し、その範囲内での意味づけを1文加えさせる（取得できない銘柄は従来通り事実のみ）。`ratio_change_pct()`が同一銘柄・同一提出者の過去開示（直近400日）から算出した保有比率の変化幅（ポイント）をfact_sheetへ渡し、過去開示が有れば「これまでの開示からXポイント増加/減少」、無ければ「直近400日以内に開示が確認できず実質的な新規保有（または大幅な保有再開）とみられる」という事実をプロンプトに含める（記事本文が同一投資家・同一銘柄でも毎回同じ言い回しの薄い内容にならないよう、既存で計算済みだが本文生成には使っていなかった実データを追加投入するSEO対策。GSC「クロール済み-インデックス未登録」対策として2026-08-14導入）。`get_company_description()`が対象企業の事業内容をClaudeの一般知識（`jpx_stock_list.description`にキャッシュ）から1文取得できた場合は冒頭の紹介文と保有比率の規模感（時価総額の一角を占める大株主、等）に自然に織り込む。`get_filer_profile()`が提出者のプロフィール（設立時期・運用方針・著名な投資事例など、800〜1000字程度）をClaudeの一般知識から取得し`edinet_filer_classification.profile`にキャッシュする（kujira-watch側`/investors/[filer]`の解説文として表示。情報が乏しい個人名義等は空文字のまま創作しない）。本文の最後には「この取得/売却が今後どんな意味を持ちうるか」の推測を必ず1文加えさせるが、事実と混同しないよう文頭に「※推測:」ラベルを付けさせ、事実として存在しない具体的計画やコメントの引用は創作しないよう明示的に指示する。金額が概算である旨・大量保有報告書制度の一般的な説明・「今後の動向を注視する必要がある」等の定型的な結びは、既に見出しや事実で伝わっているため本文で繰り返さないよう指示する（人間は事後にmicroCMS管理画面で修正する運用）。`build_price_chart_for_article()`が`yahoo_price_cache`から直近3ヶ月の終値を取得し、PIL（Pillowのみ、追加依存なし）で簡易な折れ線チャートPNGを描画してmicroCMSへアップロードし、本文HTML末尾に`<img>`タグとして埋め込む（株価取得・生成・アップロードのいずれかが失敗すればチャート無しで記事のみ投稿）。サイト上部のカテゴリフィルターはdealTypeの値をそのままカテゴリ名として使う構成にしており、microCMSに`category`フィールドは持たない（CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない）。記事タイトルはClaudeの自由生成ではなく`build_article_titles()`の決定的テンプレート（`銘柄名（コード）、提出者が保有比率X%に引き上げ/引き下げ｜大量保有報告書`、新規保有は`X%を新規保有`。60字超過時は提出者名を`…`で短縮）で組み立て、「銘柄名（コード）」「保有比率」「大量保有報告書」という検索語が必ず入ることを保証する（SEO/AIO 30日計画P1、2026-08-15）。本文の1文目も検索クエリへの直答文（`〜が保有比率をX%まで引き上げたことが大量保有報告書（EDINET）で分かりました。`）に固定してプロンプトで指示する。保有比率の変化幅は`ratioChangePct`（ポイント、売りは負値）としてmicroCMSにも送信し、フロントのファクトボックス表示に使う。`bodyEn`（kujira-watch `/en`向け英訳）は同一回のClaude呼び出しでJA本文と同時生成し（事実のズレとAPI呼び出し回数増加を防ぐ）、英語タイトル用のローマ字名`stockNameEn`/`filerNameEn`も同時に返させて英語版テンプレタイトル（`titleEn`）に使う。同一銘柄・同一開示日の重複投稿は`dealAmount`も突き合わせて判定（実運用で発生: 同日に複数提出者の別開示があるケースを誤って重複扱いしないため）。買い方向の記事には`lib.attention_score.compute_attention_score()`で算出した**クジラ注目度**（`attentionScore`0-100・`attentionReasons`）を付与する（詳細は`lib/attention_score.py`の行を参照。売り方向は対象外）。既存記事の更新（`update_article()`、`tools/reclassify_blog_articles.py`等が使用）は2026-08-14よりPUT（完全上書き）からPATCH（差分更新）に切替（APIキーの権限変更でPUTが拒否されるようになったため）。アイキャッチ画像は`PEXELS_API_KEY`が設定されていれば、投資家分類に応じたPexels写真（`EYECATCH_QUERY_BY_CATEGORY`、銘柄固有の写真は現実的でないため分類のイメージに合う汎用写真を使用）に黒帯＋ニュースカード型テキスト（売買方向バッジ＋開示日／提出者名／銘柄名＋保有比率、Noto Sans CJK Bold太字白文字の3段組み。自由記述のタイトル文字列ではなく構造化した事実を焼き込むことでGoogle Discoverのカード面での視認性を狙う。2026-08-15、`generate_eyecatch_image()`/`build_eyecatch_for_article()`のシグネチャを`(category, card)`に変更）を合成し、microCMSのメディアアップロードAPI(`{domain}.microcms-management.io`)へアップロードして`eyecatch`フィールドへ設定する（`PEXELS_API_KEY`未設定・取得失敗時は画像無しで記事のみ投稿）。`--dry-run`で投稿せず内容確認のみ可（アイキャッチ生成もスキップ）。`MICROCMS_SERVICE_DOMAIN`/`MICROCMS_API_KEY`（書き込み権限）必須、未設定ならスキップ |
| `web/x_client.py` | **ブログ新着記事のX(Twitter)自動投稿**。`publish_blog_articles.py`の`main()`から投稿完了後に呼び出され、その回に投稿した記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページの「注目」枠`getFeaturedArticles()`と同じロジック：直近プール(日付優先→同日内はdealAmount降順で取得)の中からクジラ注目度`attentionScore`が高い順に先頭3件採用、未算出はプール内最下位・同点は`dealAmount`で比較。プール自体は金額だけで並べ替えないため、投稿数が少ない日に数日前の大型取引が「注目」を占有し続けることもない）にも含まれる記事だけを、金額規模(`dealAmount`)が大きい順にX API v2(`POST /2/tweets`)へ投稿する（その日たまたま一番大きいというだけでサイト上は目立っていない小粒な開示がXにだけ投稿される事態を防ぐため。積集合が無い地味な日は0件のこともある）。`publish_blog_articles.generate_price_chart_image()`（記事本文に埋め込むのと同じPillow製の株価チャートPNG）を再生成し、v1.1 `media/upload`エンドポイントへアップロードして`media_ids`として添付する（チャート生成・アップロードのいずれかに失敗すればテキストのみで投稿を続行）。加えて**「本日のクジラ」日次サマリー投稿**（`post_daily_summary()`、SEO/AIO 30日計画P4）: 毎時バッチのうち21時JST（12時UTC）の最終便のみ、その日(dealDate)の全記事をmicroCMSから金額降順で取得し、件数・合計金額・最大買い増し・最大売却を定型フォーマットで1ポストする（`/date/{日付}`へのUTM付きリンクを添付。時刻ガードが外部ストレージ無しの1日1回重複ガードを兼ねる。0件の日は投稿しない）。認証はOAuth 1.0a User Context（`X_API_KEY`/`X_API_KEY_SECRET`/`X_ACCESS_TOKEN`/`X_ACCESS_TOKEN_SECRET`、X Developer Portalで「Read and Write」権限のAppを作成し取得。ツイート投稿・画像アップロードの両エンドポイントで共用）。いずれか未設定なら投稿をスキップ（他のステップに影響しない）。`--dry-run`実行時は呼び出されない |
| `video/build_script.py` | **自動動画投稿の台本生成**（video_post.yml、平日21:30 JST・1日1回）。microCMSに直近36時間で新規公開された記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページ「注目」枠と同じロジック）にも含まれるものを`dealAmount`降順で1件だけ選び（`pick_article()`。X投稿と同じ「新着×注目」の積集合で、サイト上目立っていない小粒な開示だけが動画化される事態を防ぐ。積集合が空の日は動画を作らない）、記事本文＋Supabaseにキャッシュ済みの補足事実（`get_company_description()`の事業内容・`get_filer_profile()`の投資家プロフィール、どちらもpublish_blog_articles.pyが生成したもの）だけを根拠に、Claudeで**ナレーション付き台本**を生成する。台本は hook→company（どんな会社か）→deal（金額・保有比率）→filer（どんな投資家か）→change（前回からの変化）→outlook（今後の推測）→cta の7シーン（`SECTION_SPEC`）で、各シーンは `narration`（読み上げ文、50〜90字）と `caption`（画面に出す字幕、26字以内）の対。字数超過は一度だけ作り直し、それでも超える場合はcaptionは末尾を詰め、narrationは句点境界で切る（`_trim_narration()`、文の途中切りは読み上げが不自然になるため）。kindはClaudeの出力に頼らず期待順で上書きする（`_flatten_scenes()`） |
| `video/background.py` | **背景映像の調達（Pexels Videos API）**。自然系9クエリ＋人物系3クエリ（オーナー指定）のプールから、人物枠1本を確保しつつ縦向き・7秒以上・80MB以下の動画を最大4本ダウンロードし（`fetch_pool()`）、各シーンへランダム割当する（`assign_backgrounds()`、プールが2本以上あれば隣接シーンで同じ映像を使わない＝セリフの区切りごとに背景が切り替わりカット感が出る）。Pexelsは無料・商用可・クレジット不要で、人物素材も装飾背景としての利用はライセンス上許可されている（人物がサービスを推奨しているかのような見せ方のみ禁止）。`PEXELS_API_KEY`未設定・全滅時は空リストを返し、Remotion側が紺のグラデーション背景にフォールバックする |
| `video/remotion/` | **縦動画のRemotionプロジェクト**（React/TypeScript）。コンポジション`ArticleShort`は1080x1920・30fpsで、**尺は固定ではなく各シーンのナレーション音声の長さで決まる**（`calculateMetadata`と`ArticleShort.tsx`が同じ式`sceneDurationSec()`で総フレーム数を算出。音声が無い場合は読み上げ文字数から概算）。TikTok/Shorts運用の定石を反映: (1)表示はすべて`safeArea`内（下部470px・右190pxはアプリのUIに隠れるため）、(2)冒頭はブランドや日付の前置き無しで金額を画面いっぱいに叩き込む（最初の1秒で離脱が決まるため）、(3)無音視聴者向けにナレーション要約の大型字幕を常時表示、(4)上部に進行バー（残り時間が見えると完走率が上がる）、(5)背景はズームドリフト＋波で1フレームも完全静止させない、(6)締めはブランド画面でループ再生に自然に繋がる構成。outlookシーンの字幕には「※ここから先は推測」ラベルを付けて事実と区別する。配色は`src/theme.ts`が`kujira-watch/src/app/globals.css`と同じブランド色を持ち、買い=金・売り=赤のアクセント。日本語フォントはOS側のNoto Sans CJK（CIはapt導入、macOSはHiragino）を使い、レンダリングがネットワークに依存しない |
| `video/render.py` | props JSONを`npx remotion render`へ渡してmp4を書き出す薄いラッパ。ナレーション音声（tts.pyが生成したwav）はRemotionの`staticFile()`経由でしか参照できないため、`video/remotion/public/`へコピーしてからレンダリングし、終了後に削除する（`_stage_audio()`。見つからない音声はそのシーンだけ無音にして続行）。`articleId`/`articleTitle`は投稿テキスト専用でコンポジションのpropsには無いため除外して渡す（`NON_PROP_KEYS`）。初回実行時のみRemotionがChrome Headless Shell(約150MB)を自動ダウンロードする |
| `video/youtube_client.py` | **YouTube Shortsへの自動アップロード**（YouTube Data API v3のresumable upload）。認証はOAuth 2.0リフレッシュトークン方式（`YOUTUBE_CLIENT_ID`/`YOUTUBE_CLIENT_SECRET`/`YOUTUBE_REFRESH_TOKEN`。ローカルで`video/youtube_auth.py`を1回実行して取得）。YouTubeは縦長かつ3分以内の動画を自動的にShortsとして扱うため専用フラグは不要だが、保険としてタイトル・説明文に`#Shorts`を入れる。説明文の記事URLには`utm_source=youtube`を付与しGA4でShorts経由の流入を識別できるようにする。いずれかの環境変数が未設定ならスキップ（TikTok側は止めない） |
| `video/tiktok_client.py` | **TikTokへの自動アップロード**（Content Posting API v2）。認証はOAuth 2.0リフレッシュトークン方式（`TIKTOK_CLIENT_KEY`/`TIKTOK_CLIENT_SECRET`/`TIKTOK_REFRESH_TOKEN`。ローカルで`video/tiktok_auth.py`を1回実行して取得）。**TikTokはアプリ審査を通るまで一般公開の投稿ができない**ため、既定ではinbox（下書き）へアップロードしTikTokアプリの通知からオーナーが手動公開する運用にする。審査通過後に`TIKTOK_DIRECT_POST=1`を設定すると直接公開に切り替わり、その際は必須の`creator_info`クエリでアカウントが一般公開を許可しているか確認し、許可が無ければ`SELF_ONLY`へ落とす。キャプション内のURLはTikTokではリンクにならないためプロフィール誘導の文言にする。未設定ならスキップ（YouTube側は止めない） |
| `video/tts.py` | **ナレーション音声の合成（VOICEVOX）**。無料・登録不要・商用利用可（クレジット表記のみ必須）の日本語音声合成エンジンで、既定の話者はずんだもん（speaker=3、`TTS_SPEAKER`で変更可、`TTS_SPEED`既定1.15倍速）。エンジンはHTTPサーバ（既定 http://127.0.0.1:50021、`VOICEVOX_URL`で変更可）として動き、CIでは公式Dockerイメージ`voicevox/voicevox_engine:cpu-ubuntu20.04-latest`をジョブ内で起動する。`narrate_sections()`が各シーンのnarrationをwav化して`audio`/`durationSec`（ffprobe計測、無ければ文字数から概算）を書き込み、この長さがそのままシーンの尺になる。エンジンに繋がらない・1シーンでも合成に失敗した場合は全体を無音扱いにして動画生成は続行する（一部だけ音が出る動画はかえって不自然なため）。クレジット表記「VOICEVOX:ずんだもん」はCTAシーン・YouTube説明文・TikTokキャプションに自動で入る。※当初はGoogle Cloud TTSで実装したが、GCPプロジェクトに請求先が未設定でAPIを有効化できずVOICEVOXへ切替（2026-08-15） |
| `video/publish_video.py` | 自動動画投稿のオーケストレーター（台本生成→VOICEVOXでナレーション合成→レンダリング→YouTube/TikTokへ投稿）。対象記事が無い日は何も投稿せず正常終了する。ナレーション合成に失敗しても無音で続行する。片方のプラットフォームが未設定・失敗でももう片方は続行し、投稿先のSecretsが1つも無い場合は動画生成のみで正常終了（設定漏れの誤検知で毎日赤くならないように）、Secretsがあるのに全滅した場合のみ異常終了する。`--dry-run`（台本まで）/`--render-only`（mp4書き出しまで）/`--keep-video`（投稿後もmp4を残す）で段階的に確認できる |
| `video/youtube_auth.py` / `video/tiktok_auth.py` | リフレッシュトークンを取得する**ローカル1回きり**のスクリプト（CIでは使わない）。YouTubeはloopback(http://localhost:8765)で認証コードを受け取り、TikTokはhttpのリダイレクトURIを受け付けないため、登録済みhttps URLへのリダイレクト後に`?code=`を手で貼り付ける方式にしている |
| `tests/test_fundamentals.py` | point-in-timeファンダ（`lib/fundamentals.py`）のユニットテスト。先読みバイアス防止（as_of日より後の開示を含めない）を確認（6件）|
| `tests/test_earnings_quality.py` | 利益の質フィルター（化粧・赤字・減益・加減点）のユニットテスト（8件）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（銘柄コード絞り込み正規表現の新形式コード対応込み、13件）|
| `tests/test_fetch_history.py` | 株価キャッシュ更新の銘柄コード収集ロジック（既存コード+JPX最新リストの和集合・JPX取得失敗時のフォールバック・J-REIT含む市場フィルター）のユニットテスト（4件）|
| `tests/test_data_sanity.py` | QA（データ整合性・価格凍結検知）のユニットテスト（14件）|
| `tests/test_market_compare.py` | 日経 vs S&P500 相対強弱アドバイザーのユニットテスト（4件）|
| `tests/test_market_timing_alert.py` | LINE通知の大口保有動向セクション（開示日優先ソート・根拠なき買い/売り推測の抑制込み）・ウォッチリストdp閾値判定（ランキング本体の推奨ラベルとの矛盾防止・売り閾値ギャップの上書き・通知疲れ対策の要約表示・前日比表示込み）・投資家ウォッチ（提出者名の部分一致照合・大口保有動向セクション生成）のユニットテスト（26件）|
| `tests/test_scan_large_holdings.py` | EDINET大量保有スキャナーの判定ロジック（売却検知・保有比率増減による方向判定・個人名判定・過半数超除外・訂正報告書除外・ノイズ除外）のユニットテスト（11件）|
| `tests/test_reclassify_blog_articles.py` | 既存ブログ記事の投資家分類一括再分類ツールのユニットテスト（microCMSのdealType配列/空配列/None正規化。空配列でIndexErrorになっていた本番バグの再発防止・PUT用ペイロードのメタ情報除去）のユニットテスト（5件）|
| `tests/test_rewrite_thin_blog_articles.py` | 既存ブログ記事の本文リライトツールのユニットテスト（HTMLタグ・株価チャート`<figure>`を除いた可視文字数算出・PATCH用ペイロードが本文のみを含むこと）のユニットテスト（4件）|
| `tests/test_publish_blog_articles.py` | ブログ記事自動投稿の判定ロジック（金額概算・発行済株式数取得のリトライ/impliedSharesOutstandingフォールバック・記事生成JSONパース・投資家分類マスター参照とClaudeフォールバック/保存・売り方向のtagsタグ付け/プロンプト分岐・重複防止・権限エラー時の早期打ち切り・投稿/更新のセレクト配列形式への自動リトライ・非文字列フィールド(eyecatch等)の型不一致時は除外して再送信・PIT文脈(株価/下落リスク水準)のプロンプト反映・保有比率変化幅(ポイント)/新規保有のプロンプト反映・事業内容の取得/キャッシュ・投資家プロフィールの取得/キャッシュ/空欄時の非創作・「※推測:」ラベル付き推測文の要求・英訳(bodyEn/ローマ字名)のプロンプト要求・決定的テンプレによるタイトル組み立て（買い/売り/新規保有・60字超過時の提出者名短縮・英語名フォールバック）・冒頭アンサー文のプロンプト指示・ratioChangePct（売りは負値）のpayload付与・dealAmount照合による重複判定・PATCH更新への型不一致自動リトライ・アイキャッチ画像生成/アップロード・株価チャート画像生成/アップロード/本文埋め込み・買い方向のみクジラ注目度attentionScore/attentionReasonsを付与・ホームページ「注目」枠と同じattentionScore降順（同点はdealAmount）での注目記事id抽出）のユニットテスト（68件、ネットワークは全てモック）|
| `tests/test_attention_score.py` | クジラ注目度スコア（`lib/attention_score.py`）のユニットテスト（分位ビンの境界値判定・パーセンタイル変換の単調性とクリップ・星評価の閾値・実績データ上有利な組み合わせがスコア高くなること・未知カテゴリのフォールバック）（7件）|
| `tests/test_supabase_client.py` | Supabase REST APIクライアントのリトライ挙動（一時的なネットワーク失敗時のバックオフ再試行・最終失敗時に呼び出し元を落とさないこと）のユニットテスト（3件）|
| `tests/test_x_client.py` | X(Twitter)自動投稿のユニットテスト（ツイート本文の組み立て・文字数超過時の切り詰め・認証情報未設定時のスキップ・「注目」記事idとの積集合に絞った上での金額規模順選定・「注目」に含まれない記事の除外・dry-run記事の除外・チャート画像のアップロード/投稿への添付・チャート生成失敗時のテキストのみ投稿へのフォールバック・「本日のクジラ」日次サマリー（本文組み立て・0件時スキップ・totalCountでの件数補正・最終便の時刻ガード）（23件、ネットワークは全てモック）|
| `tests/test_video_pipeline.py` | 自動動画投稿パイプライン（`video/`）のユニットテスト（「新着×注目」の積集合からの金額規模順選定・積集合が空なら動画を作らない・microCMSセレクト型dealTypeの配列アンラップ・tagsからの売り方向判定・filerName未設定時の空文字化・本文末尾の保有比率抽出とフォールバック・台本JSON(hook/sections/closing)のフラットなシーン列への展開・Claudeが誤ったkindの期待順上書き・字数超過時の1回だけの作り直しと最終的な切り詰め・ナレーションの句点境界での切り詰め・sections不足時の破棄・APIキー未設定時のスキップ・VOICEVOXエンジン未接続時の無音フォールバック・合成成功時のaudio/durationSec書き込み・1件でも失敗したら全体無音・YouTubeタイトル/説明文の組み立てとUTM付与・タイトル超過時も#Shortsを残す切り詰め・TikTokキャプションの組み立てと切り詰め・既定でinbox(下書き)エンドポイントを使うこと・直接投稿時に一般公開が許可されていなければSELF_ONLYへ落とすこと・認証情報未設定時のスキップ）（29件、ネットワークは全てモック）|

---

## S買い 発令条件（passes_buy_filter + rank_stocks.py フェーズ5・7・8）

下落モデルのみに一本化済み（上昇モデル・netスコアは廃止。詳細は `dev_log.md` 参照）。

品質フィルター（`passes_buy_filter`）:

| 条件 | 値 | 意図 |
|---|---|---|
| 株価 ≥ | 300円 | 低位株除外 |
| 3ヶ月モメンタム ≥ | +8% | 上昇トレンド確認（5%→8%: 10期間BTで勝率+7pp）|
| 2年モメンタム | プラス | 長期下落株を除外（2年<0は勝率25%・avg-3.1%）|
| 2年トレンド R²（504日） ≥ | 0.4 | 長期トレンド一貫性確保（R²<0.4は勝率18%・avg-2.8%）|
| RSI（14日） | < 75 | 過熱除外のみ（下限撤廃: 30〜45帯が有効と判明）|
| 出来高比 vr2060 ≥ | 1.0 | 出来高増加トレンド確認 |
| 直近20日ボラ (vol20) ≤ | 22% | 高ボラ時は見送り（BT: vol>22%は平均▼0.9pp）|
| 連続下落日数 ≤ | 3日 | 急落継続銘柄の除外 |
| 60日ドローダウン ≥ | −15% | 深い下落銘柄の除外 |
| 20日平均売買代金 ≥ | 50百万円 | 流動性確保（板薄銘柄除外）|

モデル予測フィルター（`recommend_from_scores`）:

| 条件 | S買い |
|---|---|
| 下落確率 | < 8% |
| 年率ボラティリティ | ≤ 20% |

フェーズ5・7・8 追加フィルター（`rank_stocks.py`）:
- フェーズ5: 株主優待権利落ち21日前以内の銘柄はS買い→方向感なしに降格
- フェーズ7: 対応する米国セクターETF（XLK/XLF/XLI/XLB/XLV/XLY）の前日リターンがマイナスならS買い→方向感なしに降格。リードラグ効果（US→JP翌日）を活用。21,416サンプル(2023-2026)で全26ペア正相関・avg +0.64pp効果を確認。キャッシュは `data/sector_map.json`。
- フェーズ8: 相場リスク管制官がリスクオフ地合いと判定した日は、S買いを全件見送り（自動防御）

**推奨ラベル**:
- 💎 買い: QV条件+ファンダ品質+モデルスコア全条件クリア
- 🔴 売り検討: drop_prob≥10% / drawdown60<-20% / 連続下落≥5日
- —: それ以外

## スクリーナー条件（screener.py、現在は手動実行専用）

`core/screener.py` は2026-08-01に日次パイプラインから除外済み（出力`data/screeners/*.csv`が
`rank_stocks.py`から読まれておらず、全銘柄価格取得を二重に行うだけの無駄な処理だったため）。
以下は`screener.py`単体を手動実行した場合の条件で、現在の自動配信ランキングには**適用されない**
（自動配信の実フィルターは上記「S買い 発令条件」表のみ）。

| 条件 | 値 |
|---|---|
| 株価 ≥ | 300円 |
| 20日平均売買代金 ≥ | 50百万円 |
| セクター集中除外 | 同一業種3銘柄以上でセクター全除外（バブル兆候回避）|

> 3ヶ月相対強度フィルター（`rel_strength_min`引数）は`apply_screener_v1`内で既に未使用（実装上の
> デッドパラメータ）だったため、上表からは削除した。下落モデル一本化後のバックテスト再検証は未実施
> （別環境でのbacktest.py実行が必要。詳細は `dev_log.md`）。

---

## モデル詳細（core/rf_train_v3.py）

### 学習データ
- 対象：東証プライム・スタンダード全銘柄（約3,500〜4,000銘柄）
- 期間：過去5年分（約1,800日）
- サンプリング：20営業日ごと（自己相関低減）
- 分割：cutoff日より前→学習 / 以降→テスト（ウォークフォワード）

### 特徴量（61次元 = 54基本 + クロスセクション7次元）

**テクニカル10**: ret5, ret20, ret60, ret90, ma5_25, ma25_75, rsi, vol20, vol60, pos52

**トレンド反転5**: drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir

**出来高3**: vr520, vr2060, vsurge

**日経マクロ3**: 日経225の5/20/60日リターン

**60日系列要約7**: autocorr_lag1, skew, max_ret, min_ret, pos_ratio, trend_slope, recent_vs_early

**日経相対アルファ4**: rel5, rel20, rel60, alpha_momentum

**ファンダメンタル11**: per, pbr, roe, days_to_earnings, days_since_div_ex, sin/cos_month, div_yield, eps_growth, dps_growth

**マクロ拡張4**: vix, us5, us20 (SP500), jpy5 (USD/JPY)

**新規IB特徴量8**: amihud_f (非流動性), fx_beta, jpy5, eps_surprise, bps_growth, piotroski, payout, accruals (Sloan正確版)

**EDINET1**: edinet_hold_f（大量保有報告書の保有比率）

**クロスセクショナルランク7**: cs_ret5, cs_ret20, cs_ret60, cs_rsi, cs_vol20, cs_pos52, cs_sector_ret60

### 予測ラベル
- 下落モデル：63日後（約3ヶ月）に **15%以上下落**（絶対リターン、DROP_THRESHOLD=15.0）
- 上昇モデル（rf_model.pkl）は廃止済み。下落モデルのみ学習・保存する

### ウォークフォワードモデル

先読みバイアスなしのバックテストのため、期間開始日に応じて学習済みモデルを切り替える。

| 期間開始日 | 使用モデル（cutoff） | テストAUC（下落）|
|---|---|---|
| ≥ 2025-07-01 | rf_drop_model_2025-07-01.pkl | 0.818 |
| ≥ 2025-05-01 | rf_drop_model_2025-05-01.pkl | 0.803 |
| ≥ 2025-03-01 | rf_drop_model_2025-03-01.pkl | 0.806 |
| < 2025-03-01 | rf_drop_model.pkl（cutoff 2026-01-01）| 0.766 |

キャリブレーション：IsotonicRegression で確率値を実績頻度に補正済み

---

## ランキングロジック（core/rank_stocks.py）

下落確率(%)の昇順（低い順）でランキングし、`drop_prob < 8%` を買い候補の主条件とする（詳細は上の「S買い 発令条件」）。

> **下落確率の表示について**：モデルの確率はIsotonic較正の特性上、数十段の階段値（例: 3,566銘柄が約31個の値に収束）になり、小数第1位まで出すと多数の銘柄が同じ値（例「20.3%」）に見えてしまう。そのためWeb・メール・LINEの画面表示では小数%ではなく **高 / やや高 / 中 / やや低 / 低** の5段階で示す（しきい値: 30/22/14/7%）。並び順・スコア計算は引き続き数値の下落確率を使用。

### ハードフィルター（除外）
- 連続下落日数 > 3日（down_streak > 0.15）
- 直近60日高値から-15%超（drawdown60 < -0.15）

---

## データ永続化（Supabase / Postgres）

全データを **Supabase（Postgres）** に一元管理する（旧 `stock_alert.db` SQLite から全面移行済み）。
`lib/db.py` が Supabase REST API（`lib/supabase_client.py`）経由で読み書きする。GitHub Actions の
DBキャッシュは廃止。

| テーブル | 内容 |
|---|---|
| `gen_rankings` | 毎日のランキングスコア（コード・下落確率・推奨・rank）|
| `jpx_stock_list` | 業種分類・優待月ほかメタ |
| `gen_market_compare` | 日経 vs S&P500 相対強弱判定 |
| `jquants_fin_summary` | 四半期財務サマリ（EDINET決算XBRLから抽出。テーブル名は旧J-Quants由来）|
| `yahoo_price_cache` | 株価履歴キャッシュ（バックテスト高速化用）|
| `yahoo_market_index` | VIX/S&P500/USDJPY 日次 |
| `edinet_large_holdings` | EDINET大量保有/変更報告書の日次蓄積（先回り突合用）|
| `edinet_filer_classification` | EDINET提出者(投資家)の分類マスター（個人/創業家の資産管理会社/公益・一般財団法人/プライムブローカー/アクティビスト/VC/PE・メザニンファンド/独立系ブティックAM/国内アセットマネジメント/外資系伝統運用会社/日系証券銀行/事業会社/その他。Web検索で確認済みのconfidence='high'と、Claude推測のみのconfidence='low'を区別。publish_blog_articles.pyのdealType分類とバックテスト分析で共用）|
| `edinet_filer_summary`（ビュー） | `edinet_large_holdings`×`edinet_filer_classification`を投資家(filer_name)単位に集計したビュー（保有開示件数・最終開示日・分類）。kujira-watch（`kujira-watch/src/lib/investors.ts`）の`/investors`一覧・サイトマップ生成が参照。投資家は600件超あり`edinet_large_holdings`の生データを直接集計すると1000行上限に掛かるため、1投資家1行に事前集計したビュー経由で取得する |
| `ext_tdnet_disclosures` | TDnet適時開示（やのしん・⚠️個人運営ソースのため `ext_` で隔離）|
| `jpx_short_selling` | JPX空売り残高報告（0.5%以上）|
| `jpx_margin_balance` | JPX個別銘柄信用取引週末残高 |
| `line_chat_history` | LINE Bot会話履歴（直近3往復、文脈保持用） |
| `line_users` | LINE Bot登録ユーザー |
| `dp_watchlist` | ユーザー別ウォッチ銘柄・dp閾値（LINE Bot）|
| `filer_watchlist` | ユーザー別ウォッチ投資家（EDINET提出者名。銘柄は問わずその投資家の保有比率増減を通知、LINE Bot）|
| `filer_win_rate` | `tools/filer_win_rate.py`が週次で再計算する投資家別「乗っかり勝率」（買い開示後63営業日の勝率・収縮後勝率・平均リターン・大勝率）。kujira-watchの`/ranking`ページが参照 |

全銘柄スクリーン（カタリスト候補）は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計する
（REST per-code を避け高速化）。

ランキングCSV（`data/rankings/`）と スクリーナーCSV（`data/screeners/`）も日付付きで保存されるがgitignore対象。

---

## 外部API・データソース一覧

### 利用API

| API | 取得データ | 用途 | 利用ファイル |
|---|---|---|---|
| **Yahoo Finance** (非公式REST) | 株価OHLCV（日次）、日経225/VIX/S&P500/USD/JPY | テクニカル特徴量・マクロ特徴量・バックテスト | `lib/utils.py` (`get_prices`, `get_market_index_df`) |
| **kabutan.jp** (スクレイピング) | PER/PBR/ROE、株主優待月、業績テキスト | ファンダ特徴量・NLP感情分析 | `lib/utils.py`, `lib/alt_data.py`, `lib/kabutan_earnings.py` |
| **EDINET API v2** | 大量保有報告書(350)/変更報告書(360)、有報/四半期報の決算XBRL(BS/PL/CF) | 先回りシグナル・財務サマリ（EPS/BPS/ROE/CFO/売上/営業益/予想）本体 | `lib/edinet.py`, `lib/edinet_financials.py`, `tools/scan_large_holdings.py`, `tools/fetch_edinet_financials.py` |
| **TDnet適時開示** (やのしんWEB-API・⚠️個人運営) | 適時開示（業績修正/増配/自社株買い/M&A等のカタリスト） | 企業イベント情報（LINE通知用）。停止リスク隔離のため `ext_` テーブルに保存 | `lib/tdnet.py`, `tools/fetch_tdnet.py` |
| **JPX 空売り残高/信用取引残高** (公式Excel/CSV) | 空売り残高報告(0.5%以上)、個別銘柄信用週末残高 | 需給シグナル（逆張り/買い残） | `lib/jpx_market_data.py`, `tools/fetch_jpx_market.py` |
| **JPX 東証上場銘柄一覧** (Excel) | 銘柄コード・名前・市場区分・33業種分類 | スクリーニング母集団・セクター分類 | `lib/utils.py`, `core/screener.py` |
| **yfinance** | セクターマッピング（米国ETF対応用） | 米国ETFリードラグフィルター（フェーズ7） | `core/rank_stocks.py` |
| **Supabase REST API** | 全テーブルCRUD | データ永続化（DB一元管理） | `lib/supabase_client.py` |
| **Claude API** (Anthropic) | テキスト生成 | 決算テキスト感情分析（Haiku × kabutan） | `lib/nlp_sentiment.py` |

### 特徴量が使うデータと出所

| カテゴリ | 特徴量 | データ出所 |
|---|---|---|
| **テクニカル (10)** | ret5/20/60/90, ma5_25, ma25_75, rsi, vol20/60, pos52 | Yahoo Finance 株価 |
| **トレンド反転 (5)** | drawdown60, from_hi52, down_streak, momentum_accel, ma_cross_dir | Yahoo Finance 株価 |
| **出来高 (3)** | vr520, vr2060, vsurge | Yahoo Finance 出来高 |
| **日経マクロ (3)** | nk5, nk20, nk60 | Yahoo Finance 日経225 |
| **60日系列要約 (7)** | autocorr, skew, max/min_ret, pos_ratio, slope, recent_vs_early | Yahoo Finance 株価 |
| **相対アルファ (4)** | rel5/20/60, alpha_momentum | Yahoo Finance 株価＋日経 |
| **ファンダメンタル (11)** | per, pbr, roe, earn_feat, div_ex_feat, sin/cos_month, div_yield, eps/dps_growth, dividend_relevant | jquants_fin_summary (EPS/BPS/ROE/決算日), kabutan (優待月) |
| **マクロ拡張 (4)** | vix, us5, us20, jpy5 | Yahoo Finance (^VIX, ^GSPC, JPY=X) |
| **IB特徴量 (8)** | amihud, fx_beta, jpy5, eps_surprise, bps_growth, piotroski, payout, accruals | Yahoo Finance (株価/出来高/為替), jquants_fin_summary (CFO/NP/TA/equity) |
| **EDINET (1)** | edinet_hold_f | edinet_large_holdings（大量保有報告書の保有比率） |
| **クロスセクショナル (7)** | cs_ret5/20/60, cs_rsi, cs_vol20, cs_pos52, cs_sector_ret60 | 上記テクニカル特徴量の日次グループ内正規化 |

### フィルターが使うデータ

| フィルター | 条件 | データ出所 |
|---|---|---|
| **品質フィルター** (`passes_buy_filter`) | 株価≥300, drawdown60≥-20%, down_streak≤4日, RSI<80, 売買代金≥50M | Yahoo Finance 株価・出来高 |
| **💎買い条件** (`recommend_from_scores`) | QV条件(Piotroski≥6/9, pos52<45%, EPS surprise>2% or BPS成長+) + 品質(CFOマージン>0, レバレッジ<5x) + drop_prob<8%, vol≤20%, ret90>-25%, 売買代金≥50M, bear時は💎抑制 | モデル予測＋jquants_fin_summary |
| **🔴売り検討** (`recommend_from_scores`) | drop_prob≥10% / drawdown60<-20% / 連続下落≥5日（いずれか該当で警告） | モデル予測＋株価データ |
| **優待フィルター** (フェーズ5) | 権利落ち21日前以内→S買い降格 | kabutan 優待月 |
| **米国ETFフィルター** (フェーズ7) | 対応セクターETF前日リターン<0→S買い降格 | Yahoo Finance (XLK/XLF/XLI等) |
| **レジーム調整** | 日経20日<-5%→下落相場、VIX>30→高恐怖 | Yahoo Finance (日経/VIX) |
| **カタリストスクリーン** (RPC) | PBR<1.0, ROE<8%, 自己資本比率>50%, 売買代金≥指定値 | jquants_fin_summary |
| **利益の質フィルター** (A/B) | 営業赤字/化粧決算/本業減益を除外 | jquants_fin_summary (営業益/売上/純利益) |
| **EDINET突合** | 大量保有報告×カタリスト候補マッチ（自己申告・過半数超(51%以上)除外。売りは方向性表示のため除外しない） | EDINET API |

---

## セットアップ

### 必要なSecrets（GitHub Settings → Secrets）

| Secret名 | 内容 |
|---|---|
| `GMAIL_ADDRESS` | 送信元Gmailアドレス |
| `GMAIL_APP_PASSWORD` | Gmailアプリパスワード |
| `SUPABASE_URL` | Supabase プロジェクトURL（全データ永続化の宛先）|
| `SUPABASE_SERVICE_KEY` | Supabase service_role キー（バックエンド書込用）|
| `EDINET_API_KEY` | EDINET API v2 サブスクリプションキー（daily_alert.yml Step 2d + edinet_blog.yml毎時の大量保有スキャン用。未登録ならスキャンはスキップ）|
| `MICROCMS_SERVICE_DOMAIN` | `kujira-watch`（大口投資家の監視ブログ）用microCMSサービスドメイン（edinet_blog.yml: ブログ記事自動投稿。未登録ならスキップ）|
| `MICROCMS_API_KEY` | 同上・書き込み権限付き＋メディアアップロード権限付きAPIキー（アイキャッチ画像のアップロードに使用）|
| `PEXELS_API_KEY` | ブログのアイキャッチ画像生成用（Pexels検索API）。未登録ならアイキャッチ無しで記事のみ投稿 |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` | YouTube Shorts自動投稿用（video_post.yml）。Google Cloud ConsoleでYouTube Data API v3を有効化し「デスクトップアプリ」のOAuthクライアントを作成、ローカルで`python video/youtube_auth.py`を1回実行してリフレッシュトークンを取得する。未登録ならYouTube投稿のみスキップ |
| `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET` / `TIKTOK_REFRESH_TOKEN` | TikTok自動投稿用（video_post.yml）。TikTok for DevelopersでContent Posting APIを追加したアプリを作成し、ローカルで`python video/tiktok_auth.py`を1回実行してリフレッシュトークンを取得する。未登録ならTikTok投稿のみスキップ |
| `TIKTOK_DIRECT_POST` | TikTokアプリの審査通過後に`1`を設定すると直接公開になる（既定は下書き送信。審査前は一般公開の投稿ができないため）|

### 依存パッケージ
```
requests pandas numpy scikit-learn joblib xgboost lightgbm python-dotenv openpyxl xlrd yfinance
```

### パス設定（ローカル実行）

各スクリプトはデフォルトで「実行中のプロジェクトディレクトリ」を参照する。  
別ディレクトリに `.env` / モデル / CSV を置く場合は `STOCK_ALERT_HOME` を設定する。

```bash
export STOCK_ALERT_HOME=/path/to/stock-alert
```

### 手動実行コマンド
```bash
python3 screener.py              # スクリーニング（全銘柄、約30分）
python3 screener.py --test       # テストモード（5銘柄のみ）
python3 rf_train_v3.py           # モデル学習（40〜70分）
python3 rf_train_v3.py --cutoff 2025-07-01  # ウォークフォワード用モデル学習
python3 rank_stocks.py           # ランキング生成
python3 backtest.py              # バックテスト（通常期）→ simulations/backtests/ に保存
python3 backtest.py bear         # 下落相場テスト（2024年8月クラッシュ期）
python3 backtest.py --start 2025-01-01 --end 2025-04-01  # 任意期間指定
python3 backtest.py --start 2025-03-01 --end 2025-06-01 --model-cutoff 2025-03-01  # ウォークフォワード指定

python3 multi_backtest.py        # 33期間一括バックテスト＋フィルター比較（ウォークフォワード）
python3 multi_backtest.py --skip-run  # 既存CSVのみ集計（バックテスト実行なし）

python3 tests/test_screener.py     # スクリーナーユニットテスト
python3 tests/test_fetch_history.py  # 株価キャッシュ銘柄コード収集ユニットテスト
```

---

## 設計上の注意点

- **モデルの限界**：AUC 0.766（下落）はランダム（0.50）よりわずかに良い程度。参考指標として使い、最終判断は自分で行う。
- **多段フィルターが必須**：モデル単体を全銘柄に適用しても効果なし。`rank_stocks.py`内のハードフィルター→モデル→下落確率フィルターの順で使うことでアルファが出る（`core/screener.py`による事前スクリーニングは2026-08-01に廃止。詳細は上の「スクリーナー条件」参照）。
- **下落相場では慎重に**：日経20日 < -5% のとき赤バナー警告（`lib/risk_regime.py`の相場リスク管制官がS買いを自動見送り）。
- **日経急騰時の限界**：大型株主導の急騰相場（例：2025年7月 日経+21%超）では中小型株主体の選定が相対的に不利。日経60日 ≥ +15% のときオレンジバナーで警告（新規の日経超え率: 7% vs 通常時59%）。
- **季節性**：3〜5月エントリーが最も好成績（avg+8〜10%、勝率75〜82%）。8〜9月は低調（avg−2.6〜+2.7%）。
- **主要特徴量**：下落モデルはcs_vol20（ボラ相対ランク、8%）・sin_month（7%）・div_ex_feat（7%）が上位。

---

## 付録: `kujira-watch/`

トレーディングシステム本体とは別デプロイの公開ブログサイト「大口投資家の監視ブログ」（Next.js + Tailwind CSS + microCMS、
https://kujira-watch.com/ ）。EDINET大量保有報告書をもとに機関投資家・インサイダー・自社株買いなど「クジラ」
（大口投資家の俗称）の動きを解説する。SEO/AIO対策・独自ドメイン対応済み。詳細は `kujira-watch/README.md` を参照。

---

## 付録: Claude Code スキル（`.claude/skills/`）

| スキル | 役割 |
|---|---|
| `bear-backtest` | 暴落耐性チェックの bear バックテストを実行し、マージ可否を判定 |
| `design-consult` | **デザインコンサルタント**。kujira-watchサイト・アイキャッチ画像・Remotion動画のデザインレビューと改善提案。ブランドトークン（globals.css / theme.ts）と技術制約（和文ウェブフォント禁止・MUI規約・RSC境界）を踏まえ、375px/1280px × ja/en のスクリーンショット確認込みで優先度付きの指摘を出す |
| `manim-video` | Manimによる数学アニメーション動画の生成 |
| `note-cover` | note記事用の表紙画像（1280x670）生成 |

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

【平日19:30 JST・1日1回】ショート動画パイプライン（video_post.yml）
video/publish_video.py（microCMSの新着記事×注目枠から1件選定 → Claudeで縦動画の台本生成 →
Remotionで1080x1920の縦動画mp4を書き出し → 音量を配信基準の-14 LUFSへ正規化 →
YouTube Shorts へ投稿）。
edinet_blog.ymlがその日の記事を出し切ったあとに走らせる。Remotionのレンダリングは
Chrome Headlessが必要で毎時回すには重いため、記事投稿とは別の1日1回のバッチにしている。
対象記事が無い日は何も投稿しない。workflow_dispatchの`article_id`入力に記事ID
（記事URL `https://kujira-watch.com/articles/xxxx` の `xxxx`。URLを丸ごと貼ってもよい）を入れると、
通常の新着×注目枠選定を使わずその記事を動画にできる（気に入った記事を後から動画化する手動実行用。
`--article-id`でローカル実行も可）。

その他ワークフロー: ci.yml（テスト）、
ops.yml（運用系2本立て。平日06:00 UTC=keepalive空コミット／平日14:30 UTC=watchdog=daily_alert.ymlが今日成功していなければ再実行。`github.event.schedule`でジョブを分岐し、手動実行は`job`入力で選ぶ）、
backfill.yml（手動遡及を1本に統合。`targets`入力にカンマ区切りで jpx / tdnet / edinet / prices（株価キャッシュ更新）/ rankings（ランキング遡及、`start_date`必須）を指定）、
x_post.yml（X関連を1本に統合。土曜18:00 JST=急増ランキング`web/x_weekly_trending.py`／日曜18:00 JST=今週のアクティビストの動き`web/x_weekly_activists.py`／水曜21:00 JST=「答え合わせ」投稿`web/x_followup.py`／毎日10:00 JST=インプレッション等の収集`web/x_metrics.py`／手動のみ=Xトークンの実権限確認`web/x_client --verify`。cronの値または`target`入力でステップを分岐）
```

ユーザー向けの通知・操作は LINE Messaging API 経由（Supabase Edge Function `supabase/functions/line-webhook`）で提供する。Web/Vercelアプリは廃止済み。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | **手動実行専用ツール**（日次パイプラインからは2026-08-01に除外済み）。`get_tse_stock_list()`（JPX全銘柄取得）のみ`rank_stocks.py`/`backfill_history.py`が再利用。銘柄コード絞り込みは`STOCK_CODE_PATTERN`（`^\d{3}[0-9A-Z]$`）で、旧4桁数字に加えTSEが2024年以降に発行する新形式（末尾1桁が英字。例: 151A）も含める（旧`^\d{4}$`では新形式コードが全銘柄スキャンから恒久的に漏れていた）。`apply_screener_v1`によるスクリーニング自体は現在ほぼ価格・流動性のみで`rank_stocks.py`のハードフィルターと重複しており、出力する`data/screeners/*.csv`はどこからも読まれない（下落確率ランキングは`rank_stocks.py`が全銘柄取得〜フィルターまで単独で実施）。手動での銘柄スクリーニング確認用に残置 |
| `tools/fetch_history.py` | Yahoo Finance で全銘柄株価四本値を取得し `yahoo_price_cache` を差分更新（daily_alert.yml Step 0で毎日 `--years 1` 実行。`rank_stocks.py`の「直近株価」の鮮度に直結。既存(code,date)は insert_ignore で保護されるため初回10年分バックフィルにも日次更新にも使える）。`get_all_codes()`はyahoo_price_cache既存コードだけで打ち切らず、毎回JPX最新銘柄リストとの和集合を対象にする（新規上場銘柄が価格キャッシュに永久に追加されない事態を防止。JPX取得失敗時は既存コードのみにフォールバック）。対象は内国株式に加えJ-REITも含む（`_fetch_jpx_codes()`、ブログ記事の金額推定でJ-REIT銘柄の株価が引けるようにするため。コア銘柄スクリーニングの対象銘柄は`core/screener.py`側で別途REITを除外しており本変更の影響を受けない） |
| `tools/refresh_investor_returns.py` | Supabaseのマテリアライズドビュー`investor_return_positions_3m`（明細）と`investor_returns_3m`（投資家別の3ヶ月リターン集計）を明細→集計の順で再計算（daily_alert.yml Step 0b）。RPC`refresh_investor_returns_3m()`を叩くだけの薄いバッチで、集計ロジックは`supabase/create_investor_returns_3m.sql`側にある |
| `tools/backfill_history.py` | 指定期間の過去営業日ぶんランキングを再生成し`gen_rankings`へupsert（アラート送信はしない。`--start`/`--end`指定可。既存日付は既定でスキップするため、価格データ修正後に再生成したい場合は`--force`で上書き。生成後に`check_price_freshness`で複数日にまたがるclose凍結（更新漏れ）を検査）|
| `core/rf_train_v3.py` | XGBoostの下落モデルを東証全銘柄×5年データで学習（金曜のみ。上昇モデルは廃止済み）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に下落確率をつけてランキング生成・DB保存。フェーズ5(優待権利落ち)→フェーズ7(米国ETFリードラグフィルター)→フェーズ8(相場リスク管制官) |
| `web/export_to_web.py` | Supabaseへランキング・日経 vs S&P500判定をエクスポート（Step 4）|
| `web/market_timing_alert.py` | LINE Messaging APIで日次プッシュ通知（Step 5b）。N225シグナル（平均下落確率→投資/キャッシュ）・🌐日経 vs S&P500相対強弱・🏦直近のEDINET大口保有動向（自己申告・過半数超(51%以上、スクイーズアウト対象で上値が見込めない)は除外、譲渡/売却も📈買い・📉売りを明示して表示。同一提出者の開示が期間内に複数あれば保有比率の変化を「5.2%→10.1%」で表示。開示日が新しい順を最優先し、同日内はウォッチ銘柄→法人/ファンド→保有比率が大きい順に優先し最大3件（通知疲れ防止のためLINEは絞り、残りはmicroCMSブログ「大口投資家の監視ブログ」（`kujira-watch/`、https://kujira-watch.com/ の詳細解説記事）のURLに委ねる。各行の下にはその銘柄の`/stocks/{code}`へのディープリンクを添える（`blog_stock_url()`。トップURLだけだと読者が銘柄を探し直す必要があったため）。流入はGA4で識別できるよう`utm_source=line&utm_medium=push`付き）、個人名の提出者は後回し）・🔍ユーザー別ウォッチ投資家の動き（`filer_watchlist`に登録した提出者名で部分一致照合し、その投資家がどの銘柄を動かしても通知。自己申告・過半数超は除外しない）・ユーザー別ウォッチリストのdp閾値アラート（ランキング本体の推奨が「🔴 売り検討」の銘柄は、個人のdp_sell_threshold設定値に関わらず必ず⚠️売り検討を表示。既定値20%はシステム全体の売り検討基準(drop_prob≥10%等)より緩いため、この上書きが無いと10〜20%の間で警告が沈黙するギャップが生じていた。閾値未達で変化のない銘柄は個別表示せず件数のみ要約し、前日比のdrop_prob変化があれば表示：通知疲れ対策）を配信 |
| `config.py` | 戦略パラメータ（`BASE_DIR`・下落相場判定 `BEAR_MARKET_THRESHOLD`・市場タイミング `MARKET_TIMING_20D_THRESH`）。学習時スクリーニングの閾値は`core/rf_train_v3.py`の`_SC_*`、バックテストは`tools/backtest.py`の`_SC_*`が保持する |
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_scores 等）|
| `lib/db.py` | Supabase永続化層（gen_rankings / jpx_stock_list / yahoo_price_cache ほか）。`lib/supabase_client.py` のREST API経由（タイムアウト等の一時的なネットワーク失敗は指数バックオフで自動リトライ）|
| `lib/fundamentals.py` | point-in-time（先読みバイアスなし）ファンダメンタル再構成。`rank_stocks.py`/`rf_train_v3.py`/`backtest.py`で共用。`get_pit_fundamentals()`等は`rows`（銘柄のjquants_fin_summary全履歴）を渡すとDB問い合わせせずメモリ上でas_ofフィルタする（`rf_train_v3.py`が銘柄あたり約60サンプル日で呼ぶため、都度クエリだと数時間かかっていたのを銘柄ごと1クエリに削減）|
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（下落確率レンジ・予測多様性等の行レベル、rank_stocks/export_to_webで使用）＋`check_price_freshness`（複数日にまたがるclose凍結=更新漏れ検知、backfill_historyで使用）（alert-only：違反でも更新は止めずメール通知）|
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `lib/market_compare.py` | **日経 vs S&P500 相対強弱アドバイザー**。日経225とS&P500の20日・60日リターン差から「日本株優位／米国株優位／拮抗」を判定(売買シグナルには影響しない参考情報)。rank_stocksのフェーズ8bで判定し `data/market_compare.json` に保存、`gen_market_compare`経由でLINE(`market_timing_alert.py`)に表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。下落確率が低い順に選定。結果は `simulations/backtests/` に保存。`--drop-max`で下落確率上限、`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋下落確率閾値比較分析（ウォークフォワード対応） |
| `tools/screen_catalyst_candidates.py` | カタリスト候補スクリーン（GARP補助）。PBR<1.0・ROE<8%・自己資本比率>50%・流動性の「安い箱」抽出は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計（J-Quants財務データ使用）。通過候補に **利益の質フィルター(A/B)** で化粧決算（営業赤字・純利益>営業益×1.5）と斜陽事業（本業減益）を除外し、売上CAGR・営業利益率・会社予想方向で加減点。`data/catalyst_candidates.csv`（残）＋ `data/catalyst_excluded.csv`（除外理由付き・レビュー用）。`--no-quality` で品質フィルター無効 |
| `tools/catalyst_backtest.py` | カタリスト候補スクリーンのヒストリカルBT（point-in-time・disc_date≤基準日）。A/Bあり/なしで平均・勝率・大勝率を比較。データは J-Quants財務＋yahoo_price_cache |
| `lib/earnings_quality.py` | カタリスト候補の利益の質・本業方向性を判定（年次の営業益/売上/純益から化粧決算/斜陽を機械判定）。データ源は kabutan 優先、取れない環境（クラウドはkabutanがIPブロック）では J-Quants 実績にフォールバック |
| `lib/edinet.py` + `tools/scan_large_holdings.py` | **EDINET大量保有スキャナー**（イベント駆動）。EDINET APIから大量保有関連報告書(doc_type_code 350/360)を毎時スキャン（350は新規・変更の両方を含み360は訂正のため、種別は`doc_description`の接頭辞で新規/変更/訂正を判定＝`lib/edinet.py`の`disclosure_kind_label()`/`disclosure_doc_label()`。kujira-watch側`disclosureKindLabel()`と同一ロジックで、記事のfact_sheetにもこの判定を使う）（edinet_blog.yml、平日9:00-19:00 JST）して `edinet_large_holdings` に蓄積し、カタリスト候補と突合（構造的候補×実際の買い集め＝先回り候補）。突合時に自己申告（提出者≒対象企業）・過半数超(51%以上)・訂正報告書（既存開示の事後修正で実際の持分変動ではない。ただし届出比率が3pt以上動く大幅訂正＝`is_material_correction()`は既報の保有比率自体が誤りだったという情報のため記事化側では除外しない）・譲渡/売却の報告を除外し、外部の買い集めだけ残す（`--no-exclude` で無効化可）。`is_sell_disclosure`/`is_individual_filer` は `market_timing_alert.py` のLINE通知セクションでも再利用（売却を除外せず方向性表示、個人名提出者を優先度で後回し）。買い/売りの方向判定はXBRLの直前保有割合(`holding_ratio_prior`)と現在の保有割合を比較して行い（概要欄の「譲渡/売却」等の文言が無い開示でも保有比率の減少を正しく売りと判定）、取得できない場合のみ概要欄のキーワードにフォールバックし、どちらも取得できない場合は買い/売りを推測せず方向性を表示しない。`EDINET_API_KEY` 必須 |
| `tools/reclassify_blog_articles.py` | **既存ブログ記事の投資家分類（dealType）一括再分類**（手動実行専用）。旧dealType体系（インサイダー買い/日系ファンド買い等）で公開済みの記事を、`classify_filer()`が返す新13分類へ移行する。各記事のstockCode+dealDateから`edinet_large_holdings`を逆引きしてfiler_nameを特定（同一銘柄・同一開示日に複数提出者がいて一意特定できない記事はスキップし一覧表示）、記事を全フィールド取得しdealTypeだけ書き換えて`update_article()`でmicroCMSをPATCH更新。`--dry-run`で変更内容の確認のみ可 |
| `tools/fix_misreported_blog_articles.py` | **誤って「新規保有」として公開された記事の是正**（手動実行）。EDINET開示の`holding_ratio - holding_ratio_prior`を正として、`ratioChangePct`・`dealAmount`・タイトル・tags・本文を作り直しmicroCMSへPATCHする（本文は`--keep-body`で据え置き可。株価チャートの`<figure>`は引き継ぐ）。対象は(a)`ratioChangePct`が実データとズレている記事、(b)前回比率が0より大きいのにタイトルが「新規保有」の記事の2条件のいずれかに当たるものだけ。是正後に`is_worth_publishing()`の基準を割る記事は`--delete`指定時のみバックアップ(logs/)を取ってから削除する。既定はdry-run、`--apply`で実行 |
| `tools/cleanup_duplicate_blog_articles.py` | **ブログ重複記事クリーンアップ**（edinet_blog.ymlの投稿ステップ後に毎時`--delete`で実行）。同一銘柄・同一開示日・同一提出者(`filerName`)・同一比率変化幅(`ratioChangePct`)＝`already_published()`と同じキーの記事が複数あれば先発1件（X投稿等でリンク済みの可能性が高い）を残して後発を削除する（同一提出者が同日に複数報告書を出す実例＝2936 2025-08-13 橋本舜2件 は別イベントとして残す）。`already_published()`はmicroCMS API失敗時にFalseを返す設計のため重複投稿は稀に発生しうるが、このステップが自動回収する。filerNameが空の旧記事（2026-08-15以前）は判定不能のため対象外。`--days`（既定3）でdealDateの遡り日数指定、`--delete`無しはdry-run |
| `tools/rewrite_thin_blog_articles.py` | **既存ブログ記事の本文リライト**（手動実行専用、Google Search Console「クロール済み-インデックス未登録」対策）。可視文字数（HTMLタグ・末尾の株価チャート`<figure>`を除いた文字数）が閾値未満の記事、または`--ids`で明示指定した記事を対象に、stockCode+dealDateから`edinet_large_holdings`を逆引きしてfact_sheetを再構築し、現行の`generate_article_body()`（保有比率の変化幅つき・650〜900字目標）で本文だけを再生成する。既存タイトル（アイキャッチ画像に焼き込み済み）と株価チャート`<figure>`はそのまま維持し、`update_article()`でmicroCMSをPATCH更新。閾値判定だけでは対象記事数（359件中357件）がGSC実測（104件）と大きく乖離するため、`--ids`でSearch Console提示のURL一覧に絞って実行するのが実運用上の想定（閾値判定は目安・動作確認用）。`--dry-run`・`--limit`併用可 |
| `tools/backfill_article_filer_name.py` | **既存記事の`filerName`バックフィル**（手動実行専用、カニバリゼーション対策の前提）。`filerName`は2026-08-15追加のフィールドで、それ以前の記事791本中326本が未設定だった。`stockCode`+`dealDate`で`edinet_large_holdings`を逆引きし、同日・同銘柄の提出者が1人ならそのまま採用、複数いる場合は**記事タイトル→記事本文**の順に提出者名（法人格・全角/半角・中黒を正規化して突合、4文字未満の社名は誤爆するため対象外）が含まれるかで一意に絞り、絞れなければスキップする（タイトルが「個人投資家が3.5億円規模を売却」のように提出者名を出さない回でも、本文には「個人投資家の森久保哲司氏が」と書かれている。本文まで見ることで2026-08-21時点の未設定13件のうち10件が埋まる）。Supabaseの取得は`order=doc_id`を付けて1クエリでまとめて行う（order未指定だとPostgRESTのページングで行が取りこぼされ「候補なし」が73件に膨らむ）。既定は`--dry-run`相当で、`--apply`指定時だけmicroCMSへPATCHする。2026-08-19実行で313件を補完し未設定は13件まで減少 |
| `tools/backfill_investor_profiles.py` | **既存投資家の分類・プロフィール一括バックフィル**（手動実行専用）。`edinet_large_holdings`に登場する提出者のうち、kujira-watch `/investors/[filer]`の解説文（`edinet_filer_classification.profile`、800〜1000字程度）が未設定の投資家をまとめて埋める。`edinet_filer_classification`に未登録（未分類）の提出者は`classify_filer()`で分類してから、分類済みだが`profile`未生成の提出者は`get_filer_profile()`のみを呼び出す。日次パイプラインは新規に記事化した提出者のみ都度分類・生成するため、記事化されずに大量保有履歴だけ残っている既存提出者を埋めるためのスクリプト。`--limit`件数上限、`--sleep`秒間隔（デフォルト1秒、レート制限対策） |
| `web/publish_blog_articles.py` | **ブログ記事自動生成・投稿**（edinet_blog.yml、平日9:00-21:00 JST毎時、microCMSブログ「大口投資家の監視ブログ」`kujira-watch/`向け）。株価更新パイプライン(daily_alert.yml)からは独立しており、開示当日のうちに検出・記事化する。`market_timing_alert.get_recent_large_holdings`（自己申告・過半数超・訂正報告書を除外。ただし届出比率が3pt以上動く大幅訂正（`is_material_correction()`、実例: 2026-08-18の太陽誘電6976が15.22%→4.41%で株価-11.5%）は残す。訂正記事は売買を伴わないため推定金額を付けず`dealAmount=0`・`tags`に"訂正"を立てて投稿し、タイトルも`〜が保有比率をX%に訂正｜訂正報告書`テンプレート、本文プロンプトも「売買があったと断定しない／訂正理由を推測しない」に切り替える（kujira-watch側は`isCorrectionArticle()`で金額欄に「訂正」と表示）。銘柄名は`data/code_name_map.json`優先、未収載の新規上場銘柄はEDINET開示のissuer_nameから法人格を除去して補う）からネタを取得し、保有比率の増減（取得できない場合のみ概要欄キーワード）で取得(買い)/売却(売り)の方向を判定（`is_sell_disclosure()`）した上で両方向とも記事化し、yfinanceの発行済株式数×株価×保有比率変化で取得・売却金額(億円)を概算（`shares_outstanding()`はyfinance側の一時的なレート制限対策として最大3回リトライし、`sharesOutstanding`が空ならJ-REIT等を想定して`impliedSharesOutstanding`にもフォールバックする。株価は`yahoo_price_cache`（スクリーニング対象ユニバースのみ）に無ければ`close_price_from_yfinance()`でyfinanceから直接取得する（新規上場銘柄・ユニバース外銘柄が「金額を概算できない」で落ちるのを防ぐ）。それでも株価・株式数のいずれかが取得できない銘柄はスキップ）。売り方向の記事はmicroCMSのスキーマ変更を避けるため`tags`に"売り"を追加して区別する（買い方向は従来通りtags不変）。プロンプト・見出しラベル（推定取得金額/推定売却金額）・末尾の推測文（「この取得が」/「この売却が」）も方向に応じて分岐させる。`classify_filer()`が提出者の投資家分類（個人/創業家の資産管理会社/公益・一般財団法人/プライムブローカー/アクティビスト/VC/PE・メザニンファンド/独立系ブティックAM/国内アセットマネジメント/外資系伝統運用会社/日系証券銀行/事業会社/その他）をSupabaseの`edinet_filer_classification`マスター（Web検索で確認済みの投資家分類テーブル、バックテスト分析とも共用）から参照し、未登録の提出者のみClaudeの一般知識で判定して結果をマスターへ保存（キーワード一致だけでは日系/外資やスペース無し個人名を判定できないため）。Claude（`ANTHROPIC_API_KEY`）には事実と分類済みdealTypeのみを渡して解説記事本文を生成しmicroCMSへ即時公開。事実の並置だけで終わらず投資家への示唆(so what)を加えられるよう、開示日の終値（`disclosure_close_price()`＝推定金額の概算に使うのと同じ値。以前は`gen_rankings`の株価を使っていたため、記事本文の「◯月◯日時点の株価」とサイトの「基準終値」が食い違っていた＝2026-08-19に統一）と`gen_rankings`から取得したPIT(point-in-time)の下落リスク水準(高/やや高/中/やや低/低)をプロンプトに文脈として渡し、その範囲内での意味づけを1文加えさせる（取得できない銘柄は従来通り事実のみ）。`ratio_change_pct()`が保有比率の変化幅（ポイント）をfact_sheetへ渡し（開示自体が持つ直前保有割合`holding_ratio_prior`を優先して使い、無い開示のみ同一銘柄・同一提出者の過去開示（直近400日）から算出する。ただし**変更報告書なのに`holding_ratio_prior`がまだ取れていない開示は`should_wait_for_prior_ratio()`がその便での記事化を見送る**（EDINETはメタデータ公開とXBRL本文の可用性にラグがあり、提出直後の便では前回比率が取れないことがある。その状態で記事化すると変化幅＝今回比率の全量となり「X%を新規保有」という誤ったタイトルと過大な推定金額が公開されたまま残る＝2026-08-19の監査で直近14日の照合可能56件中13件を検出。`PRIOR_RATIO_WAIT_DAYS`=2日を過ぎても埋まらない開示はXBRLの書式差とみなし、従来どおり履歴からの再導出で記事化する）。履歴からの再導出だけでは全売却（比率0%）や履歴に同値が残るケースで変化幅0となり記事化されずに落ちていた＝2026-08-17の三菱商事によるＴＯＹＯ ＴＩＲＥ 20%→0%「短期大量譲渡」の取りこぼし）、過去開示が有れば「これまでの開示からXポイント増加/減少」、無ければ「直近400日以内に開示が確認できず実質的な新規保有（または大幅な保有再開）とみられる」という事実をプロンプトに含める（記事本文が同一投資家・同一銘柄でも毎回同じ言い回しの薄い内容にならないよう、既存で計算済みだが本文生成には使っていなかった実データを追加投入するSEO対策。GSC「クロール済み-インデックス未登録」対策として2026-08-14導入）。`get_company_description()`が対象企業の事業内容をClaudeのweb検索（`web_search`ツールで会社概要を裏取りし`jpx_stock_list.description`にキャッシュ。一般知識のみで書かせていた頃は中小型株の約2/3が「不明」で空文字になり、`/trending`や`/stocks/[code]`で事業内容が出ない銘柄が大量に残っていたため2026-08-18にweb検索へ変更）。会社四季報の【特色】欄と同程度の密度＝2〜3文90〜130字で、主力事業・売上構成・製品/ブランド名・シェアや展開地域まで書かせる（1文40字では「何の会社か分からない」ままだったため2026-08-18に拡充。裏が取れない数値・シェア・順位の推測は禁止））から取得できた場合は冒頭の紹介文と保有比率の規模感（時価総額の一角を占める大株主、等）に自然に織り込む。`get_filer_profile()`が提出者のプロフィール（設立時期・運用方針・著名な投資事例など、800〜1000字程度）をClaudeの一般知識から取得し`edinet_filer_classification.profile`にキャッシュする（kujira-watch側`/investors/[filer]`の解説文として表示。情報が乏しい個人名義等は空文字のまま創作しない）。本文の最後には「この取得/売却が今後どんな意味を持ちうるか」の推測を必ず1文加えさせるが、事実と混同しないよう文頭に「※推測:」ラベルを付けさせ、事実として存在しない具体的計画やコメントの引用は創作しないよう明示的に指示する。金額が概算である旨・大量保有報告書制度の一般的な説明・「今後の動向を注視する必要がある」等の定型的な結びは、既に見出しや事実で伝わっているため本文で繰り返さないよう指示する（人間は事後にmicroCMS管理画面で修正する運用）。`build_price_chart_for_article()`が`yahoo_price_cache`から直近3ヶ月の終値を取得し、PIL（Pillowのみ、追加依存なし）で簡易な折れ線チャートPNGを描画してmicroCMSへアップロードし、本文HTML末尾に`<img>`タグとして埋め込む（株価取得・生成・アップロードのいずれかが失敗すればチャート無しで記事のみ投稿）。サイト上部のカテゴリフィルターはdealTypeの値をそのままカテゴリ名として使う構成にしており、microCMSに`category`フィールドは持たない（CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない）。記事タイトルはClaudeの自由生成ではなく`build_article_titles()`の決定的テンプレート（`銘柄名（コード）、提出者が保有比率X%に引き上げ/引き下げ｜大量保有報告書`、新規保有は`X%を新規保有`。60字超過時は提出者名を`…`で短縮）で組み立て、「銘柄名（コード）」「保有比率」「大量保有報告書」という検索語が必ず入ることを保証する（SEO/AIO 30日計画P1、2026-08-15）。本文の1文目も検索クエリへの直答文（`〜が保有比率をX%まで引き上げたことが大量保有報告書（EDINET）で分かりました。`）に固定してプロンプトで指示する。保有比率の変化幅は`ratioChangePct`（ポイント、売りは負値）としてmicroCMSにも送信し、フロントのファクトボックス表示に使う。`bodyEn`（kujira-watch `/en`向け英訳）は同一回のClaude呼び出しでJA本文と同時生成し（事実のズレとAPI呼び出し回数増加を防ぐ）、英語タイトル用のローマ字名`stockNameEn`/`filerNameEn`も同時に返させて英語版テンプレタイトル（`titleEn`）に使う。重複投稿の判定（`already_published()`）は銘柄コード＋開示日＋提出者名`filerName`＋比率変化幅`ratioChangePct`で突き合わせる（いずれも開示データから決まる値。以前は`dealAmount`で突き合わせていたが、推定金額は株価から都度概算されるため株価キャッシュ更新をまたぐと全銘柄でズレて重複判定が全滅する事故が2026-08-17に発生し17件が重複投稿された。同一提出者が同日に複数の報告書を出す実例もあるためratioChangePctの一致まで確認して別イベントを区別し、filerName未保存の旧記事に対してのみ`dealAmount`±0.05億円のフォールバックで判定する。その日その提出者の開示が1件だけの場合は変化幅の一致を問わず同一開示とみなす＝`unique_filing`、変化幅の算出ロジック変更で既報記事と再投稿がぶつかるのを防ぐ）。すり抜けた重複は`tools/cleanup_duplicate_blog_articles.py`が毎時回収する。既存記事の更新（`update_article()`、`tools/reclassify_blog_articles.py`等が使用）は2026-08-14よりPUT（完全上書き）からPATCH（差分更新）に切替（APIキーの権限変更でPUTが拒否されるようになったため）。アイキャッチ画像は`PEXELS_API_KEY`が設定されていれば、投資家分類に応じたPexels写真（`EYECATCH_QUERY_BY_CATEGORY`、銘柄固有の写真は現実的でないため分類のイメージに合う汎用写真を使用）に黒帯＋ニュースカード型テキスト（売買方向バッジ＋開示日／提出者名／銘柄名＋保有比率、Noto Sans CJK Bold太字白文字の3段組み。自由記述のタイトル文字列ではなく構造化した事実を焼き込むことでGoogle Discoverのカード面での視認性を狙う。2026-08-15、`generate_eyecatch_image()`/`build_eyecatch_for_article()`のシグネチャを`(category, card)`に変更）を合成し、microCMSのメディアアップロードAPI(`{domain}.microcms-management.io`)へアップロードして`eyecatch`フィールドへ設定する（`PEXELS_API_KEY`未設定・取得失敗時は画像無しで記事のみ投稿）。`--dry-run`で投稿せず内容確認のみ可（アイキャッチ生成もスキップ）。`MICROCMS_SERVICE_DOMAIN`/`MICROCMS_API_KEY`（書き込み権限）必須、未設定ならスキップ |
| `web/x_client.py` | **ブログ新着記事のX(Twitter)自動投稿**。`publish_blog_articles.py`の`main()`から投稿完了後に呼び出され、その回に投稿した記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページの「注目」枠と同じロジック）にも含まれる記事を金額規模順に**1回1件**（`ARTICLES_PER_RUN`。以前は3件で同一時刻に同型の投稿が3連続していた）投稿する。訂正報告書の記事（`tags`に"訂正"）は`dealAmount=0`で「注目」枠に入り得ないが既報の前提を覆すため**件数制限なしで全件**投稿し、同じ銘柄の直近の投稿（`x_posts`）が見つかればその投稿への自己リプライとしてぶら下げる。投稿してよいのはJST 8〜22時のみ（`within_posting_hours()`）。本文は記事タイトルの流用ではなく**1行目を「誰が・どの銘柄(証券コード)を・どうした」のフック**、2行目に「約N億円・保有比率 X%→Y%」、3行目に提出者の文脈（`web/x_insight.py`、文字数に収まらなければ落とす）、末尾は`#日本株 #大量保有報告書`の2タグのみ（`#EDINET`や記号を除いた`#社名`は検索母数が無いため廃止し、銘柄は本文に`社名(コード)`として素で書く）。URLは**本文に入れる**（親投稿にURLが無いとスレッドを開かない読者にリンクが一切届かないため。`X_LINK_IN_REPLY=1`で自己リプライ（2投稿目）へ移すA/Bが可能で、どちらだったかは`x_posts.variant`に記録される）。**添付画像は数字カード（`web/x_card_image.py`）1枚だけ**（altつき）。かつては2枚目に株価チャートを付けていたが、Xは複数画像を左右に並べて両方とも切り落とすため、タイムラインで銘柄名も数字もチャートも読めなくなっていた（2026-08-19に1枚へ変更）。チャートはリンク先の記事に任せ、カードを作れなかった場合（フォント欠如等）だけチャートを代替として1枚添える。カードの銘柄名は社名が長くてもフォントを段階的に下げて`社名（証券コード）`を丸ごと収め、それでも入らない場合だけ社名側を削る（コードは銘柄検索の手掛かりなので消さない）。投稿に成功すると`post_tweet()`がtweet_idを返し`log_post()`がSupabaseの`x_posts`へ記録する（`web/x_metrics.py`が日次でインプレッション等を追記。これが無いとフォーマット変更の効果を検証できない）。加えて**「本日のクジラ」日次サマリー投稿**（`post_daily_summary()`）: 毎時バッチのうち**21時JST（12時UTC）**の便のみ（19時JSTはタイムラインが薄いため2026-08-19に移動。時刻ガードが外部ストレージ無しの1日1回重複ガードを兼ねる）、その日の全記事をmicroCMSから金額降順で取得し、件数・合計金額・最大買い増し・最大売却を一覧カード画像付きで1ポストする（0件の日は投稿しない）。認証はOAuth 1.0a User Context（`X_API_KEY`/`X_API_KEY_SECRET`/`X_ACCESS_TOKEN`/`X_ACCESS_TOKEN_SECRET`）。いずれか未設定なら投稿をスキップ。401/403で失敗した場合は`verify_auth()`がv1.1 `account/verify_credentials`の`x-access-level`ヘッダからトークンの実権限（read / read-write）を引いてログに添える（実例: 2026-08-18）。`python -m web.x_client --verify`で手動確認も可（x_verify.yml）。`--dry-run`実行時は呼び出されない |
| `web/x_post_format.py` | 週末のX投稿（下記2本）で共用する整形ヘルパー。Xの投稿上限280「単位」（全角2・半角1・URLは一律23）に収めるための`weighted_len()`/`fits()`、EDINET正式名称を表示用に短くする`clean_name()`（全角英数のNFKC半角化＋和英の法人格除去）、単位数基準で切り詰める`label()`、記号を除いた`label()`の単位計算 |
| `web/x_weekly_activists.py` | **週次「今週のアクティビストの動き」のX投稿**（x_weekend_post.yml、日曜18:00 JST・週1回）。`edinet_filer_classification.category='アクティビスト'`の提出者（kujira-watch `/activists`と同じ母集団）について、直近7日(JST)の開示から提出者×銘柄ごとに「週初の保有比率→週末の保有比率」を集計し、変化幅(pt)の大きい買い増し・売却を載せて`/activists`へ誘導する。訂正報告書は持分変動でないため除外、変更報告書なのに直前保有割合が取れない行は「新規」と誤表示しないよう除外、週内に複数開示がある提出者×銘柄は正味の変化1行にまとめる。変化0.5pt未満は載せない。`--dry-run`で本文確認のみ可 |
| `web/x_weekly_trending.py` | **週次「大口投資家の取引急増ランキング」のX投稿**（2026-08-18に「クジラ急増ランキング」から改名。投稿見出しは「🐋 大口投資家の取引急増ランキング（前週比）」）（x_weekend_post.yml、土曜18:00 JST・週1回）。平日の記事投稿・日次サマリーが無い週末のタイムラインを埋める枠。kujira-watch `/trending`（`src/lib/trendingStats.ts`）の期間比較ロジックをPythonへ移植し、`edinet_large_holdings`から増加件数の多い銘柄（最大3件）・投資家（最大2件）を集計して投稿する。比較窓は/trendingの30日ではなく**7日（前週比）**にしている（30日窓だと隣り合う日曜の投稿で23日分のデータが重複し、毎週ほぼ同じランキングが並ぶため）。社名の「株式会社」等は表示用に除去し、Xの280単位制限（全角2単位）に収まるよう投資家→銘柄の順で行を自動削減。急増銘柄が無い週は投稿しない。`--dry-run`で本文確認のみ可 |
| `web/x_card_image.py` | **X投稿に添付する数字カード画像の生成**（1200x675、Pillowのみ）。配色は`kujira-watch/src/app/globals.css`・`video/remotion/src/theme.ts`と同じブランドトークンのみ（navy #16213a／paper #fffdf8／section-tint #f1ece1／rule #ded5c0／gold #b8863a／買い #047857／売り #be123c）を使い、独自色は増やさない。記事投稿の1枚目に使う`build_deal_card()`（バッジ＋提出者→銘柄名（コード）→地色を敷いた帯に「保有比率 X%→Y%」と推定金額の3段。訂正報告書は金額を持たないため帯の右に訂正幅(pt)を出す）と、日次サマリー/週次ランキング/答え合わせで使う`build_list_card()`（見出し＋最大6行の一覧。行数が2件でも下半分が空かないよう行ブロックを領域の中央に置く）。以前の添付は記事本文と同じ株価チャートだけで、投稿の主張（誰が何%いくら）を画像が伝えていなかった。フォントは`fonts-noto-cjk`（CI）→ヒラギノ（ローカル）の順に探し、見つからなければNoneを返し画像なしで投稿を続行する |
| `web/x_insight.py` | X投稿の**解釈行**（3行目）用のデータ取得。`edinet_filer_summary`の開示件数と`edinet_large_holdings`の同一提出者×銘柄の件数から「この提出者の開示は過去N件、〇〇ではM回目」「この提出者がEDINETに登場するのは初」を組み立てる。事実だけの自動投稿はbot扱いされてフォローされないため。推定損益ベースの「乗っかり実績」(`filer_win_rate`)は算出が誤っていたため2026-08-18に廃止済みで、ここでは使わない。取得失敗時は空文字＝行を出さない |
| `web/x_metrics.py` | **X投稿のメトリクス収集**（x_metrics.yml、毎日10:00 JST）。`x_posts`に記録済みのtweet_id（直近30日）をX API v2 `GET /2/tweets`で引き、インプレッション・いいね・リポスト・返信・引用・ブックマーク・**リンククリック・プロフィールクリック**を`x_posts`（最新値）と`x_post_metrics`（日次スナップショット）へ保存する。`non_public_metrics`が権限で取れない場合は`public_metrics`のみで再取得する。あわせて`GET /2/users/me`で**アカウントのフォロワー数**を引き`x_followers`（1日1行）へ記録する。フォロワーは「インプレッション→プロフィールクリック→フォロー」の順にしか増えないため、投稿単位のプロフィールクリック率とアカウント単位のフォロワー増減の両方を測る。`--report`でフォロワー推移（7日前比・30日前比。記録が飛んでいる日はその日以前で最も新しい記録と比較）と種別×variantごとの平均（プロフィールクリック率込み）を表示し、投稿フォーマット変更の効果判定に使う |
| `web/x_followup.py` | **週次「答え合わせ」投稿**（x_followup.yml、水曜21:00 JST）。約3ヶ月前(91日前付近)に大量保有報告書が出た銘柄群について、`yahoo_price_cache`から「開示日の終値→直近終値」の騰落率を計算し、平均・中央値・上昇銘柄数・最も上げた銘柄・最も下げた銘柄を投稿する（勝った銘柄だけを出さない）。基準終値が開示日から7日以上離れる銘柄と、日次±40%以上動く銘柄（株式分割・併合で終値が不連続）は除外する。対象日に開示が5件未満なら投稿しない。`--dry-run`で本文確認のみ可 |
| `video/build_script.py` | **自動動画投稿の台本生成**（video_post.yml、平日19:30 JST・1日1回）。microCMSに直近36時間で新規公開された記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページ「注目」枠と同じロジック）にも含まれるものを`dealAmount`降順で1件だけ選び（`pick_article()`。X投稿と同じ「新着×注目」の積集合で、サイト上目立っていない小粒な開示だけが動画化される事態を防ぐ。積集合が空の日は動画を作らない）、記事本文＋Supabaseにキャッシュ済みの補足事実（`get_company_description()`の事業内容・`get_filer_profile()`の投資家プロフィール、どちらもpublish_blog_articles.pyが生成したもの）だけを根拠に、Claudeで**ナレーション付き台本**を生成する。台本は hook→company（どんな会社か）→deal（金額・保有比率）→filer（どんな投資家か）→change（前回からの変化）→cta の6シーン（`SECTION_SPEC`）＋株価チャートで、各シーンは `narration`（読み上げ文、hookは22〜30字・本編は35〜55字・締めは14〜20字）と `caption`（画面に出す字幕、26字以内）の対。字数超過は最大3回まで作り直し（作り直しでは「何字の文が長すぎたか」をプロンプトに足して伝える。同じ指示をそのまま投げ直すと同じ長さが返り、2026-08-19に台本が2回とも86〜93字で戻って投稿0件になったため）、それでも超える場合はcaptionは末尾を詰め、narrationは句点境界で切る（`_trim_narration()`）。作り直しでも文の途中で終わったシーンは、**シーンごと落とすか記事の事実だけの定型文に差し替える**（`salvage_scenes()`）。hook/deal/change/ctaは定型文で組み直し、言い換えが必要なcompany/filerは落とす。動画そのものを諦めるのはhook/ctaを組み直せない場合だけ（1シーンの失敗で動画を丸ごと捨てていたため2026-08-19・20と2日続けて投稿0件になった）。切れた「…」を画面と読み上げに出さないこと（`is_broken_narration()`）と毎日1本出すことを両立させる。前回の保有比率は本文の末尾から2番目の`◯.◯◯%`から拾い（`extract_prev_holding_ratio()`）、取引の向きと矛盾する場合はNoneにしてchangeシーンごと落とす。※outlook（今後の推測）シーンは2026-08-19に廃止（中央ビジュアルが無く尺だけ食っていたのと、投資助言に寄るリスクのため）。kindはClaudeの出力に頼らず期待順で上書きする（`_flatten_scenes()`）。提出者名は`resolve_filer_name()`が返す: microCMSの`filerName`が空の旧記事（2026-08-16以前）は、同じ銘柄・同じ開示日の大量保有報告書の提出者を候補に挙げ、**記事本文に名前が書かれているもの**だけを採る（同一銘柄・同一開示日に複数の提出者がいるのが普通で、開示データだけでは一意に決まらないため。誤った提出者名を動画タイトルに載せないことを優先し、決まらなければ空文字→「大口投資家」という総称にフォールバック） |
| `video/background.py` | **背景映像の調達（Pexels Videos API）**。海4＋抽象/都市/自然の8クエリのプールから、縦向き・7秒以上・80MB以下・**人物が写っていない**動画を最大2本ダウンロードし（`fetch_pool()`）、`company`と`filer`の2シーンにだけ割り当てる（`assign_backgrounds()`。金額・保有比率・株価チャートを読ませるシーンはRemotion側のブランドグラデーション背景に固定し、実写の明部に数字が沈む事故を構造的に無くす。2026-08-19）。人物の除外はPexelsの動画URLスラッグを語単位で判定する（`has_rejected_subject()`。部分一致だと`germany`の`man`で誤爆するため）。Pexelsは無料・商用可・クレジット不要。`PEXELS_API_KEY`未設定・全滅時は空リストを返し、全シーンがグラデーション背景になる |
| `video/remotion/` | **縦動画のRemotionプロジェクト**（React/TypeScript）。コンポジション`ArticleShort`は1080x1920・30fpsで、**尺は固定ではなく各シーンのナレーション音声の長さで決まる**（`calculateMetadata`と`ArticleShort.tsx`が同じ式`sceneDurationSec()`で総フレーム数を算出。音声が無い場合は読み上げ文字数から概算。実データで約40秒）。ショート動画運用の定石を反映: (1)表示はすべて`safeArea`内（上200px・下470px・左右160pxの**左右対称**。以前は左70/右190で中央寄せが60px左にずれていた）、(2)冒頭は約0.35秒で金額を叩き込み、社名→動詞→提出者ラベル→保有比率と約2.7秒で4回情報を足す（静止画で待たせない）、(3)無音視聴者向けの字幕は26字の要約1本を68pxで出す（ナレーション全文の同時表示は読めず音声と競合するため2026-08-19に廃止）、(4)文字の可読性は影ではなく不透明の下地（`PLATE_BG`）で担保、(5)背景はKen Burnsとシーン内マイクロビート（2.6秒周期）で1フレームも完全静止させない、(6)締めの末尾0.8秒は冒頭と同じ金額組版に戻してループ再生で頭と繋げる（`LOOP_TAIL_FRAMES`。`HookVisual`の完成形を再利用するので1pxもずれない）、(7)EDINETの書類名・提出日とサイトURL・免責を全編常時表示。締めの名乗りは`kujira-watch/src/lib/site.ts`の`SITE_NAME`（大口投資家の監視ブログ）に合わせ、動画側で別名を作らない。検索誘導はしない（サイト名で検索上位を取れていないため辿り着けない）。効果音とBGMは`props.sfx`が真のときだけ鳴る（BGMは`volume`関数で頭20フレームをフェードイン・末尾14フレームをフェードアウトし、`loopVolumeCurveBehavior="extend"`でループしても音量カーブが動画全体の時間軸で効くようにする）。配色は`src/theme.ts`が`kujira-watch/src/app/globals.css`と同じブランド色を持ち、買い=金・売り=赤のアクセント。日本語フォントはOS側のNoto Sans CJK（CIはapt導入、macOSはHiragino）を使い、レンダリングがネットワークに依存しない |
| `video/render.py` | props JSONを`npx remotion render`へ渡してmp4を書き出す薄いラッパ。ナレーション音声（tts.pyが生成したwav）はRemotionの`staticFile()`経由でしか参照できないため、`video/remotion/public/`へコピーしてからレンダリングし、終了後に削除する（`_stage_audio()`。見つからない音声はそのシーンだけ無音にして続行）。`articleId`/`articleTitle`は投稿テキスト専用でコンポジションのpropsには無いため除外して渡す（`NON_PROP_KEYS`）。書き出し前に`video/audio_gen.py`が効果音とBGMのwavを生成して`public/`へ置き、`props.sfx`で鳴らすかどうかを伝える。読み上げ文が文の途中で切れているprops JSONは書き出しを拒否する（`has_broken_narration()`。古いpropsの再レンダリング経路の保険）。初回実行時のみRemotionがChrome Headless Shell(約150MB)を自動ダウンロードする。**音量正規化にはffmpegが必要**で、Ubuntu 24.04のランナーイメージには入っていないためワークフローで明示的に導入する（未導入だと無言でスキップされ-25 LUFSのまま投稿される。2026-08-21に発覚） |
| `video/audio_gen.py` | **効果音とBGMの自前生成（numpy）**。カット頭の無音が「再生バグに聞こえる」・BGM無しが「未完成品に見える」という指摘が最多だったため、シーンの切り替わり（`se_whoosh.wav`）・金額の着地（`se_impact.wav`）・カウントアップ完了（`se_tick.wav`）・全編のBGM（`bgm.wav`）を波形合成で作る。フリー素材を毎日ダウンロードするとライセンス確認が自動化できず規約変更にも気づけないため、外部素材は一切持たない。BGMはAm→F→C→Gの12秒アンビエントパッドで、和音を枠からはみ出させて配列の先頭へ回り込ませることで継ぎ目なくループする（ローパスの内部状態も2周ぶん通して定常化させてから採る。そうしないと先頭だけ音が痩せてループのたびにプチッと鳴る）。乱数を固定してあるので毎回同じファイルになる。numpyが無い環境では`False`を返し、音の追加なしで動画を書き出す |
| `video/post_text.py` | **投稿テキストの共通部品**。サイトの名乗り（`SITE_NAME`＝大口投資家の監視ブログ。`kujira-watch/src/lib/site.ts`と対）・URL・UTM付き記事URL（`article_url()`）・ハッシュタグの整形（`hashtag()`）を1箇所に集約する。名乗りを各クライアントに直書きすると、動画側で実在しない「クジラウォッチ」を名乗った事故（2026-08-19）と同じことが投稿文でも起きるため。`hashtag()`は銘柄名の空白・「．」・中黒を落とす（例: `Ｊ．フロント リテイリング`をそのまま`#`に続けるとタグが途中で切れて残りが本文として漏れる） |
| `video/youtube_client.py` | **YouTube Shortsへの自動アップロード**（YouTube Data API v3のresumable upload）。認証はOAuth 2.0リフレッシュトークン方式（`YOUTUBE_CLIENT_ID`/`YOUTUBE_CLIENT_SECRET`/`YOUTUBE_REFRESH_TOKEN`。ローカルで`video/youtube_auth.py`を1回実行して取得）。YouTubeは縦長かつ3分以内の動画を自動的にShortsとして扱うため専用フラグは不要だが、保険としてタイトル・説明文に`#Shorts`を入れる。説明文は**記事URLを先頭3行以内**に置く（Shortsは冒頭しか畳まずに見えず、以前は8行目にあって折りたたみを開かないと導線に到達できなかった。2026-08-19）。記事URLには`utm_source=youtube`を付与しGA4でShorts経由の流入を識別できるようにする。タイトルには保有比率も入れる。説明文の先頭3つのハッシュタグはタイトル上部に表示される枠なので、`#Shorts`のような機能タグではなく検索される語（`#日本株` `#大量保有報告書` `#銘柄名`）を先に置く。いずれかの環境変数が未設定ならスキップ |
| `video/tts.py` | **ナレーション音声の合成（VOICEVOX）**。無料・登録不要・商用利用可（クレジット表記のみ必須）の日本語音声合成エンジンで、既定の話者はずんだもん（speaker=3、`TTS_SPEAKER`で変更可、`TTS_SPEED`既定1.22倍速（1.3超は金額・比率の聞き取りが落ちるためこれを上限とする））。エンジンはHTTPサーバ（既定 http://127.0.0.1:50021、`VOICEVOX_URL`で変更可）として動き、CIでは公式Dockerイメージ`voicevox/voicevox_engine:cpu-ubuntu20.04-latest`をジョブ内で起動する。`narrate_sections()`が各シーンのnarrationをwav化して`audio`/`durationSec`（ffprobe計測、無ければ文字数から概算）を書き込み、この長さがそのままシーンの尺になる。エンジンに繋がらない・1シーンでも合成に失敗した場合は全体を無音扱いにして動画生成は続行する（一部だけ音が出る動画はかえって不自然なため）。クレジット表記「VOICEVOX:ずんだもん」はCTAシーン・YouTube説明文に自動で入る。※当初はGoogle Cloud TTSで実装したが、GCPプロジェクトに請求先が未設定でAPIを有効化できずVOICEVOXへ切替（2026-08-15） |
| `video/publish_video.py` | 自動動画投稿のオーケストレーター（台本生成→VOICEVOXでナレーション合成→レンダリング→YouTube Shortsへ投稿→LINEで完了通知）。対象記事が無い日は何も投稿せず正常終了する。ナレーション合成に失敗しても無音で続行する。YouTubeのSecretsが未登録の場合は動画生成のみで正常終了（設定漏れの誤検知で毎日赤くならないように）、Secretsがあるのに投稿できない場合のみ異常終了する。`--dry-run`（台本まで）/`--render-only`（mp4書き出しまで）/`--keep-video`（投稿後もmp4を残す）/`--stock-code`（銘柄指定の手動実行）で段階的に確認できる。※TikTok投稿は2026-08-20に完全撤去 — 自アカウントへの投稿用途はTikTokの本番審査ポリシー（personal/internal use不可）の対象外で承認されないため（経緯はdocs/tiktok_review.md）。LINE通知はTikTokキャプション連携用だったものを投稿完了通知として残置 |
| `video/youtube_auth.py` | YouTubeのリフレッシュトークンを取得する**ローカル1回きり**のスクリプト（CIでは使わない）。loopback(http://localhost:8765)で認証コードを受け取る |
| `tests/test_fundamentals.py` | point-in-timeファンダ（`lib/fundamentals.py`）のユニットテスト。先読みバイアス防止（as_of日より後の開示を含めない）を確認（6件）|
| `tests/test_earnings_quality.py` | 利益の質フィルター（化粧・赤字・減益・加減点）のユニットテスト（8件）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（銘柄コード絞り込み正規表現の新形式コード対応込み、13件）|
| `tests/test_fetch_history.py` | 株価キャッシュ更新の銘柄コード収集ロジック（既存コード+JPX最新リストの和集合・JPX取得失敗時のフォールバック・J-REIT含む市場フィルター）のユニットテスト（4件）|
| `tests/test_data_sanity.py` | QA（データ整合性・価格凍結検知）のユニットテスト（14件）|
| `tests/test_market_compare.py` | 日経 vs S&P500 相対強弱アドバイザーのユニットテスト（4件）|
| `tests/test_market_timing_alert.py` | LINE通知の大口保有動向セクション（開示日優先ソート・根拠なき買い/売り推測の抑制込み）・ウォッチリストdp閾値判定（ランキング本体の推奨ラベルとの矛盾防止・売り閾値ギャップの上書き・通知疲れ対策の要約表示・前日比表示込み）・投資家ウォッチ（提出者名の部分一致照合・大口保有動向セクション生成）・code_name_map未収載銘柄のEDINET issuer_nameフォールバック・大幅訂正報告書の通過/軽微な訂正の除外のユニットテスト（28件）|
| `tests/test_scan_large_holdings.py` | EDINET大量保有スキャナーの判定ロジック（売却検知・保有比率増減による方向判定・個人名判定・過半数超除外・訂正報告書除外／大幅訂正の判定・ノイズ除外）のユニットテスト（13件）|
| `tests/test_reclassify_blog_articles.py` | 既存ブログ記事の投資家分類一括再分類ツールのユニットテスト（microCMSのdealType配列/空配列/None正規化。空配列でIndexErrorになっていた本番バグの再発防止・PUT用ペイロードのメタ情報除去）のユニットテスト（5件）|
| `tests/test_cleanup_duplicate_blog_articles.py` | ブログ重複記事クリーンアップの削除対象選定ロジック（先発を残し後発を削除・別提出者/別日は対象外・同一提出者でも比率変化幅が違えば対象外・filerName空の旧記事は対象外）のユニットテスト（4件）|
| `tests/test_rewrite_thin_blog_articles.py` | 既存ブログ記事の本文リライトツールのユニットテスト（HTMLタグ・株価チャート`<figure>`を除いた可視文字数算出・PATCH用ペイロードが本文のみを含むこと）のユニットテスト（4件）|
| `tests/test_backfill_article_filer_name.py` | 既存記事の`filerName`バックフィルツールのユニットテスト（法人格・全角/半角・中黒の正規化、候補1件の即採用、タイトル一致による絞り込み、タイトルで決まらないときの本文一致へのフォールバック、タイトルを本文より優先すること、本文に複数候補が出る場合・一意にならない場合・社名が短すぎる場合のスキップ、HTMLタグの除去）のユニットテスト（10件）|
| `tests/test_publish_blog_articles.py` | ブログ記事自動投稿の判定ロジック（金額概算・発行済株式数取得のリトライ/impliedSharesOutstandingフォールバック・記事生成JSONパース・投資家分類マスター参照とClaudeフォールバック/保存・売り方向のtagsタグ付け/プロンプト分岐・重複防止・権限エラー時の早期打ち切り・投稿/更新のセレクト配列形式への自動リトライ・非文字列フィールド(eyecatch等)の型不一致時は除外して再送信・PIT文脈(株価/下落リスク水準)のプロンプト反映・保有比率変化幅(ポイント)/新規保有のプロンプト反映・事業内容の取得/キャッシュ/web_searchツール付与と検索結果ブロック混在時のJSON抽出/生の改行を含むJSONの許容・投資家プロフィールの取得/キャッシュ/空欄時の非創作・「※推測:」ラベル付き推測文の要求・英訳(bodyEn/ローマ字名)のプロンプト要求・決定的テンプレによるタイトル組み立て（買い/売り/新規保有・60字超過時の提出者名短縮・英語名フォールバック）・冒頭アンサー文のプロンプト指示・ratioChangePct（売りは負値）のpayload付与・filerName＋ratioChangePct照合による重複判定（旧記事はdealAmountフォールバック）・PATCH更新への型不一致自動リトライ・アイキャッチ画像生成/アップロード・株価チャート画像生成/アップロード/本文埋め込み・ホームページ「注目」枠と同じdealAmount降順での注目記事id抽出・直前保有割合を優先した変化幅算出／全売却の扱い・株価キャッシュ欠損時のyfinanceフォールバック・比率不変時のスキップ・大幅訂正記事の投稿(dealAmount=0/tags訂正/タイトル)・単一開示日の重複判定緩和・直前保有割合が未取得の変更報告書の持ち越し）のユニットテスト（95件、ネットワークは全てモック）|
| `tests/test_supabase_client.py` | Supabase REST APIクライアントのリトライ挙動（一時的なネットワーク失敗時のバックオフ再試行・最終失敗時に呼び出し元を落とさないこと）のユニットテスト（3件）|
| `tests/test_x_client.py` | X(Twitter)自動投稿（`web/x_client.py`・`web/x_insight.py`・`web/x_followup.py`）のユニットテスト（1行目フックの組み立て・新規取得/買い増し/売却/訂正の出し分け・検索母数のある2タグのみの付与・解釈行の挿入と文字数超過時の削除・金額の丸め・数字カードのalt文言・tweet_idの返却とログ記録・添付画像を数字カード1枚に限ること/カード生成失敗時のみチャートで代替・社名が長くてもカードに証券コードを残すこと・リンクを自己リプライに置くこと/A/Bフラグで本文に戻せること・alt設定リクエスト・JST 8〜22時外のスキップ・1回1件の投稿・「注目」に含まれない記事の除外・訂正記事の全件投稿と既報へのリプ・「本日のクジラ」日次サマリー（本文組み立て・0件時スキップ・totalCount補正・21時JST便の時刻ガード・カード画像添付）・動画クロス投稿・アクセストークン実権限の判定・解釈行の文言・答え合わせの統計と本文・分割銘柄と基準日ギャップの除外）（54件、ネットワークとSupabaseは全てモック）|
| `tests/test_x_metrics.py` | X投稿メトリクス収集（`web/x_metrics.py`）のユニットテスト（プロフィールクリック/リンククリックのパース・`non_public_metrics`が無い場合のフォールバック・フォロワー数の取得と認証未設定/APIエラー時の空返却・記録が飛んだ日を跨ぐ7日前比/30日前比の算出・記録ゼロ時の表示）のユニットテスト（7件、ネットワークとSupabaseは全てモック）|
| `tests/test_video_pipeline.py` | 自動動画投稿パイプライン（`video/`）のユニットテスト（「新着×注目」の積集合からの金額規模順選定・積集合が空なら動画を作らない・microCMSセレクト型dealTypeの配列アンラップ・tagsからの売り方向判定・filerName未設定時の空文字化・本文末尾の保有比率抽出とフォールバック・台本JSON(hook/sections/closing)のフラットなシーン列への展開・Claudeが誤ったkindの期待順上書き・字数超過時の1回だけの作り直しと最終的な切り詰め・ナレーションの句点境界での切り詰め・sections不足時の破棄・APIキー未設定時のスキップ・VOICEVOXエンジン未接続時の無音フォールバック・合成成功時のaudio/durationSec書き込み・1件でも失敗したら全体無音・YouTubeタイトル/説明文の組み立てとUTM付与・タイトル超過時も#Shortsを残す切り詰め・64MB超のファイルのチャンク分割とContent-Rangeの連続性・書き出し後の-14 LUFS正規化・文途中切れ台本の作り直しと破棄・前回保有比率の抽出と向き矛盾時のNone・チャートの日付ラベルと開示日位置・人物クリップの除外・実写背景をcompany/filerに限定すること・効果音とBGMのwav生成と再現性・BGMのループ継ぎ目に段差が無いこと・BGMのピーク余裕・説明文の記事URLが先頭3行以内にあること・サイト名の名乗り・検索される語を先頭に置くハッシュタグ順・投稿文のサイト導線・ハッシュタグの整形・作り直し時のプロンプトへの問題フィードバック・「買収」等の開示内容を超える語と新規保有を「買い増し」と書くことの禁止・作り直しでcaptionとnarrationの超過を言い分けること・壊れたシーンの定型文への差し替えと切り落とし・ffmpeg不在時に音量正規化のスキップを警告すること・旧記事の提出者名を本文と開示データの突き合わせで特定すること／候補が複数一致するときは総称に落とすこと）（88件、ネットワークは全てモック）|

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

> 3ヶ月相対強度フィルター（`rel_strength_min`引数）とモメンタム/ボラ/RSIの閾値定数は
> `apply_screener_v1`から一切参照されないデッドコードだったため、2026-08-19に実装ごと削除した
> （`MIN_MOMENTUM`/`MIN_VOLATILITY`/`MAX_VOLATILITY`/`MIN_MOMENTUM_20D`/`MIN_VOL_RATIO`/
> `MIN_RSI`/`MAX_RSI`/`MAX_FROM_HI20`/`MIN_REL_STRENGTH`/`BEAR_REL_STRENGTH`/`BEAR_NKK_20D`）。
> 下落モデル一本化後のバックテスト再検証は未実施（別環境でのbacktest.py実行が必要。詳細は `dev_log.md`）。

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
| `investor_return_positions_3m`（マテビュー） | EDINET買い開示1件＝1行の明細。開示日の終値から63営業日後の終値までの騰落率と日経平均比を持つ。kujira-watch `/investors/[filer]`の開示テーブルの「3ヶ月後」列に`doc_id`で突き合わせて使う |
| `investor_returns_3m`（マテビュー） | 上を投資家単位に等ウェイト集計したもの（開示3件以上の投資家のみ。平均・中央値・勝率・日経平均比・最高/最低銘柄）。kujira-watch `/ranking/returns`と`/investors/[filer]`の成績パネルの集計元。定義は`supabase/create_investor_returns_3m.sql`、更新はdaily_alert.yml Step 0b |
| `edinet_filer_summary`（ビュー） | `edinet_large_holdings`×`edinet_filer_classification`を投資家(filer_name)単位に集計したビュー（保有開示件数・最終開示日・分類）。kujira-watch（`kujira-watch/src/lib/investors.ts`）の`/investors`一覧・サイトマップ生成が参照。投資家は600件超あり`edinet_large_holdings`の生データを直接集計すると1000行上限に掛かるため、1投資家1行に事前集計したビュー経由で取得する |
| `ext_tdnet_disclosures` | TDnet適時開示（やのしん・⚠️個人運営ソースのため `ext_` で隔離）|
| `jpx_short_selling` | JPX空売り残高報告（0.5%以上）|
| `jpx_margin_balance` | JPX個別銘柄信用取引週末残高 |
| `line_chat_history` | LINE Bot会話履歴（直近3往復、文脈保持用） |
| `line_users` | LINE Bot登録ユーザー |
| `dp_watchlist` | ユーザー別ウォッチ銘柄・dp閾値（LINE Bot）|
| `filer_watchlist` | ユーザー別ウォッチ投資家（EDINET提出者名。銘柄は問わずその投資家の保有比率増減を通知、LINE Bot）|

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
| **EDINET API v2** | 大量保有関連報告書(350/360)、有報/四半期報の決算XBRL(BS/PL/CF) | 先回りシグナル・財務サマリ（EPS/BPS/ROE/CFO/売上/営業益/予想）本体 | `lib/edinet.py`, `lib/edinet_financials.py`, `tools/scan_large_holdings.py`, `tools/fetch_edinet_financials.py` |
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
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` | YouTube Shorts自動投稿用（video_post.yml）。Google Cloud ConsoleでYouTube Data API v3を有効化し「デスクトップアプリ」のOAuthクライアントを作成、ローカルで`python video/youtube_auth.py`を1回実行してリフレッシュトークンを取得する。未登録なら未登録なら投稿をスキップ |

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

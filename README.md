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

【平日9:00 / 15:00 / 18:00 JST・1日3便】EDINETブログパイプライン（edinet_blog.yml）
tools/scan_large_holdings.py（EDINET大量保有スキャン）→ web/publish_blog_articles.py
（ブログ記事自動生成・投稿。1日の投稿は`DAILY_MAX_ARTICLES`=**2本**まで）→ tools/enrich_buybacks.py → web/publish_buyback_articles.py
（TDnet自社株買い決定の記事化。上限10億円以上 or 発行済3%以上）→ **取りこぼしのbackfill**（専用cron `30 12 * * 1-5` の便と手動実行のみ。GitHubのscheduleは大きく遅延することがあり（実測: 2026-08-27の12:00 UTCの便が16:57に起動）、実行時の`date -u +%H`で「当日最終便か」を判定するとbackfillが永久にスキップされるため、`github.event.schedule`＝起動したcron式そのもので分岐する。手動実行の`backfill_max`入力で1便あたりの投稿上限を上書きできる（積み残しをまとめて消化する用。未指定なら`BACKFILL_MAX_ARTICLES`。シェルへの注入を避けるため環境変数で受けて整数か検証する）。ジョブの`timeout-minutes`は120（実測で1本あたり約40秒＝15本で16分）。
両スクリプトを`--backfill`で走らせ、直近30日のうちまだ記事が無い開示だけを古い順に拾い直す。通常運転は直近3日しか見ないため、
API上限やワークフロー障害で3日を超えて生成が止まるとその期間の開示が二度と記事化されなかった＝2026-08-13〜08-20の自社株買い決定12件を
2026-08-27に手作業で復旧した）。株価更新パイプライン(daily_alert.yml)から完全に独立した
別ワークフローとして実行する（2026-08-15、EDINET記事投稿を日次16:00の1回だけでなく
開示当日のうちに検出・記事化するため分離）。2026-08-30にkujira-watchを「GEOのA/B実験台」へ
切り替え、便数を毎時13便（`0 0-12 * * 1-5`）から3便（`0 0,6,9 * * 1-5`）へ、記事を
**大量保有・自社株買いそれぞれ日2本**（`DAILY_MAX_ARTICLES`）へ絞った。母集団を固定しないと
A/Bの効果が測れず、毎時走らせても枠を使い切った便が空回りするだけでAPI課金だけ増えるため。
記事1本ごとのX投稿の呼び出しもこの日に外した（Xは土曜の週次まとめ1本だけ残す。後述のx_post.yml）。edinet_large_holdings はSupabase経由で
daily_alert.ymlのランキング生成（EDINET大量保有の特徴量）からも参照される。
各ステップは`continue-on-error: true`で「落ちても後続は動かす」が、最後の
「パイプラインの結果を判定」ステップが各ステップの`outcome`を集計してジョブ自体を赤くする
（2026-08-26、全ステップがcontinue-on-errorだったためrunが常にsuccessになり、開示が1件も
保存できていない日もワークフローは緑だった）。開示の保存失敗は終了コード3、記事生成・投稿の
失敗は終了コード4で、どちらもスクリプト側が原因つきでLINE通知する。

【停止中・手動実行のみ】ショート動画パイプライン（video_post.yml）
video/publish_video.py（microCMSの新着記事×注目枠から1件選定 → Claudeで縦動画の台本生成 →
Remotionで1080x1920の縦動画mp4を書き出し → 音量を配信基準の-14 LUFSへ正規化 →
YouTube Shorts へ投稿）。
**定期実行は2026-08-30に停止した**（旧設定: 平日19:30 JST＝`30 10 * * 1-5`）。実測が
公開9本・総再生6,909回・登録者5人・コメント0（`docs/progress_video_post.md`）で、GEOのA/B実験に
無関係なままAnthropic APIとActionsの時間を消費するため。コードは残してあるので、
`video_post.yml`にscheduleを戻せば再開できる。手動実行（workflow_dispatch）は従来どおり動く。
対象記事が無い日は何も投稿しない。workflow_dispatchの`article_id`入力に記事ID
（記事URL `https://kujira-watch.com/articles/xxxx` の `xxxx`。URLを丸ごと貼ってもよい）を入れると、
通常の新着×注目枠選定を使わずその記事を動画にできる（気に入った記事を後から動画化する手動実行用。
`--article-id`でローカル実行も可）。

その他ワークフロー: ci.yml（main へのpush時にテスト全件を実行。サムネイル・画像生成のテストが日本語フォントを要求するため、edinet_blog.yml / video_post.yml と同じ `fonts-noto-cjk` を入れてから走らせる。`tests/test_ga4_clicks.py` が `google.oauth2` をmock.patchするため `google-auth` も入れる＝未導入だと `module 'google' has no attribute 'oauth2'` で1件だけ落ちる）、
ops.yml（運用系4本立て。平日06:00 UTC=keepalive空コミット／平日08:00 UTC=watchdog_blog=edinet_blog.ymlが当日1本も**起動**していなければ手動起動＋LINE通知（成否は問わない。赤で終わった便はpublish側のゲートと`if: failure()`通知が既に拾っている。cron自体の不発はどのワークフローも赤にならず22:00 JSTのheartbeatまで13時間気づけないため。9:00 JSTの初便から8時間空けるのは、scheduleの遅延が実測で最大5時間あり、これより早いと遅延を不発と誤判定して余計なAPI課金を生むから）／平日13:00 UTC=heartbeat=`tools/output_heartbeat.py`でその日の成果物（ブログ記事）と素材（EDINET APIから直接）を数え欠けていればLINE通知（起動がJSTの正午より前にずれ込んだ便は前日を判定する）／平日14:30 UTC=watchdog=daily_alert.ymlが今日成功していなければ再実行。`github.event.schedule`でジョブを分岐し、手動実行は`job`入力で選ぶ）、
backfill.yml（手動遡及を1本に統合。`targets`入力にカンマ区切りで jpx / tdnet / edinet / prices（株価キャッシュ更新）/ rankings（ランキング遡及、`start_date`必須）/ eyecatch（ブログ記事のアイキャッチ画像を作り直す。`eyecatch_mode`=replace|missing・`eyecatch_limit` 既定50・全件は0）を指定）。**eyecatch はローカルmacで実行しないこと**（本番のアイキャッチは Noto Sans CJK Bold で組まれており、macのヒラギノで焼くと過去記事と新規記事で書体が混在する。このジョブは fonts-noto-cjk を入れてから走る）、
x_post.yml（X関連を1本に統合。**2026-08-30に定期実行を週1本へ削減**。残したのは土曜18:00 JSTの「1週間のまとめ」＝急増ランキング`web/x_weekly_trending.py`（直近7日と前7日を比較した週次集計。記事1本ごとの投稿ではない）だけで、残りは`target`入力での手動実行のみ＝／アクティビストの動き`web/x_weekly_activists.py`／「答え合わせ」投稿`web/x_followup.py`／「本日の自社株買い決定」`web/x_buyback.py`／インプレッション等の収集`web/x_metrics.py`／開示原文の事実`web/x_disclosure_facts.py`／Xトークンの実権限確認`web/x_client --verify`／フォロー候補の抽出`tools/x_follow.py discover`／フォロー実行`tools/x_follow.py follow`（`usernames`入力＋`follow_execute`がONのときだけ実行）。停止の根拠は実測（2026-08-19〜08-29）で34投稿・総インプレッション43・いいね0・リポスト0・URLクリック1・フォロワー10日連続0。`docs/x_operation_rules.md`が挙げるリーチの3経路（フォロワーTL／X内検索／他チャネル流入）が全て塞がっており時間では解決しないため。アカウントと既存投稿は消さない。週1本だけ残すのはE-E-A-Tのため（Xが`Organization`スキーマの`sameAs`と`contactPoint`＝実名・メール非公開の運営方針下では唯一の連絡窓口で、`/about`もXでの発信を明記している。URL無し投稿は$0.015/本＝月約$0.06）。全面再開の条件は`docs/x_operation_rules.md`の0章）
```

各ワークフロー（daily_alert / edinet_blog / video_post / x_post / ops-heartbeat）には
`if: failure()` のLINE通知ステップが入っており、ジョブが落ちたら実行ログURL付きでスマホに届く
（`python -m lib.notify`）。ワークフローの赤は誰も見ていないという前提で運用する。

ユーザー向けの通知・操作は LINE Messaging API 経由（Supabase Edge Function `supabase/functions/line-webhook`）で提供する。Web/Vercelアプリは廃止済み。

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `core/screener.py` | **手動実行専用ツール**（日次パイプラインからは2026-08-01に除外済み）。`get_tse_stock_list()`（JPX全銘柄取得）のみ`rank_stocks.py`/`backfill_history.py`が再利用。銘柄コード絞り込みは`STOCK_CODE_PATTERN`（`^\d{3}[0-9A-Z]$`）で、旧4桁数字に加えTSEが2024年以降に発行する新形式（末尾1桁が英字。例: 151A）も含める（旧`^\d{4}$`では新形式コードが全銘柄スキャンから恒久的に漏れていた）。`apply_screener_v1`によるスクリーニング自体は現在ほぼ価格・流動性のみで`rank_stocks.py`のハードフィルターと重複しており、出力する`data/screeners/*.csv`はどこからも読まれない（下落確率ランキングは`rank_stocks.py`が全銘柄取得〜フィルターまで単独で実施）。手動での銘柄スクリーニング確認用に残置 |
| `tools/fetch_history.py` | Yahoo Finance で全銘柄株価四本値を取得し `yahoo_price_cache` を差分更新（daily_alert.yml Step 0で毎日 `--years 1` 実行。`rank_stocks.py`の「直近株価」の鮮度に直結。既存(code,date)は insert_ignore で保護されるため初回10年分バックフィルにも日次更新にも使える）。`get_all_codes()`はyahoo_price_cache既存コードだけで打ち切らず、毎回JPX最新銘柄リストとの和集合を対象にする（新規上場銘柄が価格キャッシュに永久に追加されない事態を防止。JPX取得失敗時は既存コードのみにフォールバック）。対象は内国株式に加えJ-REITも含む（`_fetch_jpx_codes()`、ブログ記事の金額推定でJ-REIT銘柄の株価が引けるようにするため。コア銘柄スクリーニングの対象銘柄は`core/screener.py`側で別途REITを除外しており本変更の影響を受けない） |
| `tools/refresh_holding_amounts.py` | Supabaseのマテリアライズドビュー`edinet_holding_amounts`（EDINET大量保有報告書1件ごとの推定売買金額＝保有比率の変化幅×発行済株式数×開示日終値の概算、億円）を再計算（edinet_blog.ymlの開示スキャン直後＝毎時と、daily_alert.yml Step 2e＝株価キャッシュ・財務サマリー更新後）。RPC`refresh_edinet_holding_amounts()`を叩くだけの薄いバッチで、集計ロジックは`supabase/create_edinet_holding_amounts.sql`側にある。kujira-watchの銘柄ランキング(`/trending`)の並べ替え軸（2026-08-27に開示件数順から金額順へ変更）の元データ |
| `tools/refresh_investor_returns.py` | Supabaseのマテリアライズドビュー`investor_return_positions_3m`（明細）と`investor_returns_3m`（投資家別の3ヶ月リターン集計）を明細→集計の順で再計算（daily_alert.yml Step 0b）。RPC`refresh_investor_returns_3m()`を叩くだけの薄いバッチで、集計ロジックは`supabase/create_investor_returns_3m.sql`側にある |
| `tools/backfill_history.py` | 指定期間の過去営業日ぶんランキングを再生成し`gen_rankings`へupsert（アラート送信はしない。`--start`/`--end`指定可。既存日付は既定でスキップするため、価格データ修正後に再生成したい場合は`--force`で上書き。生成後に`check_price_freshness`で複数日にまたがるclose凍結（更新漏れ）を検査）|
| `core/rf_train_v3.py` | XGBoostの下落モデルを東証全銘柄×5年データで学習（金曜のみ。上昇モデルは廃止済み）。`--cutoff YYYY-MM-DD` でウォークフォワード用モデルも生成可能 |
| `core/rank_stocks.py` | スクリーナー通過銘柄に下落確率をつけてランキング生成・DB保存。フェーズ5(優待権利落ち)→フェーズ7(米国ETFリードラグフィルター)→フェーズ8(相場リスク管制官) |
| `web/export_to_web.py` | Supabaseへランキング・日経 vs S&P500判定をエクスポート（Step 4）|
| `web/market_timing_alert.py` | LINE Messaging APIで日次プッシュ通知（Step 5b）。N225シグナル（平均下落確率→投資/キャッシュ）・🌐日経 vs S&P500相対強弱・🏦直近のEDINET大口保有動向（自己申告・過半数超(51%以上、スクイーズアウト対象で上値が見込めない)は除外、譲渡/売却も📈買い・📉売りを明示して表示。同一提出者の開示が期間内に複数あれば保有比率の変化を「5.2%→10.1%」で表示。開示日が新しい順を最優先し、同日内はウォッチ銘柄→法人/ファンド→保有比率が大きい順に優先し最大3件（通知疲れ防止のためLINEは絞り、残りはmicroCMSブログ「大口投資家の監視ブログ」（`kujira-watch/`、https://kujira-watch.com/ の詳細解説記事）のURLに委ねる。各行の下にはその銘柄の`/stocks/{code}`へのディープリンクを添える（`blog_stock_url()`。トップURLだけだと読者が銘柄を探し直す必要があったため）。流入はGA4で識別できるよう`utm_source=line&utm_medium=push`付き）、個人名の提出者は後回し）・🔍ユーザー別ウォッチ投資家の動き（`filer_watchlist`に登録した提出者名で部分一致照合し、その投資家がどの銘柄を動かしても通知。自己申告・過半数超は除外しない）・ユーザー別ウォッチリストのdp閾値アラート（ランキング本体の推奨が「🔴 売り検討」の銘柄は、個人のdp_sell_threshold設定値に関わらず必ず⚠️売り検討を表示。既定値20%はシステム全体の売り検討基準(drop_prob≥10%等)より緩いため、この上書きが無いと10〜20%の間で警告が沈黙するギャップが生じていた。閾値未達で変化のない銘柄は個別表示せず件数のみ要約し、前日比のdrop_prob変化があれば表示：通知疲れ対策）を配信 |
| `config.py` | 戦略パラメータ（`BASE_DIR`・下落相場判定 `BEAR_MARKET_THRESHOLD`・市場タイミング `MARKET_TIMING_20D_THRESH`）。学習時スクリーニングの閾値は`core/rf_train_v3.py`の`_SC_*`、バックテストは`tools/backtest.py`の`_SC_*`が保持する |
| `lib/utils.py` | 共通関数（get_prices, extract_features, add_cs_rank_features, recommend_from_scores 等）|
| `lib/db.py` | Supabase永続化層（gen_rankings / jpx_stock_list / yahoo_price_cache ほか）。`lib/supabase_client.py` のREST API経由（タイムアウト等の一時的なネットワーク失敗は指数バックオフで自動リトライ）。`upsert()`は全バッチ成功でTrue／1バッチでも失敗でFalseを返す。書けたかどうかがジョブの成否そのものである処理（`web/x_metrics.py`等）は必ず戻り値を見ること。送信直前に`_group_by_keys()`がキー構成の同じ行どうしへ分割する（PostgRESTは1リクエスト内でキーが不一致だとPGRST102で400を返しバッチ丸ごと落ちる。「値があるときだけ送る」列＝`issuer_name`・`short_term_transfers`等が混ざると必ず踏む。実例: 2026-08-26〜27、EDINET大量保有の全件が保存されずブログ記事が0件になった）。書き込みが失敗したら`_record_write_failure()`がテーブル別に記録し**その場でLINEへ通知**する（テーブルごと1プロセス1回。各ステップが`continue-on-error`で緑のまま進むため、`if: failure()`の通知では鳴らない）。`write_failures()`で失敗したテーブルと行数を取得でき、`tools/scan_large_holdings.py`は1行でも落ちていれば終了コード1で終わる。**テスト実行中は本番プロジェクトへの書き込み（upsert/insert_ignore/update/delete）を握りつぶす**（`_block_production_write()`。読み取りは通す。テストがURLを差し替えている場合も通す）。`tests/test_api_usage.py`がatexitの`flush()`で本番`api_usage`へ合成行（task="x" / $1.35）を書き、その1行だけで当日合計が日次予算を超えて翌営業日の記事生成が全便打ち切られる状態になった（2026-08-29）ため、テスト側のモックではなく書き込みの出口で止める|
| `lib/api_budget.py` | **Claude APIの利用上限フェイルファスト**。400の "You have reached your specified API usage limits" を検知すると同一プロセス内にフラグを立て、以降の呼び出しをAPIに投げる前にスキップさせる（`reached()`／`note(exc)`）。上限はリトライで直らないため、1件目で気づいて残りを諦めるのが正しい（2026-08-24の毎時実行では上限後も候補ごとに叩き続け、1回の実行で十数回失敗した末に記事が無言で欠落していた）。429や529などの一時的失敗では打ち切らない（SDKのリトライを殺さないため）。同じフラグは**日次予算の打ち切り**でも立てる（`stop_for_daily_cap()`。判定は`lib/api_usage.py`側）。上限を初めて検知した時点で`lib/notify.py`経由でLINEへ1回だけ通知する（同一プロセス内は`_notified`、プロセスを跨ぐ連投は`notify.push_once`の`dedupe_key`で抑制。毎時13便が同じ理由で落ちても1日1通）。**通知本文には原因と対処まで書く**（`_build_message()`）: 「Consoleで月間の**使用上限**を引き上げる」「**クレジット追加では解除されない**（残高不足ではなく上限設定に当たっている）」と、`regain_access_at()`がエラー文言から抜いた自動復旧日時（`You will regain access on ...`）。2026-08-24に実際にチャージで復旧を試みて空振りし、上限引き上げに気づくまで時間を要したため |
| `lib/api_usage.py` | **Claude API利用量の記録**。`record(resp, task=...)` が`messages.create()`のレスポンスからトークン数・キャッシュ・`server_tool_use.web_search_requests`を拾い、(UTC日付, ジョブ, タスク, モデル)単位でプロセス内に集計して、プロセス終了時(atexit)に`api_usage`テーブルへ追記する（`flush()`）。1呼び出しごとにHTTPを足さないため既存の処理時間に影響しない。コストは公開単価表（Haiku 4.5=入力$1.00/出力$5.00 per 1Mトークン、キャッシュ書込×1.25・読出×0.1、web_search $10/1,000検索）からの**推定値**でありAnthropicの請求額そのものではない。記録は計測であって目的ではないため、集計も書き込みも失敗を呼び出し側へ伝播させない（usageを持たないテストのMockを渡しても落ちず行も作らない）。書き込みのついでに当月(UTC)の累計を月次上限（`DEFAULT_MONTHLY_BUDGET_USD`=**$15**、環境変数`ANTHROPIC_MONTHLY_BUDGET_USD`で上書き、0で監視オフ）と突き合わせ、**50%/80%/100%を超えたらLINEへ通知**する（`check_budget()`）。通知は(月, 水準)単位で`notify.once()`が重複排除するため、毎時ジョブが同じ警告を何十通も送ることはない。さらに**日次予算**（`DEFAULT_DAILY_BUDGET_USD`=**$0.15**、環境変数`ANTHROPIC_DAILY_BUDGET_USD`で上書き、0で無効）を持ち、当日(UTC)の記録済み＋未送信の合計がこれを超えたら`flush()`してから`api_budget.stop_for_daily_cap()`で**その日の以降のClaude呼び出しを打ち切る**（`check_daily_cap()`、`record()`の中から毎回呼ぶ＝呼び出し側の付け忘れを無くすため）。当日の記録済みぶんはプロセス内で1回だけ読む（毎回読むと記事26本の便でSupabaseへ26往復する）。月次上限に当たると復旧まで1ヶ月止まるのに対し、日次なら被害が翌日UTC 0時までに収まる。上限は2026-08-30に$1.2から$0.15へ下げた（記事を日2本に絞ったため。記事本文1本の実測$0.0092×2本＋未取得の会社説明2社ぶん≒$0.12が定常の上振れ）。`lib/api_budget.py`は上限に**到達してから**止めるだけなので、その手前で気づくために置く。上限到達時に「どの用途がいくら使ったか」を示す記録が無く、バックフィルのログをgrepして犯人を推定するしかなかった（2026-08-23）ため2026-08-29に導入 |
| `lib/notify.py` | **異常のLINE通知の共通口**（LINE Messaging API push）。`notify.error()` / `notify.warn()` / `notify.push()`。認証情報（`LINE_CHANNEL_ACCESS_TOKEN` / `LINE_USER_ID`）が無ければ黙ってFalse、送信失敗も握りつぶして本処理を止めない。`python -m lib.notify "本文" --url <run_url> --dedupe-key <key>` でGitHub Actionsの`if: failure()`からも呼ぶ。**同じ原因の連投は`push_once(dedupe_key, text, window_hours=20)`が抑える**（Supabaseの`notify_log`に`dedupe_key`ごとの最終送信時刻を持ち、窓の内側なら送らず`sent_count`だけ積む＝何便ぶん黙ったかを追える）。毎時のワークフローが同じ理由で失敗し続けると本来13通届き、通知疲れで無視されれば検知の意味が無くなるため（2026-08-24のAPI上限超過が該当）。「一度きり」の状態通知（残枠50/80/100%）は`once()`＝**窓が無限の`push_once()`への委譲**で、実体は1つ（無限窓では時刻ではなく行の存在で判定する）。**重複判定ができないとき（Supabase未設定・障害）はどちらも送る側に倒す**——見張りの通知は「多い」より「来ない」ほうが致命的。全ワークフローの`if: failure()`通知に`--dedupe-key`を付与済み（x_postのみターゲット別）。Claudeにもワークフローの成否判定にも依存しない経路にしてある（2026-08-24、Anthropic APIの利用上限でブログ生成が全滅→記事0件→動画も投稿対象なしで無言停止した際、唯一の見張りだった日次ログレビューもClaude依存のため一緒に落ちて誰も気づけなかった）|。`once(dedupe_key, text)` は同じキーの通知を1回しか送らない（送信済みかどうかを`notify_log`に残す。毎時のジョブから残枠警告のような「状態」を鳴らすと同じ内容が1日に何十通も届き、本当に見てほしい1通が埋もれる）
| `lib/publish_ledger.py` | **「候補>0なのに公開0」の切り分け台帳**。記事生成の候補1件ごとに結末を1つ記録し、理由を正常な見送り（既報・基準未満・比率変化なし等）と異常（生成失敗・投稿失敗・権限エラー）に分類する。異常が1件でもあればLINE通知＋終了コード4、正常な見送りだけなら0。通知は理由の組み合わせを`dedupe_key`にして`notify.push_once()`へ渡す（毎時13便が同じ原因で落ちても20時間に1通）。件数だけを見る監視にしない理由は、edinet_blog.ymlが平日13便回り、そのほとんどが「候補数十件→公開0件」の正常な便だから（毎便鳴らすと誰も見なくなる）。理由を記録しないまま脱落した候補は`unclassified`として異常に倒すので、将来`continue`を足して記録を書き忘れても監視が腐らない |
| `lib/fundamentals.py` | point-in-time（先読みバイアスなし）ファンダメンタル再構成。`rank_stocks.py`/`rf_train_v3.py`/`backtest.py`で共用。`get_pit_fundamentals()`等は`rows`（銘柄のjquants_fin_summary全履歴）を渡すとDB問い合わせせずメモリ上でas_ofフィルタする（`rf_train_v3.py`が銘柄あたり約60サンプル日で呼ぶため、都度クエリだと数時間かかっていたのを銘柄ごと1クエリに削減）|
| `lib/data_sanity.py` | **Quality Assurance (QA)** ロール。リリースのたびにデータを検証。`check_ranking`（下落確率レンジ・予測多様性等の行レベル、rank_stocks/export_to_webで使用）＋`check_price_freshness`（複数日にまたがるclose凍結=更新漏れ検知、backfill_historyで使用）（alert-only：違反でも更新は止めずメール通知）|
| `lib/kabutan_earnings.py` | kabutan.jpから決算業績を取得（AI解析プロンプト用）|
| `lib/writing_style.py` | **AI生成文の「機械っぽさ」対策の共通文体ルール**。ブログ記事（publish_blog_articles/publish_buyback_articles）・動画ナレーション（build_script）のプロンプトに埋め込む禁止表現ルール（`JA_STYLE_RULES`/`NARRATION_STYLE_RULES`: 「注目が集まっています」「〜と言えるでしょう」等の常套句・同一文末の連続・接続詞頼みの文つなぎを禁止）と、生成後にAI常套句・単調文末（同一文末4連続）を検出する`find_ai_tells()`を集約。**再生成の判定には`include_monotone=False`（常套句と字数不足だけを見る）を使う**（2026-08-28の実測で再生成412回中335回が「ます。」4連続だけを理由にしており、引き直しても164記事中63記事で解消していなかった＝です・ます調では文末の55%が「ます」に偏るため。単調さの抑制はプロンプト側の「「ます。」で終わる文を4つ以上続けない」で行い、検出結果はログに出すだけにする）。検出時は記事生成側が1回だけ再生成し、字数充足 > 常套句の少なさ > 字数の優先度でマシな方を採用する（`body_quality_key()`）|
| `lib/article_text.py` | 既存ブログ記事の本文を扱う共通ヘルパー（可視文字数`visible_text_len()`＝HTMLタグと`<figure>`を除いた文字数、薄い記事の閾値`THIN_TEXT_THRESHOLD`(1000字)、銘柄コード＋開示日からの提出者逆引き`find_filer_names()`、旧本文の図の付け替え`restore_figures()`＝解説図は本文中へ・株価チャートは末尾へ戻す）。`tools/export_article_fact_cards.py` と `tools/apply_rewritten_articles.py` が共有する。元は `tools/rewrite_thin_blog_articles.py` に置いていたが、Anthropic APIに本文を書かせる同ツールを廃止した（2026-08-29）ため、APIを使わない側だけが残るようlibへ移した |
| `lib/article_redirects.py` | **削除した記事URLの引き継ぎ先の記録**。記事を消す3ツール（`cleanup_duplicate_blog_articles.py`＝残した方の記事へ、`delete_low_value_blog_articles.py`・`delete_articles_by_id.py`＝その銘柄ページへ）から呼ばれ、Supabase `deleted_article_redirects` に登録する。kujira-watchの記事詳細ページがmicroCMS 404のときだけ引いて308を返す。A→B→Cの多段リダイレクトを作らないよう、消した記事を指していた既存行は新しい行き先へ付け替える |
| `lib/gcp_auth.py` | GCPサービスアカウントのアクセストークン取得（`tools/ga4_clicks.py`のGA4と`tools/gsc_report.py`のSearch Consoleで共用）。鍵はローカルの`gcp_key.json`と環境変数`GCP_SERVICE_ACCOUNT_JSON`（CI用Secret）の両方から読む。スコープはAPIごとに違うため引数で渡す |
| `lib/risk_regime.py` | **相場リスク管制官**。日経20日・VIX・ドル円・S&P500からリスクオン/オフを判定。rank_stocksのフェーズ8でリスクオフ日はS買いを自動見送り、判定を `data/risk_regime.json` に保存しメールに警告表示 |
| `lib/market_compare.py` | **日経 vs S&P500 相対強弱アドバイザー**。日経225とS&P500の20日・60日リターン差から「日本株優位／米国株優位／拮抗」を判定(売買シグナルには影響しない参考情報)。rank_stocksのフェーズ8bで判定し `data/market_compare.json` に保存、`gen_market_compare`経由でLINE(`market_timing_alert.py`)に表示 |
| `tools/backtest.py` | バックテスト（先読みバイアスなし）。下落確率が低い順に選定。結果は `simulations/backtests/` に保存。`--drop-max`で下落確率上限、`--model-cutoff YYYY-MM-DD` でウォークフォワード用モデル指定可能 |
| `tools/multi_backtest.py` | 33期間一括バックテスト＋下落確率閾値比較分析（ウォークフォワード対応） |
| `tools/screen_catalyst_candidates.py` | カタリスト候補スクリーン（GARP補助）。PBR<1.0・ROE<8%・自己資本比率>50%・流動性の「安い箱」抽出は Postgres RPC `screen_catalyst_candidates()` でサーバーサイド集計（J-Quants財務データ使用）。通過候補に **利益の質フィルター(A/B)** で化粧決算（営業赤字・純利益>営業益×1.5）と斜陽事業（本業減益）を除外し、売上CAGR・営業利益率・会社予想方向で加減点。`data/catalyst_candidates.csv`（残）＋ `data/catalyst_excluded.csv`（除外理由付き・レビュー用）。`--no-quality` で品質フィルター無効 |
| `tools/catalyst_backtest.py` | カタリスト候補スクリーンのヒストリカルBT（point-in-time・disc_date≤基準日）。A/Bあり/なしで平均・勝率・大勝率を比較。データは J-Quants財務＋yahoo_price_cache |
| `lib/earnings_quality.py` | カタリスト候補の利益の質・本業方向性を判定（年次の営業益/売上/純益から化粧決算/斜陽を機械判定）。データ源は kabutan 優先、取れない環境（クラウドはkabutanがIPブロック）では J-Quants 実績にフォールバック |
| `lib/edinet.py` + `tools/scan_large_holdings.py` | **EDINET大量保有スキャナー**（イベント駆動）。EDINET APIから大量保有関連報告書(doc_type_code 350/360)を毎時スキャン（350は新規・変更の両方を含み360は訂正のため、種別は`doc_description`の接頭辞で新規/変更/訂正を判定＝`lib/edinet.py`の`disclosure_kind_label()`/`disclosure_doc_label()`。kujira-watch側`disclosureKindLabel()`と同一ロジックで、記事のfact_sheetにもこの判定を使う）（edinet_blog.yml、平日9:00-19:00 JST）して `edinet_large_holdings` に蓄積し、カタリスト候補と突合（構造的候補×実際の買い集め＝先回り候補）。突合時に自己申告（提出者≒対象企業）・過半数超(51%以上)・訂正報告書（既存開示の事後修正で実際の持分変動ではない。ただし届出比率が3pt以上動く大幅訂正＝`is_material_correction()`は既報の保有比率自体が誤りだったという情報のため記事化側では除外しない）・譲渡/売却の報告を除外し、外部の買い集めだけ残す（`--no-exclude` で無効化可）。「短期大量譲渡」（法第27条の25第2項）に該当する開示だけは原文に**譲渡の相手方・単価・数量**の表が付くため、`parse_short_term_transfers()`が解析して`edinet_large_holdings.short_term_transfers`(jsonb)へ保存し、`summarize_disposals()`が「誰にいくらで売ったか」と実額（単価×株数）に集計する（EDINETは通常「比率」しか開示しないため、金額が概算でなく実額で出せる唯一のケース。実例: 2026-08-25の日立製作所→日立建機9.98%売却は開示日終値ベースの概算1,274.9億円に対し、開示単価5,227円×21,462,310株の実額は1,121.8億円で、相手方はSMBC日興証券）。表に取得と処分が混在する開示（60日間の売買記録がそのまま並ぶケース）は差引きを復元できないため実額は採らず概算へフォールバックする。`is_sell_disclosure`/`is_individual_filer` は `market_timing_alert.py` のLINE通知セクションでも再利用（売却を除外せず方向性表示、個人名提出者を優先度で後回し）。買い/売りの方向判定はXBRLの直前保有割合(`holding_ratio_prior`)と現在の保有割合を比較して行い（概要欄の「譲渡/売却」等の文言が無い開示でも保有比率の減少を正しく売りと判定）、取得できない場合のみ概要欄のキーワードにフォールバックし、どちらも取得できない場合は買い/売りを推測せず方向性を表示しない。XBRL本表からは保有割合に加えて**保有目的・取得資金の内訳・保有株数・報告義務発生日**も取得する（`parse_holding_details()`）。取得資金の総額÷保有株数で**平均取得単価**（`average_acquisition_price()`、開示ベースの取得原価で現在株価と比べると含み損益の目安になる）、借入金÷取得資金で**レバレッジ比率**（自己資金0＝全額借入の買いが実在する）が出せる。保有目的の自由記述は`classify_purpose()`が重要提案行為等/経営参加/政策保有/安定株主/純投資の5区分に寄せる（「純投資及び状況に応じて重要提案行為なども行う」のような留保付きの記載は重要提案行為等側に寄せる）。共同保有では株数・発行済株式総数はXBRLの合算contextを使い、取得資金は合算contextが無いので提出者ごとの値を足し上げる。保存先は`edinet_large_holdings`の`purpose_of_holding`/`important_proposal`/`shares_held`/`shares_outstanding`/`funding_total`/`funding_own`/`funding_borrowings`/`obligation_date`（`supabase/add_holding_purpose_and_funding.sql`）。`EDINET_API_KEY` 必須 |
| `tools/reclassify_blog_articles.py` | **既存ブログ記事の投資家分類（dealType）一括再分類**（手動実行専用）。旧dealType体系（インサイダー買い/日系ファンド買い等）で公開済みの記事を、`classify_filer()`が返す新13分類へ移行する。各記事のstockCode+dealDateから`edinet_large_holdings`を逆引きしてfiler_nameを特定（同一銘柄・同一開示日に複数提出者がいて一意特定できない記事はスキップし一覧表示）、記事を全フィールド取得しdealTypeだけ書き換えて`update_article()`でmicroCMSをPATCH更新。`--dry-run`で変更内容の確認のみ可 |
| `tools/fix_misreported_blog_articles.py` | **誤って「新規保有」として公開された記事の是正**（手動実行）。EDINET開示の`holding_ratio - holding_ratio_prior`を正として、`ratioChangePct`・`dealAmount`・タイトル・tags・本文を作り直しmicroCMSへPATCHする（本文は`--keep-body`で据え置き可。株価チャートの`<figure>`は引き継ぐ）。対象は(a)`ratioChangePct`が実データとズレている記事、(b)前回比率が0より大きいのにタイトルが「新規保有」の記事の2条件のいずれかに当たるものだけ。是正後に`is_worth_publishing()`の基準を割る記事は`--delete`指定時のみバックアップ(logs/)を取ってから削除する。既定はdry-run、`--apply`で実行 |　検出条件は当初「ratioChangePctがEDINETとズレている」「前回比率>0なのにタイトルが新規保有」の2つだったが、2026-08-27の全965記事の再照合で取りこぼしが判明し2つ追加した: **(c)直前保有割合が欠損した開示で、記事が保有比率の全量を変化幅として使った痕跡があるもの**（過去開示から前回比率を補って判定。ratioChangePctを保存していなかった時代の記事はこの経路でしか検出できない。実例: AGC(5201)「293.7億円で取得」→実際33.2億円、電通グループ(4324)「189.8億円で取得」→11.1億円、日東紡績(3110)「73.8億円分を取得」→実際は4.2億円の売り）、**(d)保有比率が前回と同一（変化幅0pt）なのに推定取得/売却金額を出しているもの**（担保契約や共同保有者の変更で再提出された変更報告書で売買は起きていない。実例: 「Jトラスト、KeyHolderの約30%を取得—41.4億円」「日本レイが東テクの14.88%を取得、240億円超を投下」）。前回比率の参照は`fetch_disclosures()`が組み立てる`HISTORY`索引を引くだけで追加クエリを出さない。2026-08-27の実行で14件を訂正・12件を削除。2026-08-30に**(e)保有比率そのものがズレていた記事**を追加した: 共同保有の合算対応（`lib.edinet._aggregate_ratio`）で`holding_ratio`が「筆頭保有者の1枠」から「提出者＋共同保有者の合算」に変わったため、**新規の大量保有報告書**の記事でも見出しの比率・推定金額がズレている（実測: 2026-08-20以降の開示430件のうち212件＝49%で比率が変わった）。従来は前回比率が取れない新規開示を「全量が正しい」として対象外にしていたが、**前回0%として組み直し**、記事の`ratioChangePct`とのズレ(a)で拾う。比率が変わっていない記事は変化幅が一致するので対象外のまま。変更報告書で前回比率が取れないものは0%扱いすると全量が動いたことになるため従来どおり除外する。**本文の再生成は`generate_article_body_checked()`経由でAnthropic API（claude-haiku-4-5）を最大2回呼ぶ**（`--keep-body`のときだけ呼ばない）。実績で1本$0.0092なので、実行前に必ず件数と概算コストを出してオーナーの許可を取ること（CLAUDE.md）。**`--delete-untrafficked`**（2026-08-30）は、GA4で直近`--traffic-days`日（既定28日）のPVが0の是正対象を、本文を直さずに削除対象へ回す。是正対象が数百本あり本文の作り直しにAPIか人手が要る一方、実測で全1,119記事の28日PV合計は326・PVがあったのは135本だけだったため、読まれていない記事は誤った数字を載せたまま残すより消す（オーナー判断）。GA4のPVが取れない場合は「全記事がPV0」に見えて全部消えるため、取得失敗時は実行そのものを中止する。※GA4は人のアクセスだけを拾うのでPV0はインデックス無しを意味しない。2026-08-30に**非課金の経路`--fix-body-numbers`**を追加した（(e)の是正が数百本規模でオーナーが非課金運用を選択）。APIを呼ばず、本文中の保有比率・変化幅・推定金額の数字を新しい値へ文字列置換する（`rewrite_body_numbers()`。「1.22ポイント」「1.22pt」「0.7億円」の表記ゆれと、変化幅が本文では符号を持たないこと＝`ratioChangePct`は売りが負値、を吸収する）。置換できた項目だけでも本文へ反映し、旧値が本文に無かった項目は件数として出す（1項目でも見つからないからと本文ごと据え置くと、直せるはずの比率・金額まで古いまま残るため）。この経路は数字しか直さないので、**「約半分を占める」「過半」のような規模を語る記述が新しい比率と食い違う記事は`scale_phrase_conflicts()`が検出して一覧に出すだけで直さない**（言い回しの作り直しにはAPIが要る。対応表は`_SCALE_PHRASES`＝過半50-100%・約半分40-60%・3分の1 28-38%・4分の1 22-28%・4割37-43%・3割28-33%・2割17-23%・1割8-13%）。旧比率は記事側がフィールドに持たないため`_title_ratio()`が決定的テンプレートのタイトルから読む。
| `tools/strip_drop_model_mentions.py` | **既存記事から下落モデルへの言及を削除**（単発運用、2026-08-25）。記事本文の54%（583/921本）が「弊社モデルでは下落リスク水準を◯◯と評価」の類を書いていた一方、モデルの説明ページがサイトに無く、YMYLで検証不能な独自指標を判断材料として提示している状態だった。LLMには書き直させず**文を消すだけ**（該当文が「〜株価は1,234円で、」で始まる場合のみ株価の節を残して締め直す。それ以外は文ごと削除。空になった`<p>`は段落ごと落とす。タグを跨ぐ文は対応が壊れるため触らない）。送信前に**出力が入力の部分集合であること**（書き足しが無いこと）を検証する。`--apply`無しはdry-run、`-v`で変更前後を表示、反映前に`logs/`へバックアップ |
| `tools/cleanup_duplicate_blog_articles.py` | **ブログ重複記事クリーンアップ**（edinet_blog.ymlの投稿ステップ後に毎時`--days 30 --delete`で実行）。同一開示から二重投稿された記事の先発1件（X投稿等でリンク済みの可能性が高い）を残し後発を削除する。突き合わせキーは記事の世代で分ける: `filerName`入り（2026-08-15以降）は**銘柄コード＋開示日＋提出者名＋比率変化幅(`ratioChangePct`)**＝`already_published()`と同じキー（同一提出者が同日に複数報告書を出す実例＝2936 2025-08-13 橋本舜2件 は別イベントとして残す）、`filerName`が空の旧記事（2026-08-15以前）は**銘柄コード＋開示日＋タイトル**（旧記事は`already_published()`が概算金額`dealAmount`±0.05億円でしか突き合わせておらず、開示当日の終値が価格キャッシュに入る前後で金額が変わるたびに同じ開示が再投稿されたため、1開示あたり10件超の重複が残っている＝実例: 9706 日本空港ビルデングの`/stocks/9706`に同一記事が11件。旧記事はタイトルに提出者名と比率が入るためタイトル一致で同一開示と判定できる）。**自社株買い記事は発行体自身の開示で`filerName`を持たない**ため、`tags`に"自社株買い"を含む記事は銘柄コード＋開示日をキーに重複判定する（旧実装ではfilerName空を理由に丸ごと対象外にしており、`already_published()`のすり抜けを回収できずコンヴァノ6574の同一開示から13本が公開された＝2026-08-25修正）。銘柄コードか開示日が空の記事、旧記事でタイトルも空の記事は突き合わせ不能のため対象外。`already_published()`はmicroCMS API失敗時にFalseを返す設計のため重複投稿は稀に発生しうるが、このステップが自動回収する。`--days`（既定3、CIは30）でdealDateの遡り日数指定、`--all`で全期間（旧世代の重複回収に使う）、`--code 9706`で銘柄を1つに絞る、`--delete`無しはdry-run |
| `tools/backfill_article_redirects.py` | **削除済み記事URLの引き継ぎ先を過去ログから復元**（手動実行専用）。`logs/deleted_*.json`（削除ツールが残す全フィールドのバックアップ）から`id`と`stockCode`を拾い、`/stocks/<code>`へのリダイレクトを`deleted_article_redirects`に登録する。`--write`で実行（無指定はdry-run）。2026-08-29に257件を登録済み |
| `tools/backfill_short_term_transfers.py` | **短期大量譲渡の「譲渡の相手方・単価」バックフィル**（手動実行専用）。`short_term_transfers`列の取り込みは2026-08-26に実装したため、それ以前の開示は空。`doc_description`に「短期大量譲渡」を含む開示のXBRLを取り直して解析・保存する（EDINET APIのみでAnthropic課金は発生しない）。`--limit`で件数を絞れ、`--dry-run`で保存せず内容だけ確認できる。upsertには`issuer_code`(NOT NULL)を必ず同送する（PostgreSQLはON CONFLICT解決の前にNOT NULLを評価するため、`doc_id`だけの部分upsertは既存行があっても23502で落ちる）|
| `tools/backfill_article_publish_ledger.py` | **開示側の台帳`article_published_at`を実績から埋め直す**（手動実行。2026-08-27の一度きりのシード＋必要になったときの再構築用）。「この開示から記事を作ったことがあるか」をmicroCMSの記事有無だけで判定していると、低品質・リライト不能・誤報として**意図的に削除した記事**を取りこぼしと誤認して作り直してしまう（実測でbackfill候補424件中72件）。今ある記事（microCMS）と削除時バックアップ`logs/deleted_*.json`の両方を実績とみなし、`edinet_large_holdings`は(issuer_code, disc_date, filer_name)＋`ratioChangePct`と保有比率変化幅の一致で、`tdnet_buybacks`は(code, disclosed_at日付)で開示に紐づけて`article_published_at`を立てる。**開示を一意に決められない記事は印を付けない**（1本の記事で複数の開示を作成済みにすると、まだ記事の無い開示を永久に作れなくなる）。既定はdry-run、`--apply`で書き込み。2026-08-27実行で大量保有875行・自社株買い27行に記録 |
| `tools/backfill_article_filer_name.py` | **既存記事の`filerName`バックフィル**（手動実行専用、カニバリゼーション対策の前提）。`filerName`は2026-08-15追加のフィールドで、それ以前の記事791本中326本が未設定だった。`stockCode`+`dealDate`で`edinet_large_holdings`を逆引きし、同日・同銘柄の提出者が1人ならそのまま採用、複数いる場合は**記事タイトル→記事本文**の順に提出者名（法人格・全角/半角・中黒を正規化して突合、4文字未満の社名は誤爆するため対象外）が含まれるかで一意に絞り、絞れなければスキップする（タイトルが「個人投資家が3.5億円規模を売却」のように提出者名を出さない回でも、本文には「個人投資家の森久保哲司氏が」と書かれている。本文まで見ることで2026-08-21時点の未設定13件のうち10件が埋まる）。Supabaseの取得は`order=doc_id`を付けて1クエリでまとめて行う（order未指定だとPostgRESTのページングで行が取りこぼされ「候補なし」が73件に膨らむ）。既定は`--dry-run`相当で、`--apply`指定時だけmicroCMSへPATCHする。2026-08-19実行で313件を補完し未設定は13件まで減少 |
| `tools/backfill_blog_eyecatch.py` | **既存ブログ記事のアイキャッチ画像バックフィル**（手動実行専用）。2026-08-15〜08-22の`build_eyecatch_for_article()`が画像を`{"url":...}`オブジェクトで送っていたためmicroCMSに除外され、950件全件が画像なしで公開されていた（コミット61d2d2a4で文字列URLに修正済み、新規記事は付く）。microCMSを`filters=eyecatch[not_exists]`で新しい順に取得し、記事ごとに`generate_eyecatch_image()`→`upload_eyecatch()`→`update_article()`（PATCH）で`eyecatch`を設定する。保有比率は`edinet_large_holdings`を`stockCode`+`dealDate`+`filerName`で1クエリ引き、無ければタイトルの「保有比率X%」から正規表現で取る。バッジはtags「売り」→▼売却、「訂正」→■訂正、タイトル「新規保有」→▲新規取得、それ以外▲買い増し。Pexels無料枠（200件/時）対策で`--max-per-hour`（既定180）の間隔制御、`--limit`/`--days N`/`--index-only`（dealAmount≥3億円 or \|ratioChangePct\|≥1pt、articleIndexability.tsおよび`is_indexable_article()`と同基準。新規記事の足切りは2026-08-29に5億円/1.5ptへ上げたが、index基準は据え置きなのでそちらを見る）/`--dry-run`。**`--replace`** は逆に`filters=eyecatch[exists]`で「画像が既に付いている記事」を対象に画像を作り直す（2026-08-27に`search_pexels_photo()`の写真重複バグを直したため、それ以前に生成した964件を差し替える用途。写真は提出者名+銘柄名+開示日のハッシュで決まるので同じ記事は何度流しても同じ画像になる。保有比率が取れない記事も比率なしの表記で作り直す）。候補写真のキャッシュでPexels検索APIは分類の数しか叩かなくなったため、`--max-per-hour`の律速は写真CDNとmicroCMSへの連投側で、差し替え時は600程度まで上げてよい。 `--skip-log PATH`（複数可）は過去の実行ログの「→ OK」行から記事IDを拾って対象から外す（`--replace`は毎回microCMSから対象を取り直すため、中断して再開すると先頭からやり直しになり同じ画像のメディアが二重に積まれる）。ローカルMacにはNoto CJKが無いので`--font`または`EYECATCH_FONT_PATH`で「/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc」等を指定する |
| `tools/api_usage_report.py` | **API利用実績レポート**（手動実行専用）。`api_usage`テーブルを日別・タスク別・ジョブ別・モデル別に集計し、呼び出し回数／入出力トークン／Web検索回数／推定コスト（USD）と当月累計を表示する。`--days`（既定30、UTC基準）、`--by task|job|model|all`。当月の累計と月次上限に対する消化率・残枠も表示する。月次上限はUTC月初に戻るためJSTではなくUTCで集計する |
| `tools/traffic_report.py` | **アクセスログの機械/人の切り分け**（手動実行専用）。`blog_crawler_log`の"Browser"判定（既知botのUAでなくブラウザUAのもの）には機械が大量に混ざるため、4段階で除外して残りを表示する: ①**self**=自サイトからの自己リクエスト（`::1`/`127.0.0.1`）②**bot_ua**=UAに`bot`/`crawler`/`spider`/`externalagent`/`+URL`を含む自己申告（`crawlers.ts`の`BOT_PATTERNS`に載っていない新顔をここで拾い、名前を集計して表示するので登録漏れに気づける）③**heavy_ip**=1IPあたりPVが閾値（`--max-pv-per-ip`、既定100）超 ④**cookieless_ua**=同一UAで`--min-ua-pv`（既定20）以上のPVがあるのに`visitor_id`がPVとほぼ1:1（比0.9以上）＝クッキーを保持しない機械。残りのPV・IP数・2PV以上見た訪問者数・JST時間帯分布・記事ページ率・人気pathを表示する。**実測2026-08-28（1日）**: Browser 3,573PVの内訳がself 187 / bot_ua 1,384（meta-externalagent 1,031・Amazonbot 337）/ heavy_ip 1,421 / cookieless_ua 525で、残った人の候補は50〜100PV。同日の全リクエスト19,658件のうち81.8%はUAでbotと名乗っており、最大はGoogleOtherの13,497件。クッキー(visitor_id)の**有無**とJS実行(`/api/counter`)の有無は判別に使えないことを実データで確認済み（どちらも1人あたり74〜75PVでクローラーが同条件を満たす）ため、同じ検証を繰り返さないようdocstringに記録している（cookieless_uaは「有無」ではなく「PVに対する発行数の比」で見るので別物） |
| `tools/ga4_clicks.py` | **GA4クリックログの取得とPDCA比較**（手動実行専用）。`GaClickTracker`が全ボタン・リンクに仕込んでいる`click`イベントをGA4 Data APIで取り出し、①ページ別のクリック数とクリック率（**そのページを見た人のうち何かを押した人の割合**。分母をPVにすると1人が同じページで複数回押すぶんで率が100%を超えて読めなくなる＝実測で`/ranking/sells`が216.7%）②CTA別（ボタン文言=`label`）のクリック数 ③流入元別（`utm_source=line`/`youtube`のセッションとエンゲージメント率）を、いずれも**前の同じ日数との差分付き**で表示する。サーバーログ（`blog_crawler_log`）ではクリックを測れないため必要: 直近30日206,678PVのうち86.7%が1IPで100PV超の機械アクセスで、残りも1IP15.3PV、滞在1PVの訪問者15,092人に対し2〜10PVが76人という二極分布になる（`tools/traffic_report.py`）。認証はサービスアカウント（`gcp_key.json`、`GOOGLE_APPLICATION_CREDENTIALS`で差し替え可）。事前設定が3つ必要で、いずれも未了なら実行時に手順を表示する: ①GCPで Google Analytics Data API を有効化 ②GA4のプロパティのアクセス管理にサービスアカウントのメールを「閲覧者」で追加 ③`GA4_PROPERTY_ID`を設定。`label`はイベントパラメータなのでGA4のカスタムディメンション登録が別途必要で、未登録の場合はCTA別だけを飛ばして残りを表示する（2026-08-27に登録済み。カスタムディメンションは**登録日以降のデータしか集計されない**ため、それ以前のクリックは`(not set)`にまとまる）。`--days`で期間、`--limit`で表示行数。クリックが0件のときは終了コード1を返す（計測断を「異常なし」で流さないため）。認証はローカルの鍵ファイルに加え、環境変数`GCP_SERVICE_ACCOUNT_JSON`（SecretのJSON文字列）からも読む（鍵は`.gitignore`でリポジトリに入れていないため、CIはこちらを使う）。`collect_pdca_metrics()`は回遊指標を返す（2026-08-29の日次ログレビュー削除で定期的な呼び出し元は無くなり、現在は手動集計用。`page_group()`でページをTOP/記事/銘柄ページ/投資家ページ/データ・一覧などに畳み、種別ごとのPV・入口・内部・直帰率・滞在と、**1セッションあたりの内部移動回数**＝(全PV−入口セッション)÷入口セッション、**エンゲージセッション率**を前期間比つきで算出。内部移動回数はPV/セッションと違い入口の1ページ目を含まないので導線改善の効果が出やすい反面、**平均なので1人が何十ページも見た日に跳ねる**（実測: 26人の日=10.20回 / 56人の日=0.19回で、7日集計の前週比が-62%と出た裏でエンゲージ率は+23%・訪問者+20%だった）。**回遊の判定はエンゲージセッション率を主、内部移動回数を従とする**|
| `tools/gsc_report.py` | **Search Consoleの検索パフォーマンス取得**（手動実行専用）。SEO施策はバックテストできず判定材料がGSCの前後比較しかないのに、これまで数値はスクリーンショット共有頼みで定点観測できなかったため作った。Search Analytics APIから①全体（クリック/表示/CTR/平均掲載順位、CTRと順位は**表示回数で加重**＝行ごとの単純平均だと表示1回のクエリが上位ページと同じ重みになる）②ページ種別（記事/銘柄/投資家/ランキング…）別の内訳③上位クエリ④**CTR改善候補**（10位以内・表示20回以上・CTRがサイト平均未満＝titleの書き換えで最も早く効く層。「平均CTRなら+Nクリック」の取りこぼし順に並べる）⑤**あと一歩**（11〜20位・表示回数順）⑥上位ページと表示のあったURL数（インデックスされて検索に出ているページ数の下限）を、すべて前期間比つきで出す。集計が2〜3日遅れるため既定の期間は3日前まで。`--days`/`--limit`/`--sites`（権限のあるプロパティ一覧）。認証は`lib/gcp_auth.py`（GA4と同じサービスアカウント）。**事前設定2つ**: GCPで Search Console API を有効化 / GSC > 設定 > ユーザーと権限 でサービスアカウントを「制限付き」で追加 |
| `tools/geo_report.py` | **GEO（生成AI検索での引用最適化）のPDCAレポート**（手動実行専用）。「AIに読まれているか」と「AIから人が来ているか」を1画面で前期間と比較する。AIクローラーはJSを実行しないためGA4に一切載らず、巡回の実態は`blog_crawler_log`にしか無い。逆にAIの回答からリンクを踏んだ人の訪問はGA4（`sessionMedium=ai-assistant`）にしか無いので、両方を1つのレポートに並べる。出すもの: ①AIクローラー別の巡回数 ②ページ種別の内訳 ③`ChatGPT-User`/`PerplexityBot`が取りに来たページTOP（**回答生成のためのその場取得＝引用の代理指標**。学習・インデックス目的の`GPTBot`/`OAI-SearchBot`とは意味が違うので分ける）④AI経由のセッション・エンゲージメント率・着地ページ（GA4）⑤AIクローラーが当たっている**存在しないURL**とリダイレクト経由の旧URL。⑤はルートの形（`EXACT_PATHS`/`PREFIX_PATTERNS`）と突き合わせる実装で、ページを増減したらここも直す。**実測2026-08-29（14日）**: OAI-SearchBot 6,087回・PerplexityBot 1,777回・ChatGPT-User 1,141回に対し、AI経由の実訪問はGA4で6セッション（全体の1.4%）。廃止済みの`/disclosures`に58回・存在しない`/articles`に18回当たっていたのをこれで発見した。`--days`で期間、`--limit`で表示行数。GA4の鍵・プロパティIDが無い場合は④だけスキップして残りを表示する |
| `tools/x_follow.py` | **Xのフォロー候補抽出とフォロー実行**（手動実行専用。x_post.yml の `discover` / `follow` ターゲット）。2026-08-30時点でアカウントはフォロワー0・**フォロー0**・直近30日33投稿の平均インプレッション0〜3で、投稿フォーマットを変えても届く先が無い。`discover`はX API v2 `GET /2/tweets/search/recent`で「大量保有報告書」「アクティビスト/物言う株主」「自社株買い/株主還元」「日本株の決算」を検索し、発言しているアカウントをヒット数順に並べる（鍵アカと、フォロワー数が`DEFAULT_MIN_FOLLOWERS`〜`DEFAULT_MAX_FOLLOWERS`（300〜20万）の外は除外。読み取りのみでフォローはしない）。`follow`は`--usernames`で明示的に渡したアカウントだけを`POST /2/users/:id/following`でフォローし、`--execute`が無ければ対象を表示するだけ。**cronには絶対に載せない**（Xの自動化ポリシーは大量・無差別な自動フォローとフォロー/リムーブの反復を禁止しており凍結リスクがある）。1回`MAX_FOLLOWS_PER_RUN`=50件上限・3秒間隔で、401/403が返ったらプランか権限の問題として以降を中止する。認証は`web/x_client.py`と同じOAuth 1.0a（書き込み権限が必要）|
| `web/x_disclosure_facts.py` | **開示原文にしか無い事実のX投稿**（x_post.yml の `facts`。2026-08-30のX停止で手動のみ。該当が無い週は投稿しない不定期枠）。同ジャンル879投稿の実測で「誰が何%取得した」を流すアカウント5つが全て中央値エンゲージメント0〜2だったため、本表(XBRL)を開かないと分からない事実だけを出す枠として追加した。出すのは①**全額借入**（`funding_borrowings`>0 かつ `funding_own`==0）②**30日超の遅延開示**（`obligation_date`と`disc_date`の差）の2種類。**株価を一切使わないので、`web/x_followup.py`のような騰落率の不確実性が構造的に発生しない**。**「今年N件」のような希少性は書かない**（取得資金も報告義務発生日も本表を解析済みの開示にしか無く、それは2026年の開示の約2%。母数を全開示として件数を出すと実態と違う数字になる＝実測で「今年4件」と書きかけたが直近30日だけで5件あった）。**個人名義の提出者は除外**（`looks_like_individual()`。分類マスターに載っていない個人がいるため、法人と分かる語を含まない提出者は個人とみなす保守判定。私人を名指しで問題視する投稿を避ける）。全額借入は**新規報告書に限る**（変更報告書の取得資金欄は保有分全体の資金で、その回の買い増し分ではないため）。同じ銘柄は30日以内に繰り返さない。Anthropic APIは使わない。`--dry-run`/`--days`/`--list` |
| `tools/x_benchmark.py` | **同ジャンルのXアカウントの投稿を集めて伸びる型を見る**（手動実行専用。x_post.yml の `benchmark` ターゲット、読み取りのみ）。自分の投稿は平均インプレッション0〜3で母数がゼロのため、型の良し悪しを自分の数字では判定できない。同じ題材（大量保有報告書・適時開示・アクティビスト）を扱っていて既に読者がいる9アカウント（`DEFAULT_ACCOUNTS`）のタイムラインを`GET /2/users/:id/tweets`で取り、①アカウント別②エンゲージメント上位の投稿本文③条件別（画像の有無/URLの有無/文字数帯/ハッシュタグ数/投稿時間帯JST）の中央値を出す。**平均ではなく中央値**で比較する（1本のバズが平均を持ち上げると、たまたま伸びた1本の型を全体の傾向と誤認するため）。件数3未満のグループは表示しない。`--usernames`で対象、`--per-account`で1アカウントの取得件数、`--top`で本文を出す上位件数 |
| `tools/backfill_investor_profiles.py` | **既存投資家の分類・プロフィール一括バックフィル**（手動実行専用）。`edinet_large_holdings`に登場する提出者のうち、kujira-watch `/investors/[filer]`の解説文（`edinet_filer_classification.profile`、800〜1000字程度）が未設定の投資家をまとめて埋める。`edinet_filer_classification`に未登録（未分類）の提出者は`classify_filer()`で分類してから、分類済みだが`profile`未生成の提出者は`get_filer_profile()`のみを呼び出す。日次パイプラインは新規に記事化した提出者のみ都度分類・生成するため、記事化されずに大量保有履歴だけ残っている既存提出者を埋めるためのスクリプト。`--limit`件数上限、`--sleep`秒間隔（デフォルト1秒、レート制限対策） |
| `web/publish_blog_articles.py` | **ブログ記事自動生成・投稿**（edinet_blog.yml、平日9:00 / 15:00 / 18:00 JSTの3便、microCMSブログ「大口投資家の監視ブログ」`kujira-watch/`向け）。通常運転の投稿は**1日`DAILY_MAX_ARTICLES`=2本**まで（`daily_quota()`が当日UTCの`article_published_at`を数えて残り枠を出し、既存の`max_articles`へ渡す。上限で見送った候補は台帳に`SKIP_MAX_ARTICLES`＝正常な見送りとして残る。`--max-articles`の明示と`--backfill`（`BACKFILL_MAX_ARTICLES`件/便）は従来どおり優先）。2026-08-30に「GEOのA/B実験台」へ切り替えた際、母集団を日ごとに揺らさないため導入した。X投稿の呼び出しも同日に外した。株価更新パイプライン(daily_alert.yml)からは独立しており、開示当日のうちに検出・記事化する。`market_timing_alert.get_recent_large_holdings`（自己申告・過半数超・訂正報告書を除外。ただし届出比率が3pt以上動く大幅訂正（`is_material_correction()`、実例: 2026-08-18の太陽誘電6976が15.22%→4.41%で株価-11.5%）は残す。訂正記事は売買を伴わないため推定金額を付けず`dealAmount=0`・`tags`に"訂正"を立てて投稿し、タイトルも`〜が保有比率をX%に訂正｜訂正報告書`テンプレート、本文プロンプトも「売買があったと断定しない／訂正理由を推測しない」に切り替える（kujira-watch側は`isCorrectionArticle()`で金額欄に「訂正」と表示）。銘柄名は`data/code_name_map.json`優先、未収載の新規上場銘柄はEDINET開示のissuer_nameから法人格を除去して補う）からネタを取得し、保有比率の増減（取得できない場合のみ概要欄キーワード）で取得(買い)/売却(売り)の方向を判定（`is_sell_disclosure()`）した上で両方向とも記事化し、yfinanceの発行済株式数×株価×保有比率変化で取得・売却金額(億円)を概算（`shares_outstanding()`はyfinance側の一時的なレート制限対策として最大3回リトライし、`sharesOutstanding`が空ならJ-REIT等を想定して`impliedSharesOutstanding`にもフォールバックする。株価は`yahoo_price_cache`（スクリーニング対象ユニバースのみ）に無ければ`close_price_from_yfinance()`でyfinanceから直接取得する（新規上場銘柄・ユニバース外銘柄が「金額を概算できない」で落ちるのを防ぐ）。それでも株価・株式数のいずれかが取得できない銘柄はスキップ）。短期大量譲渡の開示で`summarize_disposals()`が実額を返した場合は概算を使わずその実額を`dealAmount`とし、見出しラベルからも「推定」を外す（`deal_amount_label()`）。あわせて**譲渡の相手方・単価・市場内外**を`format_transfer_facts()`が事実行としてプロンプトへ渡し、本文に「誰に売ったか」を書かせる。売り方向の記事はmicroCMSのスキーマ変更を避けるため`tags`に"売り"を追加して区別する（買い方向は従来通りtags不変）。プロンプト・見出しラベル（推定取得金額/推定売却金額）・末尾の推測文（「この取得が」/「この売却が」）も方向に応じて分岐させる。`classify_filer()`が提出者の投資家分類（個人/創業家の資産管理会社/公益・一般財団法人/プライムブローカー/アクティビスト/VC/PE・メザニンファンド/独立系ブティックAM/国内アセットマネジメント/外資系伝統運用会社/日系証券銀行/事業会社/その他）をSupabaseの`edinet_filer_classification`マスター（Web検索で確認済みの投資家分類テーブル、バックテスト分析とも共用）から参照し、未登録の提出者のみClaudeの一般知識で判定して結果をマスターへ保存（キーワード一致だけでは日系/外資やスペース無し個人名を判定できないため）。**分類が「個人」の提出者には`description`を持たせない**（Claudeの一般知識で書かせると同姓同名や似た名前の有名人と取り違えた経歴が生成され、実在の個人についての誤情報が記事本文に載る。実害: 2026-08-27に1,370件中45件で誤った経歴が保存され記事3本に載っていた。マスターにも空文字で保存する）。Claude（`ANTHROPIC_API_KEY`）には事実と分類済みdealTypeのみを渡して解説記事本文を生成しmicroCMSへ即時公開。事実の並置だけで終わらず投資家への示唆(so what)を加えられるよう、開示日の終値（`disclosure_close_price()`＝推定金額の概算に使うのと同じ値。以前は`gen_rankings`の株価を使っていたため、記事本文の「◯月◯日時点の株価」とサイトの「基準終値」が食い違っていた＝2026-08-19に統一）をプロンプトに文脈として渡す（取得できない銘柄は従来通り事実のみ）。**下落モデル（`drop_prob`）の水準は記事に渡さない**（かつては`gen_rankings`からPITで取った下落リスク水準(高/やや高/中/やや低/低)を渡して意味づけを1文書かせていたが、モデルの説明ページがサイトに無いまま検証不能な独自指標をYMYL（金融）の判断材料として提示する形になっており、2026-08-25の再監査で全記事の30%が「弊社モデルでは下落リスク水準を◯◯と評価」と書いていたため`dp_level_label()`ごと廃止した）。`ratio_change_pct()`が保有比率の変化幅（ポイント）をfact_sheetへ渡し（開示自体が持つ直前保有割合`holding_ratio_prior`を優先して使い、無い開示のみ同一銘柄・同一提出者の過去開示（直近400日）から算出する。ただし**変更報告書なのに`holding_ratio_prior`がまだ取れていない開示は`should_wait_for_prior_ratio()`がその便での記事化を見送る**（EDINETはメタデータ公開とXBRL本文の可用性にラグがあり、提出直後の便では前回比率が取れないことがある。その状態で記事化すると変化幅＝今回比率の全量となり「X%を新規保有」という誤ったタイトルと過大な推定金額が公開されたまま残る＝2026-08-19の監査で直近14日の照合可能56件中13件を検出。`PRIOR_RATIO_WAIT_DAYS`=2日を過ぎても埋まらない開示は、DBに同一提出者の過去開示があればそこから変化幅を再導出して記事化し、それも無い場合は**変化幅を確定できないものとして記事化を見送る**（待っても直前保有割合が入らない開示は特例報告に多く、2026-08-19の実測で直近90日に7件。変更報告書の`holding_ratio_prior`充填率は99.6%。従来はここで今回比率の全量を変化幅にしていたため、待ち日数を過ぎた開示が「X%を新規保有」＋過大な推定金額のまま公開されていた）。あわせて`is_new_holding()`は**報告書種別が変更報告書なら常に新規保有と判定しない**（変更報告書は提出者が既に5%以上を保有している届出なので、前回比率が取れなくても新規保有ではありえない））。履歴からの再導出だけでは全売却（比率0%）や履歴に同値が残るケースで変化幅0となり記事化されずに落ちていた＝2026-08-17の三菱商事によるＴＯＹＯ ＴＩＲＥ 20%→0%「短期大量譲渡」の取りこぼし）、過去開示が有れば「これまでの開示からXポイント増加/減少」、無ければ「直近400日以内に開示が確認できず実質的な新規保有（または大幅な保有再開）とみられる」という事実をプロンプトに含める（記事本文が同一投資家・同一銘柄でも毎回同じ言い回しの薄い内容にならないよう、既存で計算済みだが本文生成には使っていなかった実データを追加投入するSEO対策。GSC「クロール済み-インデックス未登録」対策として2026-08-14導入）。さらに`build_context_facts()`が**開示を横断して初めて書ける周辺事実**（①この提出者×この銘柄の開示履歴＝何回目・初回開示日と比率、②同じ提出者が同時点で5%以上持つ他の銘柄＝銘柄名・業種・比率の上位5件、③同じ銘柄の他の大株主＝提出者名・比率の上位5件、④開示日時点のPER/PBR/52週レンジ位置と業種）をSupabaseから集めてプロンプトへ投入する（2026-08-25追加。すべて`disc_date`以前に絞ったpoint-in-timeで取得。他銘柄の業種は必ず`jpx_stock_list`から引く＝社名から推測させると「業種別では唯一の化学メーカー」のような事実に無い記述が混ざるため）。**2026-08-24にAdSense審査が「有用性の低いコンテンツ」で不承認**になった際の実測で、全976記事が1,000字未満・中央値455字で、どの記事も「誰が何%取得・推定何億円・※推測」の3段落テンプレートになっていたことが原因だった（Googleの言う cookie cutter pages）。競合の大量保有報告書データベースには無い「投資家を軸にした横断的な事実」を材料として渡すことで独自性を出す。本文の目標字数も650〜900字から1,300〜1,700字に引き上げ、観点ごとに独立したセクション（各3文以上・200字以上）を書かせる（事実が無い観点は飛ばさせ、薄い文で水増ししない）。各セクションは`<h2>`見出しで始めさせ、見出しはその段落で何が分かるかを示す10〜20字の具体的な日本語にする（「まとめ」「考察」等の中身の無い語は禁止）。見出しを入れる前は全記事の94%が見出しゼロの一枚壁テキストで、AdSense再監査で「読者が拾い読みできない」と指摘された（2026-08-27）。既存記事の本文差し替えは`tools/export_article_fact_cards.py`（事実カード書き出し）→人間/Claude Codeが執筆→`tools/apply_rewritten_articles.py`（反映）のAPIを使わない経路で行う（APIに書かせる`tools/rewrite_thin_blog_articles.py`は2026-08-29に廃止）。`get_company_description()`が対象企業の事業内容をClaudeのweb検索（`web_search`ツール（`max_uses`=1。検索料$10/1,000検索＋検索結果が入力トークンとなり1社あたり約$0.017。2026-08-29に2→1へ削減）で会社概要を裏取りし`jpx_stock_list.description`にキャッシュ。生成できず空文字だった場合も`description_checked_at`に試行日時を刻み、`RECHECK_DAYS`(90日)以内は再試行しない＝空振りを記録していなかったため、バックフィルのたびに同じ「不明」社群へフル課金し直しており2026-08-23に月次上限へ到達した。一般知識のみで書かせていた頃は中小型株の約2/3が「不明」で空文字になり、`/trending`や`/stocks/[code]`で事業内容が出ない銘柄が大量に残っていたため2026-08-18にweb検索へ変更）。会社四季報の【特色】欄と同程度の密度＝2〜3文90〜130字で、主力事業・売上構成・製品/ブランド名・シェアや展開地域まで書かせる（1文40字では「何の会社か分からない」ままだったため2026-08-18に拡充。裏が取れない数値・シェア・順位の推測は禁止））から取得できた場合は冒頭の紹介文と保有比率の規模感（時価総額の一角を占める大株主、等）に自然に織り込む。`get_filer_profile()`が提出者のプロフィール（設立時期・運用方針・著名な投資事例など、800〜1000字程度）をClaudeの一般知識から取得し`edinet_filer_classification.profile`にキャッシュする（kujira-watch側`/investors/[filer]`の解説文として表示。情報が乏しい個人名義等は空文字のまま創作せず、`profile`キー自体を書かない＝空文字で確定情報として残さないため。ただし`profile_checked_at`は刻み、同じ提出者を記事のたびに引き直さない）。本文の最後には「この取得/売却が今後どんな意味を持ちうるか」の推測を必ず1文加えさせるが、事実と混同しないよう文頭に「※推測:」ラベルを付けさせ、事実として存在しない具体的計画やコメントの引用は創作しないよう明示的に指示する。金額が概算である旨・大量保有報告書制度の一般的な説明・「今後の動向を注視する必要がある」等の定型的な結びは、既に見出しや事実で伝わっているため本文で繰り返さないよう指示する（人間は事後にmicroCMS管理画面で修正する運用）。`build_price_chart_for_article()`が`yahoo_price_cache`から直近3ヶ月の終値を取得し、PIL（Pillowのみ、追加依存なし）で簡易な折れ線チャートPNGを描画してmicroCMSへアップロードし、本文HTML末尾に`<figure>`＋`<figcaption>`として埋め込む（株価取得・生成・アップロードのいずれかが失敗すればチャート無しで記事のみ投稿）。さらに`attach_figures()`が`web/article_figures.py`の解説図（保有比率の推移・同じ銘柄の他の大株主・提出者のポートフォリオ）を生成・アップロードし、**本文末尾ではなくその話をしている段落の直後**に差し込む（2026-08-25追加。画像がアイキャッチと末尾チャートの2枚だけで本文が文字の壁になっていたため）。サイト上部のカテゴリフィルターはdealTypeの値をそのままカテゴリ名として使う構成にしており、microCMSに`category`フィールドは持たない（CMS側の選択肢リストをdealTypeの分類と別途同期させる必要が無く、選択肢の同期漏れによる不具合が起きない）。記事タイトルはClaudeの自由生成ではなく`build_article_titles()`の決定的テンプレート（`銘柄名（コード）、提出者が保有比率X%に引き上げ/引き下げ｜大量保有報告書`、新規保有は`X%を新規保有`。60字超過時は提出者名を`…`で短縮）で組み立て、「銘柄名（コード）」「保有比率」「大量保有報告書」という検索語が必ず入ることを保証する（SEO/AIO 30日計画P1、2026-08-15）。本文の1文目も検索クエリへの直答文（`〜が保有比率をX%まで引き上げたことが大量保有報告書（EDINET）で分かりました。`）に固定してプロンプトで指示する。保有比率の変化幅は`ratioChangePct`（ポイント、売りは負値）としてmicroCMSにも送信し、フロントのファクトボックス表示に使う。英訳（`bodyEn`/`titleEn`）は2026-08-29に廃止した（英語版`/en`の廃止に伴う。1本あたりの出力トークンが約3割減る）。重複投稿の判定（`already_published()`）は銘柄コード＋開示日＋提出者名`filerName`＋比率変化幅`ratioChangePct`で突き合わせる（いずれも開示データから決まる値。以前は`dealAmount`で突き合わせていたが、推定金額は株価から都度概算されるため株価キャッシュ更新をまたぐと全銘柄でズレて重複判定が全滅する事故が2026-08-17に発生し17件が重複投稿された。同一提出者が同日に複数の報告書を出す実例もあるためratioChangePctの一致まで確認して別イベントを区別し、filerName未保存の旧記事に対してのみ`dealAmount`±0.05億円のフォールバックで判定する。その日その提出者の開示が1件だけの場合は変化幅の一致を問わず同一開示とみなす＝`unique_filing`、変化幅の算出ロジック変更で既報記事と再投稿がぶつかるのを防ぐ）。すり抜けた重複は`tools/cleanup_duplicate_blog_articles.py`が毎時回収する。照会は**開示日(`dealDate`)まで絞り込む**（銘柄コードだけで引いて`limit=50`を被せていた頃は、記事が50件を超える銘柄で既報が応答に入らず重複と判定できない穴があった）。照会に失敗したときは**既報扱いにして投稿を見送る**（`publish_buyback_articles.py`の同名関数と同じ方針。以前はFalseを返して投稿していたため、判定不能のまま重複記事がサイトに恒久的に残っていた＝実例: 9706 日本空港ビルデングに同一記事11件。見送っても本スクリプトは毎時走り直近`LARGE_HOLDINGS_DAYS`日の開示を毎回見直すので次の便で取り直せる）。既存記事の更新（`update_article()`、`tools/reclassify_blog_articles.py`等が使用）は2026-08-14よりPUT（完全上書き）からPATCH（差分更新）に切替（APIキーの権限変更でPUTが拒否されるようになったため）。アイキャッチ画像は`PEXELS_API_KEY`が設定されていれば、投資家分類に応じたPexels写真（`EYECATCH_QUERY_BY_CATEGORY`、銘柄固有の写真は現実的でないため分類のイメージに合う汎用写真を使用。検索結果の**候補80枚**から提出者名＋銘柄名＋開示日のハッシュで1枚を決定的に選ぶ＝常に`photos[0]`を使っていた2026-08-27以前は同じ分類の記事が全部同じ写真になっていた（実測40記事中8種類・1枚が25%を占有）。候補リストはクエリ単位でプロセス内キャッシュし、Pexels無料枠200req/時を消費しない。分類「個人」は実在の別人の顔写真が個人投資家の氏名と並んで本人の写真に見えるため、人物ポートレートではなく街並みのクエリにしている）に黒帯＋ニュースカード型テキスト（売買方向バッジ＋開示日／提出者名／銘柄名＋保有比率、Noto Sans CJK Bold太字白文字の3段組み。自由記述のタイトル文字列ではなく構造化した事実を焼き込むことでGoogle Discoverのカード面での視認性を狙う。2026-08-15、`generate_eyecatch_image()`/`build_eyecatch_for_article()`のシグネチャを`(category, card)`に変更）を合成してJPEG（約100KB）で保存し、microCMSのメディアアップロードAPI(`{domain}.microcms-management.io`)へアップロードして`eyecatch`フィールドへ**メディアURL文字列**で設定する（microCMSの画像フィールドは`{"url": ...}`のオブジェクトを受け付けず`'eyecatch' has unexpected data type`で除外される。2026-08-15〜08-22はこの型ズレで全記事が画像なしのまま投稿されていた＝2026-08-23修正。バッジの絵文字はNoto Sans CJKに無く豆腐になるため▲/▼/■に置換して焼き込む。保有比率0%の銘柄行は`_stock_line_text()`が出し分ける（売り記事は全株売却の事実なので「全株売却」、自社株買い記事で比率を取れず既定値0.0になった場合は数字を出さず銘柄名のみ。素の「0.00%」はどちらもデータ欠損に見えるため焼き込まない）。折り返しは数値トークン（`0-9.,%`の連続）の途中で割らない＝「13.41%」が「13.」「41%」に分かれて別の数字に読めていた。`PEXELS_API_KEY`未設定・取得失敗時は画像無しで記事のみ投稿）。`--dry-run`で投稿せず内容確認のみ可（アイキャッチ生成もスキップ）。`--backfill`は窓を`BACKFILL_DAYS`(30日)へ広げ、**まだ記事の無い開示だけを古い順（窓から外れる寸前のものから）に**拾い直す取りこぼし復旧モード（通常運転は直近3日しか見ないため、生成が3日を超えて止まるとその期間の開示は永久に落ちる）。30日窓は候補が1,000件を超えるので、①`fetch_published_index()`が既報記事の(銘柄コード, 開示日, 提出者名)を**1回のページング取得**でまとめて引き（候補1件ずつ`already_published()`を叩かない）、②`edinet_holding_amounts`（推定売買金額ビュー）で足切り基準に届かない開示を株価・発行済株式数を引く前に落とす（`is_backfill_target()`。ビューに行が無い開示は判定できないので残す）。既報インデックスが引けない/ページ上限で打ち切った場合はbackfillごと中止する（既報が分からないまま30日分を投稿し直す事故を避ける）。投稿数は`--max-articles`未指定時`BACKFILL_MAX_ARTICLES`(15件)で頭打ちにする（上限なしだとAPI月次上限に一撃で到達し古い記事が一度に並ぶ。15件は2026-08-27の実測＝直近30日の実取りこぼし約142件を1日1便で10日以内に消化できる線）。backfillではXへ流さない（数日前の開示がタイムラインに並ぶため）。**記事を作ったことがあるかは`edinet_large_holdings.article_published_at`（開示側の台帳）で判定する**（`is_backfill_target()`／通常運転のループ双方）。microCMSに記事があるかどうかだけで判定していると、低品質・リライト不能・誤報として**意図的に削除した記事**（2026-08-18に129件、08-25に74件、08-27に12件）を取りこぼしと誤認して作り直してしまう（実測でbackfill候補424件中72件が削除済み記事の復活だった）。台帳は投稿成功時に`lib.db.mark_article_published()`がPATCHで立て、記事を消しても消えない。`MICROCMS_SERVICE_DOMAIN`/`MICROCMS_API_KEY`（書き込み権限）必須、未設定ならスキップ |
| `web/article_figures.py` | **記事本文に差し込む解説図の生成**（Pillowのみ、API課金なし）。`build_context_facts()`が既に集めている事実をそのまま図にする: ①保有比率の推移（提出者×銘柄の開示ごとの縦棒。今回の開示だけ金色）②同じ銘柄に大量保有報告書を出している投資家の比較（横棒、今回の提出者を強調）③提出者が5%以上を保有する主な銘柄（横棒、今回の銘柄を強調）④自社株買い記事用の取得上限金額の推移（`buyback_article_figures()`、過去の決議が無ければ図なし）。配色・フォントは`web/x_card_image.py`と同じブランドトークン（navy/gold/paper/rule）を共用し、図の下端に出典（EDINET提出書類／適時開示（TDnet））と`kujira-watch.com`を焼き込む。データが2点未満の図は作らない（1本だけの棒グラフは情報量が無い）。`insert_figures_into_body()`が本文を`</p>`で分割し、図のanchor語（他の大株主名・保有銘柄名・初回開示年など）を最も多く含む段落の直後へ差し込む（見つからなければ均等配置）。1段落目＝検索クエリへの直答文の前と、最終段落＝「※推測」の締めの後には入れない |
| `tools/backfill_company_descriptions.py` | `jpx_stock_list.description`（会社情報カードの事業内容）が未設定の既存銘柄を`get_company_description()`で埋める。`--dry-run`／`--limit N`（分割実行）／`--codes 7203,9984`／`--recent-days N`（直近N日に開示があった銘柄のみ）／`--recheck-days N`（既定90日。過去に試して空文字だった銘柄はN日経つまで再挑戦しない。`0`で試行済みも全件やり直すが、1社あたり約$0.05かかるため`--limit`と併用すること）。**全件実行は月次上限を使い切るので必ず区切る**（2026-08-15〜18に4回走らせて上限到達） |
| `web/x_client.py` | **ブログ新着記事のX(Twitter)自動投稿**（**2026-08-30に停止**。`publish_blog_articles.py`の`main()`からの呼び出しを外した。定期投稿は土曜の週次まとめ（`x_weekly_trending.py`）1本だけ残っている。実測でインプレッション43・フォロワー0のため。コードは再開に備えて残す＝再開条件は`docs/x_operation_rules.md`の0章）。呼ばれていた頃の挙動: `main()`から投稿完了後に呼び出され、その回に投稿した記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページの「注目」枠と同じロジック）にも含まれる記事を金額規模順に**1回1件**（`ARTICLES_PER_RUN`。以前は3件で同一時刻に同型の投稿が3連続していた）投稿する。訂正報告書の記事（`tags`に"訂正"）は`dealAmount=0`で「注目」枠に入り得ないが既報の前提を覆すため**件数制限なしで全件**投稿し、同じ銘柄の直近の投稿（`x_posts`）が見つかればその投稿への自己リプライとしてぶら下げる。投稿してよいのはJST 8〜22時のみ（`within_posting_hours()`）。本文は記事タイトルの流用ではなく**1行目を「誰が・どの銘柄(証券コード)を・どうした」のフック**、2行目に「約N億円・保有比率 X%→Y%」、3行目に**開示日時点のPBR・ROE・配当性向**（`web/x_insight.py`の`build_valuation_line()`）、4行目に提出者の文脈（同`build_insight_line()`）。文字数に収まらないときは提出者の文脈→開示時点の数字の順に落とす（「なぜこの銘柄が狙われたか」を運ぶのは開示時点の数字のほうなので後に残す）。末尾は`#日本株 #大量保有報告書`の2タグのみ（`#EDINET`や記号を除いた`#社名`は検索母数が無いため廃止し、銘柄は本文に`社名(コード)`として素で書く）。**URLは一切入れない**（X APIの従量課金はリンク入り投稿が$0.20/本とリンク無し$0.015の13倍で、投稿コストのほぼ全部がこの加算だった。2026-08-22、14本で$2.2消費しクレジット枯渇。自己リプライにURLを置いてもそのリプライが$0.20になるため回避にならない）。誘導は末尾の`PROFILE_CTA`「詳細はプロフィールのリンクから」に統一し、URLはXプロフィールの固定リンク（`?utm_source=x&utm_medium=profile`付き）に集約する。`x_posts.variant`は`no_link`。動画クロス投稿（`build_video_tweet_text()`）もYouTubeのURLを入れない。**添付画像は数字カード（`web/x_card_image.py`）1枚だけ**（altつき）。かつては2枚目に株価チャートを付けていたが、Xは複数画像を左右に並べて両方とも切り落とすため、タイムラインで銘柄名も数字もチャートも読めなくなっていた（2026-08-19に1枚へ変更）。チャートはリンク先の記事に任せ、カードを作れなかった場合（フォント欠如等）だけチャートを代替として1枚添える。カードの銘柄名は社名が長くてもフォントを段階的に下げて`社名（証券コード）`を丸ごと収め、それでも入らない場合だけ社名側を削る（コードは銘柄検索の手掛かりなので消さない）。投稿に成功すると`post_tweet()`がtweet_idを返し`log_post()`がSupabaseの`x_posts`へ記録する（`web/x_metrics.py`が日次でインプレッション等を追記。これが無いとフォーマット変更の効果を検証できない）。加えて**日次サマリー投稿**（`post_daily_summary()`）: 毎時バッチのうち**9時JST（0時UTC）**の便のみ（時刻ガードが外部ストレージ無しの1日1回重複ガードを兼ねる）、**前営業日**（`summary_target_date()`。朝の便では当日の開示がまだ出ていないため。土日は金曜まで遡る）の全記事をmicroCMSから金額降順で取得し、件数・合計金額・最大買い増し・最大売却を一覧カード画像付きで1ポストする（0件の日は投稿しない）。**21時JSTから9時JSTへ移したのは**、同ジャンル9アカウント879投稿の実測（2026-08-30、`tools/x_benchmark.py`）で朝7-11時JSTの中央値インプレッションが2,080、夕16-19時が710、夜20-23時が1,094だったため。認証はOAuth 1.0a User Context（`X_API_KEY`/`X_API_KEY_SECRET`/`X_ACCESS_TOKEN`/`X_ACCESS_TOKEN_SECRET`）。いずれか未設定なら投稿をスキップ。401/403で失敗した場合は`verify_auth()`がv1.1 `account/verify_credentials`の`x-access-level`ヘッダからトークンの実権限（read / read-write）を引いてログに添える（実例: 2026-08-18）。`python -m web.x_client --verify`で手動確認も可（x_post.yml の `verify` ターゲット）。`--dry-run`実行時は呼び出されない |
| `web/x_post_format.py` | 週末のX投稿（下記2本）で共用する整形ヘルパー。Xの投稿上限280「単位」（全角2・半角1）に収めるための`weighted_len()`、EDINET正式名称を表示用に短くする`clean_name()`（全角英数のNFKC半角化＋和英の法人格除去）、単位数基準で切り詰める`label()`、記号を除いた`label()`の単位計算 |
| `web/x_weekly_activists.py` | **週次「今週のアクティビストの動き」のX投稿**（x_weekend_post.yml、日曜18:00 JST・週1回）。`edinet_filer_classification.category='アクティビスト'`の提出者（kujira-watch `/activists`と同じ母集団）について、直近7日(JST)の開示から提出者×銘柄ごとに「週初の保有比率→週末の保有比率」を集計し、変化幅(pt)の大きい買い増し・売却を載せて`/activists`へ誘導する。訂正報告書は持分変動でないため除外、変更報告書なのに直前保有割合が取れない行は「新規」と誤表示しないよう除外、週内に複数開示がある提出者×銘柄は正味の変化1行にまとめる。変化0.5pt未満は載せない。`--dry-run`で本文確認のみ可 |
| `web/x_weekly_trending.py` | **週次「大口投資家の取引急増ランキング」のX投稿**（2026-08-18に「クジラ急増ランキング」から改名。投稿見出しは「🐋 大口投資家の取引急増ランキング（前週比）」）（x_weekend_post.yml、土曜18:00 JST・週1回）。平日の記事投稿・日次サマリーが無い週末のタイムラインを埋める枠。kujira-watch `/trending`（`src/lib/trendingStats.ts`）の期間比較ロジックをPythonへ移植し、`edinet_large_holdings`から前週より開示が増えた銘柄（最大3件）・投資家（最大2件）を集計し、**推定売買金額（`edinet_holding_amounts`をdoc_idで引く）の大きい順**に投稿する（2026-08-27に増加件数順から変更。`/trending`と同じ並び）。金額を推定できない開示（訂正報告書等）だけ`+N件`にフォールバックする。金額の表記が「+N件」より長くなったぶんラベル上限を銘柄28→22・投資家48→42単位へ下げた（それでも280単位に収まらない週は従来どおり投資家の行から落ちる）。比較窓は**7日（前週比）**（30日窓だと隣り合う日曜の投稿で23日分のデータが重複し、毎週ほぼ同じランキングが並ぶため。2026-08-27に`/trending`側も7日窓へ揃えたので現在は両者同じ窓）。社名の「株式会社」等は表示用に除去し、Xの280単位制限（全角2単位）に収まるよう投資家→銘柄の順で行を自動削減。急増銘柄が無い週は投稿しない。`--dry-run`で本文確認のみ可 |
| `web/x_card_image.py` | **X投稿に添付する数字カード画像の生成**（1200x675、Pillowのみ）。レイアウトはCanvaで生成したデザイン案（2026-08-23）を移植したもので、**左45%をネイビーのパネル（ブランド名・バッジ・提出者・銘柄名、隅にクジラのシルエット）、右を紙色（「保有比率 旧%→」の上に112pxの新比率、罫線の下に推定金額）、下端を濃紺のフッター帯（免責・日付）**にした。以前はヘッダー帯＋白地のみでタイムライン上で他のテキストに埋もれていたため、面積の半分近くを濃色にして数字側との明暗差で視線を誘導する。CIはCanva APIを呼べない（認証なし・毎日自動生成）ためPillowで同じ見た目を描く。投資家行の「👤」は絵文字がNoto CJKで豆腐になるため付けない。配色は`kujira-watch/src/app/globals.css`・`video/remotion/src/theme.ts`と同じブランドトークンのみ（navy #16213a／navyDeep #0d1526／paper #fffdf8／section-tint #f1ece1／rule #ded5c0／gold #b8863a／買い #047857／売り #be123c）を使い、独自色は増やさない。記事投稿の1枚目に使う`build_deal_card()`（訂正報告書は金額を持たないため右下に訂正幅(pt)、バッジはネイビー地に沈まないよう金色）と、日次サマリー/週次ランキング/答え合わせで使う`build_list_card()`（ネイビーの見出し帯＋最大6行の一覧。行数が2件でも下半分が空かないよう行ブロックを領域の中央に置く）。銘柄名は左パネル幅に最大2行で折り返し、証券コードは「（7 / 203）」のように行またぎで割らず最終行の末尾に丸ごと残す（コードは銘柄検索の手掛かり）。提出者名はフォントを26→20pxまで段階的に下げて収める。フォントは`fonts-noto-cjk`（CI）→ヒラギノ（ローカル）の順に探し、見つからなければNoneを返し画像なしで投稿を続行する |
| `web/x_insight.py` | X投稿の**解釈行**（3行目）用のデータ取得。`edinet_filer_summary`の開示件数と`edinet_large_holdings`の同一提出者×銘柄の件数から「この提出者の開示は過去N件、〇〇ではM回目」「この提出者がEDINETに登場するのは初」を組み立てる。事実だけの自動投稿はbot扱いされてフォローされないため。推定損益ベースの「乗っかり実績」(`filer_win_rate`)は算出が誤っていたため2026-08-18に廃止済みで、ここでは使わない。取得失敗時は空文字＝行を出さない。加えて`fetch_valuation_context()`/`build_valuation_line()`が**開示日時点のPBR・ROE・配当性向**を返す（PBRは`gen_rankings`の開示日以前で最新の行、ROEは`jquants_fin_summary`の開示日以前で最新の**本決算(FY)**行の`np/equity`。四半期行の`np`は累計利益でROEが過小になるため使わない＝CLAUDE.mdのPIT規律）。アクティビスト・資本コストのテーマで既存アカウントに対して常に勝てるのは「全開示に必ず開示時点の数字が付く」ことなので型にした（直近実データで96/96件中92件に行が出る）。PBRが取れない銘柄（REIT等）はROE単独では資本コストの文脈にならないため行を出さない。「割安」等の評価的な語は付けない（`docs/x_operation_rules.md` 2） |
| `web/x_metrics.py` | **X投稿のメトリクス収集**（`x_post.yml` の `metrics` ターゲット。2026-08-30のX停止で定期実行は無くなり手動のみ）。`x_posts`に記録済みのtweet_id（直近30日）をX API v2 `GET /2/tweets`で引き、インプレッション・いいね・リポスト・返信・引用・ブックマーク・**リンククリック・プロフィールクリック**を`x_posts`（最新値）と`x_post_metrics`（日次スナップショット）へ保存する。`non_public_metrics`が権限で取れない場合は`public_metrics`のみで再取得する。あわせて`GET /2/users/me`で**アカウントのフォロワー数**を引き`x_followers`（1日1行）へ記録する。フォロワーは「インプレッション→プロフィールクリック→フォロー」の順にしか増えないため、投稿単位のプロフィールクリック率とアカウント単位のフォロワー増減の両方を測る。`--report`でフォロワー推移（7日前比・30日前比。記録が飛んでいる日はその日以前で最も新しい記録と比較）と種別×variantごとの平均（プロフィールクリック率込み）を表示し、投稿フォーマット変更の効果判定に使う。**Supabaseへの保存に失敗した場合も`SaveFailed`を投げて終了コード3を返す**（`sb.upsert()`は失敗してもログを出すだけなので、戻り値を見ていなかった2026-08-24〜25はkindのNOT NULL違反で18行が毎日落ちてもworkflowはsuccessだった）。**401/402/403のように待っても直らない失敗（認証NG・APIクレジット切れ・プラン外）は`MetricsUnavailable`を投げ、`run()`が終了コード2を返してCIを落とす**（2026-08-22に、402 credits depletedのまま4日連続でworkflowがsuccess・メトリクス0件という状態を検出したため。空dictを返して成功扱いにすると「毎日動いているのに数字が無い」ことに気付けない）。500等の一時的な失敗は従来どおり例外にせず次回の便に任せる。`x_posts`への保存は**tweet_idと一緒に`kind`も送る**（PostgRESTのupsertは`INSERT ... ON CONFLICT DO UPDATE`で、既存行の更新でもINSERT側の行に対してNOT NULL制約が先に評価されるため、`kind`を省くと全行まとめて23502で落ちる。2026-08-24〜25に18行が全滅し最新値が欠測した）。取得元の行に`kind`が無い場合だけ`unknown`で埋める |
| `web/x_followup.py` | **週次「答え合わせ」投稿**（x_followup.yml、水曜21:00 JST）。約3ヶ月前(91日前付近)に大量保有報告書が出た銘柄群について、`yahoo_price_cache`から「開示日の終値→直近終値」の騰落率を計算し、平均・中央値・上昇銘柄数・最も上げた銘柄・最も下げた銘柄を投稿する（勝った銘柄だけを出さない）。基準終値が開示日から7日以上離れる銘柄と、日次±40%以上動く銘柄（株式分割・併合で終値が不連続）は除外する。対象日に開示が5件未満なら投稿しない。`--dry-run`で本文確認のみ可 |
| `lib/buyback.py` | **自社株買い開示の分類・取得枠抽出**。`ext_tdnet_disclosures`（category=自社株買い）のタイトルで「決定」（取得枠の新設・ToSTNeT-3買付）と「進捗」（月次の取得状況報告）を分け、決定開示の原文PDF（TDnet）を`pypdf`でテキスト化して上限株数・上限金額・発行済比率・取得期間・方法・消却有無を正規表現で抽出（取れない開示のみClaude Haikuで補助）し`tdnet_buybacks`に保存。TDnetのPDFは公開から約1ヶ月で404になるため日次で回す。kujira-watch側は `src/lib/buybacks.ts` が `/stocks/[code]` の履歴表と TOPタブ `/buybacks`（直近30日の決定を開示日の新しい順に一覧、FAQ）で参照。タイトル分類`classify_buyback_title()`は`amendment`（変更・訂正・中止）→`progress`（取得状況・買付結果）→`decision`（新規の取得枠決議）の順に判定する。`_DECISION_RE`の「取得に係る事項」が「…の一部変更」「（訂正）「…決定に関するお知らせ」の一部訂正」にもマッチするため、決定より先に弾かないと既存決議の後追い開示が新規決議として記事化される |
| `tools/enrich_buybacks.py` | `lib/buyback.py`のCLI（daily_alert.yml Step 2f2・edinet_blog.yml・x_post.yml(buyback)から実行）。`--days N`／`--retry-failed`（抽出失敗行の再試行）／`--dry-run` |
| `web/x_buyback.py` | **「本日の自社株買い決定」のX投稿**（x_post.yml の `buyback`。2026-08-30のX停止で手動のみ）。`tdnet_buybacks`の当日分（JST）を上限金額順に並べ、上限1億円未満は除外。自社株買いは引け後（15:30〜17:00）に集中するため、投稿前に`lib/tdnet.scan_disclosures`と`lib/buyback.enrich`を回して当日分を取り込む。カード画像は`x_card_image.build_list_card` |
| `web/publish_buyback_articles.py` | **自社株買い決定のブログ記事自動生成・投稿**（edinet_blog.yml、`publish_blog_articles.py`の後段）。通常運転は**1日`DAILY_MAX_ARTICLES`=2本**まで（`publish_blog_articles.daily_quota()`に`tdnet_buybacks`を渡して当日の実績を数える。大量保有の記事とは別枠）。上限10億円以上 or 発行済3%以上の決定開示を対象に、対象は`lib.buyback.classify_buyback_title()`が`decision`と判定したものだけ（取り込み時だけでなく`fetch_candidates()`でも判定する。`tdnet_buybacks`に取り込み済みの行は取り込み時の分類のまま残るため。実例: 8560 2026-08-18「自己株式取得に係る事項の一部変更」は取締役会決議が2026-02-09なのに上限11億円・9.12%で閾値を超えており、記事にすると「2026-08-18に決議した」という誤報になる）、事実のみからHaikuで本文を生成。`dealType=自社株買い`（提出者分類ではなく発行体自身の取得。kujira-watch側も同日追加）、`dealAmount`=上限金額（億円）、`ratioChangePct`=発行済比率、`filerName`は付けない。X個別投稿はしない（`x_buyback.py`が担う）。アイキャッチ・株価チャート・解説図（`web/article_figures.buyback_article_figures()`＝過去の決議と今回の取得上限金額の比較）は`publish_blog_articles`の共通処理を使う。銘柄名は`jpx_stock_list`優先、載っていなければ`lib/tdnet.fetch_company_name()`のTDnet会社名で補う（`jpx_stock_list`はJPX＝東証の一覧なので福証・名証単独上場が載らず、銘柄名が引けないだけで決定開示を永久に取りこぼしていた＝実例8560 宮崎太陽銀行・3066 JBイレブン。TDnet側は取引所の略称「宮崎太銀」だが、開示を丸ごと落とすよりは略称で載せる）。`--backfill`は`publish_blog_articles`と同じ取りこぼし復旧モード（窓30日・既報インデックスは`tags[contains]自社株買い`で引く・古い開示から順に・`BACKFILL_MAX_ARTICLES`件で頭打ち・インデックスが引けなければ中止）。記事を作ったことがあるかは`tdnet_buybacks.article_published_at`（開示側の台帳）でも判定し、投稿成功時に立てる（意図的に削除した記事を作り直さないため） |
| `video/build_script.py` | **自動動画投稿の台本生成**（video_post.yml。2026-08-30に定期実行を停止し手動のみ）。microCMSに直近36時間で新規公開された記事のうち`publish_blog_articles.get_featured_article_ids()`（ホームページ「注目」枠と同じロジック）にも含まれるものを`dealAmount`降順で1件だけ選び（`pick_article()`。X投稿と同じ「新着×注目」の積集合で、サイト上目立っていない小粒な開示だけが動画化される事態を防ぐ。積集合が空の日は動画を作らない）、記事本文＋Supabaseにキャッシュ済みの補足事実（`get_company_description()`の事業内容・`get_filer_profile()`の投資家プロフィール、どちらもpublish_blog_articles.pyが生成したもの）だけを根拠に、Claudeで**ナレーション付き台本**を生成する。台本は hook→filer（どんな投資家か）→change（前回からの変化）→deal（金額・保有比率）→company（どんな会社か）→cta の6シーン（`SECTION_SPEC`）＋株価チャートで、**この並びは維持率の実測で決めている**（2026-08-30、8本）: 3秒残存は91〜99%でhookは落ちておらず、全8本が10秒地点で53〜75%まで落ち、最急の落ち込みは実時間5〜9秒＝hook直後の第2シーンに集中していた。そこに置いていたのが company（開示の続報ではなく背景説明）だったため、続報（filer→change）を前へ、背景説明（company）を後ろへ回し、hookと内容が重複する deal は中盤に下げた。2番目を change ではなく filer にしたのは、change は前回比率が取れない回にシーンごと落ちて測りたい枠が回によって変わってしまうから。各シーンは `narration`（読み上げ文、hookは22〜30字・本編は35〜55字・締めは14〜20字）と `caption`（画面に出す字幕、26字以内）の対。字数超過は最大3回まで作り直し（作り直しでは「何字の文が長すぎたか」をプロンプトに足して伝える。同じ指示をそのまま投げ直すと同じ長さが返り、2026-08-19に台本が2回とも86〜93字で戻って投稿0件になったため）、それでも超える場合はcaptionは末尾を詰め、narrationは句点境界で切る（`_trim_narration()`）。作り直しでも文の途中で終わったシーンは、**シーンごと落とすか記事の事実だけの定型文に差し替える**（`salvage_scenes()`）。hook/deal/change/ctaは定型文で組み直し、言い換えが必要なcompany/filerは落とす。動画そのものを諦めるのはhook/ctaを組み直せない場合だけ（1シーンの失敗で動画を丸ごと捨てていたため2026-08-19・20と2日続けて投稿0件になった）。切れた「…」を画面と読み上げに出さないこと（`is_broken_narration()`）と毎日1本出すことを両立させる。前回の保有比率は本文の末尾から2番目の`◯.◯◯%`から拾い（`extract_prev_holding_ratio()`）、取引の向きと矛盾する場合はNoneにしてchangeシーンごと落とす。※outlook（今後の推測）シーンは2026-08-19に廃止（中央ビジュアルが無く尺だけ食っていたのと、投資助言に寄るリスクのため）。kindはClaudeの出力に頼らず期待順で上書きする（`_flatten_scenes()`）。提出者名は`resolve_filer_name()`が返す: microCMSの`filerName`が空の旧記事（2026-08-16以前）は、同じ銘柄・同じ開示日の大量保有報告書の提出者を候補に挙げ、**記事本文に名前が書かれているもの**だけを採る（同一銘柄・同一開示日に複数の提出者がいるのが普通で、開示データだけでは一意に決まらないため。誤った提出者名を動画タイトルに載せないことを優先し、決まらなければ空文字→「大口投資家」という総称にフォールバック） |
| `video/background.py` | **背景映像の調達（Pexels Videos API）**。海4＋抽象/都市/自然の8クエリのプールから、縦向き・7秒以上・80MB以下・**人物が写っていない**動画を最大2本ダウンロードし（`fetch_pool()`）、`company`と`filer`の2シーンにだけ割り当てる（`assign_backgrounds()`。金額・保有比率・株価チャートを読ませるシーンはRemotion側のブランドグラデーション背景に固定し、実写の明部に数字が沈む事故を構造的に無くす。2026-08-19）。人物の除外はPexelsの動画URLスラッグを語単位で判定する（`has_rejected_subject()`。部分一致だと`germany`の`man`で誤爆するため）。Pexelsは無料・商用可・クレジット不要。`PEXELS_API_KEY`未設定・全滅時は空リストを返し、全シーンがグラデーション背景になる |
| `video/remotion/` | **縦動画のRemotionプロジェクト**（React/TypeScript）。コンポジション`ArticleShort`は1080x1920・30fpsで、**尺は固定ではなく各シーンのナレーション音声の長さで決まる**（`calculateMetadata`と`ArticleShort.tsx`が同じ式`sceneDurationSec()`で総フレーム数を算出。音声が無い場合は読み上げ文字数から概算。実データで約40秒）。ショート動画運用の定石を反映: (1)表示はすべて`safeArea`内（上200px・下470px・左右160pxの**左右対称**。以前は左70/右190で中央寄せが60px左にずれていた）、(2)冒頭は約0.35秒で金額を叩き込み、社名→動詞→提出者ラベル→保有比率と約2.7秒で4回情報を足す（静止画で待たせない）、(3)無音視聴者向けの字幕は26字の要約1本を68pxで出す（ナレーション全文の同時表示は読めず音声と競合するため2026-08-19に廃止）、(4)文字の可読性は影ではなく不透明の下地（`PLATE_BG`）で担保、(5)背景はKen Burnsとシーン内マイクロビート（2.6秒周期）で1フレームも完全静止させない、(6)締めの末尾0.8秒は冒頭と同じ金額組版に戻してループ再生で頭と繋げる（`LOOP_TAIL_FRAMES`。`HookVisual`の完成形を再利用するので1pxもずれない）、(7)EDINETの書類名・提出日とサイトURL・免責を全編常時表示。締めの名乗りは`kujira-watch/src/lib/site.ts`の`SITE_NAME`（大口投資家の監視ブログ）に合わせ、動画側で別名を作らない。検索誘導はしない（サイト名で検索上位を取れていないため辿り着けない）。効果音とBGMは`props.sfx`が真のときだけ鳴る（BGMは`volume`関数で頭20フレームをフェードイン・末尾14フレームをフェードアウトし、`loopVolumeCurveBehavior="extend"`でループしても音量カーブが動画全体の時間軸で効くようにする）。配色は`src/theme.ts`が`kujira-watch/src/app/globals.css`と同じブランド色を持ち、買い=金・売り=赤のアクセント。日本語フォントはOS側のNoto Sans CJK（CIはapt導入、macOSはHiragino）を使い、レンダリングがネットワークに依存しない |
| `video/render.py` | props JSONを`npx remotion render`へ渡してmp4を書き出す薄いラッパ。ナレーション音声（tts.pyが生成したwav）はRemotionの`staticFile()`経由でしか参照できないため、`video/remotion/public/`へコピーしてからレンダリングし、終了後に削除する（`_stage_audio()`。見つからない音声はそのシーンだけ無音にして続行）。`articleId`/`articleTitle`は投稿テキスト専用でコンポジションのpropsには無いため除外して渡す（`NON_PROP_KEYS`）。書き出し前に`video/audio_gen.py`が効果音とBGMのwavを生成して`public/`へ置き、`props.sfx`で鳴らすかどうかを伝える。読み上げ文が文の途中で切れているprops JSONは書き出しを拒否する（`has_broken_narration()`。古いpropsの再レンダリング経路の保険）。初回実行時のみRemotionがChrome Headless Shell(約150MB)を自動ダウンロードする。**音量正規化にはffmpegが必要**で、Ubuntu 24.04のランナーイメージには入っていないためワークフローで明示的に導入する（未導入だと無言でスキップされ-25 LUFSのまま投稿される。2026-08-21に発覚） |
| `video/audio_gen.py` | **効果音とBGMの自前生成（numpy）**。カット頭の無音が「再生バグに聞こえる」・BGM無しが「未完成品に見える」という指摘が最多だったため、シーンの切り替わり（`se_whoosh.wav`）・金額の着地（`se_impact.wav`）・カウントアップ完了（`se_tick.wav`）・全編のBGM（`bgm.wav`）を波形合成で作る。フリー素材を毎日ダウンロードするとライセンス確認が自動化できず規約変更にも気づけないため、外部素材は一切持たない。BGMはAm→F→C→Gの12秒アンビエントパッドで、和音を枠からはみ出させて配列の先頭へ回り込ませることで継ぎ目なくループする（ローパスの内部状態も2周ぶん通して定常化させてから採る。そうしないと先頭だけ音が痩せてループのたびにプチッと鳴る）。乱数を固定してあるので毎回同じファイルになる。numpyが無い環境では`False`を返し、音の追加なしで動画を書き出す |
| `video/post_text.py` | **投稿テキストの共通部品**。サイトの名乗り（`SITE_NAME`＝大口投資家の監視ブログ。`kujira-watch/src/lib/site.ts`と対）・URL・UTM付き記事URL（`article_url()`）・ハッシュタグの整形（`hashtag()`）を1箇所に集約する。名乗りを各クライアントに直書きすると、動画側で実在しない「クジラウォッチ」を名乗った事故（2026-08-19）と同じことが投稿文でも起きるため。`hashtag()`は銘柄名の空白・「．」・中黒を落とす（例: `Ｊ．フロント リテイリング`をそのまま`#`に続けるとタグが途中で切れて残りが本文として漏れる） |
| `video/thumbnail.py` | **YouTubeカスタムサムネイルの合成（Pillow）**。Canvaで作ったブランド台紙 `video/assets/thumbnail_base.png`（ネイビー地・クジラのラインアート・上昇チャート、下半分は無地）に、半透明の下地を敷いて「銘柄名（証券コード）」「買い/売り ◯億円」「提出者」を重ねる。Shortsフィードでは動画のフレームが使われるが、検索結果・チャンネルページ・横長のおすすめ枠ではカスタムサムネイルが出るため、銘柄と金額が一目でわかる絵にする。投稿後に`youtube_client.set_thumbnail()`（thumbnails.set API）で設定し、失敗しても投稿の成否には影響させない。台紙や日本語フォントが無ければNoneを返してスキップ。締めシーンのエンドカード `video/assets/cta_endcard.png`（同じくCanva製、1080x1920）も同じ素材群で、`render.py`の`_stage_end_card()`が`remotion/public/`へコピーして`props.endCard`に書き込み、Remotion側（`ArticleShort.tsx`の`EndCard`）がctaシーンの背景に敷く（ループ末尾の冒頭再現区間は除く）。画像に名乗りとピルが描き込み済みなので、ctaではヘッダと「大口投資家の監視ブログ」の文字を出さずURLと音声クレジットだけを重ねる。素材が無い場合は従来のテキストだけの締めにフォールバック。素材の再生成はCanva MCP（`generate-design` → `export-design`）で行い、生成物だけをリポジトリに置く（CIからCanvaは呼ばない）（2026-08-23） |
| `video/youtube_client.py` | **YouTube Shortsへの自動アップロード**（YouTube Data API v3のresumable upload）。認証はOAuth 2.0リフレッシュトークン方式（`YOUTUBE_CLIENT_ID`/`YOUTUBE_CLIENT_SECRET`/`YOUTUBE_REFRESH_TOKEN`。ローカルで`video/youtube_auth.py`を1回実行して取得）。YouTubeは縦長かつ3分以内の動画を自動的にShortsとして扱うため専用フラグは不要だが、保険としてタイトル・説明文に`#Shorts`を入れる。説明文は**1行目を「▼{銘柄名}の保有推移・提出者の全開示はこちら」、2行目を記事URL**にする（Shortsは畳まれた状態だと先頭1行しか見えない。2026-08-19に8行目→3行目へ上げ、2026-08-30に1行目を動画内字幕の再掲から誘導文へ差し替えた）。投稿直後に`post_comment()`で記事URLのコメントを自チャンネルから1件残す（Shortsの説明文は開かれないため。実測で再生に対するサイト流入が約0.5%だった。`utm_source=youtube_comment`で説明文経由と区別する。scope`youtube.force-ssl`が要る。ピン留めはData APIに機能が無いのでStudioで手動）。記事URLには`utm_source=youtube`を付与しGA4でShorts経由の流入を識別できるようにする。タイトルには保有比率も入れる。説明文の先頭3つのハッシュタグはタイトル上部に表示される枠なので、`#Shorts`のような機能タグではなく検索される語（`#日本株` `#大量保有報告書` `#銘柄名`）を先に置く。いずれかの環境変数が未設定ならスキップ。投稿後に`set_thumbnail()`でカスタムサムネイル（`video/thumbnail.py`）を設定する |
| `video/tts.py` | **ナレーション音声の合成（VOICEVOX）**。無料・登録不要・商用利用可（クレジット表記のみ必須）の日本語音声合成エンジンで、既定の話者はずんだもん（speaker=3、`TTS_SPEAKER`で変更可、`TTS_SPEED`既定1.22倍速（1.3超は金額・比率の聞き取りが落ちるためこれを上限とする））。エンジンはHTTPサーバ（既定 http://127.0.0.1:50021、`VOICEVOX_URL`で変更可）として動き、CIでは公式Dockerイメージ`voicevox/voicevox_engine:cpu-ubuntu20.04-latest`をジョブ内で起動する。`narrate_sections()`が各シーンのnarrationをwav化して`audio`/`durationSec`（ffprobe計測、無ければ文字数から概算）を書き込み、この長さがそのままシーンの尺になる。エンジンに繋がらない・1シーンでも合成に失敗した場合は全体を無音扱いにして動画生成は続行する（一部だけ音が出る動画はかえって不自然なため）。クレジット表記「VOICEVOX:ずんだもん」はCTAシーン・YouTube説明文に自動で入る。※当初はGoogle Cloud TTSで実装したが、GCPプロジェクトに請求先が未設定でAPIを有効化できずVOICEVOXへ切替（2026-08-15） |
| `video/publish_video.py` | 自動動画投稿のオーケストレーター（台本生成→VOICEVOXでナレーション合成→レンダリング→YouTube Shortsへ投稿→LINEで完了通知）。対象記事が無い日は何も投稿せず正常終了する。ナレーション合成に失敗しても無音で続行する。YouTubeのSecretsが未登録の場合は動画生成のみで正常終了（設定漏れの誤検知で毎日赤くならないように）、Secretsがあるのに投稿できない場合のみ異常終了する。**レンダリング前にYouTubeの認証を1リクエストで確認**し、リフレッシュトークンが失効していればそこで中止する（2026-08-25は230秒かけて書き出した74.9MBの動画の行き先が失効で消えた）。`--dry-run`（台本まで）/`--render-only`（mp4書き出しまで）/`--keep-video`（投稿後もmp4を残す）/`--stock-code`（銘柄指定の手動実行）で段階的に確認できる。※TikTok投稿は2026-08-20に完全撤去 — 自アカウントへの投稿用途はTikTokの本番審査ポリシー（personal/internal use不可）の対象外で承認されないため（経緯はdocs/tiktok_review.md）。LINE通知はTikTokキャプション連携用だったものを投稿完了通知として残置 |
| `video/youtube_metrics.py` | **YouTube（Shorts）の再生数・登録者数の記録**（手動実行専用）。チャンネル(`@kujira-watch`)の公開動画の統計をYouTube Data API v3から取得し、Supabaseの`youtube_videos`（最新値）・`youtube_video_metrics`（日次スナップショット）・`youtube_channel_stats`（登録者・総再生・本数）に保存する（`supabase/create_youtube_metrics.sql`）。**動画パイプラインは毎営業日アップロードしているのに成果を1つも記録していなかった**（動画IDすら保存しておらず、何本出して何回見られたかを追えなかった＝2026-08-27に発覚）。実測は総再生4,747回・登録者3人・公開7本で、X（60日で34インプレッション・フォロワー0人）より桁違いに届いている一方、GA4上のサイト流入は28日で22セッション＝再生の0.46%だった。認証は**サービスアカウント**（`gcp_key.json`）。公開動画の統計はこれで読めるため、アップロード用のOAuthリフレッシュトークン（scopeが`youtube.upload`だけで統計を読めない）を取り直す必要はない。動画IDはチャンネルのuploadsプレイリストから辿るので過去分も後追いで集計できる。加えて`record_upload()`を`video/publish_video.py`が投稿直後に呼び、公開した事実だけ先に`youtube_videos`へ残す（この収集は手動実行なので、待つと当日のハートビートに動画0本と見える。既存行は上書きしないinsert_ignore）。加えて`fetch_engagement()`が**視聴維持率**（`video/youtube_analytics.py`）を各動画に足し、`youtube_videos`の`avg_view_pct`/`avg_view_sec`/`subscribers_gained`/`hook_survival`と`youtube_video_retention`（維持率カーブ）に保存する。`--report`で尺別の平均再生（実測2026-08-30: 60秒以下5本が平均1,171回、60秒超4本が平均582回）と1本ごとの内訳（再生・高評価・平均視聴率・hook3秒の残存率・登録者獲得）を表示。**初回の実測（2026-08-30、8本）: 平均視聴率63.7%、3秒残存91〜99%（hookは落ちていない）、10秒地点で53〜75%。最急の落ち込みは実時間5〜9秒＝hook終わりからcompanyシーンに入る境目**。直近1〜2本はAnalytics側の反映待ちで欠けることがある。1本も取れないときは終了コード1を返す |
| `video/youtube_auth.py` | YouTubeのリフレッシュトークンを取得する**ローカル1回きり**のスクリプト（CIでは使わない）。loopback(http://localhost:8765)で認証コードを受け取る。scopeは3つ（`youtube.upload`=投稿 / `youtube.force-ssl`=記事URLコメント / `yt-analytics.readonly`=視聴維持率）。**2026-08-30にscopeを拡張したので、既存のupload専用トークンではコメント投稿と維持率取得が403になる**。取り直すと両方が有効になり、投稿側の動作は変わらない |
| `video/youtube_analytics.py` | **視聴維持率の取得**（YouTube Analytics API v2）。平均視聴率・平均視聴秒・登録者獲得数（`video_stats()`）と、尺に対する経過割合0.00〜1.00での維持率カーブ（`retention_curve()`）を読む。`survival_at()`はカーブを尺で割り戻して「先頭3秒で何割残ったか」を本ごとに比較できる形にする（カーブは秒ではなく割合で返るため、尺の違う動画をそのまま並べても比べられない）。サービスアカウントでは読めずチャンネル所有者本人のOAuthが要るので、投稿用のリフレッシュトークンを流用する。**再生数・高評価しか記録できていなかったので「hookで何割消えたか」が分からず、演出を変えても良し悪しを言えなかった**（2026-08-30に追加）。scope不足の403では例外を投げず空を返し、再生数の記録は成立させる |
| `tools/output_heartbeat.py` | **成果物ハートビート**（ops.yml heartbeat、平日22:00 JST）。ワークフローの成否ではなく成果物そのもの（microCMSの当日公開記事数・素材側は**EDINET API を直接叩いた当日の大量保有件数**（`count_edinet_disclosures()`、`EDINET_API_KEY`必須）と`tdnet_buybacks`）を数え、「EDINETに開示があるのにDBが0件（保存の故障）」「素材があるのに記事0件」をLINEへ通知する。X投稿と動画は2026-08-30に定期実行を止めたため数えない（止めた成果物を数え続けると毎日「X投稿0件」で誤報になる）。素材をSupabaseの`edinet_large_holdings`から数えていた頃は、保存側が壊れると素材も成果物も同時に0になり「開示が無い静かな日」と区別できず無言で通していた（2026-08-26〜27）。EDINETを引けなかった場合(-1)のみDBの件数にフォールバックする。各ワークフローが`continue-on-error`で緑のまま止まる無言停止を検知するための見張りで、Claudeを一切使わない。開示が無い日（祝日等）の0件は異常としない。取得できなかった件数(-1)では判定しない。**対象日は「起動時刻のJST日付」ではない**。JSTの正午より前に起動した便は前日を判定する（GitHubのscheduleは数時間遅れることがあり、13:00 UTCの便が翌8/28 07:40 JSTに起動して、始まったばかりの当日を「0件」と誤報した＝2026-08-28）。件数の集計もJSTの当日0時〜24時で必ず閉じる（上限が無いと前日判定の便が当日ぶんのbackfill記事まで数える）。`--always`で正常時も1通、`--dry-run`で送信せず本文表示、`--strict`で異常時に終了コード1、`--date`で対象日を明示 |
| `tests/test_fundamentals.py` | point-in-timeファンダ（`lib/fundamentals.py`）のユニットテスト。先読みバイアス防止（as_of日より後の開示を含めない）を確認（6件）|
| `tests/test_earnings_quality.py` | 利益の質フィルター（化粧・赤字・減益・加減点）のユニットテスト（8件）|
| `tests/test_screener.py` | スクリーナー条件のユニットテスト（銘柄コード絞り込み正規表現の新形式コード対応込み、13件）|
| `tests/test_fetch_history.py` | 株価キャッシュ更新の銘柄コード収集ロジック（既存コード+JPX最新リストの和集合・JPX取得失敗時のフォールバック・J-REIT含む市場フィルター）のユニットテスト（4件）|
| `tests/test_data_sanity.py` | QA（データ整合性・価格凍結検知）のユニットテスト（14件）|
| `tests/test_fix_body_numbers.py` | 記事是正の非課金経路（`tools/fix_misreported_blog_articles.py --fix-body-numbers`）のユニットテスト。本文中の比率・変化幅・金額の置換（表記ゆれ・符号なし変化幅・pt表記）、旧値が本文に無いときの報告、規模を語る記述と新しい比率の矛盾検出、タイトルからの旧比率の読み取りを確認（17アサーション）|
| `tests/test_market_compare.py` | 日経 vs S&P500 相対強弱アドバイザーのユニットテスト（4件）|
| `tests/test_market_timing_alert.py` | LINE通知の大口保有動向セクション（開示日優先ソート・根拠なき買い/売り推測の抑制込み）・ウォッチリストdp閾値判定（ランキング本体の推奨ラベルとの矛盾防止・売り閾値ギャップの上書き・通知疲れ対策の要約表示・前日比表示込み）・投資家ウォッチ（提出者名の部分一致照合・大口保有動向セクション生成）・code_name_map未収載銘柄のEDINET issuer_nameフォールバック・大幅訂正報告書の通過/軽微な訂正の除外のユニットテスト（28件）|
| `tests/test_scan_large_holdings.py` | EDINET大量保有スキャナーの判定ロジック（売却検知・保有比率増減による方向判定・個人名判定・過半数超除外・訂正報告書除外／大幅訂正の判定・ノイズ除外・保存失敗時の`HoldingsSaveFailed`送出／1日ぶん失敗しても残りの日は取得して最後に投げること）のユニットテスト（16件）|
| `tests/test_holding_details.py` | 保有目的・取得資金パーサ（`lib/edinet.py`の`parse_holding_details()`/`classify_purpose()`/`average_acquisition_price()`）のユニットテスト（単独提出者の株数・資金・報告義務発生日の抽出／共同保有3名で取得資金・自己資金・借入金を足し上げ、タグが飛んだ提出者のぶんを隣の提出者の値で埋めないこと／全部売却（株数0）で平均取得単価を出さないこと／空XBRL／保有目的5区分の判定／平均取得単価のゼロ除算ガード、6件）|
| `tests/test_short_term_transfers.py` | 短期大量譲渡の「譲渡の相手方・単価」パーサ（`lib/edinet.py`）のユニットテスト（相手方・単価・数量の抽出／開示単価×株数の実額算出／表で比率変化を説明できないときの実額不採用／和暦・自己終了タグの空セル・「4,730円」表記・取得と処分の混在／端数処分ではなく主取引を代表値にすること／「不明」表記を相手方として扱わないこと／表が無い開示／見出しラベルの「推定」有無／プロンプト事実行の組み立て／新株予約権の単価を株価として扱わないこと／連続譲渡で今回ぶんの行だけを使うこと、10件）|
| `tests/test_reclassify_blog_articles.py` | 既存ブログ記事の投資家分類一括再分類ツールのユニットテスト（microCMSのdealType配列/空配列/None正規化。空配列でIndexErrorになっていた本番バグの再発防止・PUT用ペイロードのメタ情報除去）のユニットテスト（5件）|
| `tests/test_strip_drop_model_mentions.py` | 下落モデル言及の削除ロジック（株価の節を残す/残せない場合は文ごと削除・残す部分にモデルの話が混ざる場合は残さない・段落内の該当文だけを落とす・空になった段落の削除・タグを跨ぐ文は触らない・タグを空白に置換して段落を分離する・ピリオド直後に空白が無い英文の分割・英文は文ごと削除・書き足しの検出）のユニットテスト（10件）|
| `tests/test_cleanup_duplicate_blog_articles.py` | ブログ重複記事クリーンアップの削除対象選定ロジック（先発を残し後発を削除・別提出者/別日は対象外・同一提出者でも比率変化幅が違えば対象外・filerName空の旧記事は同一銘柄/同一開示日/同一タイトルなら対象・旧記事でもタイトルが違えば対象外・銘柄コード/開示日/タイトルが欠けた記事は対象外・自社株買い記事はtagsで判別し銘柄×開示日で重複判定）のユニットテスト（8件）|
| `tests/test_article_redirects.py` | 削除済み記事のリダイレクト（`lib/article_redirects.py` / `tools/backfill_article_redirects.py`）のユニットテスト（銘柄コードが無い記事は登録しないこと、`upsert`の`on_conflict=article_id`、A→B→Cの2ホップを作らない付け替え、バックアップJSONの読み取り＝同一idは先勝ち・リストでないJSONと壊れたJSONを飛ばすこと、dry-runで書き込まないこと。10件） |
| `tests/test_article_figures.py` | 記事本文の解説図（`web/article_figures.py`）のユニットテスト（anchor語に一致した段落の直後への差し込み・1段落目の前と最終段落の後に入れないこと・anchor無しの均等配置と順序保持・段落が少ない本文への末尾追加・図が2点未満なら作らないこと・PNGバイト列の生成・自社株買い図の過去決議の要否、13件）|
| `tests/test_backfill_article_filer_name.py` | 既存記事の`filerName`バックフィルツールのユニットテスト（法人格・全角/半角・中黒の正規化、候補1件の即採用、タイトル一致による絞り込み、タイトルで決まらないときの本文一致へのフォールバック、タイトルを本文より優先すること、本文に複数候補が出る場合・一意にならない場合・社名が短すぎる場合のスキップ、HTMLタグの除去）のユニットテスト（10件）|
| `tests/test_backfill_blog_eyecatch.py` | アイキャッチ画像バックフィルツールのユニットテスト（tags/タイトルからのバッジ判定4種、直近N日・index対象・件数上限・新しい順の候補抽出、タイトルからの保有比率抽出とSupabase値の優先、カード組み立て、再開用の実行ログからの処理済みID抽出（OK行のみ・ログ欠損の許容）、9件）|
| `tests/test_publish_blog_articles.py` | ブログ記事自動投稿の判定ロジック（金額概算・発行済株式数取得のリトライ/impliedSharesOutstandingフォールバック・記事生成JSONパース・投資家分類マスター参照とClaudeフォールバック/保存・売り方向のtagsタグ付け/プロンプト分岐・重複防止・権限エラー時の早期打ち切り・投稿/更新のセレクト配列形式への自動リトライ・非文字列フィールド(eyecatch等)の型不一致時は除外して再送信・メディアURLが不正と言われたとき（`'eyecatch' field invalid`）は同じURLで一度待って再送信し、直らなければ画像なしで投稿・PIT文脈(株価)のプロンプト反映と下落モデル水準を渡さないこと・保有比率変化幅(ポイント)/新規保有のプロンプト反映・事業内容の取得/キャッシュ/web_searchツール付与と検索結果ブロック混在時のJSON抽出/生の改行を含むJSONの許容・投資家プロフィールの取得/キャッシュ/空欄時の非創作・「※推測:」ラベル付き推測文の要求・英訳を求めないこと・決定的テンプレによるタイトル組み立て（買い/売り/新規保有・60字超過時の提出者名短縮）・冒頭アンサー文のプロンプト指示・ratioChangePct（売りは負値）のpayload付与・filerName＋ratioChangePct照合による重複判定（旧記事はdealAmountフォールバック）・PATCH更新への型不一致自動リトライ・アイキャッチ画像生成/アップロード・株価チャート画像生成/アップロード/本文埋め込み・ホームページ「注目」枠と同じdealAmount降順での注目記事id抽出・直前保有割合を優先した変化幅算出／全売却の扱い・株価キャッシュ欠損時のyfinanceフォールバック・比率不変時のスキップ・大幅訂正記事の投稿(dealAmount=0/tags訂正/タイトル)・単一開示日の重複判定緩和・直前保有割合が未取得の変更報告書の持ち越し・待っても直前保有割合が入らない変更報告書の変化幅None／記事化スキップ／新規保有と判定しないこと・AI常套句検出時の再生成と字数優先の採用・事業内容/プロフィールのネガティブキャッシュ（試行済みはClaudeを呼ばない・RECHECK_DAYS経過後は再挑戦・空文字でもchecked_atを刻む・空文字でdescription/profileを上書きしない・`max_uses`が1であること・タイムゾーン無しや不正な値の扱い）・利用上限の検知（後続呼び出しを止める・上限エラーはネガティブキャッシュに含めない・529等の一時的失敗では止めない））のユニットテスト・アイキャッチ写真の記事ごとの選び分け（seedで候補80枚から決定的に選択・同一seedは同じ写真・クエリ単位の候補キャッシュで検索APIを1回に抑える）・数値トークンを割らない折り返し・保有比率0%の銘柄行（売り記事は「全株売却」・自社株買いの欠損値は数字を出さない）・既報インデックスのページング/失敗時None・backfillの事前足切り（既報と基準未満を落とす／ビュー未収載は残す）・backfillの窓拡大と古い開示優先・既報インデックスが引けないときの中止・`BACKFILL_MAX_ARTICLES`による投稿数の頭打ち・開示側の台帳（記事を作ったことがある開示はbackfill・通常運転とも作り直さない／投稿成功時に記録する）・本文HTMLに生の改行が入ったJSONのパース・個人の提出者は説明文を落とし法人は残す・台帳（`lib/publish_ledger.py`）との結線（基準未満だけの便は正常扱い・生成失敗/投稿失敗は異常として終了コード4・公開できた候補も台帳に乗り未分類が残らないこと）・引き上げ後の足切り(5億円/1.5pt)で旧基準を落とすこと・公開済み記事のindex基準(3億円/1.0pt)は据え置きで足切り≧index基準であること・既報照会が判定不能なときの投稿見送り（HTTPエラー／例外）・既報照会が開示日まで絞り込まれること・日次上限（当日の実績を引いた残り枠で止まること・当日ぶんが上限に達していれば0件・取得できなければ上限いっぱい・`--max-articles`の明示は当日の実績を見に行かないこと））のユニットテスト（144件、ネットワークは全てモック）|
| `tests/test_writing_style.py` | AI常套句・単調文末の検出（`lib/writing_style.py`）のユニットテスト（常套句の検出・自然な記事文の非検出・同一文末4連続の検出・文末が変化する文の非検出・HTMLタグ除去・空入力・プロンプト用ルール文の波括弧非含有、10件）|
| `tests/test_api_budget.py` | Claude API利用上限フェイルファスト（`lib/api_budget.py`）のユニットテスト（実際に返ってきた400文言の検知・クレジット残高不足の検知・429/529/500/接続断を上限と誤判定しないこと・フラグのラッチと解除、6件）|
| `tests/test_api_usage.py` | API利用量の記録（`lib/api_usage.py`）のユニットテスト（トークン単価の計算・Web検索の$10/1,000検索を別建てで数えること・キャッシュ書込×1.25/読出×0.1・同一タスクの集約とタスク別の行分け・日付サフィックス付きモデルIDで単価表が引けること・未知モデルでも検索料は数えること・usageを持たないMockを無視すること・flushでバッファが空になること・書き込み失敗を伝播させないこと・残枠の警告水準判定（50/80/100%）・上限0で監視しないこと・環境変数による上限の上書き・当月以外を数えないこと・通知本文にコスト上位のタスクが入ること・監視の失敗を伝播させないこと・日次予算の環境変数上書き/0で無効/予算内では止めないこと/超過時に打ち切って1回だけ通知すること/打ち切り前にflushして次の便に残すこと/取得失敗で止めないこと・atexitのflushが本番へPOSTしないこと（2026-08-29の合成行の再発検知）、24件）|
| `tests/test_notify.py` | LINE通知の共通口（`lib/notify.py`）と当日ハートビート（`tools/output_heartbeat.py`）のユニットテスト（認証情報なしで送らない・本文の組み立て・送信例外を投げない・上限文字数での切り詰め・API利用上限を検知した瞬間に1回だけ通知すること・一時的エラーでは通知しないこと・素材ありで記事0件/X0件/記事ありで動画0本の検知・EDINETに開示があるのにDBが0件の検知・素材をDBではなくEDINETから数えること・EDINET不明時のDBフォールバック・開示が無い日の0件を異常としないこと・件数不明(-1)で誤報しないこと・JST日境界のUTC変換・重複抑制（窓の内側で送らない／窓が明けたら送る／未送信なら送る／判定失敗時は送る／dedupe_key無しは従来どおり毎回送る）・上限通知の本文（復旧日時の抽出・クレジット追加では直らない旨・日時不明時は復旧行を出さない・dedupe_key付きで送ること）・遅延して翌朝に起動した便が前日を判定すること・数えるのはブログ記事と素材だけでX投稿・動画を数えないこと（2026-08-30に定期実行を停止したため）、35件）|
| `tests/test_supabase_client.py` | Supabase REST APIクライアントのリトライ挙動（一時的なネットワーク失敗時のバックオフ再試行・最終失敗時に呼び出し元を落とさないこと）・キー構成が違う行の分割送信（upsert/insert_ignore）・書き込み失敗の記録とLINE通知（テーブルごと1回）・**テスト実行中に本番プロジェクトへ書き込まないこと**（upsert/insert_ignore/update/deleteを握りつぶし読み取りは通す／URLを差し替えたテストの書き込みは止めない）のユニットテスト（9件）|
| `tests/test_publish_ledger.py` | 「候補>0なのに公開0」の切り分け台帳（`lib/publish_ledger.py`）のユニットテスト（正常な見送りだけなら鳴らさない・生成失敗/投稿失敗/権限エラーは異常・理由を記録しないまま脱落した候補を異常に倒すこと・max_articles打ち切りの残りを未分類にしないこと・未知の理由は安全側に倒すこと・内訳1行の組み立て・同じ原因の通知を連投しないこと、9件）|
| `tests/test_fix_misreported_blog_articles.py` | 公開済み記事の数字の是正（`tools/fix_misreported_blog_articles.py`）の対象判定のユニットテスト（保有比率そのものがズレた新規開示を前回0%で拾うこと・比率が一致する記事を触らないこと・前回比率が取れない変更報告書を除外すること・履歴から前回比率を補う経路・売りの符号・訂正報告書の金額0・保有比率欠損時のスキップ）・GA4のPVから読まれている記事だけを抜き出す判定（PV0・記事以外のパス・クエリ付きURL・壊れた行・認証情報が無いときにNoneを返すこと））（13件、株価の再概算とGA4はモック）|
| `tests/test_x_client.py` | X(Twitter)自動投稿（`web/x_client.py`・`web/x_insight.py`・`web/x_followup.py`）のユニットテスト（1行目フックの組み立て・新規取得/買い増し/売却/訂正の出し分け・検索母数のある2タグのみの付与・解釈行の挿入と文字数超過時の削除・金額の丸め・数字カードのalt文言・tweet_idの返却とログ記録・添付画像を数字カード1枚に限ること/カード生成失敗時のみチャートで代替・社名が長くてもカードに証券コードを残すこと・本文にURLを入れずプロフィール誘導行で終わること・alt設定リクエスト・JST 8〜22時外のスキップ・1回1件の投稿・「注目」に含まれない記事の除外・訂正記事の全件投稿と既報へのリプ・「本日のクジラ」日次サマリー（本文組み立て・0件時スキップ・totalCount補正・21時JST便の時刻ガード・カード画像添付）・動画クロス投稿・アクセストークン実権限の判定・解釈行の文言・**開示時点のバリュエーション行**（PBR/ROE/配当性向の整形・PBR欠損時に行を出さないこと・PIT条件（`date=lte.`／`doc_type=eq.FY`）でのみ引くこと・比率と%が混在する`payout_ratio`の正規化・異常値の除外）・答え合わせの統計と本文・分割銘柄と基準日ギャップの除外）（62件、ネットワークとSupabaseは全てモック）|
| `tests/test_x_metrics.py` | X投稿メトリクス収集（`web/x_metrics.py`）のユニットテスト（プロフィールクリック/リンククリックのパース・`non_public_metrics`が無い場合のフォールバック・フォロワー数の取得と認証未設定/APIエラー時の空返却・記録が飛んだ日を跨ぐ7日前比/30日前比の算出・記録ゼロ時の表示・401/402/403など待っても直らない失敗で`MetricsUnavailable`を投げること・500等の一時的失敗では投げないこと・恒久失敗時に`run()`が0以外を返すこと・`save()`が`kind`を送ってNOT NULL違反を起こさないこと）のユニットテスト・upsertが失敗した日にSaveFailedで落として終了コード3を返すこと（成果物ゼロのまま緑にしない）（14件、ネットワークとSupabaseは全てモック）|
| `tests/test_x_follow.py` | Xのフォロー（`tools/x_follow.py`）のユニットテスト（鍵アカとフォロワー数が範囲外のアカウントを候補から外すこと・ヒット数→フォロワー数の順で並べること・説明文を1行80字に畳むこと・1回の上限50件を超える指定をAPIに投げる前に止めること・`--execute`が無いときPOSTしないこと・403で以降を中止すること）（6件、X APIは全てモック）|
| `tests/test_x_benchmark.py` | 同ジャンル投稿のベンチマーク（`tools/x_benchmark.py`）のユニットテスト（字数・行数・タグ/URL/画像の有無とエンゲージメントの抽出・投稿時刻のJST変換・欠損フィールドで落ちないこと・3件未満のグループを表示しないこと・文字数帯の境界・24時間すべてが時間帯に入ること）（6件、X APIは全てモック）|
| `tests/test_x_disclosure_facts.py` | 開示事実の投稿（`web/x_disclosure_facts.py`）のユニットテスト（マスター未登録の個人名を除外すること・自己資金欄が空の開示を全額借入と判定しないこと・しきい値以下を遅延としないこと・全額借入に変更報告書を使わないこと・直近投稿済みの銘柄を避けること・希少性の件数を書かないこと・遅延の本文で増減を断定しないこと・遅延日数が測れない開示で本文を作らないこと）（8件、SupabaseとX APIは呼ばない）|
| `tests/test_youtube_metrics.py` | YouTubeメトリクス収集（`video/youtube_metrics.py`）のユニットテスト（ISO8601の再生時間→秒の変換と読めない値を0にして残すこと、尺別集計の平均・空配列でゼロ除算しないこと・尺不明(0秒)を短尺に数えないこと、チャンネル未検出時の例外、統計の読み取り、playlistItemsのページング追跡と古い順ソート、Supabase未設定時の保存スキップ、最新値と日次スナップショットの両方を書くこと、鍵なし/動画0本/API失敗で終了コード1、投稿直後の記録をinsert_ignoreで書くこと・動画IDなし/Supabase未設定では書かないこと、15件、YouTube API・Supabaseはモック）|
| `tests/test_youtube_analytics.py` | 視聴維持率の取得（`video/youtube_analytics.py`）のユニットテスト（動画ID別の統計への変換、scope不足の403で例外を投げず空を返すこと、維持率カーブの経過割合順ソート、秒→経過割合の割り戻し、カーブなし/尺0/尺より後ろの時点でNoneを返すこと、5件、Analytics APIはモック）|
| `tests/test_buyback.py` | 自社株買い開示の分類・取得枠抽出（`lib/buyback.py`）のユニットテスト（決定/進捗/変更・訂正・中止(amendment)のタイトル分類・ToSTNeT-3単日買付の日付・億円単位・一部変更の変更後列の採用・LLM補助の条件、10件）|
| `tests/test_x_buyback.py` | 平日「本日の自社株買い決定」X投稿（`web/x_buyback.py`）のユニットテスト（1億円未満の除外・社名解決・270単位収容・投稿kind、7件）|
| `tests/test_publish_buyback_articles.py` | 自社株買い記事の自動投稿（`web/publish_buyback_articles.py`）のユニットテスト（規模閾値・検索クエリ型タイトル・payload（dealType/filerName無し/消却タグ）・既報スキップ・AI常套句検出時の再生成・既報インデックスをtagsで引くこと（dealTypeでは常に0件になり重複投稿する）・インデックスが引けないときのbackfill中止・backfillが既報を落として古い開示から消化すること・銘柄名のTDnetフォールバック（マスターにあればTDnetを叩かない）・開示側の台帳（記事化済みの開示を作り直さない／投稿成功時に記録する）・既存決議の一部変更／訂正を候補にしないこと・台帳（全件既報での0件公開は正常扱い・生成失敗は異常・未分類が残らないこと）・日次上限（当日の残り枠で止まり`tdnet_buybacks`の実績を数えること）、22件）|
| `tests/test_traffic_report.py` | アクセスログ切り分け（`tools/traffic_report.py`）のユニットテスト（1IPあたりPV閾値による除外、ip_addressがNULLの行を落とさずまとめて数えること、UA自己申告によるbot検出、クッキー不保持UAの判定に比率と母数の両方が要ること、visitor_idがNULLの行を別IDとして数えること、5分類で行が1つも消えず重複もしないこと、bot由来のPVで共有IPをheavy扱いにしないこと、UTC→JSTの時間帯変換と末尾Z表記の受理、記事ページ率、2PV以上の訪問者数、機械を除いた集計の表示とクローラー名の明示、行ゼロ時とSupabase未設定時の扱い）のユニットテスト（15件、Supabaseはモック）|
| `tests/test_ga4_clicks.py` | GA4クリックログ取得（`tools/ga4_clicks.py`）のユニットテスト（GA4応答の整形＝1ディメンションは文字列キー/複数はタプルキー、前期間0からの増加を`(新規)`と出すこと、APIエラーの手順への翻訳＝API無効/プロパティ権限なし/プロパティID誤り、eventNameを完全一致で絞ること、期間とフィルターのリクエスト組み立て、GA4_PROPERTY_ID未設定、閲覧者数を分母にしたクリック率の算出と前期間比、クリック数がPVを上回るページでも率が100%を超えないこと、labelカスタムディメンション未登録時の手順表示、クリック0件とAPI失敗で終了コード1、ページ種別への畳み込み、1セッションあたり内部移動回数とエンゲージ率の算出・入口0件でのゼロ除算回避、環境変数の鍵をファイルより優先すること、18件、GA4 APIはモック）|
| `tests/test_gsc_report.py` | Search Consoleレポート（`tools/gsc_report.py`）のユニットテスト（ページ種別の畳み込み＝ランキング等のハブを一覧と分けること、CTR・平均掲載順位を表示回数で加重すること、CTR改善候補の抽出条件＝10位以内かつ表示20回以上かつCTRがサイト平均未満と「平均CTRなら増えたクリック数」順の並び、あと一歩＝11〜20位のみ、APIエラーの手順への翻訳＝API無効/権限なし/プロパティ名誤り、プロパティ名のURLエンコードと`rowLimit`の25,000上限、全セクションの出力。15件） |
| `tests/test_geo_report.py` | GEOレポート（`tools/geo_report.py`）のユニットテスト（実在パスをokと判定すること、証券コードに英字を含む銘柄=603A等をokにすること、存在しないURL=`/articles`・`/watchlist`・`/stocks/{code}/feed.xml`をmissingとして拾うこと、301で受けている旧URLをmissingと分けてredirectにすること、クエリ・末尾スラッシュを無視すること、ページ種別への畳み込み、GA4のAI流入判定をmedium=ai-assistantとホスト名の両方で行うこと、引用の代理指標にChatGPT-User/PerplexityBotだけを入れて一括クロールを混ぜないこと、ログ0件時の表示、GA4の鍵が無くてもクローラー側は出すこと）のユニットテスト（10件、Supabase/GA4はモック）|
| `tests/test_video_pipeline.py` | 自動動画投稿パイプライン（`video/`）のユニットテスト（「新着×注目」の積集合からの金額規模順選定・積集合が空なら動画を作らない・microCMSセレクト型dealTypeの配列アンラップ・tagsからの売り方向判定・filerName未設定時の空文字化・本文末尾の保有比率抽出とフォールバック・台本JSON(hook/sections/closing)のフラットなシーン列への展開・Claudeが誤ったkindの期待順上書き・字数超過時の1回だけの作り直しと最終的な切り詰め・ナレーションの句点境界での切り詰め・sections不足時の破棄・APIキー未設定時のスキップ・VOICEVOXエンジン未接続時の無音フォールバック・合成成功時のaudio/durationSec書き込み・1件でも失敗したら全体無音・YouTubeタイトル/説明文の組み立てとUTM付与・タイトル超過時も#Shortsを残す切り詰め・64MB超のファイルのチャンク分割とContent-Rangeの連続性・書き出し後の-14 LUFS正規化・文途中切れ台本の作り直しと破棄・前回保有比率の抽出と向き矛盾時のNone・チャートの日付ラベルと開示日位置・人物クリップの除外・実写背景をcompany/filerに限定すること・効果音とBGMのwav生成と再現性・BGMのループ継ぎ目に段差が無いこと・BGMのピーク余裕・説明文の記事URLが先頭3行以内にあること・サイト名の名乗り・検索される語を先頭に置くハッシュタグ順・投稿文のサイト導線・ハッシュタグの整形・作り直し時のプロンプトへの問題フィードバック・「買収」等の開示内容を超える語と新規保有を「買い増し」と書くことの禁止・作り直しでcaptionとnarrationの超過を言い分けること・壊れたシーンの定型文への差し替えと切り落とし・ffmpeg不在時に音量正規化のスキップを警告すること・旧記事の提出者名を本文と開示データの突き合わせで特定すること／候補が複数一致するときは総称に落とすこと・Canva製エンドカードのpublic/へのコピーとpropsへの書き込み／素材が無いときのテキスト締めへのフォールバック・カスタムサムネイルの1280x720生成と台紙欠如時のスキップ・金額表記のRemotionとの一致・thumbnails.setへのPOSTと認証未設定時のスキップ）・YouTubeトークン失効時にレンダリング前で中止すること/Secrets未登録なら従来どおり動画だけ作ること/公開に成功したらyoutube_videosへ記録すること（98件、ネットワークは全てモック）|

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
| `edinet_holding_amounts`（マテビュー） | EDINET大量保有報告書1件＝1行の推定売買金額（億円）。保有比率の変化幅×発行済株式数（`jquants_fin_summary.sh_out`のPIT値）×開示日終値（`yahoo_price_cache`）の概算。訂正報告書・株価/株式数が取れない開示は行を作らない（金額不明と金額ゼロを混ぜないため）。kujira-watchの銘柄ランキング(`/trending`)の並べ替え軸（推定売買金額順）の元。定義は`supabase/create_edinet_holding_amounts.sql`、再計算は`tools/refresh_holding_amounts.py` |
| `edinet_filer_ids` | 投資家(filer_name)→連番ID（`/investors/<番号>`のURL用、`supabase/create_edinet_filer_ids.sql`）。`edinet_large_holdings`へのINSERT/UPDATEトリガー`trg_assign_edinet_filer_id`で新しい提出者に自動採番。以前のURLは提出者名そのもの（日本語・全角・空白）で、Search Consoleカバレッジで投資家ページ603件がインデックス未登録だったため2026-08-23に番号化した |
| `edinet_filer_summary`（ビュー） | `edinet_large_holdings`×`edinet_filer_classification`×`edinet_filer_ids`を投資家(filer_name)単位に集計したビュー（保有開示件数・最終開示日・分類・`filer_id`）。kujira-watch（`kujira-watch/src/lib/investors.ts`）の`/investors`一覧・サイトマップ生成が参照。投資家は600件超あり`edinet_large_holdings`の生データを直接集計すると1000行上限に掛かるため、1投資家1行に事前集計したビュー経由で取得する |
| `ext_tdnet_disclosures` | TDnet適時開示（やのしん・⚠️個人運営ソースのため `ext_` で隔離）|
| `jpx_short_selling` | JPX空売り残高報告（0.5%以上）|
| `jpx_margin_balance` | JPX個別銘柄信用取引週末残高 |
| `notify_log` | LINE通知の重複排除台帳（`dedupe_key`がPK）。`lib/notify.py`の`once()`が「この警告はもう送った」を記録し、毎時ジョブが同じ内容を何十通も送るのを防ぐ |
| `api_usage` | Anthropic API利用量（UTC日付・ジョブ・タスク・モデル別のトークン/Web検索/推定コスト）。`lib/api_usage.py`が追記専用で書き、`tools/api_usage_report.py`がSUMして読む |
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

#### Claude APIのコスト管理

月次利用上限に到達すると全パイプラインが同時に壊れる（2026-08-23に到達、復帰9/1）ため、
以下を常に守る。

- **空振りは必ず記録する**: 事業内容・投資家プロフィールは、生成できず空文字だった場合も
  `description_checked_at` / `profile_checked_at` に試行日時を刻む（`RECHECK_DAYS`=90日は再試行しない）。
  記録しないと同じ対象へ何度でも課金される。`get_company_description()` は
  `web_search`（**$10/1,000検索** ＋ 検索結果が入力トークン、1社あたり約$0.05）を使うため影響が大きく、
  2026-08-15〜18のバックフィル4回で月次上限を使い切った。
- **`max_uses` は費用に直結する**: 増やすときは検索料と入力トークンの両方が増えることを踏まえて判断する（会社説明は2026-08-29に2→1へ削減、1社$0.034→$0.017）。
- **上限到達時は即座に打ち切る**: `lib/api_budget.py` が400の
  "You have reached your specified API usage limits" を検知し、同一プロセスの後続呼び出しを
  APIを叩かずにスキップさせる（429などの一時的失敗では打ち切らない）。
- **1日ぶんの予算で先に止める**: `lib/api_usage.py`の`check_daily_cap()`が当日(UTC)の推定コストを
  `ANTHROPIC_DAILY_BUDGET_USD`（既定$0.15）と突き合わせ、超えたらその日の残りを打ち切ってLINEへ1通流す。
  月次上限に当たってから止まると復旧まで1ヶ月記事が出ない（2026-08-23の実例）。
- **使った量は必ず記録する**: 各`messages.create()`の直後で`lib/api_usage.py`の`record()`がusageを拾い、`api_usage`テーブルへ残す。内訳は`python3 tools/api_usage_report.py`で見る（用途別の実績が無いと、上限に近づいていることにも、どのバッチが食っているかにも気づけない）。
- **一括バックフィルは分割する**: `--limit` と `--recent-days` で区切り、残枠を確認してから走らせる。

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
| `SUPABASE_URL` | Supabase プロジェクトURL（全データ永続化の宛先）|
| `SUPABASE_SERVICE_KEY` | Supabase service_role キー（バックエンド書込用）|
| `EDINET_API_KEY` | EDINET API v2 サブスクリプションキー（daily_alert.yml Step 2d + edinet_blog.yml毎時の大量保有スキャン用。未登録ならスキャンはスキップ）|
| `ANTHROPIC_DAILY_BUDGET_USD` | Anthropic日次予算(USD)。超えたらその日の以降のClaude呼び出しを打ち切る（`lib/api_usage.py`の`check_daily_cap()`）。未設定なら**$0.15**（`DEFAULT_DAILY_BUDGET_USD`）。`0`で無効 |
| `ANTHROPIC_MONTHLY_BUDGET_USD` | Anthropic月次利用上限(USD)。`lib/api_usage.py`の残枠警告の基準。未設定なら**$15**（`DEFAULT_MONTHLY_BUDGET_USD`）。`0`にすると残枠監視を止める |
| `GA4_PROPERTY_ID` | GA4のプロパティID（数字。`tools/ga4_clicks.py`のクリックログ取得用。未設定なら取得はスキップ）|
| `GSC_SITE_URL` | Search Consoleのプロパティ名（`tools/gsc_report.py`用。既定は`sc-domain:kujira-watch.com`。URLプレフィックス型のプロパティなら`https://kujira-watch.com/`を指定する）|
| `MICROCMS_SERVICE_DOMAIN` | `kujira-watch`（大口投資家の監視ブログ）用microCMSサービスドメイン（edinet_blog.yml: ブログ記事自動投稿。未登録ならスキップ）|
| `MICROCMS_API_KEY` | 同上・書き込み権限付き＋メディアアップロード権限付きAPIキー（アイキャッチ画像のアップロードに使用）|
| `PEXELS_API_KEY` | ブログのアイキャッチ画像生成用（Pexels検索API）。未登録ならアイキャッチ無しで記事のみ投稿 |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` | YouTube Shorts自動投稿用（video_post.yml）。Google Cloud ConsoleでYouTube Data API v3を有効化し「デスクトップアプリ」のOAuthクライアントを作成、ローカルで`python video/youtube_auth.py`を1回実行してリフレッシュトークンを取得する。未登録なら投稿をスキップ。**scopeは3つ必要**（upload / force-ssl / yt-analytics.readonly）。2026-08-16取得のトークンはuploadだけなので、記事URLコメントと視聴維持率を使うには取り直して`.env`と`gh secret set YOUTUBE_REFRESH_TOKEN`を更新する。**OAuth同意画面が「テスト」状態のままだとリフレッシュトークンは約7日で失効し、`invalid_grant: Token has been expired or revoked` で投稿が落ちる**（2026-08-16取得のトークンが2026-08-25に失効）。恒久対策はGoogle Cloud Consoleで同意画面を「本番」に公開すること |

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

python3 tools/api_usage_report.py  # Anthropic API利用実績（直近30日・用途別）

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
| `revision-review` | **修整レビュー13ラウンド**。修整指示に対する変更を、観点（指示充足/バグ/特徴量整合性/戦略規律/PIT規律/IOコスト/CI/テスト/コード規律/セキュリティ/冪等性/DBスキーマ/総合）を変えて13回レビューし、指摘を潰してから完了判定する。1観点1体の `revision-reviewer` サブエージェント（`.claude/agents/`）を並列起動する。**論点（どちらを選んでも何かを失う／指示の解釈が割れる）はAIが決めず、ラウンド1〜9の終了時にまとめてユーザーへ選択肢つきで確認する** |

サブエージェント定義は `.claude/agents/` に置く（現在は `revision-reviewer` の1体。`revision-review` スキルから起動する読み取り専用レビュアー）。

# Dev Log

## 2026-08-15 ショート動画v2: ナレーション付き・TikTok運用の定石を反映した全面改修

```
背景: 初版は「文字が順に出てくるだけ」で動画である必然性が無かった（オーナー評:
面白くない）。TikTokで再生される動画の定石に沿って全面的に作り直した。

台本の変更（video/build_script.py）:
- hook/bullets/closing の3要素 → 7シーン構成（hook→company→deal→filer→change→outlook→cta）
- 各シーンは narration（読み上げ文50〜90字）と caption（字幕26字以内）の対
- 「記事をほぼ読む」密度にするため、記事本文に加えSupabaseキャッシュ済みの
  事業内容(get_company_description)と投資家プロフィール(get_filer_profile)も
  プロンプトへ渡す（どちらもpublish_blog_articles.pyが生成済みの事実。新規の情報源は足さない）
- ナレーションの切り詰めは句点境界（_trim_narration）。文の途中切りは
  そのまま読み上げられてしまうため（実際に「…こ…」で切れる例が出た）

音声（video/tts.py）:
- Google Cloud TTSで実装 → GCPプロジェクトに請求先が無くAPI有効化不可 → VOICEVOXへ切替
- VOICEVOX（ずんだもん）は無料・登録不要・商用可（クレジット表記必須）で、
  日本のTikTok/Shorts解説動画で視聴者の馴染みが最も深い
- CIは公式Dockerイメージ voicevox/voicevox_engine:cpu-ubuntu20.04-latest をジョブ内起動
- エンジン未接続・合成失敗時は全編無音にフォールバックして投稿は続行
- クレジット「VOICEVOX:ずんだもん」はCTAシーン・YouTube説明文・TikTokキャプションに自動挿入

映像（video/remotion/）:
- 尺は固定20秒 → ナレーション音声の実測長で決まる可変尺（calculateMetadata）
- 冒頭はブランド・日付の前置きを廃止し、金額を画面いっぱいに叩き込む1秒目に変更
- 全表示をsafeArea内に（下部470px/右190pxはTikTokのUIに隠れて読めない）
- 無音視聴者向けに caption大型字幕＋narration全文の小型字幕を常時表示
- 上部に進行バー（完走率対策）、背景はズームドリフト＋波で常時動かす
- 締めはブランド画面でループ再生に繋がる構成、outlookには「※ここから先は推測」ラベル

検証: 実データ（アインホールディングス9627、Oasisの20.93%取得記事）で
台本→レンダリングまで通し確認。字幕の字数制御・safeArea・全シーンの表示を目視確認。
tests/test_video_pipeline.py を29件に更新、全216件pass。
（ローカルにVOICEVOXエンジンが無いため音声付きの通しはCI初回実行で確認する）
```

## 2026-08-15 自動動画投稿パイプライン（YouTube Shorts / TikTok）を新設

```
背景: ブログ記事(microCMS)とX投稿は自動化済みだったが、動画チャネルは未着手だった。
検索流入とAI引用に効くYouTube、リーチの広いTikTokの2つに、既存の記事資産を
そのまま再利用して展開する。

構成: video/ 以下に一式を新設。
  build_script.py  … microCMSの新着記事×ホーム「注目」枠の積集合から1件選び、
                     記事本文だけを根拠にClaudeで縦動画の台本(hook/bullets/closing)を生成
  remotion/        … Remotion(React/TS)のコンポジション ArticleShort。
                     1080x1920 / 30fps / 600フレーム(20秒)。
                     Hook → Fact(金額カウントアップ) → Point(要点3行) → CTA の4シーン
  render.py        … props JSON を npx remotion render に渡してmp4を書き出す
  youtube_client.py / tiktok_client.py … 各プラットフォームへアップロード
  publish_video.py … オーケストレーター（--dry-run / --render-only で段階確認可）
新規ワークフロー .github/workflows/video_post.yml を平日21:30 JST（cron '30 12 * * 1-5'）で
1日1回実行する。edinet_blog.yml（毎時9:00-21:00 JST）が記事を出し切ったあとに走る。

記事投稿と別バッチにした理由: Remotionのレンダリングは Chrome Headless を使うため、
毎時の記事パイプラインに相乗りさせるとCI時間が大きく伸びる。動画は1日1本で十分なので
1日1回のバッチに分離した。

対象記事の選び方: X投稿(web/x_client.py)と同じく「直近に新規公開された記事」×
「ホームページの注目枠に入っている記事」の積集合。サイト上で目立っていない小粒な開示
だけが動画になる事態を防ぐ。積集合が空の日は動画を作らない（毎日必ず出す運用にしない）。

台本の字数制約: 縦動画は1行40字を超えると3行に折り返して20秒では読み切れない。
プロンプトで字数を厳守させたうえで、超過したら一度だけ作り直し、それでも超える場合だけ
末尾を詰める（初回生成は60〜70字の行が出たため、この二段構えが必要だった）。

TikTokの制約: Content Posting API はアプリ審査を通るまで一般公開の投稿ができず、
未審査アプリは SELF_ONLY に限定される。そのため既定では inbox（下書き）へアップロードし、
TikTokアプリの通知からオーナーが手動公開する運用にした。審査通過後に
TIKTOK_DIRECT_POST=1 で直接公開へ切り替わる。

検証: 実データ（東陽テクニカ 8151 の売却記事）で台本生成→レンダリングまで通し、
4シーンの日本語表示・売り方向の赤アクセント・提出者名未設定時の行省略を目視確認。
tests/test_video_pipeline.py 22件を追加、既存を含む209件が pass。

未了: Secrets（YOUTUBE_*/TIKTOK_*）は未登録のため、実アップロードは未実施。
```

## 2026-08-15 EDINETブログ記事投稿を毎時バッチに分離（株価更新パイプラインから切り離し）

```
背景: これまで tools/scan_large_holdings.py（EDINET大量保有スキャン）と
web/publish_blog_articles.py（ブログ記事生成・投稿）は daily_alert.yml の
Step 2c/5c として、平日16:00 JSTの株価更新パイプラインに相乗りしていた。
そのため、朝一で提出されたEDINET開示も記事化されるのは同日16時以降になっていた。

変更: 新規ワークフロー .github/workflows/edinet_blog.yml を作成し、
tools/scan_large_holdings.py（--days 2）→ web/publish_blog_articles.py を
平日9:00-21:00 JST・毎時（cron '0 0-12 * * 1-5'）で独立実行するように分離。
daily_alert.yml からは Step 2c/5c を削除（未使用になった依存パッケージ
Pillow/requests-oauthlib と MICROCMS_*/PEXELS_API_KEY/X_* のenv設定も削除）。
edinet_large_holdings テーブルはSupabase経由で共有されるため、
daily_alert.ymlのランキング生成（EDINET大量保有の特徴量edinet_hold_f）は
引き続き最新の毎時スキャン結果を参照できる。

頻度の検討: このリポジトリはpublic（GitHub Actions分数は無料）で、
記事生成コスト(Claude Haiku)も新規記事1本あたりで決まり実行頻度に比例しない
（publish_blog_articles.already_published()が重複生成を防ぐため）。
よって「毎時」自体のコスト増は実質ゼロ。24時間フル稼働ではなく
平日9:00-21:00に絞ったのは、EDINETの新規開示がこの時間帯にしか
出ないため（深夜・早朝に回しても空振りするだけ）。
```

## 2026-08-07 ブログ記事の金額推定スキップ（29件中15件が原因不明）を調査・2件修正

```
発見の経緯: ユーザーから「8/6にEDINET開示222件のうち52件しか記事にならなかったのは
おかしくないか」との指摘。daily_alert.yml Step 5c の実行ログ（run 31097732254、
2026-08-06T12:24〜）を確認したところ、「金額を概算できないためスキップ」が29件あった。
Supabaseで各銘柄コードのyahoo_price_cache収録状況を確認し、原因を3系統に分類:

1. 新形式(英数字)銘柄コード7件（151A/603A×2/604A/607A/599A×2）
   → 2026-08-06付の別修正(#226)より前の実行だったため。現在は解消見込み。
2. J-REIT(投資法人)7件（スターアジア不動産投資法人・アドバンス・レジデンス投資法人等）
   → tools/fetch_history.py の _fetch_jpx_codes() が市場・商品区分を「内国株式」の
   みで絞り込んでおり、J-REIT（市場・商品区分に"REIT"を含む）がyahoo_price_cacheに
   構造的に一度も入らない状態だった。ユーザー確認の上、J-REITも対象に含めるよう変更。
3. 通常の4桁コード9件（サンリオ・三菱製紙・日本化学工業等）
   → yahoo_price_cacheには直近日付の株価が11件前後揃っており、価格データ自体は
   原因ではなかった。web/publish_blog_articles.py の shares_outstanding() が
   yfinance の Ticker.info を1回だけ呼ぶ実装で、同一プロセス内で候補ごとに繰り返し
   呼ぶ構成上レート制限にかかりやすく、これが失敗の実体だったと推定。
   （イーグランド3294のみ株価キャッシュ自体が2026-06-02で止まっており別要因、
   今回は未対応。J-REIT以外にも同様の停滞銘柄が無いか別途監視が必要。）

修正:
1. shares_outstanding() を最大3回・短い間隔（1.5s×試行回数）でリトライするように変更。
   加えて sharesOutstanding が空の場合 impliedSharesOutstanding にもフォールバック
   （J-REITの投資口数がこちらに入るケースがあるための保険）。
2. tools/fetch_history.py の _fetch_jpx_codes() の市場フィルターを
   「内国株式」→「内国株式|REIT」に拡張。core/screener.py 側のコア銘柄スクリーニング
   （drop_probモデルの対象銘柄選定）は変更していないため、REITが取引対象・ランキングに
   混ざることはない。影響は価格キャッシュの網羅性（＝ブログの金額推定）のみ。

検証: tests/test_publish_blog_articles.py（リトライ・フォールバック・リトライ枯渇の
3件を追加、46→49件）・tests/test_fetch_history.py（REIT行を含む市場フィルターの
1件を追加、3→4件）で確認、全件成功。イーグランド(3294)の価格キャッシュ停滞と、
新規コード599A/603A/604A/607Aが現時点でもyahoo_price_cache未収録である点は、
今回のリトライ・REIT対応では解消しない別問題として残っている（次回Step 0実行後に
再確認が必要）。JPXのdata_j.xlsのREIT区分表記は本サンドボックス環境からネットワーク
アクセスできず実データで検証できていない（"REIT"という部分一致文字列を含む前提）。
```

## 2026-08-06 新形式(英数字)銘柄コードが価格キャッシュ・全銘柄スキャンから恒久的に漏れる問題を修正

```
発見の経緯: ブログ記事（大口投資家の監視ブログ）でEDINET開示のある151A・4189・5644・
6025・603A・7176の6銘柄が金額推定不能でスキップされていた。調査の結果2系統の原因判明:

1. tools/fetch_history.py の get_all_codes() が「yahoo_price_cache に既にある
   コード」があれば即returnする実装で、JPX最新リストとの差分を取り込む処理が
   実質デッドパスだった（docstringには「既存コード+JPXリストの和集合」と書かれて
   いたが実装が伴っていなかった）。新規上場銘柄（6025・603A・7176等）が
   yahoo_price_cache に一度も追加されず、価格が引けない状態が続いていた。

2. core/screener.py の get_tse_stock_list()（core/rank_stocks.py が日次の全銘柄
   スキャンで再利用する銘柄コード取得元）が銘柄コードを ^\d{4}$ で絞り込んでおり、
   TSEが2024年以降に発行する新形式コード（末尾1桁が英字。例: 151A・603A）を
   構造的に除外していた。本体のランキング・LINEアラートの対象銘柄からも
   これらの銘柄が恒久的に漏れていたことになる。

修正:
1. get_all_codes() を、yahoo_price_cache 既存コードとJPX最新リストの和集合を
   毎回返すように修正（JPX取得失敗時は既存コードのみへフォールバック）。
2. core/screener.py に STOCK_CODE_PATTERN = r"^\d{3}[0-9A-Z]$" を定義し、
   get_tse_stock_list() の2箇所のフィルターをこれに置き換え。

検証: tests/test_fetch_history.py（新規、3件）・tests/test_screener.py（4件追加、
9→13件）で正規表現・和集合ロジックを確認、全件成功。bearバックテストは本セッションの
サンドボックス環境に学習済みモデル(rf_drop_model.pkl)が無くYahoo Financeへの
ネットワークアクセスも不可のため未実施。銘柄ユニバースの拡張（全銘柄スキャン対象が
新形式コード分増える）であり、モデルの特徴量定義・ハードフィルターには無変更。
マージ前にbear（2024/08下落相場）での平均リターン・勝率・大勝率のデグレが無いことを
別環境で確認することを推奨。
```

## 2026-08-03 ブログ記事が0件投稿になっていたバグを修正（eyecatchフィールドの型不一致）

```
発見の経緯: ユーザーから「記事が変わってない」との指摘。daily_alert.yml の実行ログ
（run 30803078843、2026-08-03T09:50〜）を確認したところ、Step 5c
（ブログ記事自動生成・投稿）が候補多数を処理しつつ「0件処理しました」で終わっていた。
全候補が `'dealType' を配列形式に変えて再送信します`（既存の既知対応）の後、
`⚠ 投稿失敗 HTTP 400: {"message":"'eyecatch' has unexpected data type."}` で失敗していた。

原因: web/publish_blog_articles.py の publish_article() の型不一致リトライは、
microCMSが「フィールドXの型が不一致」と1つずつ返すエラーを検知して自動修正する
仕組みだが、文字列→配列への変換にしか対応していなかった。eyecatch は
build_eyecatch_for_article() が返す {"url": ...} という辞書値であり、
配列化では直せないため無限にリトライ対象外のまま「投稿失敗」として記事ごと
破棄されていた。Pexels画像生成・microCMSへのアップロード自体は毎回成功しており
（ログにエラーなし）、記事投稿の最終ステップでだけ落ちていたため気づきにくかった。

修正: publish_article() のリトライで、型不一致フィールドが文字列でない場合は
そのフィールドを除外して再送信するようにした（画像なしで記事のみ投稿する方を、
記事ごと投稿失敗させるより優先）。tests/test_publish_blog_articles.py に
eyecatchのようなオブジェクト値フィールドを除外して再送信するケースを追加
（45→46件）。

検証: 既存テスト全10ファイル成功、デグレなし。
判定: モデル・ハードフィルター・特徴量定義には無変更のためバックテスト対象外。
```

## 2026-08-02 ブログ記事の投稿件数上限を撤廃

```
web/publish_blog_articles.py の MAX_ARTICLES_PER_RUN=10（1回の実行での投稿上限）を撤廃。
build_and_publish() の max_articles デフォルトを None にし、None のときはループを打ち切らない
（--max-articles 未指定時も同様）。テスト（tests/test_publish_blog_articles.py）は元々
max_articles=3 を明示的に渡しているため無影響、全45件成功を確認。
```

## 2026-08-02 ブログ記事に事業内容紹介・規模感・ラベル付き推測・株価チャートを追加

```
ユーザー提示のサンプル記事（例: エム・エイチ・グループ 9439）を基に、記事の情報量を増やす
方向で4点追加。ただし例にあった「アジア圏への進出やピボットが計画されている可能性が高い」
という無条件の推測は、事実にない意図の創作を禁じてきた既存方針・過去のバグ修正
（PR #164「根拠なき買い/売り推測を解消」等）と衝突するため、ユーザーに確認の上
「推測は含めるが必ず※推測:ラベルで事実と分離する」方式に調整して採用。

1. 事業内容紹介（web/publish_blog_articles.py: get_company_description）
   対象企業が何をしている会社かをClaudeの一般知識から1文取得し、jpx_stock_list.description
   列（ALTER TABLEで追加）にキャッシュ。classify_filerと同じ「Web検索確認済みマスター無し→
   Claude一般知識→保存」の方針。記事冒頭の紹介文に自然に織り込むようプロンプトに追加。

2. 保有比率の規模感（同ファイル、プロンプトのみ）
   新規データ取得はせず、既存のholding_ratio事実を使って「時価総額の一角を占める大株主」等
   実感が湧く形で触れるようプロンプトに指示を追加。

3. 「※推測:」ラベル付き推測文（同ファイル、プロンプトのみ）
   従来は開示日株価・下落リスク水準が取得できた場合のみ限定的なso what文を1文加えていたが、
   常に文末に「この取得が今後どんな意味を持ちうるか」の推測を1文加えるよう変更。事実の記述と
   混同しないよう文頭に「※推測:」を必須化し、事実として存在しない具体的計画やコメントの
   引用は禁止する指示を明記。

4. 株価チャート画像（同ファイル: generate_price_chart_image/upload_price_chart/
   build_price_chart_for_article）
   直近3ヶ月の終値（yahoo_price_cache）からシンプルな折れ線チャートPNGを生成し、
   microCMSのメディアAPIへアップロードして本文HTML末尾に<img>タグで埋め込む。
   matplotlib等の新規依存を避け、アイキャッチと同じPillowのみで直接描画する構成にした。
   アイキャッチのアップロード処理を_upload_media()に共通化し、upload_eyecatch/
   upload_price_chartの両方から使う。

検証: tests/test_publish_blog_articles.py に10件追加（35→45件）。既存テストのうち
      build_and_publish(dry_run=False)を呼ぶもの5件はbuild_price_chart_for_articleを
      未モックのままにするとPillow未インストール環境でImportErrorになるため、
      他の外部I/O（publish_article等）と同様にモックを追加。
      既存テストスイート全10ファイル実行しregressionなしを確認。
      README.mdのファイル構成・テスト件数も同一コミットで更新。
判定: モデル・ハードフィルター・特徴量定義には無変更のためバックテスト対象外。
```

---

## 2026-08-02 ブログ記事の定型文2種を削除（金額の概算注記・制度説明の定型結び）

```
ユーザーから、公開済みブログ記事の全てに以下2つの定型文が繰り返し出ていて冗長との指摘。

1. 「が、これはあくまで概算であり、実際の取得価格ではないことにご注意ください」
   web/publish_blog_articles.py の generate_article_body() プロンプトが
   「金額は...概算であり、実際の取得価格ではないことを本文中で明記してください」と
   明示的に指示していたため、Claudeが毎回この注記文を本文に書いていた。
   金額が概算である旨は見出しの「推定取得金額」表記で十分伝わるため、本文中での
   注記繰り返しをしないよう指示を変更。

2. 「大量保有報告書は、株式の5%以上の保有比率に達した場合に提出が義務付けられる
   もので、市場の透明性確保と投資家保護を目的とした制度です。今後の同社の経営方針や
   当該ファンドの投資戦略の動向に注視する必要があります。」
   プロンプトに明示指示はなかったが、Claudeが「3〜4段落」の指定に対して制度の一般的な
   説明・定型的な結び文で埋め合わせる傾向があった。この取引固有ではない一般論・定型結びを
   書かないよう明示的に禁止する指示を追加。

検証: tests/test_publish_blog_articles.py 含む既存テスト全10ファイル実行し regression なしを確認。
      プロンプト文言のみの変更のためテスト件数は増減なし。
判定: モデル・ハードフィルター・特徴量定義には無変更のためバックテスト対象外。
```

---

## 2026-08-02 LINE大口保有動向を3件に縮小、残りはWeb記事URLへ

```
通知疲れ対策の追加調整。web/market_timing_alert.py の LARGE_HOLDINGS_LIMIT を 5→3 に縮小し、
上限を超えた分は「LINEで聞けば個別回答」ではなく microCMSブログ（詳細解説記事）のURLに委ねる
方針に変更。check_catalystツールでの都度回答に頼らず、Web側に情報を寄せる。
モデル・ハードフィルター・特徴量定義には無変更（バックテスト対象外）。

検証: tests/test_market_timing_alert.py 含む既存テスト全9ファイル実行し regression なしを確認。
README.md も同一コミットで更新。
```

---

## 2026-08-02 クジラウォッチ「注目」枠を1件→取得金額上位3件に変更

```
きっかけ: 「注目の一件」が実際には単に直近の新着記事(contents[0])を機械的に表示している
だけで、取得金額の大小など「注目に値するか」の判定を一切していなかった。ユーザーから
「1件である必要はない」という指摘を受け、直近プール(20件)の中から推定取得金額
(dealAmount)が大きい順に3件を選ぶロジックに変更。

- kujira-watch/src/lib/microcms.ts: getFeaturedArticles(poolSize=20, count=3) を追加。
  直近20件を取得しdealAmount降順にソートして上位3件を返す。
- kujira-watch/src/components/FeaturedArticleCard.tsx: rankプロップを追加し、
  バッジを「注目の一件」固定から「🥇/🥈/🥉 注目N位」に変更。
- kujira-watch/src/app/page.tsx: 単一のfeatured抽出をgetFeaturedArticles()呼び出しに
  置き換え、選ばれた3件をメイン一覧(日付グループ)から重複除外。

検証: npx tsc --noEmit / npm run lint (eslint) いずれもエラーなし。next build は
MICROCMS_SERVICE_DOMAIN/API_KEY未設定（このサンドボックスに.env.localが無い）のため
ページデータ収集の直前で失敗するが、コンパイル・型チェックまでは成功しており
今回の変更に起因する失敗ではないことを確認済み。実データでの動作確認はVercel
プレビューデプロイで行う。

判定: フロントエンドの表示ロジック変更のみ（Python側のモデル・特徴量・下落確率
ロジックには一切触れていないためbacktest対象外）。
```

---

## 2026-08-01 コンサルレビューの残り5件に着手（通知疲れ・記事のso what・screener整理・可観測性・閾値整合）

```
前回(2026-07-31)のコンサルレビューで保留した改善案5件に着手。いずれもモデル・ハードフィルター・
特徴量定義には無変更（バックテスト対象外）。

1. LINE通知疲れ対策（web/market_timing_alert.py）
   ウォッチリストで閾値未達・変化なしの銘柄を毎日個別表示すると同文の繰り返しで読み飛ばされる
   ため、アクションのある銘柄（買い時/売り検討）だけ個別表示し、変化なし銘柄は末尾に件数だけ
   要約するよう変更。あわせて前営業日のdrop_probをまとめて取得し（get_previous_rankings、
   2クエリで完結）、個別表示する銘柄には前日比（pt）を添えるようにした。
   tests/test_market_timing_alert.py に4件追加。

2. ブログ記事の「だから何？」不足（web/publish_blog_articles.py）
   事実の並置だけで終わっていた記事に、開示日時点（point-in-time、記事公開時点の
   post-hocスナップショットではない）の株価・下落リスク水準（高/やや高/中/やや低/低、
   README記載の5段階表示と同じ閾値）を取得できた場合はプロンプトへ文脈として渡し、
   その範囲内での投資家への意味づけを1文加えさせるようにした（取得不可なら従来通り事実のみ）。
   post-hocの現在値ではなくPITスナップショットを使うのはCLAUDE.md PIT規律に準拠するため。
   tests/test_publish_blog_articles.py に5件追加。

3. core/screener.pyの実質デッドパス化を解消
   core/rank_stocks.pyはcore.screenerからget_tse_stock_list()のみを再利用し、
   screener.pyが生成するdata/screeners/*.csvはリポジトリ全体のどこからも読まれていない
   （grep確認済み）ことが判明。全銘柄の株価取得（約30分）を screener.py と rank_stocks.py で
   二重に行っているだけだった。daily_alert.ymlのStep 1（screener.py実行）を削除し、
   README（システム概要・ファイル構成・スクリーナー条件・設計上の注意点）を実態に合わせて
   修正。あわせてapply_screener_v1のrel_strength_min引数がロジック内で未使用（デッド
   パラメータ）だったことも判明したためREADMEから該当記述を削除。core/screener.py自体は
   get_tse_stock_list()が現役利用されているため削除せず、手動スクリーニング確認用ツールとして残置。

4. 💎買い0件時のファンダ欠損可観測性ログ追加（core/rank_stocks.py）
   piotroski/eps_surprise/bps_growthの欠損によりqv_okが常にFalseになり、下落確率が
   どれだけ低くても💎買いになり得ない銘柄がある。「相場が悪いのか」「ファンダデータが
   欠損しているのか」を運用上区別できるよう、ファンダ欠損銘柄数と現時点の💎買い件数を
   ログ出力するようにした（ロジック自体は無変更、観測性のみ追加）。

5. ウォッチリスト売り閾値デフォルト(20%)とシステム全体基準(10%)の乖離解消
   （web/market_timing_alert.py）
   dp_sell_thresholdの既定値20%は、recommend_from_scoresの売り検討基準（drop_prob≥10%等）
   より緩い。前回のコミットで追加した「dp<buy_thの時だけrecommendを見るガード」では、
   dpがbuy_th〜sell_thの間（例: 8%〜20%）にあるとシステムは既に🔴売り検討と判定していても
   ウォッチ通知は沈黙していた。判定順序を変更し、recommendが「🔴 売り検討」なら
   個人の閾値設定に関わらず必ず⚠️売り検討を表示するようにした（dp>=sell_thのelif分岐は、
   ユーザーが独自により厳しい閾値を設定した場合の補助として残す）。
   tests/test_market_timing_alert.py に1件追加。

検証: 既存テスト全9ファイル（tests/test_*.py）を実行し regression なしを確認。
      test_market_timing_alert.py 16→20件、test_publish_blog_articles.py 13→18件。
      core/rank_stocks.pyの可観測性ログ追加箇所は専用テストファイルが元々存在しない
      （DB/モデル依存の統合的処理のため）ため構文チェックのみ。

判定: マージ可（通知UX改善・記事品質改善・デッドコード整理・可観測性向上・閾値整合）。
```

---

## 2026-07-31 コンサルエージェントによる利用者価値レビュー→バグ修正3件

```
経緯: 「モデルバッチ・LINEメッセージ・web記事について、利用者にとってもっと
      役立つツールになるようアドバイスし、修正させる」という依頼を受け、
      general-purposeエージェントに3領域（core/rf_train_v3.py・screener.py・
      rank_stocks.py／web/market_timing_alert.py／web/publish_blog_articles.py）の
      実コード・テスト・README・dev_log.mdを読ませて利用者視点でレビューさせた。

指摘のうち、確認済みバグ・モデル無関係・バックテスト不要な3件を即修正:

1. core/rank_stocks.py:721 未定義変数 target_date による NameError（無音でexcept握りつぶし）
   → 2026-07-14の日経vsS&P500機能追加以来、毎日必ず失敗し data/market_compare.json が
     更新されないままLINE配信されていた疑い。datetime.now()に修正（1行）。

2. web/publish_blog_articles.py:289 is_sell_disclosure() の呼び出しが概要欄キーワードのみで
   holding_ratio/holding_ratio_prior を渡していなかった。2026-07-23に他3箇所
   （market_timing_alert.py等）で修正済みの「保有比率増減による売り判定」が
   このファイルだけ未反映で、東芝型（概要「変更報告書」のみ・実際は比率減少=売り）を
   「買い」記事として自動投稿しうる状態だった。引数を追加して修正、
   tests/test_publish_blog_articles.py に東芝型ケースを追加（既存テスト関数の強化のため件数は変更なし。
   なお同ブランチのmainマージ後は、別PR #181のcategoryフィールド廃止に伴うテスト削減で17→13件）。

3. web/market_timing_alert.py: ウォッチリストの「🔔買い時！」判定がdrop_prob閾値のみで、
   ランキング本体（品質フィルター込み）の推奨ラベルを見ておらず、同一銘柄で
   「ウォッチ通知は買い時／ランキングは🔴売り検討」という矛盾表示になりえた。
   get_today_rankings の select に recommend を追加し、build_watchlist_section で
   recommend=="🔴 売り検討" の場合は dp閾値未達でも買い時表示を抑制するガードを追加。
   tests/test_market_timing_alert.py にテスト3件追加（13→16件）。

検証: 既存テスト全件（tests/test_*.py 全9ファイル）実行し regression なしを確認。
      いずれもモデル・ハードフィルター・特徴量定義には無変更のためバックテスト対象外。

判定: マージ可（バグ修正・可観測性改善のみ）。
      レビューで指摘された残りの改善案（LINEの通知疲れ対策、web記事のso what不足、
      screener.pyの実質デッドパス化、💎買い0件時のファンダ欠損可観測性など）は
      次タスク候補として保留。
```

---

## 2026-07-31 CI (ci.yml) にanthropicが不足していて4件のテストが実CIで失敗していたのを修正

```
発見の経緯: ユーザーが実際のGitHub Actions CI (ci.yml) の失敗ログを貼り付け。
      tests/test_publish_blog_articles.py の4件が
      `ModuleNotFoundError: No module named 'anthropic'` で失敗（77 passed, 4 failed）。

原因: web/publish_blog_articles.py が anthropic を import しており、
      それをテストする tests/test_publish_blog_articles.py も間接的に
      anthropic に依存する。しかし .github/workflows/ci.yml の
      pip install 一覧に anthropic が含まれていなかった（追加時に
      CI依存関係の更新が漏れていた）。

修正: ci.yml の pip install に anthropic を追加。
      lightgbm は core/rf_train_v3.py のみが使用し、tests/ 配下から
      直接importされていないため対象外（grep で確認済み）。

検証: ローカルで anthropic をインストールし
      `python3 -m pytest tests/ -v --tb=short` を実行 → 83 passed。
```

## 2026-07-31 rf_train_v3.py の学習が数時間かかる原因（jquants_fin_summaryへの過剰クエリ）を修正

```
発見の経緯: モデルキャッシュ修正後の初回フル学習を監視中、Supabase APIログを
      確認したところ、同一銘柄に対してdisc_date違い（サンプル日ごと）で
      jquants_fin_summaryへの問い合わせが繰り返し発生していた。

原因: lib/fundamentals.py の get_pit_fundamentals()/get_pit_valuation() が、
      呼ばれるたびに lib/db.py の get_jquants_fin_history()/
      get_jquants_fin_history_fy() 経由でSupabaseへ生のネットワーク
      リクエストを送っていた（キャッシュなし）。rf_train_v3.py の
      generate_samples() は1銘柄あたり約60サンプル日をループするため、
      1銘柄で約240リクエスト（get_pit_fundamentals内3クエリ×60 + 
      get_pit_valuation重複含む）、東証全銘柄(3500超)で80万回以上の
      個別リクエストになっていた。これが「モデル学習が5時間以上かかる」
      直接の原因だった（README記載の想定所要時間「40〜70分」から大幅に乖離）。

対応:
  - lib/db.py: get_jquants_fin_history_all(code) を追加（銘柄の全開示履歴を
    1回のクエリで取得。disc_date降順）
  - lib/fundamentals.py: _filter_asof() を追加し、_jq_split_safe_bps/
    get_pit_valuation/get_pit_fundamentals/pit_fundamental_features に
    任意の rows 引数を追加。rows（銘柄の全履歴）が渡された場合はDBに
    問い合わせず、point-in-timeフィルタ（as_of日以前・件数制限・
    doc_type=FY絞り込み）をメモリ上で再現する。rows省略時は従来通り
    DB問い合わせ（rank_stocks.py/backtest.pyは1銘柄1回の呼び出しのため
    変更不要、後方互換）
  - core/rf_train_v3.py: generate_samples()にfin_rows引数を追加し
    get_pit_fundamentals()へ橋渡し。main()の銘柄ループで
    get_jquants_fin_history_all(code)を1回だけ呼び出しfin_rowsとして渡す
    ことで、銘柄あたりのjquants_fin_summaryクエリを約60回→1回に削減
  - tests/test_fundamentals.py を新規追加（_filter_asofの先読み防止・
    limit・doc_type絞り込み、get_pit_fundamentals(rows=...)の
    point-in-time正しさを検証、6件）

判定: パフォーマンス修正（特徴量の値・計算式は不変、モデル出力に影響なし
      のためbacktest対象外）。既存テスト全件+新規6件パス。次回の
      Friday retrain（またはモデルキャッシュ空の状態での実行）で
      所要時間が大幅短縮されることを実運用で確認する必要がある。
```

---

## 2026-07-31 daily_alert.yml モデルキャッシュの設計不具合を修正（LINE定期配信停止の根本原因）

```
発見の経緯: ユーザーから「LINEから定期メッセージが来ない」と報告。GitHub Actions
      の daily_alert.yml 実行履歴を調査したところ、直近2回（7/24, 7/30）とも
      Step 2「モデル学習」が数時間経っても終わらず job timeout(360分)で
      cancelled になり、以降の全ステップ（ランキング生成・Supabaseエクスポート・
      LINE配信含む）がスキップされていた。ログには
      "Cache not found for input keys: ml-models-v37-weekly-113, ml-models-v37-weekly-"
      と出ており、木曜（非retrain日）にもかかわらずキャッシュ皆無で強制フル
      再学習に入っていた。

原因: モデルキャッシュのキーが `ml-models-v37-weekly-${{ github.run_number }}`
      （実行のたびに変わる値）になっていた。actions/cache は同一キーの上書きが
      できないため、成功する実行のたびに新しいキャッシュエントリが際限なく
      積み上がる一方、復元は前方一致(restore-keys)の運任せになっていた。
      リポジトリ全体のキャッシュ容量上限(10GB)や他ワークフローの使用量と
      合わさって、肝心の「学習済みモデル」キャッシュが予測不能なタイミングで
      失われ、非retrain日でも強制フル再学習（実測5時間超）が走る状態になって
      いた。

対応:
  - .github/workflows/daily_alert.yml:
    - モデルキャッシュのキーを固定値 `rf-drop-model-v1` に変更
    - actions/cache@v4（復元+保存の複合アクション）を
      actions/cache/restore@v4 + actions/cache/save@v4 に分離
    - 保存前に、復元時にヒットしていた場合のみ `gh cache delete rf-drop-model-v1`
      で既存分を削除してから保存し直す（固定キーで常に最新の1件だけを保持）
    - 上記の`gh cache delete`実行のため `permissions: actions: write` を追加
  - .github/workflows/backfill_rankings.yml: 復元専用のキーも同じ
    `rf-drop-model-v1` に統一

判定: インフラの信頼性修正（モデル・買いフィルターへの変更なし、backtest対象外）。
      次回のFriday retrain（または手動実行）でキャッシュが正しく保存・復元
      されることを実運用で確認する必要がある。
```

---

## 2026-07-30 LINE大口保有通知にmicroCMSブログへのリンクを追加

```
背景: ユーザー指示。EDINET大口保有報告（🏦セクション）はLINEの日次プッシュ通知
      （web/market_timing_alert.py）とAIチャットのcheck_catalystツール
      （supabase/functions/line-webhook/index.ts）の2箇所に表示されるが、
      web/publish_blog_articles.py（Step 5c）が生成する詳細解説記事
      （microcms-blog-demo, https://stock-alert-lyart.vercel.app/）への
      導線がLINE側に無かった。

対応:
  - web/market_timing_alert.py: BLOG_SITE_URL定数を追加し、
    build_large_holdings_section()の末尾に詳細解説記事へのリンクを追加
    （大口保有の話がある場合のみ）
  - supabase/functions/line-webhook/index.ts: 同名の定数を追加し、
    executeCheckCatalystの🏦セクション末尾にも同じリンクを追加
  - tests/test_market_timing_alert.py: リンクが含まれることを検証するテストを
    追加（12→13件）

判定: 表示追加のみ（モデル・買いフィルターへの変更なし、backtest対象外）。
      既存テスト全件+新規1件パス。
```

---

## 2026-07-25 Supabaseクライアントのネットワークタイムアウト耐性追加

```
発見の経緯: ユーザーがPR #163（下落モデル一本化）のbacktest.py bear検証を
      ローカル環境で実行中、Supabaseへのyahoo_price_cache書き込み
      (insert_ignore)が読み取りタイムアウト(30秒)で失敗し、リトライ処理が
      無かったため例外がそのまま伝播してバックテスト全体が停止した。

原因: lib/supabase_client.py の insert_ignore/select/select_one/delete/rpc は
      requests.post/get/delete を直接呼ぶだけで例外処理が無く、単発の
      ネットワーク瞬断でも即座に呼び出し元まで例外が伝播していた
      （upsertだけは既にtry/exceptで例外を握りつぶしバッチ単位でスキップする
      作りだったが、他の関数には無かった）。

対応:
  - lib/supabase_client.py に共通ラッパー _request() を追加。
    requests.exceptions.RequestException（タイムアウト・接続エラー等）を
    最大3回、指数バックオフ(2s/4s/8s)でリトライしてから諦める
  - upsert/insert_ignore/select/select_one/delete/rpc の生requests呼び出しを
    全て _request() 経由に統一
  - insert_ignore は upsert と同様、リトライを尽くしても失敗した場合は
    そのバッチをログに記録してスキップし、呼び出し元（バックテスト等の
    長時間パイプライン）を丸ごと落とさないようにした
  - tests/test_supabase_client.py を新規追加（リトライ後の成功・リトライ尽きた
    後の例外送出・insert_ignoreが最終失敗時も呼び出し元を落とさないことを
    requests.requestをモックして検証、3件）

判定: インフラの堅牢性修正（モデル・買いフィルターへの変更なし、backtest対象外）。
      既存テスト全件+新規3件パス。
```

---

## 2026-07-23 EDINET大量保有の買い/売り誤判定バグ修正

```
発見の経緯: ユーザーがLINE通知で見た「🏦大量保有報告」で以下の誤表示を発見。
  - 株式会社東芝がキオクシアHD(285A)株を一部売却し保有比率16.10%→15.10%に
    低下（関東財務局への変更報告書で開示）したのに📈買いと表示
  - Ａｔａｒｉ　Ｃａｐｉｔａｌ（9399 ビート・ホールディングス）も保有比率
    16.93%→14.44%に低下したのに📈買いと表示

原因: 買い/売りの方向判定(`is_sell_disclosure`)が、EDINETの概要欄
      (doc_description)に「譲渡」「売却」「売出」「処分」のいずれかの
      キーワードが含まれるかどうかだけで判定していた。しかし変更報告書の
      概要欄は多くの場合「変更報告書」とだけ書かれ売買方向を示さないため、
      実際には売却でもキーワードにヒットせず📈買いにフォールバックしていた
      （lib/edinet.py, tools/scan_large_holdings.py, web/market_timing_alert.py,
      supabase/functions/line-webhook/index.ts の4箇所で同一パターン）。

対応:
  - lib/edinet.py: XBRLから直前報告時点の保有割合
    (HoldingRatioOfShareCertificatesEtcPerLastReportタグ、これまで
    パースの都合で明示的に除外していた)も取得し holding_ratio_prior として返す
  - Supabase: edinet_large_holdings に holding_ratio_prior 列を追加(migration適用済み)
  - tools/scan_large_holdings.py: is_sell_disclosure/is_noise_match に
    holding_ratio・holding_ratio_prior を渡せるようにし、両方取得できる場合は
    比率の増減で判定（現在<直前なら売り）。片方でも欠ける場合のみ従来の
    キーワード判定にフォールバック
  - web/market_timing_alert.py: 上記と同様の優先順位（holding_ratio_prior→
    同一提出者の複数開示から見た最古比率→キーワード）で方向判定
  - supabase/functions/line-webhook/index.ts (check_catalyst): isSellDisclosureを
    同じ優先順位に変更、select句にholding_ratio_priorを追加
  - tests/test_scan_large_holdings.py: 比率減少での売り判定テスト追加(7→9件)
  - tests/test_market_timing_alert.py: 東芝/キオクシアの実例を再現したテスト追加(11→12件)

判定: 表示バグの修正（モデル・買いフィルターへの変更なし、backtest対象外）。
      既存テスト全件パス。過去に蓄積済みの edinet_large_holdings 行は
      holding_ratio_prior が null のままキーワード判定にフォールバックする
      （新規スキャン分から順次埋まる。過去分の一括バックフィルは未実施）。
```

---

## 2026-07-20 下落モデル一本化（上昇モデル・Netスコア・💎買いシステムの廃止）

```
背景: ユーザーから「webを削除して、下落モデルしか使ってないから、たくさん削除できる
      ものがあると思ってる」との指摘。調査の結果、以下が判明：
      - core/rf_train_v3.py は rf_model.pkl（上昇）と rf_drop_model.pkl（下落）の
        2モデルを学習・保存していたが、rank_stocks.py/backtest.py 側で条件分岐して
        いた「4モデルアンサンブル」（alpha_rise/alpha_drop込み）は alpha モデルが
        一度も生成されていないため実質デッドコードだった
      - LINE Bot（supabase/functions/line-webhook/index.ts）はウォッチリストの
        買い時/売り時判定・ランキングソート・AIチャットの根拠のいずれにおいても
        drop_prob のみを参照し、recommend（💎買いラベル）や net はどこからも
        参照していなかった。AIチャット自体のシステムプロンプトにも「上昇モデルは
        精度が低いのでnetスコアは絶対に使わない」と明記されていた
      - つまり上昇モデル・Netスコア・QVフィルターに基づく「💎買いシステム」は
        gen_rankingsに毎日書き込まれるだけで、本番のどのユーザー接点にも一切
        到達していない完全な計算コスト・保守コストのみのオーバーヘッドだった

対応（ユーザー承認: 上昇モデル/Net/💎買いシステム全廃止 + backtest系ツールの
      大幅書き換えも許可、との3点確認に「あってます」で合意）:
  - lib/utils.py: recommend_from_scores() から net 引数を削除。net起因の
    売買判定（net<-5売り、net≥10買いゲート）を撤廃し、drop_prob<8%を唯一の
    確率ベース買いゲートとした
  - core/rank_stocks.py: rise_model/alpha_rise_model/alpha_drop_model の
    読み込み・net計算を削除。ソート順をdrop_prob昇順に変更。判定ラベルを
    drop_prob閾値ベース（🟢安全圏<8%/🟡通常<15%/🔴危険）に変更。
    既存の4防御フィルター（優待権利落ち・米国ETFリードラグ・NLP感情・
    相場リスク管制官）は元々 recommend文字列（"💎 買い"）で判定していたため
    ロジック変更不要だった
  - core/rf_train_v3.py: 上昇モデル(rf_model.pkl)の学習・保存を停止。
    generate_samples() から label_rise/alpha_rise/alpha_drop の計算を削除
  - lib/data_sanity.py: net整合性チェック（net=rise-drop、このQAモジュール
    創設のきっかけとなった過去バグの検知ロジック）を廃止。下落確率のみの
    検査に一本化（予測多様性/縮退検知の対象もdrop_probに変更）
  - tools/backtest.py・multi_backtest.py: --net-min→--drop-max、
    nlargest(net)→nsmallest(drop_prob) に全面書き換え
  - tools/optimize_net_weights.py・tools/simulate_monthly.py: 目的が消滅した
    ため削除（simulate_monthly.pyは調査中に buy_labels 未定義のクラッシュ
    バグも発見済み）
  - tools/export_report_to_sheets.py: net/rise_prob列を削除、モデル説明を
    「XGBoost（下落モデルのみ）」に修正（ついでに従来の誤記だった
    「21日後±5%」を実際の「63日後±15%」に修正）
  - web/export_to_web.py・lib/db.py: select/order を drop_prob 基準に変更
  - config.py: 未使用だった FORECAST/RISE_THRESHOLD/MAX_BUY_VOL20 と、
    simulate_monthly.py専用だった新規候補フィルター定数群を削除
  - tests/test_data_sanity.py: drop_prob単一のフィクスチャに全面書き換え
    （TestNetIntegrity削除、他は移植）。README.md/CLAUDE.mdのモデル説明・
    S買い条件・ファイルマップ・AUC表記も同時更新
  - 影響なし（変更不要と確認済み）: tools/catalyst_backtest.py（元々net/rise
    非依存）、export_all_to_supabase()（過去日付のrise_prob/net値を読み戻す
    処理は正しい挙動のため維持）

制約・注意: このサンドボックス環境には xgboost/scikit-learn/学習済みモデル
      (.pkl)/DB接続が無いため、CLAUDE.mdの「改善マージ規律」が要求する
      tools/backtest.py bear での効果検証はここでは実行不可能だった。
      コード変更後、GitHub Actions等の別環境でbacktest.py bear（および
      通常期間）を実行し、平均リターン・勝率・大勝率が既存と同等以上で
      あることを確認してからのマージが必須（未検証のままdeploy厳禁）。

判定: 検証なしでマージ承認 — 本来はCLAUDE.mdの改善マージ規律によりbacktest.py bear
      での効果確認が必須だが、ユーザーローカル環境での検証が環境要因
      （lightgbm未インストール→Supabaseタイムアウト、いずれも別PRで対応済み）で
      難航したため、ユーザーが2026-07-27に「検証なしでも今回はマージしていい」と
      明示的に許可し例外的にマージした。次回以降の変更は通常通り検証必須。
```

---

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

## 2026-08-15 kujira-watch: スマホ表示改善（最終形）

当初はウェブフォント除去＋MUI除去の両方をやっていたが、並行してmainで
「Material Design化 Phase 7/8」「ボタンUIをMUI Button統一」が進んでおり、
MUI除去は真っ向から対立していた（試しにマージすると11ファイルが方針レベルで衝突）。
ユーザー判断で**MUI除去は取り下げ、MUIと無関係な性能修正のみ**に絞った。

さらに作業中に別セッションが `/investors` の軽量化（PostgREST 1000行上限の
ページング・200件ずつのページ送り・行の軽量化）をmainへ先にマージしたため、
こちらで用意していた同等の修正（getAllFilersのrange ページング・
/investorsのページネーション・Pagination.tsx）は不要になり破棄した。
※こちらが「1000行上限で黙って切られている可能性」と指摘した点は実害として確認され、
サイトマップから投資家ページ約1,900件が漏れていたと判明している。

### 最終的に残した修正
1. **和文ウェブフォントの廃止（最大の効果）**
   `next/font/google` の `subsets` はプリロード範囲の指定でしかなく、生成CSSから
   CJKの `@font-face` は落ちない。Noto Sans JP 4ウェイトで `@font-face` が496個・
   378KB（gzip 130KB）のレンダリングブロッキングCSSになり、さらに本文の漢字に
   応じて70〜90KBのwoff2スライスをウェイトごとに追加ダウンロードしていた。
   和文は端末内蔵フォント（iOS: Hiragino Sans / Android: 内蔵のNoto Sans CJK JP）
   に委ね、欧文のGeistのみ残す。
2. **`getFilerWinRates()` を `unstable_cache` に載せる**
   `/ranking` も `searchParams` を読むためdynamic renderingになりページ側の
   `revalidate` が効かない。`getAllFilers()` は別セッションがキャッシュ済みだったが
   こちらは未対応のままだった。
3. **RippleEffectのpointerdownをpassiveに**（スクロール開始をブロックしうるため）
4. UI: ヘッダーの月別アーカイブを一番右へ / `/about` のタイトルを「このサイトについて」へ

### 結果（gzip後・main 8f95386 との比較）
| 指標 | 前 | 後 |
|---|---|---|
| CSS 合計 | 138 KB | 7 KB (-95%) |
| `@font-face` 宣言数 | 510 | 13 |
| woff2 ファイル | 135個 / 5.5MB | 11個 / 184KB |
| 初期JS | 410 KB | 410 KB（MUIを残したため変化なし） |

## 2026-08-15 kujira-watch: MUIを残したまま初期JSを削る（遅延読み込み）

「MUI入れても早くする方法はあるでしょ」の指摘を受けて再検討。あった。
MUIを外さずに、**閉じているのが既定なのに全ページの初期JSに積まれていた**
コンポーネントを`next/dynamic`で遅延化する。

- `StockSearch` → パネル(Autocomplete/TextField/CircularProgress)を
  `StockSearchPanel.tsx`へ切り出し。虫眼鏡をタップするまで読み込まない。
- `HeaderMenu` → ドロワー本体(Drawer/List/ListItemButton/Divider/ChevronRight)を
  `HeaderMenuDrawer.tsx`へ切り出し。MUI DrawerはModal/Portal/Backdrop/Slide一式を
  引き連れている。

使うMUIコンポーネントも見た目も挙動も一切変えていない。読み込みのタイミングを
後ろにずらしただけなので、Material Design化の方針とは衝突しない。

### 結果（初期ロードJS・gzip後）
| ページ | 前 | 後 |
|---|---|---|
| `/` | 321.4 KB | 288.3 KB (-33.1 KB) |
| `/articles/[id]` | 318.1 KB | 285.3 KB (-32.8 KB, -10.3%) |
| 他の主要ページ | 約317 KB | 約284 KB |

※ `next build` の `/monthly/[month]` の collect 失敗は、ダミーAPIキーでの403による
ローカル環境の制約。main でも同様に出るため今回の変更とは無関係（tsc・eslintはクリーン）。

### まだ残っている手（未着手）
emotionのランタイムが全ページに残っている。MUIのゼロランタイム版(Pigment CSS)へ
移行すればMUIの見た目を保ったまま更に削れるが、全コンポーネントに影響する移行なので
別途判断が要る。

## 2026-08-15 デザインコンサルスキルの導入

- `.claude/skills/design-consult/SKILL.md` を新規追加。kujira-watchサイト・アイキャッチ画像・Remotion動画のデザインレビュー/改善提案を行うデザインコンサルタントのスキル。
- ブランドトークン（`globals.css` CSS変数 / `theme.ts` brandパレット）を唯一の色ソースとして明文化し、技術制約（和文ウェブフォント禁止・MUI Button既定値・RSC境界・プレス演出二重化禁止・ja/en両対応）を焼き込み。
- レビュー手順は Playwright スクリーンショット（375px/1280px × ja/en）による目視必須。報告は P1/P2/P3 の優先度付きフォーマット。
- Think Small 原則: 大規模リデザインは提案せず、既存デザインシステム内での調整に限定。実装はユーザー依頼時のみ。
- モデル・パイプラインのコード変更なし（ドキュメント/スキルのみ。バックテスト対象外）。

## 2026-08-15 kujira-watch: デザインコンサルレビューと改善（design-consultスキル初回実行）

外部ネットワーク遮断環境のため、本番Supabaseの実データ抜粋＋microCMS/PostgREST互換モックで
ローカル再現し、Playwrightで375px/1280px × ja/en 全主要ページを目視監査（モックはscratchpadのみ、
コミット対象外）。レビュー詳細は `kujira-watch/docs/progress_design_consult.md`。

### P1（可読性の実害）
- 注目カード: アイキャッチ焼き込みテキストとカード文字の二重表示 → 画像opacity 0.5＋スクリム強化
- 注目カード上の分類/売りバッジがダーク地で沈む → `onDark` プロパティ追加（文字白・色はドットのみ）
- モバイルヘッダーで「クジラウォッチ」が3行折返し → ロゴnowrap＋訪問者数カウンターをsm以上のみに
- 記事詳細で分類バッジが二重（DealTypeBadge＋CategoryBadge同一ラベル）→ リンク付きChipへ一本化(ja/en)

### P2（一貫性・洗練）
- /rankingの「+107,900.0億円」→ 1兆円以上は兆円へ繰り上げ（`formatAmountParts`、/weeklyタイルと共用）
- EDINET全角英数名（ＢＣＰＥ　Ｐａｎｇｅａ…）→ 表示専用 `displayFilerName()`（href/DB照合は原文維持）を
  ranking/investors一覧・詳細/trending/記事メタに適用
- /weekly金額タイルの数字折返し → 数字+単位分離＋レスポンシブfontSizeで左タイルと構造統一
- 投資家一覧の行レイアウト揺れ → 名前/メタの2行構造に統一

検証: `npx tsc --noEmit`・eslint（変更12ファイル）クリーン、全主要ルート200、
修正後スクリーンショットで before/after 目視確認済み。Python側パイプラインへの変更なし。

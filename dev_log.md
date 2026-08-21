# Dev Log

## 2026-08-20 TikTok投稿機能を完全撤去

```
背景: TikTokアプリ審査がリジェクト（「App will not be approved for personal or company
internal use」）。自アカウントへの自動投稿という用途自体がTikTok for Developersの
本番承認ポリシーの対象外で、書類・デモの改善で通る余地が無い。Sandboxでの下書き運用は
可能だが、毎晩の手動公開が必要なことからオーナー判断で撤去を決定。

削除: video/tiktok_client.py・video/tiktok_auth.py・関連テスト・video_post.ymlのTIKTOK_*
env・GitHub SecretsのTIKTOK_*（3件）・README/コメントの言及。LINE通知はTikTokキャプション
連携用だったが、YouTube投稿完了通知として簡素化して残置（line_notify.notify(props, youtube_id)）。

学び: プラットフォームAPIの審査ポリシー（誰向けのアプリを承認するか）は実装前に確認する。
TikTokは「自社サービスのユーザーに使わせるアプリ」のみ承認し、自社アカウント運用の
自動化ツールは承認しない。経緯の詳細は docs/tiktok_review.md。
```

## 2026-08-19 X投稿カードのブランド配色化とURLの本文復帰（オーナー指摘）

```
オーナー指摘2件を修正した。

1. 「写真2枚が微妙」→ web/x_card_image.py を作り直した
   - 独自色（#102A43 / #15804F 等）を使っていたのをブランドトークンだけに統一
     （globals.css・remotion theme と同値: navy #16213a / paper #fffdf8 /
     section-tint #f1ece1 / rule #ded5c0 / gold #b8863a / 買い #047857 / 売り #be123c）。
     Xから来た読者がサイトを別物と感じないよう媒体間で配色を揃える。
   - 純白背景 → 紙色(#fffdf8)、ヘッダー下に金の細線を1本。
   - 数字カード: 右半分が空白のまま残っていたので、主役の「保有比率 X%→Y%」に地色の帯を敷き、
     同じ帯の右に推定金額を置いた。訂正報告書は金額が無いので訂正幅(pt)を出す。
     帯とフッターの間にも空きが残っていたため縦の配分をやり直した。
   - 一覧カード: 行が2件の日に下半分が丸ごと空いていたので、行ブロックを領域の中央に置き、
     行の高さを残り高さから決めるようにした。最終行の下に宙ぶらりんの罫線が出る問題も解消。

2. 「URLが投稿にないのも微妙」→ link_in_reply() の既定を反転
   「本文中の外部リンクはリーチが落ちる」という定説から自己リプライを既定にしていたが、
   親投稿にURLが無いとスレッドを開かない読者にはリンクが一切届かない。本文にURLを入れる方を
   既定に戻し、X_LINK_IN_REPLY=1 で自己リプライ側に切り替えてA/Bを取る形にした
   （どちらだったかは x_posts.variant に残るので web/x_metrics.py --report で比較できる）。
```


## 2026-08-19 個人投資家1000人コンサルの12施策を実装（記事データの誤り是正が最優先）

```
docs/consult_1000_investors_20260819.md（仮想個人投資家1000人・10セグメントのコンサル）で
出した改善点を優先度順に12件、すべて実装した。事前監査で見つかった「記事の数字そのものが
違う」問題が1〜4位を占めたため、まずそこから直した。

1. 記事化の前提を守る（web/publish_blog_articles.py）
   EDINETはメタデータ公開とXBRL本文の可用性にラグがあり、提出直後の便では
   holding_ratio_prior が取れないことがある。その状態で記事化すると
   ratio_change_pct() が「今回比率の全量＝新規取得」とみなし、
     - タイトルが「X%を新規保有」（実際は変更報告書）
     - 推定金額が比率全量ぶんに膨らむ（セイコーグループ8050: 実際-0.15ptを-10.57pt・966.1億円）
   という誤りが公開されたまま残っていた。直近14日の照合可能56件中13件=23%が該当。
   → should_wait_for_prior_ratio() で、変更報告書かつ前回比率が未取得の開示は
     その便では記事化せず次の便へ持ち越す（PRIOR_RATIO_WAIT_DAYS=2を過ぎたら
     XBRLの書式差とみなして従来どおり履歴からの再導出で記事化。取りこぼしは作らない）。

2. 既存記事の是正（tools/fix_misreported_blog_articles.py 新規）
   EDINET開示を正として ratioChangePct・dealAmount・タイトル・tags・本文を作り直し
   microCMSへPATCH。対象は(a)ratioChangePctが実データとズレている記事、
   (b)前回比率>0なのにタイトルが「新規保有」の記事。是正後に is_worth_publishing() を
   割る記事は --delete 指定時のみバックアップを取って削除する。

3. 表示の正を1つに（kujira-watch/src/app/(ja)/articles/[id]/page.tsx）
   ファクトボックスの「前回比」はEDINET開示の（今回比率−直前保有割合）を優先し、
   CMSのratioChangePctはフォールバックへ降格。同じ画面に
   「保有比率10.57%（前回10.72%）」と「前回比−10.57pt」が並ぶ矛盾を解消。

4. 株価表記の一本化（disclosure_close_price()）
   本文へ渡す株価を gen_rankings（開示日に当日行が無ければ数日前を拾う）から
   開示日終値＝推定金額の概算に使うのと同じ値へ変更。サイトの「基準終値」とも同源になった。
   金額欄には「（概算）」を添える。

5〜12. フロント側（kujira-watch）
   5. 保有比率の推移グラフ（HoldingRatioChart.tsx、同一投資家×銘柄の開示を折れ線で）
   6. RSS/X導線（FollowUpdatesCta.tsx＋/investors/[filer]/feed.xml・/stocks/[code]/feed.xml。
      ウォッチリストは2026-08-18に削除済みでオーナー判断により復活させない方針のため、
      再訪の受け皿はRSSと公式Xの2つにする）
   7. /activists の軽量化（HTMLに載せる件数の上限を注目銘柄30件・直近の動き60件に。
      ShowMoreListは隠しているだけで全件HTMLに含めるため約500KBあった）
   8. 記事末尾の「自動生成」タグを非表示、「※推測:」段落を「編集部の見立て」枠へ切り出し
   9. TOPの開示サマリーに「この日の開示は確定」/「受付中」（src/lib/jst.ts）
   10. タイトルの全角英数を表示時に半角へ正規化（microcms.tsのnormalizeDealType内で1か所）
   11. 開示後の株価推移で未到来の地点に「開示から21営業日後に表示」
   12. /en/investors（Top Whales）を新設し、英語版ヘッダーとサイトマップから参照

テスト: tests/test_publish_blog_articles.py 95件（+4）
```

## 2026-08-19 X投稿の全面改修（インフルエンサー1000人コンサルの収束10施策）

```
docs/x_post_improvement_1000.md（仮想インフルエンサー1000人・10パネルのコンサル）で
出した改善点1000個を有力10施策に収束させ、全て実装した。

1. 投稿ログ＋メトリクス基盤（最優先）
   post_tweet() が bool ではなく tweet_id を返すようにし、log_post() が Supabase の
   x_posts へ記録。web/x_metrics.py（x_metrics.yml、毎日10時JST）が
   GET /2/tweets の public_metrics / non_public_metrics を引いて x_posts と
   x_post_metrics（日次スナップショット）へ保存する。--report で種別×variantの平均を出す。
   → これが無い間は「どの型が伸びたか」を一度も測れていなかった。以降の9施策は
     この数字で効果を判定してから本採用する（CLAUDE.md 改善マージ規律）。

2. 1行目をフック化: 記事タイトル（SEO用の長い定型文）の流用をやめ、
   「🟢 提出者が銘柄(コード)を買い増し」→「約120億円・保有比率 3.21%→5.02%」の2行に。

3. リンクを自己リプライへ: 本文中の外部リンクはリーチが落ちるため、親投稿はカード＋数字で
   完結させ URL は2投稿目に置く。X_LINK_IN_REPLY=0 で本文に戻せる（x_posts.variant で識別）。

4. 画像を数字カードに: web/x_card_image.py を新設。1枚目=銘柄・保有比率の変化・金額の
   カード、2枚目=従来の株価チャート。両方に alt を設定（これまで alt 未設定）。
   テキストのみだった日次サマリー・週次2本・答え合わせにも一覧カードを付けた。

5. 投稿時刻: 1回3件の連投を1件に（ARTICLES_PER_RUN=1）、JST 8〜22時以外は投稿しない、
   日次サマリーを19時JST→21時JST（edinet_blog.yml の cron を 0-10 → 0-12 UTC に延長）。

6. 答え合わせ投稿を新設: web/x_followup.py（x_followup.yml、水21時JST）。約3ヶ月前の
   開示銘柄の「開示日終値→直近終値」を yahoo_price_cache から集計し、平均・中央値・
   上昇銘柄数・最良・最悪を出す（勝ちだけ出さない）。株式分割で終値が不連続な銘柄
   （日次±40%以上）と基準日が7日以上離れる銘柄は除外。
   実データ確認: 5/20の54銘柄 → 平均+5.9%・中央値+3.3%・上昇32/54。

7. 訂正報告書を独立枠に: CORRECTIONS_PER_RUN=1 の制限を撤廃して全件投稿し、
   同じ銘柄の直近投稿（x_posts）が見つかればそこへ自己リプライでぶら下げる。

8. タグ整理: #EDINET と記号除去した #社名 を廃止し #日本株 #大量保有報告書 の2つに。
   銘柄は本文に 社名(コード) と素で書く（株クラはこの表記で検索する）。
   未使用になった web/x_post_format.hashtag() は削除。

9. 解釈行: web/x_insight.py。「この提出者の開示は過去N件、〇〇ではM回目」を1行足す。
   推定損益ベースの filer_win_rate は算出が誤っていて2026-08-18に廃止済みのため使わず、
   検証の要らない件数の事実だけで文脈を作る。文字数に収まらなければ落とす。

10. 運用ルールを docs/x_operation_rules.md に文書化（週2回の手動会話枠・禁止表現・
    訂正の扱い・A/Bの回し方）。コードで担保できない部分。

テスト: tests/test_x_client.py を50件に刷新（旧35件）。全体 373件パス。
```


## 2026-08-19 デッドコード除去（挙動不変のリファクタリング）

```
参照ゼロのコード（フロントエンド全面撤去・SQLite→Supabase移行・上昇モデル廃止で
取り残された残骸）をAST走査で洗い出し、実装ごと削除した。削除166行 / 追加23行。

削除した関数（全て参照ゼロ。commit 8dfd75e0 のフロント撤去等で消費側が消えていた）:
- lib/db.py: get_jquants_disc_dates / save_all_sectors / get_price_raw
- lib/db.py: init_db()（Supabase移行後は pass だけの空関数。呼び出し側 tools/fetch_history.py も削除）
- lib/edinet.py: fetch_holding_ratio（fetch_xbrl_details の後方互換ラッパー）
- lib/kabutan_earnings.py: format_earnings_for_prompt（LINE Bot用。消費側が撤去済み）
- lib/tdnet.py: get_recent_disclosures（同上）

削除した定数（参照ゼロ）:
- config.py: HOT_MARKET_THRESHOLD / MARKET_TIMING_ENABLED / MARKET_TIMING_SMA_DAYS /
  SCREENER_*（7個）。SCREENER_* は「screener.py/backtest.py/rf_train_v3.py と同値に保つ」
  と書かれていたが誰も読んでおらず、実際の閾値は tools/backtest.py と core/rf_train_v3.py の
  _SC_* が別々に持っている（＝黙って乖離しうる4重定義だった）。
- core/screener.py: MIN_MOMENTUM / MIN_VOLATILITY / MAX_VOLATILITY / MIN_MOMENTUM_20D /
  MIN_VOL_RATIO / MIN_RSI / MAX_RSI / MAX_FROM_HI20 / MIN_REL_STRENGTH /
  BEAR_REL_STRENGTH / BEAR_NKK_20D
- core/rank_stocks.py: TOP_SHOW / tools/backtest.py: NIKKEI_CODE / video/tts.py: CREDIT
  （クレジット文字列は video/youtube_client.py と Remotion の SceneView.tsx が直書きで持つ）

削除したデッドパス:
- core/screener.py: apply_screener_v1(rel_strength_min=...) の引数と、それを組み立てる
  is_bear 分岐。引数は関数内で一度も使われておらず、下落相場時に「相対強度閾値を5%に
  引き上げ」と表示するだけで実際には何も変わっていなかった（誤解を招くログ）。

不要import 20件超を除去（glob/time/timedelta/np/math/HEADERS/SEQ_DAYS/IsotonicCalibrated 等）。

意図的に残したもの:
- core/rf_train_v3.py の _select_features と未使用import IsotonicCalibrated
  → CLAUDE.md「金曜(再学習日)以外は触らない」に従い次の再学習日に回す
- web/x_weekly_trending.py の POST_MAX_WEIGHTED / URL_WEIGHTED_UNITS
  → 一度削除したが tests/test_x_weekly_trending.py がモジュール属性経由で参照しており失敗、復元

検証: pytest 321件 全passで削除前後とも同数（挙動不変のため bear BT は未実施）。
```

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

## 2026-08-16: 動画パイプラインに銘柄コード指定の手動実行を追加
オーナー依頼「太陽誘電とキオクシアのクジラの件を動画にして投稿」。通常の選定は
「直近36h新着×注目枠」のため、8/10-8/12開示（太陽誘電×Situational Awareness LP、
キオクシア×東芝売却）の記事は対象外になっていた。

- `video/build_script.py`: `fetch_recent_articles()` に stockCode フィルタを追加。
  `pick_targeted()`（注目枠を問わず金額最大1件）と `build(stock_code=...)` の
  銘柄指定モード（遡り幅 TARGETED_HOURS=14日）を追加
- `video/publish_video.py`: `--stock-code` 引数を追加
- `.github/workflows/video_post.yml`: workflow_dispatch に `stock_code` 入力を追加
- テスト: test_video_pipeline.py 43件（+2）、全テストpass

## 2026-08-17: ブログ重複投稿事故の対応（重複判定キーをdealAmountからfilerName+ratioChangePctへ）
edinet_blog.yml 17:48 JSTの便が、それ以前の便で投稿済みの記事17件をほぼ全て再投稿し、
トップページの記事が軒並み2件ずつ並ぶ事故。原因は `already_published()` の突き合わせキー
「銘柄コード＋開示日＋dealAmount±0.05億円」のうちdealAmountが株価からの都度概算のため、
直前(16:54-17:45)に走ったdaily_alert.ymlが株価キャッシュを当日終値で更新した瞬間に
全銘柄で推定金額がズレ（例: インフォマート19.6→18.2億円）、重複判定が全滅したこと。

- `already_published()`: 突き合わせを 銘柄コード＋開示日＋`filerName`＋`ratioChangePct` に変更
  （いずれも開示データから決まる決定的な値）。同一提出者が同日に複数報告書を出す実例
  （2936 2025-08-13 橋本舜2件）があるためratioChangePct一致まで確認。filerName未保存の
  旧記事(2026-08-16以前)のみ従来のdealAmountフォールバック
- `build_and_publish()`: is_sell判定を重複チェック前に移動し、microCMS保存値と同じ
  符号付き`signed_change`（売りは負）で突き合わせ
- 重複記事の削除: 同一タイトルの重複10組をAPIで削除試行→現行APIキーはDELETE不可
  （`DELETE is forbidden`、マネジメントAPIも403）のためmicroCMS管理画面での手動削除が必要。
  タイトルの比率が異なる同日ペア（ベースフード/fantasista）は実在する別報告のため残す
- `tools/cleanup_duplicate_blog_articles.py` 新規: すり抜けた重複を自動回収する。
  同一銘柄・開示日・提出者・比率変化幅（＝`already_published()`と同じキー）の記事が
  複数あれば先発1件を残して後発を削除。edinet_blog.ymlの投稿ステップ後に毎時 --delete で実行
  （2026-08-18にAPIキーへDELETE権限を付与済み。全909記事を走査して残存重複0件を確認）
- テスト: test_publish_blog_articles.py 73件（+5）、test_cleanup_duplicate_blog_articles.py 4件（新規）、全pass

## 2026-08-17 kujira-watchページ数増加の予算計算（docs/page_count_budget.md）

Supabase実測（edinet_large_holdings 19,830行）から増加ペースを算出: 記事約300件/月×2(ja/en)+
銘柄+100+投資家+170+日付+21 ≈ 約900ページ/月。現在クロール可能約7,300ページ→12ヶ月後約1.8万。
金銭コストはほぼ増えない（Claude API記事生成 約$5/月、microCMS無料枠上限は約32ヶ月後）。
本当の予算問題はGoogleクロールバジェット: 価値ページ約3,000に対しクロール可能7,300で、
割当の約4割が薄いページ（開示1件のみの投資家992件=34%等）に流れる構造。
対策案P1=sitemapから開示1件投資家を除外（実装は別途判断）。分析のみ、コード変更なし。

## 2026-08-19 動画v7改修（インフルエンサー100人コンサルの反映）

オーナー指示「動画を改善したい。インフルエンサー100人にコンサルをお願いして」。
10ジャンル×10人のパネルに、実データ回（アインHD×Oasis、88.9秒）のレンダー済みフレーム9枚と
実装コードを見せてレビューさせた。**スワイプ予測の中央値は13秒**（総尺88.9秒）で、
100人中70人が hook シーン（0〜12.7秒）の中で離脱。独立した全パネルが示し合わせなしに
(1) hookが12.7秒で1枚の静止画 (2) 背景が内容と無関係 (3) 字幕カードのナレーション全文表示
の3点を critical に挙げた。詳細は `docs/progress_video_v7_influencer.md`。

反映した主な変更:
- **hook 12.7秒 → 約5.4秒の4ビート構成**（金額スラム→社名→動詞→提出者ラベル→保有比率）。
  一番引きの強い「誰が買ったか」を下段の小さい字幕から中央へ引き上げた。
- **総尺 88.9秒 → 40.4秒**。outlookシーン廃止 / narration上限 90→55字 / hook 22〜30字 /
  シーン間パディング 0.45→0.18秒 / TTS 1.15→1.22倍速。ファイルサイズも55MB→9.9MB。
- **実写背景は company / filer の2シーンのみ**に限定。数字を読ませるシーンはブランドの
  グラデーション背景に固定した。Pexelsが自然系クエリでも返してくる人物クリップは
  URLスラッグの語判定で除外する。
- **字幕はcaption 1本（68px）に統一**。ナレーション全文の同時表示（34px）は廃止。
- **可読性を影から不透明の下地（PLATE_BG）へ**。safeAreaの左右非対称（70/190）を対称（160）に直し、
  中央寄せが60px左にずれていたのを解消。
- **changeシーンを前回→今回の2本バー＋差分に**。前回比率が本文から確定できない回は
  シーンごと落とす（比較対象の無い1本バーを「推移」と称して出さない）。
- **chartに不透明パネル・高値安値・期間・開示日の縦線**を追加。
- **効果音をnumpyで自前生成**（whoosh / impact / tick）。BGMはライセンス確認が人手前提のため未着手。
- **文の途中で切れた台本は破棄して投稿しない**。切れた「…」が画面にも読み上げにも出ていた。
- **ProgressBarを削除**（尺が40秒になり残量表示の価値が消え、序盤で長さを悟らせる害だけが残るため）。
- サイト名とEDINET出典（書類名・提出日）・免責を全編常時表示。CTAは「クジラウォッチで検索」に。
- 締めの末尾0.8秒を冒頭と同じ金額組版に戻し、ループ再生で頭と繋がるようにした。

不採用（rejected）: ずんだもんの語尾キャラ化・2話者掛け合い・TTS 1.32倍速・
カラオケ字幕（ワード単位同期）・投資判断を誘導する演出（「この買い、追う？」等）・
Pexels素材の手動ホワイトリスト。理由は `docs/progress_video_v7_influencer.md` に記載。

検証: 実データ（アインHD 9627）で通しレンダリング。40.4秒 / 9.9MB / -14.0 LUFS。
効果音の有無でA/B比較し、無音区間が -91 dB → -31.8 dB になることを確認。
テストは video 57→75件、リポジトリ全体 339件 pass。

### 2026-08-19 追補（オーナー指摘2件）
- **鯨アイコン（🐋）を動画から全削除**。上部の銘柄行と締めの画面の2箇所にあった。
- **「クジラウォッチで検索」を廃止**。サイトは `kujira-watch/src/lib/site.ts` の `SITE_NAME` =
  **「大口投資家の監視ブログ」**で運営しており、「クジラウォッチ」という名前では出していない。
  さらにその語で検索上位を取れていないため、検索誘導は言っても辿り着けない。
  締めは「大口投資家の監視ブログ」＋URLピル（kujira-watch.com）に変更し、
  build_script.py の closing プロンプトからも検索を促す指示を外した。

### 2026-08-19 追補2: YouTube説明文・TikTokキャプションの導線改善
動画本体を直したあと、実際に流入を生むのは投稿テキストなのでそちらも見直した。見つかった問題:
1. **記事URLが説明文の8行目**にあった。Shortsの説明文は冒頭1〜2行しか畳まずに見えないため、
   折りたたみを開かない限り導線に到達できない → **先頭3行以内**へ移動。
2. **サイト名が投稿文に一度も出ていなかった** → `SITE_NAME`（大口投資家の監視ブログ）を明示。
3. **説明文の先頭3ハッシュタグ**はタイトル上部に表示される枠なのに `#Shorts #EDINET` を置いていた
   → 検索される語（#日本株 #大量保有報告書 #銘柄名）を先頭へ。
4. **TikTokキャプションにサイトへの導線が1行も無かった** → リンクが押せない仕様なので
   `kujira-watch.com` を文字列で1行置く。
5. **銘柄名に空白や「．」が入るとハッシュタグが壊れる**（例: Ｊ．フロント リテイリング →
   `#Ｊ．フロント` でタグが切れ、残りが本文に漏れる） → `video/post_text.hashtag()` で整形。
6. YouTubeタイトルに保有比率を追加（一覧で「何%になったのか」が分かる）。

サイトの名乗り・URL・UTM・ハッシュタグ整形は `video/post_text.py` に集約した。
名乗りを各クライアントに直書きすると、動画側で「クジラウォッチ」と名乗った事故と同じことが
投稿文でも起きるため。テストは video 75→84件、全体354件pass。

### 2026-08-19 追補3: BGMをnumpyで自前生成（CC0音源の選定は不採用）
「BGMが無いと未完成品に見える」という指摘への対応。CC0音源を1本選ぶ案と自前生成の案を出し、
オーナー判断で**自前生成**を採用（外部素材を持たないのでライセンス確認が不要になり、
毎日の全自動運用と両立する）。

- `video/se.py` → `video/audio_gen.py` にリネーム。効果音に加えBGMも作るため。
- BGMはAm→F→C→Gの12秒アンビエントパッド。各和音を枠（3秒）からはみ出す長さで鳴らし、
  はみ出したぶんを配列の先頭へ回り込ませて足すことで、ループの継ぎ目に音の切れ目を作らない。
- ローパスは内部状態が0から始まるため、そのまま掛けると先頭だけ音が痩せてループのたびに
  プチッと鳴る。2周ぶん通して後半だけを採り、フィルタが定常状態の区間を使う。
  （実測: 継ぎ目の段差 0.130 → 0.0127。通常の隣接サンプル差分99%点は 0.0133）
- ArticleShort に `volume` 関数付きで敷き、頭20フレームをフェードイン・末尾14フレームを
  フェードアウト。`loopVolumeCurveBehavior="extend"` でループしても音量カーブが
  動画全体の時間軸で効くようにする。

検証: BGM有無で同じ動画をレンダリングして比較。**ナレーション中はほぼ変化なし**
（-19.8→-19.9 dB / -17.1→-16.9 dB）で、**無音の間だけ持ち上がる**（-33.1→-28.5 dB /
-30.0→-27.4 dB）。BGMは声より約13dB下で、声を邪魔せず無音だけを埋めている。
テストは video 84→86件。

### 2026-08-19 追補4: Xのフォロワー獲得を測れるようにした
「Xのフォロワーを増やしたい」（オーナー）。着手時点で、フォロワー数はどこにも記録されておらず、
投稿メトリクスもインプレッション・いいね・保存までしか取っていなかった。この状態では
どの施策がフォロワーを連れてきたか判定できない（本日入れた10施策の効果検証も含めて）。

- `x_followers`（1日1行）を新設し、`GET /2/users/me` の public_metrics を日次記録。
- `x_posts` / `x_post_metrics` に `url_link_clicks` / `user_profile_clicks` を追加。
  フォロワーは「インプレッション→プロフィールクリック→フォロー」の順にしか増えないので、
  投稿の型を判断する中間指標はプロフィールクリック率に置く（いいね数は使わない）。
- `--report` にフォロワー推移（7日前比・30日前比）とプロフィールクリック率を追加。
  記録が飛んだ日があっても、その日以前で最も新しい記録と比べる。

テストは `tests/test_x_metrics.py` 7件を新設。X投稿系は 50→57件。

### 2026-08-19 障害と修正: 定時便が投稿0件（原因は同日の台本バリデーション追加）
19:54 JSTの定時便（run 32244995158）が **投稿0件で正常終了**した。対象は
日本製鉄（5401）×JPモルガン・セキュリティーズ。ログ:
```
↻ 台本が長すぎるため作り直します（最長86字）
↻ 台本が長すぎるため作り直します（最長93字）
⚠ 読み上げ文が文の途中で切れたままのため動画を作りません
```
同日に入れた「文が途中で切れた台本は破棄する」バリデーションが正しく発火した形だが、
そもそも台本が上限55字に対して86〜93字で返っていた。原因は**プロンプトの矛盾**:
尺を縮めて上限を90→55字にしたのに、前段の枠組み説明に
「この動画は音声ナレーションで**記事の内容をほぼ読み上げ**」が残っており、
Haikuが数値の上限より枠組みの指示に従っていた。

修正:
- 枠組みの説明を「40秒前後のショート動画。記事を読み上げるのではなく要点だけを短く言い切る」に変更
- 「1文は40字以内。入り切らない場合は2文に分けるのではなく情報を削る」を明記
- **作り直しのプロンプトに「何字の文が長すぎたか」を実際の文とともに入れる**
  （同じ指示をそのまま投げ直しても同じ長さが返るため。これが2回連続で外した直接原因）
- 作り直しの上限を2回→3回に
- 「買収」「経営権を握る」「TOB」など開示内容を超える語を禁止
  （0.38%の新規保有を「今回の買収で」と読み上げる出力が実際に出たため）

実データ（日本製鉄の同じ記事）で再実行し、台本生成が通ることを確認（概算34秒）。
テストは video 86→88件、リポジトリ全体384件pass。

### 2026-08-19 追補5: X記事投稿の添付画像を1枚に戻した
「画像2枚は見づらい」（オーナー、実際の投稿のスクリーンショット）。Xは複数画像を左右に
並べて**両方とも左右を切り落とす**ため、数字カードは銘柄名の途中（「ン&アイ・ホールディ」）、
チャートは縦の1本だけという、どちらも読めない状態になっていた。

- `build_article_media`は数字カード1枚だけを返す。チャートはリンク先の記事に任せ、
  カードを作れなかった場合（フォント欠如等）だけ代替として1枚添える。
- あわせてカードの銘柄名が「セブン&アイ・ホールディングス（3…」と証券コードごと
  切れていたのを修正。フォントを60→54→48→42と下げて`社名（コード）`を丸ごと収め、
  それでも入らない場合だけ社名側を削る（`_stock_line`）。コードは銘柄検索の手掛かり。

テストは x_client 50→54件。

### 2026-08-21 定時便が2日連続で投稿0件だった件の根治
8/19（日本製鉄×JPモルガン）に続き 8/20（博報堂ＤＹ×シルチェスター）も投稿0件だった。
8/20のログ:
```
↻ 台本が長すぎるため作り直します（最長72字）
↻ 台本が長すぎるため作り直します（最長29字）   ← これは caption（上限26字）の超過
↻ 台本が長すぎるため作り直します（最長75字）
⚠ 読み上げ文が文の途中で切れたままのため動画を作りません
```
原因は2つ:
1. **作り直しのフィードバックが caption と narration を言い分けていなかった**。上限は
   26字と55字で別なのに、29字の caption 超過を「narrationは1文40字以内…」と伝えており、
   モデルに見当違いの直しをさせていた（3回とも外した直接原因）。
2. **1シーンの読み上げ文が直らないだけで動画を丸ごと捨てていた**。壊れた文を出さない
   という判断は正しいが、その代償として投稿が飛ぶのは割に合わない。

修正:
- フィードバックで caption / narration のどちらが何字超えたかを実際の文とともに個別に伝える
- `salvage_scenes()` を追加。作り直しでも直らないシーンは、hook/deal/change/cta なら
  記事の事実だけの定型文に差し替え、言い換えが必要な company/filer は落とす。
  動画を諦めるのは hook/cta を組み直せない場合だけ。
- 新規保有の開示で「買い増し」と書くことを禁止（日本製鉄の回で実際に出た）

検証: 落ちた2記事で再実行し、どちらも台本生成が通ることを確認
（博報堂ＤＹ=作り直し2回で成功・概算38.9秒、日本製鉄=一発・概算34.7秒）。
テストは video 82件、全体382件pass。

### 2026-08-21 音量正規化がCIで一度も効いていなかった
上記の修正後、投稿しないモード（`render_only`）でCIを走らせて成果物を実測したところ
**-24.49 LUFS** だった（ローカルでは -14.0 になる）。2026-08-18に入れた
`normalize_loudness()` が**本番では一度も動いていなかった**。

原因: ランナーが Ubuntu 24.04 で、このイメージには **ffmpeg が入っていない**。
`_has_audio_stream()` が `OSError` を握り潰して `False` を返していたため、
ログに1行も出ないまま音量正規化が丸ごとスキップされていた。
つまり「ずんだもんの声が聞こえない」という元の指摘は、8/18以降も本番では
解決していなかった（YouTubeに上がっている動画はすべて約11dB小さい）。

- `.github/workflows/video_post.yml` の apt に `ffmpeg` を追加
- `_has_audio_stream()` / `_measure_loudness()` で `FileNotFoundError` を個別に捕まえ、
  「音量正規化をスキップします（配信基準より約11dB小さいまま投稿されます）」と必ずログに出す

## 2026-08-20 修整レビュー10ラウンドのエージェント追加（.claude/agents + revision-review スキル）

「修整指示に対して10回ぐらいレビューしてほしい」への対応。同じレビューを10回回しても
指摘が重複するだけなので、**1体1観点**に分割して10ラウンドにした。

- `.claude/agents/revision-reviewer.md`: 読み取り専用（Edit/Write なし）のレビュアー。
  渡された観点1つだけを見て、`file:line` と根拠つきの指摘を最大5件返す。
  ファイル全文の出力・観点外の指摘・推測での指摘を禁止（Token Saving / No Hallucination）。
- `.claude/skills/revision-review/SKILL.md`: 10観点（指示充足 / バグ / 64次元特徴量の整合性 /
  戦略規律・ハードフィルター / PIT規律・リーク / Supabase往復とAPIコスト / CI・運用 /
  テストのデグレ / コード規律§7・README更新 / 総合再レビュー）を固定し、
  1〜9は3体ずつ並列 → 重複排除 → B/M を修正 → 10で潰れたかを再検証する。
  ラウンド10で Blocker が出たら修正して10をやり直す（最大3回、収束しなければ残件を報告）。
- 締めに `tests/test_*.py` 全実行、モデル・ランキングに触れた場合は bear バックテストで数値確認。

### 2026-08-21 YouTubeタイトルが「大口投資家が」になる件（filerName欠落）
チャンネルの4本中2本のタイトルが提出者名ではなく総称の「大口投資家が」になっていた。
調べたところ記事側のデータ欠落で、microCMSの記事883件中13件（2026-08-07〜08-12の旧記事）に
`filerName` が無い。2026-08-16以降の記事は全件入っているので、新規に発生する問題ではない。

素朴に開示データ（edinet_large_holdings）から埋めようとしたが、**同一銘柄・同一開示日には
複数の提出者がいるのが普通**で、13件すべてが候補2〜3件に割れて一意に決まらなかった
（誤った提出者名を記事や動画タイトルに載せるのは事実の毀損なので採用できない）。

一方、記事本文には提出者名が文章として書かれている（例:「個人投資家の久世良太氏が」）。
そこで **開示データの候補 ∩ 本文に名前が出てくるもの** を条件にしたところ、13件中10件が
一意に決まった（残る3件は該当する開示データ自体が無い）。

- `build_script.resolve_filer_name()` を追加し、`build_props` から使う。
  表記ゆれ（全角空白・中黒・「株式会社」の有無）は `_normalize_name()` で吸収する。
- 一意に決まらない場合は空文字を返し、従来どおり「大口投資家」へフォールバックする。

実データで確認: サンクゼール→久世　良太 / データセクション→アースエレメンツ・キャピタル株式会社 /
日本精蝋→特定できず（総称）。テストは video 88件、全体388件pass。

### 2026-08-21 filerNameバックフィルをタイトル一致から本文一致へ拡張
上記の調査で、未設定13件はすべて「同日・同銘柄に提出者が2〜3人いて一意に決まらない」
ケースだと分かった。既存の `tools/backfill_article_filer_name.py` は**記事タイトル**
だけで突合していたため、これらを全部スキップしていた。

タイトルが提出者名を出さない書き方の回がある（例:「個人投資家が3.5億円規模を売却、
パンチ工業の保有比率が2.44%に」）一方、本文には「個人投資家の森久保哲司氏が」と
書かれている。突合をタイトル→本文の順に広げたところ、**13件中10件が一意に特定**できた
（残る3件は該当する開示データ自体が無い）。

「ちょうど1件だけ名前が出てくる場合のみ確定する」という性質は変えていない。
本文に複数の候補が出てくる回はスキップする（誤った提出者を書き込むより総称のままが良い）。
dry-runで10件の対応表を確認済み。書き込み（--apply）はオーナー判断待ち。

※ このとき既存のテストファイルを確認せず上書きしてしまい、元の6件を復元して統合した。
   pick_filer のシグネチャを (title, candidates) から (article, candidates) に変えたため、
   既存6件も記事dictを渡す形に追従させている。テストは4件追加して10件。

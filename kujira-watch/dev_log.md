## 2026-08-27 kujira-watch: 銘柄ランキングの並び順を増加件数順→推定売買金額順へ

オーナー指示「銘柄ランキングは金額順になってる？」→「金額順を既定にする」。
`/trending` は増加件数(delta)→直近件数→金額の順で並べており、金額は同着の並べ替えにしか
効いていなかった。8/27に比較窓を30日→7日へ変えて以降、増加179銘柄のうち151銘柄が「+1件」・
うち146銘柄は直近件数も1件（2026-08-27実測）で、上位が実質ランダムになっていた。

- `trendingStats.ts` の `compareTrending()` を `amount → delta → count` の順へ変更。
  金額が同じ銘柄（多くは金額を推定できない＝0億円）だけ従来どおり増加件数で並ぶ。
  絞り込み（delta>0の銘柄のみ）は変更なし。
- `TrendingTable` のカードは「直近7日 推定◯億円 N件」の順（金額が並べ替え軸なので先頭）。
  合計0億円の銘柄は金額欄を空にせず「金額不明」と出す（下位に並ぶ理由が読めるように）。
  前期間の金額が無い場合は「―」。
- ページ説明文・InfoTip・ItemList構造化データ・下部の注記・`/faq/usage`のQ&Aを金額順に更新。
- 金額の取得に失敗した場合は全件0億円＝増加件数順にフォールバックしてページは成立させる
  （`getHoldingAmountsInRange()` の `.catch(() => ({}))` は維持）。
- `tsc --noEmit`・`eslint` パス。ローカルdevで実描画を確認（買い72件・両方179件、
  トップは推定337億円のレゾナック、金額不明63件が最下位群に集まる）。

## 2026-08-23 kujira-watch: 銘柄ランキングの「月別の開示件数トレンド」を削除

オーナー指示「月別の開示件数トレンドが銘柄ランキングにあるけど、週次トレンドと被るから消して」。
`/weekly`（大口投資家の週次トレンド）が市場全体の増減を週単位で見せており、
`/trending`下部の月別グラフは同じ「市場全体で開示が増えているか」を粗い粒度で
繰り返しているだけだった。銘柄ランキングは銘柄を選ぶページなので、市場全体の話は`/weekly`に寄せる。

- `/trending`から月別グラフのセクションを削除（`src/app/(ja)/trending/page.tsx`）
- 使い手が居なくなった`src/components/DisclosureTrendChart.tsx`と
  `src/lib/disclosures.ts`の`getMonthlyDisclosureCounts()`・`MonthlyDisclosureCount`型を削除。
  コメントアウトで残さない（CLAUDE.md §7）。`/weekly`の`AmountTrendChart`・`CategoryTrendGrid`は別実装なので影響なし
- 副次的に、ページ表示ごとに走っていた月数ぶん（十数本）の並列countクエリが無くなる

### 検証
- `npx tsc --noEmit` / `npx eslint src` エラー無し
- 残存参照の全文検索（DisclosureTrendChart / getMonthlyDisclosureCounts / MonthlyDisclosureCount）は
  過去ログの記述を除いてゼロ

## 2026-08-23 kujira-watch: 投資家ページのURLを提出者名から連番IDへ（/investors/<番号>）

Search Consoleカバレッジ（2026-08-23）で、`/investors/<提出者名>` 603件がインデックス未登録だった。
日本語・全角英字・全角空白・改行までURLに含む長いパスで、URLエンコード後は数百バイトになる
（事前生成時のENAMETOOLONGの原因でもあった）。オーナー指示で番号化。

- Supabaseに`edinet_filer_ids`（id serial, filer_name unique）を新設し2,967件を採番。
  `edinet_large_holdings`のINSERT/UPDATEトリガーで新規提出者は自動採番。
  `edinet_filer_summary`ビューに`filer_id`列を追加（`supabase/create_edinet_filer_ids.sql`）
- `/investors/[filer]`は番号なら`getFilerNameById()`で名前を引き、旧形式（名前）なら
  `getFilerIdByName()`→`permanentRedirect`で番号URLへ308転送。該当なしは404
- リンク生成は全箇所`investorPath(filerId, filerName)`に統一（`src/lib/investorPath.ts`。
  クライアント側の検索ボックス・リターンランキングからも使うためSupabase依存の`investors.ts`と分離）。
  名前しか持たない行（推移表・アクティビスト・月次・RSS）は`getFilerIdMap()`で一括解決
- `/api/stocks/search`の投資家結果に`filerId`を追加、`InvestorReturnRow`/`StockFiler`/`FilerSummary`にも`filerId`
- 記事本文の投資家名リンク化（`linkifyFilerNames`）は`{filerName, filerId}[]`を受けるよう変更

## 2026-08-22 kujira-watch: 銘柄ランキングに開示件数と推定売買金額を並べて表示

オーナー指示「銘柄ランキングには、件数と金額を乗せて」への対応。
`/trending`は開示件数だけを出しており、「何件動いたか」は分かっても「どれくらいの規模か」が
分からなかった（FAQにも「金額規模は各銘柄のページで確認してください」と逃がしていた）。

- 各カードの数値行を「直近30日 12件 推定245億円 / 前30日 3件 推定80億円 / 増加 +9件」に変更
  （`src/components/TrendingTable.tsx`）。並び順は従来どおり**増加件数**で、金額は規模の目安
- 金額は1兆円以上で兆表記に繰り上げる（`formatAmountParts()`を流用）。
  合計0億円の銘柄は「推定0億円」ではなく金額そのものを出さない（不明とゼロを混ぜない）

### 金額の作り方（EDINET開示に取引金額は無い）
Supabaseにマテリアライズドビュー`edinet_holding_amounts`を新設し、開示1件ごとに
`保有比率の変化幅(%) ÷ 100 × 発行済株式数 × 開示日の終値`で概算する
（`supabase/create_edinet_holding_amounts.sql`）。式は記事側の
`web/publish_blog_articles.py: estimate_deal_amount_oku()`と同じで、記事の「推定取得金額」と
数字が食い違わないようにしてある（違いは株式数の出どころだけ。記事はyfinanceの最新値、
ビューは`jquants_fin_summary.sh_out`の開示日時点＝PIT値）。

- 変化幅は開示自身の`holding_ratio_prior`が最優先。取れない開示だけ同一投資家×同一銘柄の
  直前開示から再導出し、それも無ければ「大量保有報告書」（新規提出）のみ前回0%扱い。
  前回比率が取れない変更報告書は規模不明なので行を作らない
  （直近60日の実測: 前回比率あり1,985件・新規提出333件・直前開示から再導出3件・不明0件）
- 訂正報告書（売買を伴わない）と、株価・発行済株式数が取れない開示も行を作らない。
  件数には入るが金額には入らないので、件数と金額は比例しない。この注記をランキング直下と
  `/faq/usage`のQ&Aに明記した
- 素のビューだと直近60日の読み取りだけで2.8秒かかったためマテビュー化（全13,011行の
  再作成で約9秒）。PostgRESTの接続ロールに8秒の上限があるので、RPC側で
  `SET statement_timeout = '300s'`を明示している
- 再計算は`tools/refresh_holding_amounts.py`。開示が増える毎時便（edinet_blog.yml）と、
  株価・財務サマリー更新後（daily_alert.yml Step 2e）の両方から叩く

### フロント側
`getHoldingAmountsInRange()`（`src/lib/investors.ts`）が`doc_id`→金額のオブジェクトを返し、
`buildTrendingIssuers()`が件数と同じバケットに足し込む（`src/lib/trendingStats.ts`）。
突き合わせキーが要るので`HoldingRow`に`doc_id`を追加し、データキャッシュのキーをv2→v3に上げた。
金額の読み取りに失敗してもランキングは件数だけで成立させる（`.catch(() => ({}))`）。

### 検証
- `npx tsc --noEmit` / `npx eslint` ともにエラー無し
- 集計ロジックを単体で実行し、買い2件=151億円・前期間1件=7億円・
  金額不明の開示は件数のみ加算（both 4件/171億円）を確認
- SQL側は直近30日で792件・合計29,365億円、上位は東芝→キオクシア2,718億円、
  Ursa4→ジェイ・エス・ビー1,763億円（TOB）など実額と整合する規模

## 2026-08-22 kujira-watch: 投資家ランキングに投資家の種類での絞り込みを追加

オーナー指示「投資家ランキングに投資家の種類別のフィルターを入れて」への対応。

- `/ranking/returns` の見出し直下に分類フィルターを`<details>`で畳んで配置
  （`src/components/InvestorReturnRanking.tsx`）。該当0名の分類はボタンごと出さず、
  各ボタンに`（N名）`を添える。実データでは12分類が該当（合計198名）
- 絞り込むと**その分類の中での順位を1位から振り直す**。全体順位のまま歯抜けの数字
  （8位・17位・30位…）を出しても「アクティビストの中で何番目か」が読み取れないため
- 絞り込み中は全行が同じ分類になるので、行ごとの分類ラベルは省く
- 脚注も「『アクティビスト』に分類される32名のうち上位30名を表示しています（2名は表示していません）」
  のように絞り込みに追従させる。上位30件で打ち切っていることを黙って隠さない

### searchParamsを使わない理由
`?category=`を読むとページがリクエストごとの動的レンダリングになりISRキャッシュが効かなくなる
（`/investors`が実際にそうなっている）。ランキングは198名と小さいので全件を最初から
クライアントに渡してその場で絞り込む方式にした（2026-08-20の`/trending`の売買方向フィルターと同じ判断）。
切り替えで追加のfetchは発生しない。HTMLは169KB→214KB（gzip前）に増えたが、
perf_checkの閾値はgzip後100KBで、gzip後は約22KB→30KB弱に収まる。
SSRされるHTMLとItemList構造化データは初期表示＝絞り込みなしの上位30名に揃えた。

### ついでに削除
`getInvestorReturnsSummary()`を削除。全件を取るようになって母数がrows.lengthで分かるため、
件数を数えるだけのクエリが不要になった。

### 検証
- devサーバで絞り込みを実操作。アクティビスト（32名）→上位30名・順位1から振り直し・
  「2名は表示していません」、VC（4名）→4名・打ち切り注記なし、すべて→198名を確認
- ボタンの合計が198名と一致（4+32+4+6+10+14+26+16+21+37+28）
- 375pxでページ全体の横スクロールなし・ボタン列はStack内でスクロール、1280pxで2列カードを実描画確認
- `tsc --noEmit`・`eslint src` パス

## 2026-08-21 kujira-watch: 投資家ページに3ヶ月リターンの成績と開示ごとの内訳を追加

前日に作った/ranking/returnsの数字が、ランキングから飛んだ先の投資家ページでは見えなかった
（オーナー指示「やる」）。

### 追加したもの
- `/investors/[filer]` のプロフィール直後に「買い開示の3ヶ月後リターン」パネル
  （`src/components/FilerReturnRecord.tsx`）。平均・中央値・勝率・日経平均比と
  「198名中◯位」＋ランキングへの導線。「この投資家についていって大丈夫か」に最初に答える
  数字なので保有銘柄より前に置いた。買い開示3件未満の投資家は非表示（n=1の「実績」を出さない）
- 「最近の取引」テーブルに「3ヶ月後」列。開示1件ごとの騰落率で、パネルの平均値の内訳になる。
  売却の開示・訂正報告書・まだ3ヶ月経っていない開示は「—」

### ビューを明細＋集計の2層に分けた
`investor_returns_3m`（集計）が`investor_return_positions_3m`（買い開示1件＝1行の明細）を
参照する構成に変更。ランキングの平均値と投資家ページの内訳を別々のクエリで計算すると、
片方だけ条件がズレても誰も気付けないため、必ず同じ行から作る。
リフレッシュは明細→集計の順（`refresh_investor_returns_3m()`）。

### 作り直しで見つかった不具合2件
どちらも「作った直後に気付いた」ものではなく、明細を出せるようにしたことで初めて見えた。

1. **同一開示の二重計上**: 買い開示を抽出するDISTINCTに`issuer_name`を含めていたため、
   同じ銘柄でも開示ごとに半角/全角がゆれると1件が2行になっていた
   （実例: 4189 KHネオケム/ＫＨネオケム、株式会社ストラテジックキャピタルの開示が51件と
   数えられていた。正しくは50件）。キー3列の`DISTINCT ON`に変更し、
   `(filer_name, issuer_code, disc_date)`のユニークインデックスで再発を検知できるようにした。
   これは2026-08-18に旧filer_win_rateを撤去した理由の1つ（重複行1,273組の二重計上）と同じ種類の
   バグで、ユニークインデックスを張った瞬間にビュー作成が失敗して発覚した
2. **訂正報告書の行に数字が付く**: 開示テーブルとの突合を(銘柄コード, 開示日)で行っていたため、
   同じ銘柄・同じ日に変更報告書と訂正報告書が並ぶと訂正報告書の行にも騰落率が表示されていた
   （実例: 2026-03-18のアーカス×SHIFT）。明細ビューに`doc_id`を持たせ、そちらで突合するよう変更。
   買い判定のルールをTS側に書き写さずに済む＝二重定義によるズレも起きない

### 検証
- 明細4,461件・投資家198名。明細から再集計した件数・平均が集計ビューと全件一致（不一致0件）
- アーカス・インベストメント・リミテッド（ランキング8位）のページで、パネルの+25.2%・12件が
  ランキングの表示と一致し、訂正報告書5行すべてが「—」になることを確認
- 買い開示2件のＡＮＲＩ株式会社でパネルが非表示になり、明細2件の数字だけが出ることを確認
- 1280px/375pxで実描画（デスクトップ4列・モバイル2列、ページの横スクロールなし、
  表はTableContainer内でスクロール）。gain/lossの色が正・負で切り替わることも確認
- `tsc --noEmit`・`eslint src` パス

### つまずき
`unstable_cache`はJSONで保存するため`Map`を返すとキャッシュ往復で`{}`に化け、
`returnPositions.get is not a function`で落ちた。素のオブジェクトを返すよう変更。
`curl`では200が返る（devのエラーオーバーレイも200）ため、ブラウザで実描画するまで気付けなかった。

## 2026-08-20 kujira-watch: 月間ランキングを投資家の3ヶ月リターンランキングに置き換え

オーナー指示「月間ランキングがあまり参考になってない。投資家ランキングにして、投資家の
3ヶ月でのリターンを出してランキングにしたい」への対応。

### 何を作ったか
`/ranking/returns`（3ヶ月リターンランキング）を新設し、タブの先頭に置いた。
EDINETの買い開示1件を1ポジションとして、開示日（休場なら直後の営業日）の終値から
63営業日後の終値までの騰落率を出し、投資家ごとに**等ウェイトで平均**した順に並べる。
各行に分類・買い開示件数・勝率・日経平均比（同期間の日経平均の騰落率との差％pt）・
最も上がった銘柄を添えた。開示3件以上の投資家198名が対象で、上位30名を表示する。

集計はSupabaseのマテリアライズドビュー`investor_returns_3m`で完結させた
（`supabase/create_investor_returns_3m.sql`）。アプリ側で組み立てると
「買い開示×株価」で数千回のクエリになるため。日次の再計算は
`tools/refresh_investor_returns.py`（daily_alert.yml Step 0b、株価キャッシュ更新の直後）。
実測でリフレッシュは約6.5秒。

### 2026-08-18に廃止した filer_win_rate（乗っかり実績）と同じ轍を踏まないための設計
当時の撤去理由を1つずつ潰した（SQLのヘッダコメントにも同じ内容を残してある）:

- `holding_ratio_prior`がNULLの開示を「新規取得」とみなさない。新規の大量保有報告書
  （doc_descriptionが「大量保有報告書」で始まる）だけprior=0として扱い、変更報告書で
  priorが欠けている88件は方向不明として捨てる。旧実装はここで売り切り（ＢＣＰＥの
  キオクシア22.53%→0）を巨額の買いに数えていた。現データで再確認したところ、
  ＢＣＰＥの開示は全てpriorが埋まっており全件が売り判定＝ランキングに載らない
- 指標は名目損益（金額×騰落）ではなく等ウェイトの騰落率。金額加重だと貸株玉の大きい
  プライムブローカーが上位を独占する
- `(filer_name, issuer_code, disc_date)`でDISTINCTして二重計上を防ぐ
- マテビューなので毎回作り直され、対象外になった行が残らない（旧実装はupsertのみで
  削除が無く、1,371行中320行が残骸だった）
- n>=3の投資家だけを載せる（旧実装は68%がn=1）
- 訂正報告書は除外（開示日が訂正日であって売買日ではないため）

### 廃止したもの
`/ranking/buys`（買い増し）・`/ranking/sells`（売却）を削除した。直近30日の推定金額を
投資家別に合計するだけで、「いちばん大きく張った投資家」が分かっても銘柄選びの
手掛かりにならず、常連の運用会社が並ぶだけだったため。インデックス済みURLなので
404にせず`/ranking/returns`へ301（旧`/ranking`のリダイレクト先も、連鎖を作らないよう
`/ranking/buys`から`/ranking/returns`へ張り替えた）。
`buildFilerRows()`とその型・`isSell`のexportも削除し、`rankingStats.ts`は
activist用の`buildStockRows()`だけになった。
h1・パンくず・ヘッダー/フッターのナビ表記は「月間ランキング」→「投資家ランキング」に戻した
（returnsは直近30日ではなく1年強の全買い開示が対象で「月間」ではないため）。

### 検証
- 実データで集計内容を確認（対象4,438ポジション・投資家198名、全体の平均+4.2%・勝率53.9%）。
  上位はディメンショナル+46.8%(3件)、ゴールドマン・サックス証券+23.8%(23件)、
  五味大輔+28.6%(6件)など
- `tools/refresh_investor_returns.py`をローカル実行しRPC経由のリフレッシュが通ることを確認
- devサーバで`/ranking/returns`・`/ranking/activist`・`/ranking/trending`が200、
  `/ranking`・`/ranking/buys`・`/ranking/sells`→`/ranking/returns`、
  `/ranking/filings`→`/ranking/trending`の301を確認。sitemapも3URLに更新済み
- 1280px/375pxで実描画。横スクロールなし・コンソールエラーなし
- `tsc --noEmit`・`eslint src` パス、`tests/test_*.py` 全件パス

## 2026-08-18 kujira-watch: PCのヘッダー検索を常時表示の検索窓に

オーナー指示「PCの検索ボタンは検索窓にしてほしい。スマホはそのままで良い」への対応。

- md以上では虫眼鏡アイコンをやめ、幅220pxの検索窓をヘッダーに常時表示する。
  モバイル（md未満）は横幅をロゴに使いたいので従来どおり🔍タップで開く。
- 平常時は素の`<input>`のまま。フォーカスした時点で`StockSearchPanel`
  （MUI Autocomplete）を`next/dynamic`で読み込んで差し替えるので、
  「重いコンポーネントは開くまで読み込まない」既存の方針を崩さずに常時表示にできる。
- 差し替え中のキー入力が落ちないよう、パネル側の`onReady`（マウント通知）が来るまで
  素のinputを残す。入力値は`StockSearch`のstateに集約し、Autocompleteには`inputValue`で
  渡す。マウント直後にMUIが`reason="reset"`で空文字を流してくるので、`onInputChange`は
  `input`/`clear`だけ拾う（この2点を入れる前は、クリック直後に打った先頭の数文字が
  消えてPlaywrightで「トヨタ」→「タ」になっていた）。

### 検証
- Playwright（1280px / 390px）でヘッダーを実描画して確認。PCは検索窓が常時表示され、
  クリック直後に「トヨタ」と打っても全文字がAutocompleteに引き継がれる（`/api/stocks/search`も発火）。
  モバイルは従来どおり🔍→ドロップダウン。
- `tsc --noEmit`・`eslint src` パス

## 2026-08-18 kujira-watch: 見出しを「月間ランキング」に改名し、報告書件数ランキングを廃止

オーナー指示2件（「タブの投資家ランキングは月間ランキングにして」「報告書の件数ランキングの
使い道がわかってない。あまり有益じゃないなら消してもいいかも」）への対応。

### 改名
`/ranking`・`/ranking/[slug]`・`/ranking/trending` 共通のh1・パンくず・ヘッダー/フッターの
ナビ表記を「投資家ランキング」→「月間ランキング」に統一。個別ランキング名は従来どおり
タブ直下のh2に置く。タブ配下は直近30日の集計が主なので「月間」を冠しているが、既定タブの
3ヶ月リターンだけは63営業日の評価である点はREADMEに注記した。

### 報告書件数ランキング(/ranking/filings)の廃止
残す価値が無いと判断して削除した。

- 中身が `/ranking/trending`（開示急増投資家）とほぼ重複する。どちらも直近30日の開示件数を
  投資家別に数えるもので、trendingはさらに前30日との差分まで出す上、集計元が記事(2026年7月〜)
  ではなくEDINET開示(1年分)で精度も高い
- 件数の絶対値は毎月ほぼ同じ常連の提出者（大量の変更報告書を出す運用会社）が並ぶだけで、
  月ごとの変化が出ず読み手の判断材料にならない
- タブ・sitemap・記事詳細の「関連ランキング」navからも導線を削除
- 2026-08-15公開でインデックス済みのURLなので404にはせず、`next.config.ts`の`redirects()`で
  `/ranking/trending`へ301。`buildFilerRows()`のslugは`buys`/`sells`のみになり、
  件数優先のタイブレーク分岐も畳んだ

### 検証
- 集計関数を素のnodeで再確認（buys/sellsの投資家別合計・代表開示・最終開示日・limit・
  activistが銘柄別のままであること）。全PASS
- `tsc --noEmit`・`eslint src` パス

## 2026-08-18 kujira-watch: 投資家ランキングのタブ3種が銘柄ランキングになっていた不具合を修正

オーナー指摘「投資家ランキングが、銘柄ランキングになってる。買い増しと売りましと報告件数」への対応。

### 症状
`/ranking/buys`・`/ranking/sells`・`/ranking/filings` は h1「投資家ランキング」＋
`RankingTabNav` のタブ配下にあるのに、中身は開示1件＝1行（buys/sells）・銘柄別集計（filings）で
並んでおり、順位が付いていたのは投資家ではなく銘柄だった。
同じタブ内で `/ranking`（3ヶ月リターン）・`/ranking/trending`（開示急増）は投資家別なので、
タブを切り替えるとランキングの軸が投資家↔銘柄で入れ替わる状態だった。

### 対応
集計を `src/lib/rankingStats.ts` に切り出し、3ランキングを投資家別に積み直した
（`/trending` の `trendingStats.ts` と同じ置き方）。

- `buys`: 買い開示を投資家ごとに合計し推定取得金額の降順（同額は件数降順）
- `sells`: 売り開示を投資家ごとに合計し推定売却金額の降順（同額は件数降順）
- `filings`: 投資家ごとの開示件数の降順（同数は合計金額降順）
- 提出者名(`filerName`)が無い過去記事は投資家別に積めないため集計対象外にした
- 各行のメタは「分類ラベル・N件の開示・代表銘柄（金額が最大の開示）・解説記事・最終開示日・
  合計金額」。投資家／銘柄／記事の3リンクは維持している（内部リンクを減らさない）
- ItemList構造化データも投資家ページを指すよう軸を合わせた
- `activist` はタブに含まれない「アクティビストが動いた銘柄」なので銘柄別のまま
  （`/activists` と対の関係。ページ内で軸が2種類あるためレンダリングを分岐させている）
- `filings`（直近30日の件数そのもの）と `/ranking/trending`（前30日比の増加件数）は
  指標が別なので併存させた

### 検証
- 集計関数を素のnodeに落として8ケースを確認（投資家別の合計・件数、同数時のタイブレーク、
  代表開示＝金額最大の開示、最終開示日、`filerName`なしの除外、`limit`打ち切り、
  activistが銘柄別のままであること）。いずれもPASS
- `tsc --noEmit`・`eslint` パス。`next build` はコンパイル・型検査を通過
  （ページデータ収集はmicroCMSキー未設定の環境のため実行不可）

## 2026-08-18 kujira-watch: /trendingをオートページャー化（HTML 1.08MB → 225KB）

オーナー指示「/trendingはオートページャーにして」への対応。
別セッションが急増銘柄の件数制限を外して全件表示にした結果、480件が一度に描画され
HTMLが1.08MB(gzip 106KB)まで膨らみ、perf_check.ymlの閾値100KBを超えていた。

### 実装
`TrendingTable`を`"use client"`にし、初回30件だけ描画して下端のsentinelが見えたら
30件ずつ増やす。TOPの`InfiniteArticleList`と同じIntersectionObserver方式（rootMargin 600px）。

`/api/articles`のような追加取得は作っていない。集計結果（480件）はサーバー側で
すでに全件手元にあり、重いのはデータではなくカードのマークアップだったため
（480件で1.08MB＝1件あたり約2.3KB）、描画する件数だけを絞れば足りる。

- `hrefOf`/`noteOf`の関数propsは廃止し、`items: TrendingItem[]`（href・noteを解決済み）
  で受け取る。クライアントコンポーネントの境界を関数は越えられないため。
- ItemList構造化データは初回SSR分の30件のみに揃える（追加分はクライアント描画で
  クロール時点のHTMLには無いため。TOPのItemListと同じ規律）。
- sentinelは同時に「もっと見る（残りN件）」ボタンにしてある。IntersectionObserverが
  働かない環境（バックグラウンドタブ等）でも先に進めるようにするため。

### 確認
| ページ | 変更前 | 変更後 |
|---|---|---|
| /trending | 1,079KB raw / 106KB gzip / カード480枚 | **225KB raw / 46KB gzip / SSRカード30枚** |
| /ranking/trending | 125KB raw / 16KB gzip | 106KB raw / 16KB gzip |

ページ送りの状態遷移は実ブラウザで確認（30件→60件、ラベル「480件中60件」・
残り件数の更新まで）。IntersectionObserverの発火自体は、検証に使うプレビューペインが
`document.visibilityState = "hidden"`のため確認できていない（この環境ではObserverの
コールバックが配送されない）。実装はTOPで本番稼働中の`InfiniteArticleList`と同じ。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。

## 2026-08-18 kujira-watch: デザインレビューP2（パンくず・チャート・分類バッジ）

デザインレビューの優先度2（一貫性・洗練）の3件を実装。P1は2026-08-17に対応済み。

### 修正内容
1. **記事パンくずが3行を占有**（375pxでフルタイトルが折り返し、本文到達前のノイズ）
   - ja/en の `articles/[id]/page.tsx`: flex＋`truncate`で1行ellipsisに。
     「トップ / 日付」側は`flex-none`で固定し、タイトルだけが縮む。
     SEO用のBreadcrumbList（JSON-LD）はフルタイトルのまま変更なし。
2. **株価チャートが線だけで水準感が読めない**
   - `CompanyInfoCard.tsx`: 期間高値・安値の点線ガイド（`--rule`）と終値のドット
     （`--color-brand-blue`）、右上に「期間高値◯円・安値◯円」のキャプションを追加。
     文字はSVG内に置くと`preserveAspectRatio="none"`の横伸縮で歪むためHTML側に出す。
     新規ライブラリ・アセットは追加していない（既存のインラインSVGのまま）。
3. **同じ「分類」が一覧と記事詳細で別デザイン**（一覧=色ドット、記事詳細=枠付きChip）
   - `CategoryBadge.tsx`: `DEAL_TYPE_COLORS[].dot` の分類色ドットを追加し
     `DealTypeBadge`と視覚的に揃えた。枠とリンクはカテゴリ一覧への導線として維持。

### 見送り
- ランキング冒頭の説明文の畳み込み: 別作業で既に2文＋FAQリンクに短縮済み。
- デスクトップの2カラム化: 効果は見込めるがRSC境界・ja/en両対応で大規模になるため、
  計測（回遊率）を整えてから判断する。

### 確認
- 375px/1280px × ja/en で目視確認。パンくずは41px(1行)、横スクロール無しを
  `scrollWidth`で実測。`tsc --noEmit`・eslintクリーン。

## 2026-08-18 kujira-watch: カード一覧の高さのばらつきを解消

オーナー指摘「カードUIの大きさのずれが気になる。統一して」への対応。
`/investors`で、同じ行の左右のカードの下端が揃っていなかった（実例: 「三光起業株式会社」
87px と「寺井 秀藏」67px が隣り合う）。

### 原因
2つ重なっていた。
1. **カードが行の高さいっぱいに伸びていなかった**。`<li><a class="card">`という構造で、
   グリッドアイテムである`li`は引き伸ばされるが、中の`a`は内容分の高さしか持たない。
2. **カード内の行数がカードごとに変わっていた**。分類ラベルと開示メタを同じ行に
   flex-wrapで流していたため、「創業家の資産管理会社」のような長い分類のときだけ
   2行に折り返り、カードが1行ぶん高くなっていた。

### 実装
- `globals.css`: `.card-grid > li:not(.card)`を`display: flex`、その直下の`.card`を
  `flex-grow: 1`に。`li`そのものが`.card`の一覧（/disclosures・/activists・/ranking/[slug]）は
  グリッドアイテムとして自動で揃うので`:not(.card)`で除外する。
- `/investors`: 「1行目=名前／2行目=分類／3行目=開示メタ」の3行固定に変更。

### 確認
1280pxで実測。`/investors`は行内の高さ不一致0件、`/monthly`は全15枚が95pxで完全に一致、
`/stocks`も行内不一致0件。行をまたいだ高さの差（`/stocks` 82px/106px、`/investors` 90px/114px）は
銘柄名・投資家名が2行に折り返すかどうかによるもので、名前を省略しない限り解消できない。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。

## 2026-08-18 kujira-watch: 文字サイズをTOPに統一し、一覧7ページをカードUI＋ページ送りに

オーナー指摘「なんかトップページだけ文字が大きい？」→「やや小さいぐらいで良い」
→「他のページもTOPにサイズ感を合わせたい」→「リスト形式の情報は全部カードUIにできる？」
→「多い場合はページネーション」への対応。

### 何が起きていたか（実測・幅1280px）
TOPだけが他ページに無い大きさを2つ持っていた。
- `TodayWhaleSummary`の件数・金額が **48px**（MUI `variant="h3"`の既定）。
  サイトの他の最大値はh1の30pxで、48pxはこの1箇所だけだった。
- `FeaturedArticleCard`のタイトルが **24/30px** で、ページのh1と同寸。
  見出し階層が潰れたカードが3枚縦に並ぶため「上半分ぜんぶ大きい」印象になっていた。

一方で通常の記事カード（見出し20px／本文14px／メタ12px）は他ページと完全に一致しており、
スクロールすると急に普通のサイズに戻る＝上だけ浮いた状態だった。

### 実装
1. **文字サイズ**
   - サマリー数値 48px → `{ xs: "1.5rem", sm: "1.75rem" }`(24/28px)、`lineHeight`も1→1.1。
   - 注目カード見出し 24/30px → `{ xs: "1.25rem", sm: "1.5rem" }`(20/24px)。
   - h2セクション見出しを ja/en 18ファイル・61箇所で `text-lg`(18px) → `text-xl`(20px)。
     TOPだけ`text-xl`、他ページは`text-lg`という2系統になっていたためTOP側に寄せた。
     `text-lg`の使用箇所は全て`text-lg font-bold text-brand-navy`のh2で、他用途への巻き込みは無い。
2. **カードUI**（`/disclosures` `/trending` `/ranking` `/ranking/[slug]` `/activists` `/investors` `/stocks` `/monthly`）
   - `src/app/globals.css`の`@layer components`に`.card`/`.card-grid`/`.card-grid-wide`を新設。
     **MUI CardやTailwindユーティリティの羅列は使わない**: `/faq`では可視HTMLの75%が
     コンテンツではなくマークアップ（`Mui〜`クラスの出現14,160回）だった前例がある。
     `@layer components`に入れるのは、レイヤー無しだとTailwindのユーティリティより優先されて
     `flex`等での上書きが効かなくなるため。
   - グリッドは`repeat(auto-fill, minmax(min(100%, 14rem), 1fr))`。`min()`により、
     カード最小幅より狭い端末では自動的に1列になるのでメディアクエリを持たない。
   - `TrendingTable`はMUI Tableをやめてカードに。`minWidth: 420px`で375px幅では
     横スクロールが必要だったため、数値にラベルを添えてカード内で折り返す形にした。
   - `MonthList`はMUI List＋`"use client"`をやめ、素のカードのサーバーコンポーネントに。
   - `/ranking`は「1行目=順位＋投資家名（全幅）／2行目=分類・件数・リターン（右端）」の2段。
     リターンを名前と同じ行に置くと、長い投資家名が細く4行に折り返されていた。
     表の見出し行（順位・投資家／トータルの推計リターン）はカード2列と対応しないため凡例1行に変更。
   - `/ranking/[slug]`（buys/sells/filings/activist）も同じ2段カードに。上位30件固定でページ送りは無い。
3. **ページ送り**
   - `/stocks`（596件を1ページに全件描画していた）に`?page=`を新設。100件/ページ。
   - `/ranking`（131件）に`?page=`を新設。順位はページをまたいで通し番号（`rankOffset`）にし、
     ItemList構造化データの`position`も合わせる。
   - `/investors`は既存のページ送りを200件→100件に。カード化で1件あたりの縦幅が増えたため。
   - 3ページとも`/investors`と同じ規律: 各ページが自分自身をcanonical、絞り込みリンクは
     ページ番号を持ち越さない、サイトマップに載せるのは1ページ目のみ。

### 確認
本番ビルド(`next start`)のHTMLをgzipして本番サイト(変更前)と比較。閾値は
`perf_check.yml`のHTML 100KB。

| ページ | 変更前 | 変更後 |
|---|---|---|
| /disclosures | 31KB | 31KB |
| /trending | 14KB | 13KB |
| /ranking | 47KB | 42KB |
| /activists | 36KB | 36KB |
| /investors | 36KB | 24KB |
| /stocks | 57KB | **21KB** |
| /monthly | 10KB | 9KB |

カード化してもHTMLは増えていない（MUIを使わなかったため）。`/stocks`はページ送りで大幅減。
`/investors`は200件→100件のページ送りで36KB→24KB。`/stocks`のページ高さは 22,110px → 4,894px。
375px/1280pxの両方で7ページを目視確認、横スクロール無しを`scrollWidth`で実測。
`tsc --noEmit`・eslintクリーン、`npm run build`成功。


## 2026-08-18 kujira-watch: 記事の無い銘柄が検索に出ない不具合を修正（6929 日本セラミック）

オーナー報告「6929の銘柄が検索で出てこない。日セラミックのはず」への対応。

### 原因
検索(`/api/stocks/search`)がmicroCMSの記事だけを引いていたため、解説記事が1本も
無い銘柄は検索から完全に消えていた。6929はSupabaseに会社情報(`jpx_stock_list`)も
EDINET大量保有4件(`edinet_large_holdings`)もあるのに記事が0件で、
`/stocks/6929`も記事0件を理由に404にしていた。
記事が無い理由は、日次投稿(`web/publish_blog_articles.py`)が `disc_date >= 今日-3日`
しか見ないのに対し、6929の開示4件は2026-08-10の1年分バックフィルで後から入った行
（`fetched_date`=2026-08-10、`disc_date`は2025-06〜2026-05）だったため。
過去分を埋める`tools/backfill_monthly_articles.py`は月あたり保有比率上位50件のみで、
6929の5.04%は入らなかった。
規模: EDINET開示のある2,982銘柄に対し、記事があるのは599銘柄だけ。
約2,380銘柄が検索・銘柄ページの両方から見えていなかった。

### 対応（記事ベース → 開示・上場銘柄ベースへ）
- `lib/companyInfo.ts`: `searchStockMaster()`（`jpx_stock_list`のコード前方一致・
  社名部分一致）と`getAllListedCodes()`（全上場コードのSet、1時間キャッシュ）を追加。
  `CompanyInfo`に`name`を追加（記事が無い銘柄の表示名の取得元）
- `/api/stocks/search`: 記事検索とマスター検索の両方を引いてコードで重複排除
  （記事のある銘柄を先頭に最大20件）
- `/stocks/[code]`: 記事0件でも会社情報＋開示履歴＋FAQでページを成立させ、
  マスターにも記事にも無いコードだけ404。記事0件のページは薄いので
  `robots: noindex, follow` を付与（sitemapは従来どおり記事ベースなので変更なし）
- `/disclosures`・`/trending`・`/activists`: 銘柄リンクの可否判定を
  「記事がある」から「上場銘柄マスターにある」へ変更（404リンクを作らない規律は維持）

### 確認
- `/api/stocks/search?q=6929` → `日本セラミック`、`?q=日本セラミック` → 6929 が返る
- `/stocks/6929` 200。会社情報（電気機器）・提出投資家3名・保有比率テーブル4行・
  FAQ・`noindex, follow` を実測
- 記事のある銘柄（9235）は `index, follow`・ItemList構造化データ・記事一覧とも従来どおり
- `/disclosures` `/trending` `/activists` `/` 200、存在しないコード（0000/abc）は404
- `tsc --noEmit`・eslintクリーン


## 2026-08-18 kujira-watch: ハンバーガーメニューを上部タブと同順＋見出し付きグループに再編

オーナー指示「ハンバーガーメニューの並びを上タブと同じにして。上タブにないものを
その下に入れて、見出しもつけて」への対応。

### 実装
- `src/lib/nav.ts` 新設: 上部タブとメニューが共用する主要ナビゲーションを一元管理。
  2箇所で別々に定義していたため、ページ改名時にメニューだけ取り残されていた
  （実例: /weekly改名後もメニューは「今週のまとめ」のまま）。今後は構造的に起きない。
- `HeaderMenu.tsx`: フラットな`siteLinks`を廃止し、見出し付き`MenuGroup[]`に。
  ja=「主要ページ（タブと同順9件）／サイト情報（about・FAQ・プライバシー・利用規約）／
  フォロー（X・YouTube・RSS）」、en=「Main pages（Top＋全分類）／Site info／Follow」。
  利用規約はどこからもリンクが無かったため追加（Footerのサイト情報と同構成）。
  enメニューにYouTubeを追加（jaと同じ導線に）。
- `HeaderMenuDrawer.tsx`: `menuGroups`を受け取り、グループごとに
  overline見出し＋Listで描画（言語セクションと同じ見た目）。

### 確認
- 375pxでja/en両ドロワーを開き、並び・見出し・全リンクを実測確認。
  `tsc --noEmit`・eslintクリーン。
- 検証は別セッションのdevサーバー(3002)を利用（Next 16は同一プロジェクトの
  多重起動不可のため）。

## 2026-08-17 kujira-watch: XカードのOGP画像が出ない不具合を修正（サムネなし）

公式X（@kujira_watch）の投稿でリンクカードのサムネイルが表示されない
（グレーのプレースホルダになる）との報告。調査の結果、全ページの`<head>`に
`og:image`/`twitter:image`メタタグ自体が出力されていなかった。

### 原因
`opengraph-image.tsx`を`src/app/`直下に置いていたが、このプロジェクトは
ページがすべて`(ja)`/`(en)`のルートグループ（それぞれルートレイアウトを持つ）
配下にあるため、appルート直下のOGP画像がどのページにも紐付かなかった
（Next.js 16.2.11で確認。`/opengraph-image`のルート自体は200で画像を返すが、
メタタグが注入されない）。`icon.tsx`（ファビコン）はルート直下でも全ページに
効くため気付きにくい。ローカルのdevサーバで`/privacy`等の`<head>`を確認して再現。

### 修正
- `src/app/opengraph-image.tsx` → `src/app/(ja)/opengraph-image.tsx` に移動
  （内容は従来どおり：🐋＋日本語サイト名＋説明文、1200x630）。
- `src/app/(en)/en/opengraph-image.tsx` を新規追加（英語版。`SITE_NAME_EN`/
  `SITE_DESCRIPTION_EN`を使用。従来は英語ページにOGP画像の仕組み自体がなかった）。

### 確認
- devサーバ実測: `/`・`/privacy`に`og:image`/`twitter:image`（ja画像）、
  `/en/about`にen画像のメタタグが出力され、両画像URLとも200/image/pngを返す。
- 記事ページの`generateMetadata`が設定するアイキャッチ（`openGraph.images`）は
  ルートグループのファイルベース画像より優先されることをテストページで実測確認
  （アイキャッチ付き記事のカードは従来どおりアイキャッチが出る）。
- `tsc --noEmit`・eslint（変更ファイル）クリーン。`npm run build`はコンパイル・
  型チェックまで通過（その先のページデータ収集はmicroCMS実キーが無い環境のため
  403で失敗。本修正とは無関係）。
- デプロイ後はX Card Validator等でカード再取得（Xはカードをキャッシュするため
  反映に時間がかかる場合がある）。

## 2026-08-17 kujira-watch: デザインレビューP1修正（重なり・コントラスト・縦割れ）

サイト全体のデザインレビュー（375px/1280px × ja/en を実機確認）で見つけた
実害ありの3件を修正。

### 修正内容
1. **ヘッダーの訪問者数がハンバーガーメニューと重なる**（sm+幅で実測30px重複）
   - `HeaderMenu.tsx`: md+の`position: absolute`配置を撤去しフロー配置に。
     「画面右端に固定」が本来の意図だったが、MUI化でAppBarに付いた
     `backdrop-filter`が包含ブロックを作るため実際はカラム右端に落ちており、
     訪問者数の上に重なるだけで意図は機能していなかった。
2. **クジラ注目度バッジのコントラスト不足**
   - `AttentionScoreBadge.tsx`: gold文字×金12%ティントは明色背景で実効2.7:1
     （WCAG AAの4.5:1未達）。明色地は文字を紺（13:1）・★のみgoldに変更。
     `onDark` propを追加（DealTypeBadgeと同じ方針）し、ダーク地はgoldBright
     （6.1:1）に。`FeaturedArticleCard`から`onDark`を渡す。
3. **記事ページのファクト欄で銘柄名が縦割れ**（375pxで1行3〜4文字×4行）
   - ja/en両方の`articles/[id]/page.tsx`: 2カラムグリッドのうち銘柄・取引企業
     （jaのみ）の項目だけ`gridColumn: 1 / -1`でxs時に全幅化。sm+は4カラム維持。

### 確認
- 375px/1280px × ja/en をブラウザで再確認（重なり解消を座標実測、色は
  computed styleで確認）。`tsc --noEmit`・eslintクリーン。
- 注: `npm run build`の型チェックは別セッションのFAQ機能WIP
  （investorFaq/stockFaq、未コミット）が型エラーで失敗する。本修正とは無関係
  （WIPを退避した状態でビルドが通ることを確認してからpush）。

## 2026-08-15 kujira-watch: 表示速度の週次チェックを追加

「定期的に遅いページを見つけて改善したい」への対応。GitHub Actions `perf_check.yml`
（毎週月曜 09:00 JST / 手動実行も可）で本番の代表9ページを計測する。

### 計測項目（すべてgzip後 = 実際に回線を流れる量）
1. **TTFB**（3回計測の最小値）: キャッシュが効いていないページはここが伸びる。
   実例: `/investors` が searchParams で dynamic rendering になりキャッシュ無しで1.6〜1.9秒。
2. **HTML転送量**: 一覧を全件描画すると膨らむ。実例: `/investors` が1.5MB。
3. **レンダリングブロッキングCSS**（`<head>`内の`<link rel=stylesheet>`のみ）:
   実例: Noto Sans JPで`@font-face`が496個・gzip 130KB。

JSは初期表示を直接ブロックしないので参考値として出すだけで閾値は設けない。
外部ドメインのJS（広告・計測タグ）は自分たちで削れないため集計対象外。

### 閾値と通知
TTFB 0.8秒 / HTML 100KB / CSS 30KB。超過したページがあれば `perf` ラベルの
Issueを立て、既にオープンなら追記する（毎週Issueが増えないように）。
全ページが閾値内に戻ったら自動クローズ。閾値は「調べる価値がある」ラインで目標値ではない。

### 検証
`tests/test_perf_check.py`（8件）。実際にHTTPサーバーを立ててend-to-endで確認する。
開発中に実際に2件バグを見つけた:
- `rel=stylesheet`/`href=`/`src=` のクォート無し属性を取りこぼしていた
  （取りこぼすと「CSSが軽い」と誤判定して肥大化を見逃す）
- `<body>`内のstylesheetを除外できているかの検証で、テストのフィクスチャが
  圧縮で潰れてしまい比較にならなかった（gzipで潰れないCSSを生成するよう修正）

## 2026-08-15 kujira-watch: /faq を分割（1.57MB → 各22KB以下）

週次チェック(perf_check)が本番で唯一フラグを立てた `/faq` を調査・修正。
本番241KB(gzip)をローカル本番ビルドで239KBまで再現できたので、内訳を確定させた。

### 何が起きていたか（実測）
HTML全体 1,570 KB (raw) / 235 KB (gzip)
- 可視HTML          873 KB
- RSCペイロード      471 KB  ← 同じ本文の2回目
- FAQPage構造化データ 225 KB  ← 同じ本文の3回目

**FAQ本文が1つの文書に3回入っていた。**
- 構造化データ: `faqJsonLd.mainEntity` が全502件のquestion/answerを持つ
- RSCペイロード: `FaqList` が `"use client"` で `faqs={FAQS}` を受け取るため、
  ハイドレーション用に全件がシリアライズされる
- 可視HTML: SSR出力

さらに可視HTMLの内訳は、タグを除いた実テキストが220KBに対し
マークアップのオーバーヘッドが653KB（`MuiAccordion-root` 502個、
`Mui〜`クラスの出現14,160回、クラス名の文字列だけで166KB）。
**可視HTMLの75%がコンテンツではなくマークアップだった。**

### 対応
502件を1ページに置いているのが根本原因なので、カテゴリ別ページに分割した。
- `src/lib/faqData.tsx`: データ本体を切り出し（`/faq`と`/faq/[category]`が共用）
- `/faq`: ハブ化。9カテゴリの件数・質問サンプル5件・カテゴリページへのボタン。
  回答本文は置かない（表示していないQ&Aを構造化データに載せるのはGoogleの
  ガイドライン違反になるため、FAQPage構造化データもカテゴリページ側へ移した）
- `/faq/[category]`: カテゴリ別Q&A + そのカテゴリ分だけのFAQPage構造化データ。
  9カテゴリをgenerateStaticParamsで事前生成。サイトマップにも追加
- 分割で「タブによる絞り込み」がURLの役割になったため `FaqList` は削除し、
  タブ無しの `FaqAccordionList` に置き換え（MUI Accordionは維持）

### 結果（gzip後・ローカル本番ビルドで実測）
| ページ | 前 | 後 |
|---|---|---|
| `/faq` | 241 KB | 18 KB (-93%) |
| `/faq/basics` | - | 22 KB |
| `/faq/terms` | - | 22 KB |

TTFBも 0.35s → 0.04s。

## 2026-08-17 kujira-watch: CDNキャッシュの明示活用（API・RSS・画像）

CDNの勉強を兼ねて、Vercel Edge Networkのキャッシュを「自動で効いている部分」だけでなく
明示的に設計した。座学メモは `docs/cdn_study.md`（一般論→自サイトの実例の対応表つき）。

### 現状整理
- ページはISR（`export const revalidate`）で既にCDNキャッシュ済み
  （ISRはCDN的には `s-maxage={revalidate} + stale-while-revalidate` として配信される）
- 一方Route HandlerはNext.js 15以降デフォルト非キャッシュで、
  `/api/articles`（無限スクロール）・`/api/stocks/search`（ヘッダー検索）・
  `/api/watchlist-latest`・`/feed.xml` は毎リクエストがオリジン
  （Vercel関数→microCMS/Supabase）まで到達していた

### 対応
1. APIルート3本に `Cache-Control: public, s-maxage=N, stale-while-revalidate=M` を付与
   - `/api/articles`: 60/300（記事ページのISR 60sと同鮮度）
   - `/api/stocks/search`: 300/3600（検索対象の増減は記事投稿時のみ＝最短毎時）
   - `/api/watchlist-latest`: 300/600（開示スキャンが毎時なので5分で十分）
   - `/api/counter` はPOST＋副作用ありのため対象外（キャッシュしない設計判断も記録）
2. `/feed.xml` を `export const revalidate = 300` でISR化（RSSリーダーの定期巡回対策）
3. `next.config.ts` の `images.minimumCacheTTL` を既定4時間→31日
   （microCMSは画像差し替えでURLが変わる＝実質immutableなので安全）

### 検証
- `tsc --noEmit`・`eslint` パス。`next build` はコンパイル・型検査を通過
  （ページデータ収集はmicroCMSキー未設定の環境のため実行不可）
- デプロイ後は `x-vercel-cache` ヘッダー（MISS→HIT→STALE）で実測する
  （手順は `docs/cdn_study.md` §5）

## 2026-08-21 /weekly「週別の売買金額トレンド」をグラフ化＋週の起点をJSTに是正

### 背景
表だけの「週別の売買金額トレンド」は、買い越し/売り越しがどちらへ振れた週なのかを
数字を読み比べないと掴めなかった（開示件数トレンドは既に棒グラフ化済み）。

### 対応
1. `src/components/AmountTrendChart.tsx` を新設。ベースラインを中央に置き、
   買いを上（`--brand-blue`）・売りを下（`--loss`）へ伸ばす上下対称の棒グラフ。
   チャートライブラリは使わずインラインSVG（`DisclosureTrendChart`・株価グラフと同じ方針）。
   2系列なので凡例あり。今週は薄色＋「集計中」、直接ラベルは最新週のみ、
   各棒の`<title>`で買い/売り/差し引き/件数をホバー表示。既存の表は
   `<details>`「数値を表で見る」へ格納（全数値はクローラー・SRからも読める）。
2. `buildWeeklyAmountRows()` を「直近N週の枠を先に作ってから記事を流し込む」方式に変更。
   開示0件の週が横軸から消えると「その週は少なかった」と誤読されるため、
   空週も枠として残す（記事データがまだ無い最古側の空週だけ落とす）。
   実データでも 7/6・7/13 週の欠落が横軸の空きとして正しく出るようになった。
3. 週の起点判定を日本時間に統一（`toLocaleDateString("sv-SE", {timeZone:"Asia/Tokyo"})`）。
   UTCのまま`new Date()`を使うと月曜0〜9時（JST）はUTCでまだ日曜で、
   今週が前週として扱われ最新週がグラフから消えていた（`/weekly`の金額側と
   `getWeeklyDisclosureCounts()`の件数側の両方）。週の区切りは月曜始まりのまま。

### 検証
- `tsc --noEmit`・`eslint` パス
- dev サーバーのHTMLで、横軸8週（6/29〜8/17）・空週2つが枠として残ること、
  ツールチップの金額・件数・差し引き・「集計中」表示を実データで確認

## 2026-08-21 /ranking/trending（開示急増投資家ランキング）を廃止

### 背景
「直近30日の開示件数 − 前30日の開示件数」の降順というランキングで、次の3点により
読者の行動につながらなかった（利用イメージが無いという指摘）。
- 件数は変更報告書の提出回数で、金額・重要性を表さない。保有先の多い常連が上位を占める
- 買いか売りかが分からない（/trendingの銘柄版には売買方向の絞り込みがあるが投資家版には無い）
- どの銘柄に対する開示なのかが一覧に出ないので、銘柄選びの手掛かりにならない
投資家軸のランキングは「この投資家が買った銘柄はその後どうなったか」を出す
/ranking/returns（3ヶ月リターン）に一本化する。

### 対応
- `src/app/(ja)/ranking/trending/page.tsx` と `src/components/RankingTabNav.tsx` を削除
  （残り1タブのタブUIは意味が無いため、コンポーネントごと削除）
- `buildTrendingFilers()`・`TrendingEntry` 型（`src/lib/trendingStats.ts`）を削除。
  `TrendingTable` の props 型は銘柄一覧向けにその場で定義し直す
- `next.config.ts`: `/ranking/trending` → `/ranking/returns` の301を追加。
  併せて `/ranking/filings` の飛び先を（廃止した）`/ranking/trending` から
  `/ranking/returns` に付け替えてリダイレクトの連鎖を解消
- `sitemap.ts` から該当URLを削除。記事詳細の「関連ランキング」navと
  `/trending` の相互リンクは銘柄ランキング／3ヶ月リターンランキングへ差し替え

### 検証
- `tsc --noEmit`・`eslint` パス
- dev で `/ranking/trending`・`/ranking/filings` が `/ranking/returns` へ308、
  `/ranking/returns`・`/ranking/activist`・`/trending` が200、
  `/sitemap/pages.xml` に該当URLが無いこと、レンダリング後のHTMLに
  「開示急増」「ranking/trending」の残骸が無いことを確認

## 2026-08-22 /weekly 投資家分類別トレンドを分類ごとのグラフへ・「集計中」判定を平日のみに
- 「集計中」が土曜にも出ていた原因: 今週判定が暦週（月〜日）のみで、`isPartial = weekStart === currentWeekStart`。
  EDINET開示も記事生成cron（平日0-12 UTC）も平日しか動かないため、土・日は確定扱いへ（`isCurrentWeekPartial()`）。
- 投資家分類別トレンド: 買い用・売り用の2枚のヒートマップ表（`CategoryWeeklyTrend`）を廃止し、
  分類ごとに買い上・売り下の小さな棒グラフを並べる `CategoryTrendGrid` へ。縦軸スケールは当初全分類共通にしたが、最大分類に引っ張られて他がほぼ見えないため分類ごとのスケール＋「目盛り」表示へ変更。
  最新週の買い/売り上位3分類の文章は1段落に統合。全数値は`<details>`内の表（買い / 売り併記）。

## 2026-08-22 /weekly 直近7日間の件数・金額タイルを削除しグラフ主体へ
- 冒頭の「開示件数（直近7日）」「推定取引金額（直近7日）」タイルは取得・売却を合算した規模で
  方向が読めず、下のグラフ（買い/売り別・分類別）と比べて伝える情報が無かったため削除。
- 専用だった `getPreviousPeriodArticles()`（microcms.ts）・前週比の算出・`formatSigned` も削除。
  `generateMetadata` は件数を含まない固定文言に変更（microCMSへの追加取得が1本減る）。
- データ不足でグラフが出せない場合のフォールバック文を追加。
- `tsc --noEmit`・`eslint` パス、dev で /weekly の描画を確認。
- 2026-08-22: /weekly 分類別トレンドの「直近の週に最も買って/売っていたのは…」文章を削除（グラフと重複し冗長）。
- 2026-08-22: /weekly の売買金額トレンド・分類別トレンドの数値ラベルを最新週のみ→全週表示に変更。

## 2026-08-22 全タブにアイキャッチ付き記事カードを展開
- TOPのアイキャッチ画像で見やすくなったのを受け、画像が無かったヘッダータブ全ページに
  タブの文脈に合った記事カード（`ArticleCard`＝アイキャッチ付き）のセクションを追加。
  共通コンポーネントは `src/components/RelatedArticles.tsx`（記事0件ならセクションごと非表示）。
- 各タブの中身（いずれも4件・取れなくてもページ本体は成立させる `.catch`）:
  - `/trending`: ランキング入り銘柄の最新記事（最新30件から1銘柄1件）
  - `/ranking/returns`: 上位30名の投資家による直近記事（最新50件から1投資家1件）
  - `/ranking/activist`: 取得済みの直近30日記事から金額上位（1銘柄1件・追加fetch無し）
  - `/weekly`: 最新20件から推定金額上位（1銘柄1件）
  - `/activists`: dealType=アクティビストの最新記事
  - `/investors`・`/stocks`・`/monthly`: 最新の解説記事
- `/buybacks` は既に解説記事カードがあるため変更なし。カテゴリ・日付・月別詳細ページは
  既存の `ArticleCard` 一覧で画像表示済み。
- `tsc --noEmit`・`eslint` パス（microCMSキーが無い環境のため実描画は未確認）。

## 2026-08-23 銘柄=業種アイコン・投資家=分類アイコンを全一覧へ、/buybacks再構成、タブ名短縮
- 会社・投資家のロゴは商標・取得元の問題で使えないため、代替アイコン2種を新設:
  - `SectorIcon`: JPX33業種→絵文字（jpx_stock_list.sector）。紺系の丸背景。未登録・"-"は💼。
  - `DealTypeIcon`: 投資家分類→絵文字＋分類色（DEAL_TYPE_COLORSのドット色を約12%アルファで背景に）。
- 適用箇所: /trending 銘柄カード、/ranking/returns 投資家カード、/ranking/activist 銘柄カード、
  /weekly 分類別トレンドのカード見出し（色ドットを置き換え）、/activists 銘柄カード、
  /buybacks 比率・金額ランキング＋決定一覧、/stocks・/investors の索引カード。
  sectorが手元に無いページ（/ranking/activist・/activists・/buybacks）は表示分だけ
  `getCompanyBriefs()`で一括取得（失敗時は空Mapでアイコンはフォールバック表示）。
- /buybacks: 「最新の自社株買い決定」をMUI Table（minWidth 720px・スマホ横スクロール）から
  他ページと同じ1件1カード（.card-grid-wide）へ変更。「自社株買いの数字の見方」セクションは
  FAQへQ&A形式で移設（新規3問。消却・枠の性質は既存Q&Aと重複するため統合）。
- ヘッダー上タブの「アクティビスト注目銘柄」→「アクティビスト」に短縮（lib/nav.ts。
  ページタイトル・フッターは据え置き）。
- `tsc --noEmit`・`eslint` パス（Supabase/microCMSキーが無い環境のため実描画は未確認）。

## 2026-08-23 /buybacks 「月別の決定件数」を削除
- 最新一覧・ランキングと比べて意思決定に使える情報が無いため、セクションごと削除。
- 専用だった `getMonthlyBuybackCounts()`・`MonthlyBuybackCount` 型（lib/buybacks.ts）と
  ページのMUI Tableインポートも削除（月別表が最後の利用箇所だった）。
- `tsc --noEmit`・`eslint` パス。

## 2026-08-23 業種・分類アイコンを絵文字から自作SVGラインアイコンへ
- 絵文字はOS・ブラウザで見た目がバラつくため、SectorIcon（33業種+汎用ブリーフケース）・
  DealTypeIcon（14分類）の中身を24x24グリッドのストローク描画（strokeWidth 2、丸キャップ）の
  自作SVGへ差し替え。ページ側のAPI（sector/dealType/size）は変更なし。
- 色はSectorIconが紺（text-brand-navy/80）＋bg-brand-navy/10、DealTypeIconが分類の
  文字色（DEAL_TYPE_COLORS.text）＋ドット色の約12%アルファ背景。
- 一時ページ(icon-preview)＋headless Chromiumのスクリーンショットで全47種の描画を確認し、
  判読しづらかった鉱業（つるはし）・繊維製品（Tシャツに変更）・海運業（コンテナ船に変更）・
  VC（ロケット）を描き直した。一時ページは確認後に削除済み。
- `tsc --noEmit`・`eslint` パス。

## 2026-08-29 英語版(/en)を全除却
- `src/app/(en)` 配下（TOP・記事・銘柄・カテゴリ・investors・about・privacy・OGP）を削除。
  `/en/*` は `next.config.ts` の redirects で対応する日本語ページ（カテゴリ等はトップ）へ301。
- 判断材料: `blog_crawler_log` 直近14日で EN記事1,046本にブラウザPV1,544（中央値1・最大7・
  10PV以上は0本）。日本語版は最大73PV・10PV以上が42本あり、EN側だけ検索流入の山が無い。
  英訳のために記事1本あたり約3割多い出力トークンを払い続ける形だったため生成ごと停止した。
- 併せて削除: hreflang（layout/about/privacy/記事/銘柄）、サイトマップの`/en`エントリと
  `articles-en`分割（`SITEMAP_IDS`は6→5種）、`getTranslatedArticlesForSitemap()`と`translatedOnly`、
  `isIndexableEnArticle()`、`Article.titleEn`/`bodyEn`、`DEAL_TYPE_EN`/`EN_SLUG_TO_DEAL_TYPE`、
  `SITE_NAME_EN`/`SITE_DESCRIPTION_EN`、言語切替UI（ヘッダーメニュー・フッター）、
  `Locale`型と`UI.en`辞書（未使用になった文言キー33件も削除）。
- Organizationの`alternateName`から "Big Investor Watch" を外した（対応するページが無くなったため）。
- `tsc --noEmit`・`eslint`・`next build` パス。

## 2026-08-30 記事ページに保有目的・平均取得単価・借入比率を出した（競合対抗）
- 発端: 競合アプリ「アクティビストウォッチャー」(@activistw_app、月300円/年3000円)の
  **プレミアム機能**が①保有比率推移グラフ ②平均取得単価（参考値）③保有目的の自動分類。
  ①は`HoldingRatioChart`で実装済みだったが、②③は当方にデータが無かった。
- データ側: EDINETのXBRLに全部入っていた（`lib/edinet.py`の`parse_holding_details()`、
  同日のstock-alert側コミット）。追加のAPIコストはゼロ。
- 表示: 記事詳細のファクトボックスに4項目を追加。
  - **保有目的**: 開示原文＋`classifyPurpose()`の5区分バッジ（`HoldingPurposeBadge.tsx`）。
    分類は自由記述からの機械判定なので原文を必ず併記する。
  - **平均取得単価（開示ベース）**: 取得資金の総額÷保有株数。EDINETは比率しか出さないため
    このサイトの金額は基本すべて概算だが、取得原価だけは実額で出せる。
  - **借入比率**: 借入金÷取得資金。競合が出していない切り口で、自己資金0＝全額借入の買いが実在する
    （成成→東京コスモス電機9.05億円、DOE5パーセント→日本フエルト14.2億円）。50%以上は警告色。
  - **報告義務発生日**: 提出まで30日超のときだけ。法定は5営業日以内で、大幅に遅れた開示は
    「株価が動いた後に出てきた開示」として読む必要がある。
- `classifyPurpose()`/`averageAcquisitionPrice()`/`borrowingRatio()`/`filingLagDays()`は
  `src/lib/disclosures.ts`。判定ロジックは`lib/edinet.py`と同一で、片方だけ直すと
  記事本文（Python生成）とサイト表示がずれる（`summarizeDisposals`と同じ運用）。
- 踏んだ点: supabase-jsの`.select()`は文字列**リテラル**から戻り値の型を推論するため、
  列が増えて長くなったからと`"..." + "..."`で連結すると型が`GenericStringError`に落ちて
  全プロパティがエラーになる。1行のリテラルで渡すこと。
- 検証: `npx tsc --noEmit` 成功。

## 2026-08-30 デザインシステムを導入（トークン層＋MUI同期＋バッジ統合）

色トークンだけがあり他の軸にシステムが無い状態だったため、`docs/design_system.md` を唯一の台帳として
トークン層を新設し、既存の場当たり値をそこへ寄せた。既存レイアウト（高密度）は変えていない。

- 文字階調: `text-foreground/30〜80` の6段（200箇所・41ファイル）→ 用途で選ぶ4段 `text-ink*` に統一。
  `/50`(3.3:1)・`/40`(2.6:1) がWCAG AA未達だったので、寄せると同時にコントラストが改善している。
  MUIの `text.secondary`(3.3:1)・`text.disabled`(2.2:1) も同じ4段に接続。
- タイポグラフィ: サイズ・行間・字間をペアで `@theme` に定義。和文向けに本文行間を広げ、見出しほど
  行間比率を下げ字間を締める。欠番だった `text-lg` を埋め、直書き 0.6875rem 用に `text-2xs` を追加。
  `theme.ts` の typography も同値で定義し、Tailwind製の画面とMUI製の画面がズレないようにした。
- 角丸: 素の `rounded`(4px) 14箇所を用途に応じ `rounded-md`(6px)/`rounded-sm` へ。スケールを `--radius-*` で定義。
- バッジ: 同じsxを複製していた4コンポーネントを `DotBadge` に統合（書式は `MuiChip` 既定）。4つとも
  Server Component 化できた。
- エレベーション/モーションのトークンも定義したが、**カードの影は入れていない**。READMEに
  「影で持ち上げず罫線で区切る」というエディトリアル意匠の決定があるため、`--card-elevation`（既定 `none`）
  という切り替え点だけ作って既定は現状維持にした。

ハマり: MUIの `palette` に `var(--ink)` を渡すと `alpha()` の色計算で
`MUI: Unsupported 'var(--ink)' color` の実行時エラーになる。paletteは実値、`styleOverrides` は `var()` でよい。

## 2026-08-30 出力HTMLのセマンティクスを整えた（記事カード＝article／見出しの入れ子）

TOPのHTMLソースを読もうとしたときに構造が追えなかったのがきっかけ。実測したところ、
記事カード30枚がすべて素の `<div class="MuiCard-root">` で、ページ内の見出しが
`h1:1 / h2:33 / h3:1` ＝ セクション見出しも取引日の区切りも記事タイトルも全部 h2 で、
アウトラインが1段に潰れていた。

- 記事カード（`ArticleCard` / `FeaturedArticleCard`）の `Card` に `component="article"` を付けた。
  カード1枚は単体で意味が通る記事の要約なので `article` が適切。TOPで `<article>` が 0 → 30。
- `ArticleCard` に `headingLevel`（既定 `h2`）、`DealDateHeading` に `level`（既定 `h2`）を足し、
  `InfiniteArticleList` は `dateHeadingLevel` を受けてカードの見出しを常にその1段下にする。
  既定値は変更前の出力と同じなので、明示的に渡したページ以外のHTMLは変わらない。
- 実際に入れ子が崩れていた3か所に渡した: TOP（`新着の取引` h2 → 取引日 h3 → 記事 h4）、
  `/stocks/[code]`（`大量保有・自社株買い履歴` h2 → 取引日 h3 → 記事 h4）、
  `RelatedArticles` と記事詳細の関連記事（セクション h2 → 記事 h3）。
  `/date/[date]` は取引日そのものが h1 なので既定の h2 のままで正しい。
- 結果（ローカルSSR実測）: `<article> 30` / `<time datetime> 31` / `h1:1 h2:6 h3:2 h4:27`。
  グリッドは 356px×2 のまま、横スクロールも出ていない（見た目の変更なし）。

なお **出力HTMLが1行になっているのは直せない**（React SSRは要素間に改行を挟まない。
本番HTMLは564KBで改行0）。ソースを読むときはDevToolsのElementsか整形ツールを通すこと。

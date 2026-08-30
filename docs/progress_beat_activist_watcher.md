# 競合「アクティビストウォッチャー」対抗施策（2026-08-30 開始）

競合: https://obaonline.net/apps/activist-watcher/ja/ （iOS/Androidアプリ、System Design Office OBA）
- データ範囲 2025-01-01〜。保有比率推移グラフ・平均取得単価・変更点ハイライト・
  ウォッチ無制限・ランキング100位・過去分PDFは **有料**。
- 株価データを一切持たない＝「買った後どうなったか」は出せない。ここがうちの勝ち筋。

## ① EDINET開示のバックフィル（2024-12-03〜2025-06-17）
現状 `edinet_large_holdings` は 2025-06-18〜（20,351件）。価格キャッシュ `yahoo_price_cache` が
2024-12-03からあるので、そこまで遡れば追加分も全件リターン計算が付き、投資家の勝率統計の
サンプル数が増える。EDINET APIは無料（Anthropic課金なし）。

- [x] `scan_large_holdings()` に `end_date` を追加（現状 start_date→当日で既存範囲を再走査してしまう）
- [x] `tools/scan_large_holdings.py` に `--end` を追加
- [~] バックフィル実行（2024-12-03 〜 2025-06-15。2025-06-16/17は動作確認で投入済み）
      → バックグラウンド実行中 `logs/edinet_backfill_20241203_20250615.log`
- [ ] 件数・期間をSQLで確認
- [ ] `investor_returns_3m` 系マテビューのリフレッシュ

## ② リターン実績の露出強化
- [x] `/investors` 一覧のカードに「開示3ヶ月後 平均±X%・勝率Y%（N件）」（`getFilerReturnMap()`を新設）
- [x] TOPに「{開示日}の買い開示は、その後どうなったか」枠（`TopReturnPreview.tsx`＋`getLatestReturnCohort()`）。
      上位3件だけでなく母数の平均・勝率を必ず併記する
- [x] README更新（`/`と`/investors`の行）
- [x] tsc / lint / build / `next start`で実データ表示確認（TOP=200・/investors=200、
      2026/05/28の買い開示10件・平均+16.8%・勝率70%・上位3件が実データで描画）
- [ ] コミット & push・本番確認（※pr284がorigin側と8対8で分岐しておりpush不可。オーナー判断待ち）

## ③ 保有目的の保存と変更ハイライト
- [x] XBRLから保有目的を抽出できるか確認（`jplvh_cor:PurposeOfHolding` / `ActOfMakingImportantProposalEtc` /
      `TotalNumberOfStocksEtcHeld` / `TotalAmountOfFundingForAcquisition` が取得可。
      contextRef が `FilingDateInstant`(接尾辞なし)＝共同保有者を含む合算、`...HolderNMember`＝保有者別）
- [x] `edinet_large_holdings` に列追加（purpose_of_holding / important_proposal / shares_held /
      shares_outstanding / funding_total / funding_own / funding_borrowings / obligation_date）
- [ ] 取得・表示・差分ハイライト


## ⚠️ 並行セッションとの競合（2026-08-30 11:14 検知）
同一作業ツリーで別セッションが 11:12〜11:14 に lib/edinet.py・lib/db.py を編集し、
`parse_holding_details()` / `classify_purpose()` / `average_acquisition_price()` /
`HOLDING_DETAIL_COLUMNS` を追加した（＝③と同じ範囲）。本セッションの `end_date` 追加の上に
乗っているので、後発はあちら。列名が食い違ったため、**DDLをあちらのコードに合わせた**
（`funding_amount` を削除し `funding_total/funding_own/funding_borrowings/shares_outstanding/obligation_date` を追加）。
stock-alert-dd / stock-alert-db に担当範囲を照会中。返答があるまで lib/ 配下は触らない。


## ④ holding_ratio の共同保有バグ修正（2026-08-30・オーナー「直して」で着手）
XBRLは保有割合を「保有者ごとのcontext」と「メンバー無しの合算context(FilingDateInstant)」の
両方に持ち、報告書の見出し数値は後者。従来は正規表現で先に見つかった値＝筆頭保有者の1枠を
拾っていた。実測1,101件中656件（60%）がズレ。

- [x] `_aggregate_ratio()` / `_normalize_ratio()` を追加（`_aggregate_or_sum`は`int(float())`で
      0.1975が0に落ちるため流用不可。別関数にした）
- [x] `holding_ratio_prior` も同じ扱いに。`jplvh_cor:`で取れない開示向けに旧正規表現をフォールバックで残す
- [x] tests/test_holding_details.py 12件（うち6件はstock-alert-dbが追加）・test_scan_large_holdings.py 13件 PASS
- [x] `tools/backfill_holding_details.py` 新設（蓄積済み行のXBRL引き直し。`--only-missing`で再開、
      `--dry-run`で差分だけ数える）
- [x] dry-run実測: 2026-08-20以降430件で比率変更212件（49%）・取得失敗0件。
      例 0.39%→5.12% / 0.66%→8.05% / 23.25%→56.26%
- [x] コミット `4929ffc3`
- [~] 全行スイープ実行中（バックフィル完走を待って自動起動する形で予約。
      `logs/edinet_detail_sweep_20260830.log`）
- [ ] スイープ後に `investor_returns_3m` / `investor_return_positions_3m` をリフレッシュ
- [ ] 公開済み記事の数字の是正（`ratioChangePct`・`dealAmount`・タイトル）
      → stock-alert-db が担当。`tools/fix_misreported_blog_articles.py` を拡張する方針。
        **Anthropic APIは使わない**（現行ツールは決定的テンプレートで本文再生成。LLM呼び出し無し）

## ⑤ git分岐（B・オーナー「相談して」）
- [x] 未コミット5本の持ち主を全セッションに照会 → 終了済みセッションの取りこぼしと判明。
      stock-alert-dd が内容確認のうえコミット（`c4e6ff08`）
- [x] こちらの担当分をコミット（`7f75e601` / `4929ffc3`）。作業ツリーをクリアにした
- [x] stock-alert-dd が `git merge origin/pr284` → push 完了（`526f0ada`）。衝突2件、
      うち `/investors` は「3ヶ月リターン表示」と「未公開投資家を一覧から外す」を両方残す形で解消（内容確認済み）
- [x] `git branch -u origin/pr284`（stock-alert-ddが実施）
- 履歴の書き換え（rebase/drop/force push）はしない方針で合意（CLAUDE.md §8・2026-07-16の消失事故の前例）

## ⑥ マージで生じた404リンクの修正（2026-08-30）
origin/pr284 が PR#290（薄い集約ページをnoindexではなく404にする）と `lib/publishedPages.ts` を
持ち込んだため、同じ時間帯に別ブランチで書いた `TopReturnPreview.tsx` だけがガード漏れになっていた。
- [x] `getPublishedStockCodes()` でガードし、未公開の銘柄は素のテキストで出す
- [x] 実証: 直近コホート3件のうち ジェネレーションパス(3195) は `/stocks/3195` が404。
      修正前はリンクを張っていた。修正後はリンクが消え、他2件(9722/9425)はリンクのまま
- [x] tsc / lint / build / `next start` で TOP=200・/investors=200 を確認


## ⑦ 記事の是正（非課金経路。オーナー「非課金で進めて」2026-08-30）
既定の本文再生成は Anthropic API（claude-haiku-4-5）を最大2回呼ぶ。対象が数百本あるため
オーナー判断で非課金運用にした。**API許可は取らない。**

- [x] `tools/fix_misreported_blog_articles.py` に `--fix-body-numbers` を追加（`fa031f52`）
      - `rewrite_body_numbers()` 比率・変化幅・金額を文字列置換（ポイント/pt/億円の表記ゆれ、
        変化幅が本文では符号を持たないことを吸収）
      - 置換できた項目だけでも本文へ反映する（1項目の取りこぼしで本文ごと据え置かない）
      - `scale_phrase_conflicts()` 数字を置換しても直らない規模の記述（約半分/過半/3分の1…）を
        新しい比率と突き合わせて検出。**この経路では直さず一覧に出すだけ**
      - `_title_ratio()` 記事は比率をフィールドに持たないためタイトルから読む
- [x] `tests/test_fix_body_numbers.py` 17アサーション・README更新・実データdry-run確認
- [ ] スイープ完了後に `--fix-body-numbers --apply` を実行
- [ ] 「規模の記述が食い違う」件数を確認（本文の言い回しはAPI無しでは直らないので、
      直さず残すか将来まとめて直すかはその件数を見て判断）

実測（2026-08-30、直近14日の開示553件）: 影響を受ける記事130本 /
うち本文に旧比率がそのまま出ている127本（98%）/ 規模を語る定型句を含む101本 /
修正幅 <1pt 17・1-5pt 56・5-20pt 30・20pt以上 27。

## ⑧ ジョブの再実行（ネットワーク断からの復旧）
バックフィルが 2025-05-09 の時点でDNS解決失敗により異常終了し、pid監視で予約していた
スイープの連鎖も同時に空振りした。再開に強いラッパー
（scratchpad/run_jobs.py。バックフィルの続き → `--only-missing` → `--all` を各8回まで再試行）で
流し直し中。ログ `logs/edinet_jobs_20260830.log`。
※ stock-alert-db セッションは終了済み。記事是正の担当はこのセッションに戻っている。


## ⑨ 実行結果（2026-08-30 夜）

### スイープ
- バックフィル 2024-12-03〜2025-06-15 完走。
- **`tools/backfill_holding_details.py` に致命的なバグがあった**（`3a585f1e` で修正）。
  PostgRESTのupsertは `INSERT ... ON CONFLICT DO UPDATE` なので、UPDATEになる行でも
  「INSERTしようとしたタプル」にNOT NULL制約が評価される。payloadを doc_id と再計算した列
  だけにしていたため `null value in column "issuer_code"` でバッチごと400になり、
  **25,000件を処理して「比率変更12,506件」と表示しながらDBは1行も変わっていなかった**。
  書き込み失敗を数えていなかったため進捗表示だけを見て成功と誤認した。
  → issuer_code をpayloadに載せ、書き込み失敗を数えて1件でもあれば終了コード1にした。
- `--with-article` を追加（`5876250f`）。記事のある開示は2026-01-05以降の1,051件だけで、
  バックフィル範囲には記事が1件も無い（実測0件）ため、記事の是正は全件スイープの完走を
  待たずに始められる。1,051件を先行スイープ（比率変更110件・書き込み失敗0件）。
- 全件スイープ（`--only-missing` 19,490件）は継続中。

### 記事の是正（非課金）
- `--fix-body-numbers --apply`: **更新238件 / 失敗0件**。
  EDINETと変化幅が食い違う記事278件、是正後に基準未満40件（削除はしていない）。
  旧値が本文に無く置換なし129件、規模の記述が新しい比率と食い違う51件。
  更新前の内容は `logs/fixed_articles_backup_20260830_112533.jsonl` に退避（`3d303b5c` で
  PATCH直前の退避を実装）。
- **方向反転の不具合を発見・対処**: 比率が直った結果、売買方向が反転した記事が28本。
  タイトルはテンプレートで組み直されるのに本文の方向語が残り、食い違っていた
  （例: マニー7730 三菱UFJ「1.64%に引き上げ」→「3.73%に引き下げ」）。
  本文1文目は決定的テンプレートでアンカーが明確なのでそこだけ修正（27本。1本は不一致で未修正。
  退避 `logs/direction_fix_backup_20260830_114728.jsonl`）。本文の他の箇所の方向語は
  文脈依存で機械的に直せないため触っていない。
- 訂正注記（`tools/annotate_ratio_correction.py`、stock-alert-57が実装）の断り書きに
  「売買の方向」を追加（`2ee4ad14`）。

### 非課金では直らない残件（記事上で開示する方針）
1. 本文が引用する**他の提出者の比率**は訂正していない → 比較・順位の結論が古いまま
2. **規模を語る記述**（約半分/過半/3分の1…）51件が新しい比率と食い違う
3. 本文1文目以外の**方向語**
4. 開示を一意に特定できない記事（`find_disclosure()` が None）は置換も注記も届かない

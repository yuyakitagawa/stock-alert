# 自社株買い（TDnet）のサイト・X・ブログ展開 進捗

開始: 2026-08-23。背景: `ext_tdnet_disclosures` に自社株買い開示は溜まっていた（直近30日73件）が、
サイト（/stocks/[code] は「大量保有・自社株買い履歴」と見出しに書きながらEDINETのみ）・X投稿・ブログの
いずれにも出ていなかった。LINE Botの check_catalyst だけが参照していた。

## 方針
- 開示タイトルで「決定」（取得枠の新設・ToSTNeT-3買付）と「進捗」（取得状況/取得結果/取得終了）を分ける。
  ニュース価値があるのは決定のみ。X・ブログは決定だけを対象にし、サイトは両方を履歴表示する。
- 決定開示はPDF（TDnet原文）を pypdf でテキスト化し、Claude Haiku で上限株数・上限金額・発行済比率・
  取得期間・方法・消却有無を抽出して `tdnet_buybacks` に保存（PK: code, disclosed_at, title）。
- 株価等の文脈は PIT 規律（開示日以前の gen_rankings）で取る。

## ステップ
- [x] 0. `tdnet_buybacks` テーブル作成（Supabase migration）
- [x] 1. `lib/buyback.py`（分類・PDF取得・Haiku抽出・enrich）+ `tools/enrich_buybacks.py` + daily_alert.yml 組み込み + tests
- [x] 2. サイト: `kujira-watch/src/lib/buybacks.ts` + `/stocks/[code]` に自社株買いセクション
- [x] 3. X: `web/x_buyback.py`（平日19:00 JST「本日の自社株買い決定」）+ x_post.yml + tests
- [x] 4. ブログ: `web/publish_buyback_articles.py`（上限10億円以上 or 比率3%以上）+ edinet_blog.yml + tests
- [x] 5. README / dev_log 更新、デプロイ確認（kujira-watch.com 本番で表示確認）

## 既知の課題
- `ext_tdnet_disclosures` の日付カバレッジが2ヶ月で16日分しか無い（daily_alert の fetch_tdnet が
  continue-on-error で落ちているか、やのしんAPIの limit=2000 で決算期に切り捨て）。要調査。
- TDnetの原文PDFは公開から約1ヶ月で404になる（2026-07-07以前の決定開示はPDF取得不可）。抽出は日次で回す前提。

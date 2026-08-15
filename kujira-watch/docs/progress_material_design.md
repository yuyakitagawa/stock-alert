# kujira-watch Material Design (MUI) 化 進捗

プラン全文: `.claude/plans/robust-herding-riddle.md` (セッションローカル。要点はこのファイルに集約)

## Phase 0 — セットアップ
- [x] 依存関係インストール (@mui/material @emotion/react @emotion/styled @mui/material-nextjs @mui/icons-material)
- [x] `src/theme.ts` 作成 (primary=紺, error=rose-700, success=emerald-700, background.default/paper, brand.{blue,blueDark,gold,goldBright}, borderRadius=6)
- [x] `src/components/ThemeRegistry.tsx` 作成 (AppRouterCacheProvider→ThemeProvider→CssBaseline)
- [x] `(ja)/layout.tsx` に ThemeRegistry 配線
- [x] `(en)/en/layout.tsx` に ThemeRegistry 配線 (他セッションが追加したRippleEffectと共存させる形で配線)
- [x] 使い捨てButton/Chipで色・ripple・Tailwind共存を目視確認 → CssBaseline維持で問題なし、削除済み
- [x] コンソール/hydrationエラーなしを確認 (mobile 375px / desktop 両方)

**メモ**: 他セッションが並行で `RippleEffect.tsx`(JS製ripple)+`globals.css`の`:active`拡張+`.ripple-span`を追加中。
MUIの`ButtonBase`(Button/Chip/IconButton等)は独自のTouchRippleを持つため、`RippleEffect`のグローバル`pointerdown`リスナー(セレクタ`button`等)と二重発火する。
Phase 1でHeader/HeaderMenuをMUI化する際、`.MuiButtonBase-root`をRippleEffectの対象から除外するか、他セッションと調整すること。

## Phase 1 — ヘッダー/ナビゲーション
- [x] Header.tsx → AppBar/Toolbar/Tabs (activeHref計算でサブページのハイライトも維持)
- [x] HeaderMenu.tsx → Drawer (List/ListItemButton)
- [x] StockSearch.tsx → Autocomplete (disablePortal必須。デフォルトのportal描画だと外側クリック判定と競合し選択遷移が効かない)
- [x] VisitCounter.tsx → Typography
- [x] ブラウザ確認: ja/en, デスクトップ/モバイル, 検索→銘柄ページ遷移, Drawer開閉, tsc/eslintクリーン

## Phase 2 — カード/バッジ ライブラリ
- [x] ArticleCard.tsx → Card/CardActionArea/CardMedia ("use client"化: component={Link}をMUIポリモーフィックpropに渡すとRSC境界エラーになるため)
- [x] FeaturedArticleCard.tsx → 同上。sxのtheme callback((theme)=>...)もRSC境界を跨げないため静的値に置換
- [x] DealTypeBadge.tsx → Chip(COLOR_MAPをhexに変換) + Tooltip ("use client"化: TooltipのSSR/クライアント差分でhydration mismatchが出るため)
- [x] DealDirectionBadge.tsx → 同上
- [x] CategoryBadge.tsx → Chip(outlined, clickable) + "use client"
- [x] CompanyInfoCard.tsx → Card + dl相当のBox Grid (サーバーコンポーネントのまま変更なし、component={Link}やTooltipを使わないため問題なし)
- [x] ブラウザ確認: home/stocks/[code]/articles/[id]、デスクトップ/モバイル、tsc/eslintクリーン
- メモ: MUIのポリモーフィックcomponent propにnext/link等の関数を渡す・Tooltipを使う場合は、Server Componentのままだとhydration/RSC境界エラーになるため一律"use client"にする方針で統一

## Phase 3 — テーブル
- [x] ranking/page.tsx (en版は存在しないため対象外)
- [x] investors/[filer]/page.tsx (en版は存在しないため対象外)
- [x] weekly/page.tsx 内訳テーブル (en版は存在しないため対象外)
- [x] ブラウザ確認: 3ページとも表示・色分け(success/error)確認、コンソールエラーなし

## Phase 4 — 残りのインタラクティブ要素
- [x] CategoryFilterDetails.tsx → Accordion ("use client"化。component={Link}をTypographyに渡すため)
- [x] FaqList.tsx → Tabs(カテゴリ絞り込み)+Accordion(Q&A)。CATEGORY_COLORSをhex/テーマトークン化
- [x] DealDateHeading.tsx → Typography+Box(区切り線)
- [x] InfiniteArticleList.tsx 「もっと読み込む」インジケータ → Typography/Box(グリッド自体はTailwindのまま)
- [x] weekly/page.tsx 統計タイル(開示件数/推定取引金額、買い/売り) → Card+Typography
- [x] ブラウザ確認: home(Accordion開閉)/faq(Tabs切替+Accordion開閉)/weekly、コンソールエラーなし

## Phase 5 — 仕上げ
- [x] stocks/page.tsx, investors/page.tsx → List/ListItem(ListItemButtonではなくListItem採用。component={Link}をServer Componentのpage.tsxで使うとasync関数のページ自体を"use client"にできないため)
- [x] articles/[id]/page.tsx メタデータdlグリッド → Box(grid)+Typography
- [x] about/page.tsx 用語集dl → Box+Typography(軽微、プロース本文は変更なし)
- [x] globals.css `:active` ルール + RippleEffect.tsx(別セッションがPR#259でマージした機能)の対象セレクタを`.MuiButtonBase-root`除外にスコープ変更(二重リップル/二重プレス演出の実害を確認して修正)
- [x] ブラウザ確認: stocks/investors一覧、記事詳細、about、他セッションの並行変更(月別アーカイブnav追加・記事関連ロジック追加)との共存も確認、tsc/eslintクリーン

## 完了
Phase 0〜5すべて完了。サイト全体がMUIコンポーネントベースのMaterial Designに移行済み。

## Phase 6 — ボタンUIの全画面統一（2026-08-15）
- [x] `src/theme.ts` に `MuiButton` の既定値/スタイルを追加（`variant="outlined"` / `size="small"`、色は`globals.css`のCSS変数を参照）
- [x] `src/components/ActionButton.tsx` 追加（内部リンク=component={Link} / 外部リンク=component="a"。Server Componentから関数propを渡せないため`"use client"`）
- [x] `src/components/FilterButtonNav.tsx` 追加（選択中はcontained。スマホ幅は`.no-scrollbar`で横スクロール1行、sm以上は折り返し）
- [x] 適用: DealDateSeeMoreLink / ShareButtons / CategoryFilterDetails / articles/[id]の銘柄履歴導線 / monthly/[month]の前後の月ナビ / ranking・stocks・investorsの絞り込みナビ
- [x] ブラウザ確認: TOP・/stocks・/stocks/[code]・/ranking(選択状態)・/investors(デスクトップ/モバイル)・/monthly/[month]・記事詳細、全ルート200、tsc/eslintクリーン
- メモ: 本文中の文脈依存リンク（/about・/faqの説明文、テーブル内の銘柄・投資家リンク、行全体がリンクのカード）はボタン化せずテキストリンクのまま。行全体リンクの内側にbuttonを置くとHTMLとして不正になるため。
- メモ: 英語ページ（/en配下）には独立したCTAが無く（/en/date/[date]未実装のためDealDateSeeMoreLinkはnull、共有ボタンはja記事詳細のみ）、テーマ側の統一のみ適用。

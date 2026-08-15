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
- [ ] stocks/page.tsx, investors/page.tsx → List
- [ ] about/page.tsx, articles/[id]/page.tsx メタデータ → Typography/Stack
- [ ] globals.css `:active` ルールの再スコープ
- [ ] 全19ページ×2ロケール横断チェック

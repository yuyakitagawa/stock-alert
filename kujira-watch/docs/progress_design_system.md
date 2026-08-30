# kujira-watch デザインシステム導入 進捗（2026-08-30〜）

## 背景（着手前の監査結果）

`src/` 全体を機械監査した結果、「色トークンだけがあり、他の軸にシステムが無い」状態だった。

| 軸 | 現状 | 問題 |
|---|---|---|
| 色（ブランド） | `globals.css` の12変数で統一済み | ✅ 唯一整っていた軸 |
| 文字色の階調 | `text-foreground/30 40 50 60 70 80` の**6段**が混在 | 段階の意味が定義されておらず、同じ役割の文字がページごとに別の濃さ |
| コントラスト | `/50`=3.3:1、`/40`=2.6:1 | **WCAG AA（4.5:1）未達**。実害あり |
| タイポグラフィ | Tailwind既定のまま。`text-lg` は**使用0**で sm→xl に飛ぶ | 和文の行間調整が無く、サイズ階層に欠番がある |
| 角丸 | `rounded` / `-sm` / `-md` / `-lg` / `-xl` / `-full` の6種＋CSSの6px | スケールが無く場当たり |
| 影 | **使用0件** | 面の前後関係が罫線1本だけで表現されている |
| モーション | `0.12s` `0.15s` `0.6s` が各所に直書き | |
| バッジ | 4コンポーネントが同じsx（`0.6875rem`/700/`0.08em`/6pxドット）を**個別に複製** | |
| 分類色 | Tailwind標準色から14色を場当たりで採用 | 明度がバラバラで、暖色ブランドの上で浮く |

## 方針

既存レイアウト（高密度・情報量重視）は変えない。トークン層を敷き、既存の場当たり値をそこへ寄せる。
大胆なヒーロー化・和文ウェブフォント追加・大規模リデザインは対象外（CLAUDE.md / design-consult スキルの制約）。

## フェーズ

- [x] P1 トークン層を `globals.css` に定義（タイポ / 文字色階調 / 面 / 罫線 / 角丸 / エレベーション / モーション）
- [x] P2 MUIテーマ（`theme.ts`）をトークンに同期（typography variants / shape / shadows / Chip既定）
- [x] P3 プリミティブ `DotBadge` を作り、バッジ4種の重複を解消
- [x] P4 既存コードをトークンへ移行（文字色6段→4段 / 角丸 / カードのエレベーション）
- [x] P5 `docs/design_system.md` に台帳を書く
- [x] P6 375px / 1280px で目視検証、`tsc --noEmit` / eslint / build
- [ ] P7 dev_log 追記・README更新・コミット

## 実施内容（2026-08-30）

### P1 トークン層（`src/app/globals.css`）
- 文字階調4段 `--ink` / `-secondary`(72%) / `-tertiary`(62%) / `-muted`(55%) ＋ダーク地用2つ。
  背景 `#faf7f0` 上の実測コントラストを併記（15.5 / 6.3 / 4.6 / 3.7:1）。
- 面 `--surface*` 4種、罫線 `--rule-subtle` / `--rule-strong`。
- エレベーション3段。影はニュートラル黒ではなくブランドネイビー透過（クリーム地で彩度が抜けないため）。
- モーション `--duration-fast/base/slow` ＋ `--ease-standard`。
- タイポグラフィスケールを `@theme` で定義。サイズ・行間・字間をペアで持ち、`text-2xs`(11px) を追加、
  欠番だった `text-lg` を埋めた。和文向けに本文行間を広げ、サイズが上がるほど行間比率を下げ字間を締める。
- 角丸スケール `--radius-xs`〜`-xl`。

### P2 MUIテーマ（`src/theme.ts`）
- `typography` の h1〜h6 / body1 / body2 / caption / overline をCSS側と同値で定義（`overline` は和文用に大文字変換なし）。
- `palette.text` をink 3段に接続。MUI既定の `text.secondary`(3.3:1) / `text.disabled`(2.2:1) はAA未達だった。
- `shadows` を3段だけに制限（4段目以降は3段目を流用）。
- `MuiChip` にバッジ書式を集約、`MuiCard` にカード質感を集約。

### P3 プリミティブ
- `src/components/DotBadge.tsx` を新設。`DealTypeBadge` / `DealDirectionBadge` /
  `HoldingPurposeBadge` / `CategoryBadge` の4つが複製していた同一sxを解消。
  4つとも `"use client"` が不要になった（DotBadgeのみクライアント）。

### P4 移行
- `text-foreground/30〜80`（6段・200箇所・41ファイル）→ ink 4段。`/50` `/40` は同時にAAへ改善。
- 素の `rounded`(4px, 14箇所) → 用途に応じ `rounded-md`(6px) / `rounded-sm`。スケール外の `text-[10px]` → `text-2xs`。
- TOPの `TopTrendingPreview` / `TopReturnPreview` だけ独自の罫線色(`border-foreground/15`)だったのをカードと同じ `border-rule` + `bg-surface-raised` に統一。
- 直書きの `fontSize: "0.6875rem"` 等5箇所をトークン参照に。
- `ArticleCard` の罫線色・hover・トランジション・見出しlineHeightの個別指定を削除（テーマ側が持つ）。

## 判断メモ
- **カードの影は入れなかった**。一度 `--elevation-1` を既定にしたが、READMEに
  「カードは影で持ち上げる代わりに罫線区切り」というエディトリアル意匠の決定が明記されていたため撤回。
  代わりに `--card-elevation` / `--card-elevation-hover`（既定 `none`）という切り替え点を1箇所作り、
  意匠を変えたくなったらここだけ変えれば `.card` とMUI Cardが一斉に切り替わるようにした。
- MUIの `palette` に `var(--ink)` を渡すと `alpha()` の色計算で実行時エラーになる
  （`MUI: Unsupported 'var(--ink)' color`）。paletteは実値、`styleOverrides` は `var()` でよい。

## 検証
- `npx tsc --noEmit` クリーン / `npx eslint`（変更ファイル）クリーン。
- ローカルで TOP・`/trending`・`/investors` を 1280px / 375px で目視。375pxで横スクロール発生なし。

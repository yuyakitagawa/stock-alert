# kujira-watch デザインシステム

このサイトの見た目を決める値の**唯一の台帳**。コンポーネントは生の色・サイズを書かず、
ここで定義されたトークンだけを参照する。

- 実体: `src/app/globals.css`（CSS変数 + Tailwindユーティリティ）と `src/theme.ts`（MUI）
- **両者は必ず対で更新する**。片方だけ変えると、Tailwindで組んだ画面とMUIで組んだ画面が別のデザインになる。
- 新しい色・サイズ・角丸・影を**直書きしない**。必要ならまずこの台帳に足す。

---

## 1. 色

### 1.1 ブランドパレット（意味を持つ生の色）

| トークン | 値 | 用途 |
|---|---|---|
| `--background` | `#faf7f0` | ページ背景（クリーム） |
| `--paper` | `#fffdf8` | カード・紙面 |
| `--foreground` | `#201d1a` | 本文テキスト |
| `--brand-navy` | `#16213a` | primary（見出し・containedボタン） |
| `--brand-blue` / `--brand-blue-dark` | `#0068b7` / `#004c87` | リンク・hover |
| `--brand-gold` / `--brand-gold-bright` | `#b8863a` / `#d9a44f` | アクセント（買い・注目） |
| `--section-tint` | `#f1ece1` | セクション背景・hover背景 |
| `--rule` | `#ded5c0` | 罫線 |
| `--gain` / `--loss` | `#047857` / `#be123c` | 買い・上昇 / 売り・下落 |

### 1.2 文字の階調（4段）

役割で選ぶ。**濃さで選ばない**。数値は背景 `#faf7f0` 上の実測コントラスト比。

| トークン | Tailwind | 不透明度 | コントラスト | 用途 |
|---|---|---|---|---|
| `--ink` | `text-ink` | 100% | 15.5:1 | 本文・見出し |
| `--ink-secondary` | `text-ink-secondary` | 72% | 6.3:1 | 補足文・リード文・説明 |
| `--ink-tertiary` | `text-ink-tertiary` | 62% | 4.6:1 | メタ情報（日付・件数・パンくず） |
| `--ink-muted` | `text-ink-muted` | 55% | 3.7:1 | **大きい文字・UI装飾のみ**。本文に使わない（AA未達） |
| `--ink-on-dark` | `text-ink-on-dark` | — | — | ダーク地（注目カード・フッター）の文字 |
| `--ink-on-dark-secondary` | `text-ink-on-dark-secondary` | — | — | 同上の補足文 |

> 導入前は `text-foreground/30 40 50 60 70 80` の6段が場当たりに混在し、`/50`(3.3:1)・`/40`(2.6:1)
> はWCAG AA（通常文字4.5:1）未達だった。4段へ寄せた結果、全ページでコントラストが改善している。

MUI側は `text.primary` / `text.secondary` / `text.disabled` が上の1〜3段目に接続済み。

### 1.3 面と罫線

| トークン | Tailwind | 用途 |
|---|---|---|
| `--surface` | `bg-surface` | ページ地 |
| `--surface-raised` | `bg-surface-raised` | カード・パネル |
| `--surface-sunken` | `bg-surface-sunken` | セクション帯・hover背景 |
| `--surface-inverse` | `bg-surface-inverse` | 注目カード・フッター |
| `--rule` | `border-rule` | 既定の罫線 |
| `--rule-subtle` | `border-rule-subtle` | 表の内側の細い区切り |
| `--rule-strong` | `border-rule-strong` | 押せる要素の枠 |

---

## 2. タイポグラフィ

サイズが上がるほど行間比率を下げ、字間を締める。和文は欧文より字面が大きいため、
Tailwind既定より本文の行間を広く取っている。

| Tailwind | サイズ | 行間 | 字間 | 用途 |
|---|---|---|---|---|
| `text-2xs` | 0.6875rem | 1.5 | +0.08em | バッジ・キッカー・ラベル |
| `text-xs` | 0.75rem | 1.6 | — | メタ情報・注記 |
| `text-sm` | 0.875rem | 1.65 | — | **本文の既定** |
| `text-base` | 1rem | 1.75 | — | 記事本文 |
| `text-lg` | 1.125rem | 1.7 | -0.005em | 小見出し |
| `text-xl` | 1.25rem | 1.55 | -0.01em | カードタイトル・セクション見出し |
| `text-2xl` | 1.5rem | 1.45 | -0.015em | ページ見出し（モバイル） |
| `text-3xl` | 1.875rem | 1.35 | -0.02em | ページ見出し |
| `text-4xl` | 2.25rem | 1.25 | -0.025em | 予備 |

MUIの `h1`〜`h6` / `body1` / `body2` / `caption` / `overline` は同じ値で定義済み。
`overline` は和文の日付・分類に使うため大文字変換しない。

フォントは**追加しない**。和文は端末内蔵フォント、欧文はGeistのみ（CSS 138KB→7KB削減の経緯による）。

---

## 3. 角丸

| トークン | 値 | 用途 |
|---|---|---|
| `rounded-xs` | 3px | — |
| `rounded-sm` | 4px | 小さいタグ・スケルトン |
| `rounded-md` | 6px | **既定**。カード・ボタン・Chip・パネル |
| `rounded-lg` | 10px | 大きいパネル |
| `rounded-xl` | 14px | ヒーロー級のブロック |
| `rounded-full` | — | ドット・アバター |

MUIの `shape.borderRadius: 6` が `rounded-md` と同じ値。

---

## 4. エレベーション（影）

影はニュートラルな黒ではなく**ブランドネイビーを透過**させる。クリーム地に黒影を落とすと
彩度が抜けて紙がくすんで見えるため。

| トークン | Tailwind | 用途 |
|---|---|---|
| `--elevation-1` | `shadow-card` | 軽く浮かせる面 |
| `--elevation-2` | `shadow-raised` | hoverで持ち上げる面 |
| `--elevation-3` | `shadow-overlay` | ドロワー・ポップオーバー |

MUIの `shadows` も同じ3段だけに絞ってある（4段目以降は3段目を流用。**段を増やさない**）。

### カードは既定で影を使わない

このサイトはエディトリアル（雑誌）系の意匠として「影で持ち上げず、罫線で区切る」を選んでいる
（README参照）。そのため `--card-elevation` / `--card-elevation-hover` の既定は `none`。

カードを紙のように浮かせたくなったら、`globals.css` のこの2つを
`var(--elevation-1)` / `var(--elevation-2)` に変えるだけで `.card` とMUI Cardが一斉に切り替わる。
**個々のコンポーネントに `shadow-*` を直接書かないこと**（意匠の方針が1箇所で切り替えられなくなる）。

---

## 5. モーション

| トークン | 値 | 用途 |
|---|---|---|
| `--duration-fast` | 120ms | タップフィードバック |
| `--duration-base` | 180ms | hover・色/影の変化 |
| `--duration-slow` | 320ms | 開閉・スライド |
| `--ease-standard` | `cubic-bezier(0.2, 0, 0, 1)` | 全般 |

カードのhover持ち上げは `prefers-reduced-motion: reduce` で無効化する。

---

## 6. プリミティブ

### `.card`（`globals.css`）
一覧の軽量カード。数百件並ぶ索引ページでHTMLが肥大化しないよう、ユーティリティの羅列でも
MUIコンポーネントでもなく素のCSSクラス1つで持つ（`/investors` をMUI行にした際にHTMLが1.5MB・
TTFB 1.9秒まで悪化した経緯。dev_log 2026-08-15）。`.card-grid` / `.card-grid-wide` / `.card-list` と対で使う。

### `DotBadge`（`src/components/DotBadge.tsx`）
「色ドット＋短いラベル」の唯一の実装。分類・売買方向・保有目的・カテゴリ導線の4バッジはすべてこれを使う。
書式（サイズ・字間・太さ）は `theme.ts` の `MuiChip` 既定が持つので、**各バッジ側でsxを複製しない**。

| 使う側 | 渡すもの |
|---|---|
| `DealTypeBadge` | 分類色ドット＋説明tooltip。`onDark` でダーク地対応 |
| `DealDirectionBadge` | 売りのみ表示。`--loss` |
| `HoldingPurposeBadge` | `tint` で薄い面を敷く |
| `CategoryBadge` | `bordered` + `href` でカテゴリ一覧への導線 |

### ボタン
`theme.ts` の `MuiButton` 既定（outlined / small）に従う。`ActionButton` / `FilterButtonNav` を再利用する。

---

## 7. やらないこと

- 和文ウェブフォントの追加、装飾フォント、重いライブラリ・大画像（レンダリングブロッキングになる）
- `.MuiButtonBase-root` への独自タップ演出の追加（`globals.css` の `:active` / `RippleEffect` と二重になる）
- 本文中の文脈依存リンクのボタン化。行全体がリンクのカード内側への `<button>` 設置（HTML不正）
- エレベーション・文字階調・角丸の**段を増やすこと**

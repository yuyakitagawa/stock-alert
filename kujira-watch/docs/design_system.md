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

### 1.4 カテゴリカルパレット（分類色）

投資家分類14種・保有目的5種・FAQカテゴリ9種は、**同じ1つのパレット**から選ぶ。
OKLCHで明度と彩度をそろえ、色相だけを振り分けてある。

| グレード | L | C | 用途 |
|---|---|---|---|
| ドット | 0.63 | 0.15 | 一覧の色ドット |
| 文字 | 0.46 | 0.13 | バッジのラベル文字（背景トークン3種すべてでWCAG AA達成。最小は `--background` 6.24:1 / `--paper` 6.57:1 / `--section-tint` **5.67:1**＝色を触るときは最も濃い `--section-tint` 上で測る） |

色相は意味のまとまりで割り当てる。並んだときにドットの色そのものが「どのグループか」を伝えるため。

| 色相 | グループ |
|---|---|
| 45〜88° | 個人・創業家・自社株買い（暖色。自社株買いはブランドの金） |
| 148〜190° | 事業会社・公益法人（緑〜青緑） |
| 228〜274° | 銀行・国内運用会社（青） |
| 295〜338° | 海外・特殊な運用会社（紫〜赤紫） |
| 355° | VC |
| 25° | **アクティビストのみ均等割りから外す**。彩度を上げた赤（L 0.60 / C 0.19）。読み手が最も注目する分類で、他と同じ彩度だと一覧で埋没するため |
| 無彩色 | その他・純投資・サイト説明。分類できない/中立なものは色相を持たせず `var(--ink-tertiary)` |

> 導入前はTailwind標準色から場当たりに選んでおり（amber-600 / orange-500 / pink-500 /
> blue-500 / blue-600 …）、同じ役割のドットなのに明度が段ごとにバラバラで、特定の分類だけが
> 不自然に目立っていた。FAQは別途 `brand.blue` / `brand.gold` を文字色に使っており、
> `brand.gold`(#b8863a) はクリーム地で約3:1しか出ずAA未達だった。

実体は `src/lib/dealTypeInfo.ts`（`DEAL_TYPE_COLORS`）・`src/lib/disclosures.ts`（`PURPOSE_COLORS`）・
`src/lib/faqData.tsx`（`CATEGORY_COLORS`）。**4つ目のカテゴリ配色を作らないこと**。

分類色に薄い面を敷くときは16進アルファの連結（`` `${color}14` ``）ではなく
`color-mix(in srgb, ${color} 8%, transparent)` を使う。中立色に `var(--ink-tertiary)` が
入りうるため、連結だと壊れる。

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

スケールが持つのは**サイズと行間だけ**。字間は「サイズ」ではなく「様式」なので別トークンに分ける。

| トークン | 値 | 用途 |
|---|---|---|
| `tracking-label` | 0.08em | バッジ・タグ |
| `tracking-kicker` | 0.14em | `.kicker` |

> 当初 `text-2xs` に 0.08em を焼き込んでいたが、そうすると同じ11pxでも字間の要らない
> 数値ラベル（チャートの軸・注記）でスケールを使えず、`text-[11px]` のような任意値が
> コード中に残る原因になった（2026-08-30に15箇所を回収）。

**任意値ユーティリティ（`text-[11px]` / `rounded-[...]` / `leading-[...]`）は使わない。**
必要な段が無いと感じたら、まずこの表を見直す。

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

## 7. 生成画像（アイキャッチ）

`web/publish_blog_articles.py` の `generate_eyecatch_image()` が焼き込む文字も、このシステムの一部として扱う。

- 帯 = `--surface-inverse`(#16213a) の205/255アルファ。サイト上の注目カード（ダーク地）と同じ色。
- バッジ文字 = `--brand-gold-bright`(#d9a44f)。
- **提出者名・銘柄名は `display_text()` を通してから焼き込む**。EDINETの登録名は英数字が全角
  （`ＩＸＧＳ，　Ｉｎｃ．`）で、そのままだと字間が間延びして読めない。
  この関数は `src/lib/format.ts` の `displayText()` と同じ規則の写しなので、**片方だけ変えないこと**。
  NFKCは使わない（全角括弧・句読点まで半角化して和文の見た目が崩れる）。

## 8. 検査

```bash
python3 tools/check_design_system.py
```

`kujira-watch/src` を走査し、次を違反として落とす。CI（`ci.yml`）の `design` ジョブが **PRとmainへのpushの両方**で同じものを走らせるので、違反はマージ前に止まる。標準ライブラリだけで動く（pip install もフォントも不要）ので数秒で終わる。テスト本体（pytest全件＋画像生成）は重いので main への push のときだけ走る。

| 検出するもの | 直し方 |
|---|---|
| `text-foreground/NN` / `bg-foreground/NN` / `border-foreground/NN` | ink 4段・`bg-surface-*`・`border-rule*` へ |
| `text-[...]` / `leading-[...]` / `tracking-[...]` / `rounded-[...]` / `shadow-[...]` | スケールの段へ。段が足りなければ**先にこの台帳へ足す** |
| Tailwind標準色（`text-red-600` / `fill-emerald-600` など） | ブランドトークンへ。上昇/下落は `text-gain` / `text-loss` |
| `.tsx` の `fontSize: "0.875rem"` のような直書き | `var(--text-*)` かMUIの typography variant へ |

`src/theme.ts` だけ対象外。実値が並ぶのが正しい場所なので。

> トークンを用意しても、コンポーネント側で場当たりの値を書けば意味が無い。実際、導入した
> その日に任意値サイズ15箇所・Tailwind標準色6箇所・`bg-foreground/10` 4箇所が残っていた。
> 目視とgrepでは取りこぼすので機械で落とす。

## 9. やらないこと

- 和文ウェブフォントの追加、装飾フォント、重いライブラリ・大画像（レンダリングブロッキングになる）
- `.MuiButtonBase-root` への独自タップ演出の追加（`globals.css` の `:active` / `RippleEffect` と二重になる）
- 本文中の文脈依存リンクのボタン化。行全体がリンクのカード内側への `<button>` 設置（HTML不正）
- エレベーション・文字階調・角丸の**段を増やすこと**
- 任意値ユーティリティ（`text-[11px]` / `rounded-[10px]` など）でスケールを迂回すること
- 上昇・下落（リターン、株価変化、保有比率の増減）を Tailwind標準色で書くこと。
  **必ず `text-gain` / `text-loss`**（`fill-` / `stroke-` も同じトークンから生成される）。
  プラスを `text-brand-blue`、マイナスを `text-red-600` と書いた箇所が別々に存在し、
  同じ「上昇」がページによって青と緑に分かれていた（2026-08-30に6箇所を統一）。

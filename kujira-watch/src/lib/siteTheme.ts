// Storyblokの「サイト設定」で選べる配色・見出し書体・カード質感。
// 色は自由入力にせず、コントラストを実測済みのプリセットから選ばせる。任意の色を許すと
// リンク色や薄い文字がWCAG AAを割り込んでも気づけないため（globals.css の文字階調の注記参照）。
// 実測値（背景上）: リンク色 5.4〜6.5:1 / --ink-tertiary 4.5〜5.0:1 / 白文字on紺 12〜18:1。
// classic は globals.css の :root と同値で、設定が無い・取得に失敗した時の見た目と一致する。

export type PaletteName = "classic" | "slate" | "forest" | "wine" | "ink";
export type HeadingFont = "sans" | "serif";
export type CardStyle = "flat" | "raised";

export type Palette = {
  background: string;
  paper: string;
  foreground: string;
  navy: string;
  blue: string;
  blueDark: string;
  gold: string;
  goldBright: string;
  tint: string;
  rule: string;
};

export const PALETTES: Record<PaletteName, Palette> = {
  // 現行（クリーム地×紺×金）
  classic: {
    background: "#faf7f0", paper: "#fffdf8", foreground: "#201d1a", navy: "#16213a",
    blue: "#0068b7", blueDark: "#004c87", gold: "#b8863a", goldBright: "#d9a44f",
    tint: "#f1ece1", rule: "#ded5c0",
  },
  // 白地のクールグレー（金融ダッシュボード風）
  slate: {
    background: "#f5f6f8", paper: "#ffffff", foreground: "#191c22", navy: "#0f1b2d",
    blue: "#1d5fd1", blueDark: "#164aa6", gold: "#a16207", goldBright: "#ca8a04",
    tint: "#eceff4", rule: "#d6dbe3",
  },
  // 深緑×生成り。リンクは上昇色(--gain)と紛れないよう緑ではなく青緑にしている
  forest: {
    background: "#f5f5ef", paper: "#fcfcf8", foreground: "#1c201b", navy: "#1e3a2f",
    blue: "#0f5f7a", blueDark: "#0b4a60", gold: "#9a6b1f", goldBright: "#c08a2e",
    tint: "#ebeee3", rule: "#d5d9c8",
  },
  // ワインレッド×温白。リンクは下落色(--loss)と紛れないよう赤系にしない
  wine: {
    background: "#f9f6f3", paper: "#fffdfb", foreground: "#221c1c", navy: "#3a1d2b",
    blue: "#1f5fa8", blueDark: "#174a85", gold: "#a5732f", goldBright: "#cf9a4a",
    tint: "#f1e9e4", rule: "#e0d3cb",
  },
  // 白黒の新聞調
  ink: {
    background: "#fafafa", paper: "#ffffff", foreground: "#171717", navy: "#171717",
    blue: "#2b55c4", blueDark: "#1f3f96", gold: "#8a6d3b", goldBright: "#b08d4f",
    tint: "#f0f0f0", rule: "#dcdcdc",
  },
};

export type SiteTheme = { palette: PaletteName; headingFont: HeadingFont; cardStyle: CardStyle };

export const DEFAULT_SITE_THEME: SiteTheme = { palette: "classic", headingFont: "sans", cardStyle: "flat" };

export function normalizeSiteTheme(raw: Record<string, unknown> | null | undefined): SiteTheme {
  const palette = String(raw?.palette ?? "");
  return {
    palette: palette in PALETTES ? (palette as PaletteName) : DEFAULT_SITE_THEME.palette,
    headingFont: raw?.heading_font === "serif" ? "serif" : "sans",
    cardStyle: raw?.card_style === "raised" ? "raised" : "flat",
  };
}

// :root のトークンを上書きするCSS。既定値（classic・sans・flat）なら空文字を返し、
// 何も出力しない（globals.css の値がそのまま効く）。
export function siteThemeCss(theme: SiteTheme): string {
  const rules: string[] = [];
  if (theme.palette !== "classic") {
    const p = PALETTES[theme.palette];
    rules.push(
      `--background:${p.background};--paper:${p.paper};--foreground:${p.foreground};` +
        `--brand-navy:${p.navy};--brand-blue:${p.blue};--brand-blue-dark:${p.blueDark};` +
        `--brand-gold:${p.gold};--brand-gold-bright:${p.goldBright};` +
        `--section-tint:${p.tint};--rule:${p.rule};`
    );
  }
  if (theme.cardStyle === "raised") {
    rules.push("--card-elevation:var(--elevation-1);--card-elevation-hover:var(--elevation-2);");
  }
  let css = rules.length > 0 ? `:root{${rules.join("")}}` : "";
  if (theme.headingFont === "serif") {
    // 和文明朝も端末内蔵フォントに任せる（ウェブフォントは読まない。layout.tsx の注記参照）。
    css +=
      'h1,h2,h3{font-family:"Hiragino Mincho ProN","Yu Mincho","YuMincho","Noto Serif JP","Noto Serif CJK JP",serif;letter-spacing:0.02em}';
  }
  return css;
}

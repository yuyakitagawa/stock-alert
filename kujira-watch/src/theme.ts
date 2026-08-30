import { createTheme } from "@mui/material/styles";
import type { Shadows } from "@mui/material/styles";

declare module "@mui/material/styles" {
  interface Palette {
    brand: {
      blue: string;
      blueDark: string;
      gold: string;
      goldBright: string;
    };
  }
  interface PaletteOptions {
    brand?: {
      blue: string;
      blueDark: string;
      gold: string;
      goldBright: string;
    };
  }
}

// globals.cssの--font-sansと同じスタック（和文は端末内蔵フォント）。
const fontFamily = "var(--font-sans)";

const theme = createTheme({
  palette: {
    primary: {
      main: "#16213a",
    },
    error: {
      main: "#be123c",
    },
    success: {
      main: "#047857",
    },
    background: {
      default: "#faf7f0",
      paper: "#fffdf8",
    },
    // globals.css の文字階調トークン --ink / --ink-secondary / --ink-tertiary と同値。
    // MUI既定の text.secondary(0.6)/text.disabled(0.38) はクリーム地でそれぞれ約3.3:1・
    // 約2.2:1しか出ずWCAG AA未達だったため、Tailwind側と同じ4段に揃える。
    // paletteには var() を書けない（MUIが alpha() で色計算するため実値が必要）。
    // globals.css 側を変えたらここも同じ値に直すこと。
    text: {
      primary: "#201d1a",
      secondary: "rgba(32, 29, 26, 0.72)",
      disabled: "rgba(32, 29, 26, 0.62)",
    },
    divider: "#ded5c0",
    brand: {
      blue: "#0068b7",
      blueDark: "#004c87",
      gold: "#b8863a",
      goldBright: "#d9a44f",
    },
  },
  // globals.css の --radius-md。カード・ボタン・Chipの既定角丸。
  shape: {
    borderRadius: 6,
  },
  // MUIは25段の影を要求するが、実際に使うのはエレベーション1〜3の3段だけに絞る
  // （globals.css の --elevation-* と同じ値）。4以上は3段目を流用し、増やさない。
  shadows: [
    "none",
    "var(--elevation-1)",
    "var(--elevation-2)",
    ...(Array<string>(22).fill("var(--elevation-3)")),
  ] as unknown as Shadows,
  // globals.css の @theme スケールと同じ値。Tailwindユーティリティで組んだ画面と
  // MUIコンポーネントで組んだ画面が別のタイポグラフィにならないよう、両者を必ず対で更新する。
  typography: {
    fontFamily,
    h1: { fontSize: "1.875rem", lineHeight: 1.35, fontWeight: 700, letterSpacing: "-0.02em" },
    h2: { fontSize: "1.5rem", lineHeight: 1.45, fontWeight: 700, letterSpacing: "-0.015em" },
    h3: { fontSize: "1.25rem", lineHeight: 1.55, fontWeight: 700, letterSpacing: "-0.01em" },
    h4: { fontSize: "1.125rem", lineHeight: 1.7, fontWeight: 700, letterSpacing: "-0.005em" },
    h5: { fontSize: "1rem", lineHeight: 1.75, fontWeight: 700 },
    h6: { fontSize: "1.25rem", lineHeight: 1.55, fontWeight: 700, letterSpacing: "-0.01em" },
    body1: { fontSize: "1rem", lineHeight: 1.75 },
    body2: { fontSize: "0.875rem", lineHeight: 1.65 },
    caption: { fontSize: "0.75rem", lineHeight: 1.6 },
    // 和文の日付・分類ラベルに使うため大文字変換はしない。
    overline: {
      fontSize: "0.6875rem",
      lineHeight: 1.5,
      fontWeight: 700,
      letterSpacing: "0.08em",
      textTransform: "none",
    },
  },
  components: {
    // サイト全体のボタンUIをここで一元定義する（CTA・フィルター・共有ボタンが
    // ページごとにバラバラな見た目にならないようにするため）。色は globals.css の
    // CSSカスタムプロパティを直接参照し、Tailwind側の配色と一致させる。
    MuiButton: {
      defaultProps: {
        variant: "outlined",
        size: "small",
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          textTransform: "none",
          fontWeight: 700,
          letterSpacing: "0.02em",
          whiteSpace: "nowrap",
        },
        outlined: {
          borderColor: "var(--rule)",
          color: "var(--brand-navy)",
          "&:hover": {
            borderColor: "var(--brand-blue)",
            color: "var(--brand-blue)",
            backgroundColor: "var(--section-tint)",
          },
        },
        contained: {
          backgroundColor: "var(--brand-navy)",
          "&:hover": {
            backgroundColor: "var(--brand-blue-dark)",
          },
        },
      },
    },
    // バッジ（分類・売買方向・保有目的・カテゴリ）の書式をここに一元化する。
    // 導入前は4つのバッジコンポーネントが同じsx（0.6875rem / 700 / 0.08em）を
    // それぞれ複製しており、片方だけ直すと見た目がズレる状態だった。
    MuiChip: {
      defaultProps: {
        size: "small",
        variant: "outlined",
      },
      styleOverrides: {
        root: {
          fontSize: "var(--text-2xs)",
          fontWeight: 700,
          letterSpacing: "0.08em",
        },
      },
    },
    // カードは .card（素のCSSクラス）と同じ質感にそろえる。面の持ち上げ方は
    // globals.css の --card-elevation が一元管理するので、ここでは値を書かない。
    MuiCard: {
      defaultProps: {
        variant: "outlined",
      },
      styleOverrides: {
        root: {
          borderColor: "var(--rule)",
          boxShadow: "var(--card-elevation)",
          transition:
            "border-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard)",
          "&:hover": {
            borderColor: "var(--brand-gold)",
            boxShadow: "var(--card-elevation-hover)",
          },
        },
      },
    },
  },
});

export default theme;

import { createTheme } from "@mui/material/styles";

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

const fontFamily =
  "var(--font-noto-sans-jp), var(--font-geist-sans), 'Hiragino Sans', 'Yu Gothic', sans-serif";

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
    brand: {
      blue: "#0068b7",
      blueDark: "#004c87",
      gold: "#b8863a",
      goldBright: "#d9a44f",
    },
  },
  shape: {
    borderRadius: 6,
  },
  typography: {
    fontFamily,
  },
});

export default theme;

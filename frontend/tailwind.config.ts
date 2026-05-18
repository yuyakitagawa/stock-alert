import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        buy:     "#16a34a",
        sbuy:    "#15803d",
        sell:    "#dc2626",
        watch:   "#6b7280",
        surface: "#111827",
        card:    "#1f2937",
        border:  "#374151",
      },
    },
  },
  plugins: [],
};
export default config;

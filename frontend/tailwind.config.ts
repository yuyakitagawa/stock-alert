import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Signal colors
        "signal-s":       "#16a34a",
        "signal-a":       "#22c55e",
        "signal-caution": "#eab308",
        "signal-neutral": "#6b7280",
        "signal-weak":    "#f97316",
        "signal-down":    "#dc2626",
        "signal-sell":    "#ef4444",
        // Surface colors
        surface: "#0a0f1a",
        card:    "#111827",
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "Liberation Mono", "Courier New", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;

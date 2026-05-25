export type Lang = "ja" | "en";

export const SIGNAL_LABELS: Record<string, { ja: string; en: string }> = {
  "S買い":        { ja: "S買い",        en: "Strong Buy" },
"方向感なし":   { ja: "方向感なし",   en: "Neutral" },
  "弱気シグナル": { ja: "弱気シグナル", en: "Weak Signal" },
  "下降シグナル": { ja: "下降シグナル", en: "Downtrend" },
  "買い継続":     { ja: "買い継続",     en: "Hold" },
  "買い増し":     { ja: "買い増し",     en: "Add" },
};

export const UI = {
  ja: {
    top: "TOP",
    rankings: "ランキング",
    rise: "上昇",
    drop: "下落",
    nikkei: "日経比",
    detail: "詳細 →",
    allSectors: "全業種",
    searchPlaceholder: "銘柄名・コードで検索",
    chartPeriods: { "1M": "1ヶ月", "3M": "3ヶ月", "6M": "6ヶ月", "1Y": "1年" },
    loading: "読み込み中...",
  },
  en: {
    top: "TOP",
    rankings: "Rankings",
    rise: "Rise",
    drop: "Drop",
    nikkei: "vs Nikkei",
    detail: "Detail →",
    allSectors: "All Sectors",
    searchPlaceholder: "Search by name or code",
    chartPeriods: { "1M": "1M", "3M": "3M", "6M": "6M", "1Y": "1Y" },
    loading: "Loading...",
  },
} as const;

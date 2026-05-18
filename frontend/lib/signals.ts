import type { Recommend } from "./types";

export const SIGNAL_COLORS: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  "S買い":    { bg: "bg-green-900/60",  text: "text-green-300",  border: "border-green-700",  dot: "bg-[#16a34a]" },
  "A買い":    { bg: "bg-green-900/40",  text: "text-green-400",  border: "border-green-800",  dot: "bg-[#22c55e]" },
  "高値警戒": { bg: "bg-yellow-900/40", text: "text-yellow-300", border: "border-yellow-700", dot: "bg-[#eab308]" },
  "方向感なし":{ bg: "bg-gray-800",     text: "text-gray-400",   border: "border-gray-700",   dot: "bg-[#6b7280]" },
  "弱気シグナル":{ bg: "bg-orange-900/40",text: "text-orange-300",border: "border-orange-800", dot: "bg-[#f97316]" },
  "下降シグナル":{ bg: "bg-red-900/50", text: "text-red-300",    border: "border-red-800",    dot: "bg-[#dc2626]" },
  "売り検討": { bg: "bg-red-900/60",   text: "text-red-200",    border: "border-red-700",    dot: "bg-[#ef4444]" },
  "買い継続": { bg: "bg-teal-900/40",  text: "text-teal-300",   border: "border-teal-800",   dot: "bg-teal-500"  },
  "買い増し": { bg: "bg-teal-900/40",  text: "text-teal-300",   border: "border-teal-800",   dot: "bg-teal-500"  },
};

export function signalStyle(recommend: Recommend) {
  return SIGNAL_COLORS[recommend] ?? { bg: "bg-gray-800", text: "text-gray-400", border: "border-gray-700", dot: "bg-gray-500" };
}

/** rank signals by importance for tab filtering */
export const FILTER_TABS = [
  { label: "全銘柄",    value: "all"       },
  { label: "S買い",    value: "S買い"      },
  { label: "A買い",    value: "A買い"      },
  { label: "高値警戒", value: "高値警戒"   },
  { label: "売り検討", value: "売り検討"   },
] as const;

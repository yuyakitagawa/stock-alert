// ネットスコア（上昇確率 − 下落確率）に応じた配色。
// シグナルラベル（S買い等）は廃止し、ネットスコアで色分けする。
export function netStyle(net: number | null | undefined): {
  bg: string; text: string; border: string; color: string;
} {
  const n = net ?? 0;
  if (n >= 15)  return { bg: "bg-green-900/60",  text: "text-green-300",  border: "border-green-700",     color: "#16a34a" };
  if (n >= 5)   return { bg: "bg-green-900/30",  text: "text-green-400",  border: "border-green-900/50",  color: "#22c55e" };
  if (n >= 0)   return { bg: "bg-gray-800",      text: "text-gray-400",   border: "border-gray-700",      color: "#9ca3af" };
  if (n >= -10) return { bg: "bg-orange-900/40", text: "text-orange-300", border: "border-orange-800",    color: "#f97316" };
  return            { bg: "bg-red-900/50",    text: "text-red-300",    border: "border-red-800",       color: "#dc2626" };
}

// 上昇/下落確率を段階ラベルに変換する。
// モデルの確率はIsotonic較正で数十段の階段値になっており、小数第1位まで出すと
// 「20.3%」のように多数の銘柄が同値になり、過度に精密な印象を与える。
// そのため画面表示は連続値ではなく粗い段階（高/やや高/中/やや低/低）で示す。
export function probBand(p: number | null | undefined): string {
  if (p == null) return "—";
  if (p >= 30) return "高";
  if (p >= 22) return "やや高";
  if (p >= 14) return "中";
  if (p >= 7)  return "やや低";
  return "低";
}

// 符号付き％を方向記号つきで返す（色覚多様性に配慮: 色だけに頼らない）。
// 例: +1.5 → "▲ +1.5%"、-2.0 → "▼ -2.0%"、null → "—"
export function signFmtArrow(n: number | null | undefined, digits = 1): string {
  if (n == null) return "—";
  const arrow = n > 0 ? "▲" : n < 0 ? "▼" : "▬";
  const sign  = n >= 0 ? "+" : "";
  return `${arrow} ${sign}${n.toFixed(digits)}%`;
}

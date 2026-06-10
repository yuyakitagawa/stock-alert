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

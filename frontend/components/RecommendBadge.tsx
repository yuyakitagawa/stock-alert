import type { Recommend } from "@/lib/types";

const styles: Record<string, string> = {
  "S買い":   "bg-green-600 text-white",
  "A買い":   "bg-green-700 text-white",
  "買い継続": "bg-emerald-800 text-emerald-200",
  "買い増し": "bg-teal-700 text-white",
  "売り検討": "bg-red-700 text-white",
  "様子見":  "bg-gray-600 text-gray-200",
};

export default function RecommendBadge({ value }: { value: Recommend }) {
  const cls = styles[value] ?? "bg-gray-700 text-gray-300";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${cls}`}>
      {value}
    </span>
  );
}

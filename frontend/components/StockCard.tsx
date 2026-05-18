import Link from "next/link";
import type { Ranking } from "@/lib/types";
import RecommendBadge from "./RecommendBadge";
import Sparkline from "./Sparkline";
import { signalStyle } from "@/lib/signals";

function fmt(n: number | null, digits = 1) {
  if (n == null) return "—";
  return n.toFixed(digits);
}

function signFmt(n: number | null) {
  if (n == null) return "—";
  return (n >= 0 ? "+" : "") + n.toFixed(1) + "%";
}

interface Props {
  r: Ranking;
  sparkline?: number[];
}

export default function StockCard({ r, sparkline }: Props) {
  const s = signalStyle(r.recommend);
  return (
    <Link
      href={`/stocks/${r.code}`}
      className={`group relative block rounded-xl border ${s.border} bg-gray-900 hover:bg-gray-800 transition-colors p-4 space-y-3 overflow-hidden`}
    >
      {/* Background sparkline — no height change */}
      {sparkline && sparkline.length >= 2 && (
        <div className="absolute bottom-0 left-0 right-0 h-3/5 opacity-[0.12] pointer-events-none">
          <Sparkline prices={sparkline} />
        </div>
      )}

      {/* Header */}
      <div className="relative flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-semibold text-sm text-white truncate">{r.name}</p>
          <p className="text-xs text-gray-500 font-mono">{r.code}</p>
        </div>
        <RecommendBadge value={r.recommend} />
      </div>

      {/* Price */}
      <div className="relative flex items-baseline justify-between">
        <span className="font-mono text-xl font-bold text-white">
          ¥{r.close?.toLocaleString() ?? "—"}
        </span>
        <span className={`font-mono text-sm font-bold ${r.net >= 0 ? "text-green-400" : "text-red-400"}`}>
          {signFmt(r.net)}
        </span>
      </div>

      {/* Probability bar */}
      <div className="relative space-y-1">
        <div className="flex justify-between text-xs text-gray-500 mb-1">
          <span>上昇 {fmt(r.rise_prob)}%</span>
          <span>下落 {fmt(r.drop_prob)}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full bg-green-500 rounded-full"
            style={{ width: `${Math.min(r.rise_prob ?? 0, 100)}%` }}
          />
        </div>
      </div>

      {/* Footer */}
      <div className="relative flex justify-between text-xs text-gray-600 font-mono pt-1 border-t border-gray-800">
        <span>日経比 {signFmt(r.rel20)}</span>
        <span className="group-hover:text-gray-400 transition-colors">詳細 →</span>
      </div>
    </Link>
  );
}

"use client";
import Link from "next/link";
import type { Ranking } from "@/lib/types";
import Sparkline from "./Sparkline";
import { netStyle, signFmtArrow, probBand } from "@/lib/signals";
import { useLang } from "@/contexts/LanguageContext";
import { UI } from "@/lib/i18n";

function fmt(n: number | null, digits = 1) {
  if (n == null) return "—";
  return n.toFixed(digits);
}

interface Props {
  r: Ranking;
  sparkline?: number[];
}

export default function StockCard({ r, sparkline }: Props) {
  const { lang } = useLang();
  const ui = UI[lang];
  const s = netStyle(r.net ?? 0);
  return (
    <Link
      href={`/stocks/${r.code}`}
      className={`group block rounded-xl border ${s.border} bg-gray-900 hover:bg-gray-800 transition-colors overflow-hidden`}
    >
      {sparkline && sparkline.length >= 2 ? (
        <div className="border-b border-gray-800/60">
          <Sparkline prices={sparkline} color={s.color} showLabel />
        </div>
      ) : (
        <div className="h-1 bg-gray-800/40" />
      )}

      <div className="p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="font-semibold text-sm text-white truncate">{r.name}</p>
            <p className="text-xs text-gray-500 font-mono">{r.code}</p>
          </div>
        </div>

        <div className="flex items-baseline justify-between">
          <span className="font-mono text-xl font-bold text-white">
            ¥{r.close?.toLocaleString() ?? "—"}
          </span>
          <span className={`font-mono text-sm font-bold ${r.net >= 0 ? "text-green-400" : "text-red-400"}`}>
            <span className="text-[10px] text-gray-500 font-sans font-normal mr-1">{ui.net}</span>
            {signFmtArrow(r.net)}
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>{ui.rise} {probBand(r.rise_prob)}</span>
            <span>{ui.drop} {probBand(r.drop_prob)}</span>
          </div>
          <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full"
              style={{ width: `${Math.min(r.rise_prob ?? 0, 100)}%` }}
            />
          </div>
        </div>

        <div className="flex justify-between text-xs font-mono pt-1 border-t border-gray-800">
          <span className={r.rel20 == null ? "text-gray-600" : r.rel20 >= 0 ? "text-blue-400" : "text-orange-400"}>
            <span className="text-gray-600 mr-1">{ui.nikkei}</span>{signFmtArrow(r.rel20)}
          </span>
          <span className="text-gray-600 group-hover:text-gray-400 transition-colors">{ui.detail}</span>
        </div>
      </div>
    </Link>
  );
}

import type { MarketCompare } from "@/lib/types";

const STYLE: Record<string, { bg: string; border: string; text: string }> = {
  us_favored: { bg: "bg-blue-950/25",   border: "border-blue-800",   text: "text-blue-300" },
  jp_favored: { bg: "bg-red-950/20",    border: "border-red-900",    text: "text-red-300" },
  neutral:    { bg: "bg-gray-900/40",   border: "border-gray-800",   text: "text-gray-300" },
};

export default function MarketCompareBanner({ compare }: { compare: MarketCompare | null }) {
  if (!compare) return null;
  const s = STYLE[compare.verdict] ?? STYLE.neutral;

  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} px-4 py-3`}>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-base">🌐</span>
        <span className="text-sm font-bold text-white">日経 vs S&amp;P500</span>
        <span className={`text-sm font-bold ${s.text}`}>{compare.label}</span>
      </div>
      {compare.reasons?.length > 0 && (
        <p className="text-xs text-gray-500 mt-1.5 leading-relaxed">
          {compare.reasons.join("・")}
        </p>
      )}
      <p className="text-[11px] text-gray-600 mt-1">
        直近の日経225とS&amp;P500のリターン差から自動判定（参考情報・投資助言ではありません）。本サイトは日本株スクリーニングが対象で、米国株のシグナルは提供していません。
      </p>
    </div>
  );
}

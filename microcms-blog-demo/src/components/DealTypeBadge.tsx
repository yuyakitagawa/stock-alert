import type { DealType } from "@/types/article";

const COLOR_MAP: Record<DealType, string> = {
  機関投資家買い: "border-brand-blue text-brand-blue",
  インサイダー買い: "border-brand-gold text-amber-700",
  自社株買い: "border-emerald-600 text-emerald-700",
  ETFフロー: "border-slate-500 text-slate-600",
  その他: "border-gray-400 text-gray-600",
};

export default function DealTypeBadge({ dealType }: { dealType: DealType }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border bg-white px-2.5 py-0.5 text-xs font-semibold ${COLOR_MAP[dealType]}`}
    >
      {dealType}
    </span>
  );
}

import type { DealType } from "@/types/article";

const COLOR_MAP: Record<DealType, string> = {
  機関投資家買い: "border-brand-blue text-brand-blue",
  インサイダー買い: "border-brand-gold text-amber-700",
  日系ファンド買い: "border-blue-600 text-blue-700",
  外資系ファンド買い: "border-violet-600 text-violet-700",
  ベンチャーキャピタル買い: "border-rose-500 text-rose-600",
  財団買い: "border-teal-600 text-teal-700",
  日系企業買い: "border-sky-600 text-sky-700",
  外資系企業買い: "border-cyan-600 text-cyan-700",
  自社株買い: "border-emerald-600 text-emerald-700",
  ETFフロー: "border-slate-500 text-slate-600",
  その他: "border-gray-400 text-gray-600",
};

export default function DealTypeBadge({ dealType }: { dealType: DealType }) {
  if (!dealType) return null;
  return (
    <span
      className={`inline-flex items-center rounded-full border bg-white px-2.5 py-0.5 text-xs font-semibold ${
        COLOR_MAP[dealType] ?? "border-gray-400 text-gray-600"
      }`}
    >
      {dealType}
    </span>
  );
}

import type { DealType } from "@/types/article";

const COLOR_MAP: Record<DealType, string> = {
  機関投資家買い: "bg-blue-100 text-blue-800",
  インサイダー買い: "bg-purple-100 text-purple-800",
  自社株買い: "bg-green-100 text-green-800",
  ETFフロー: "bg-amber-100 text-amber-800",
  その他: "bg-gray-100 text-gray-800",
};

export default function DealTypeBadge({ dealType }: { dealType: DealType }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${COLOR_MAP[dealType]}`}
    >
      {dealType}
    </span>
  );
}

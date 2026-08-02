import type { DealType } from "@/types/article";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";

const COLOR_MAP: Record<DealType, string> = {
  個人: "border-brand-gold text-amber-700",
  創業家の資産管理会社: "border-orange-500 text-orange-700",
  "公益/一般財団法人": "border-teal-600 text-teal-700",
  プライムブローカー: "border-indigo-500 text-indigo-700",
  アクティビスト: "border-rose-600 text-rose-700",
  VC: "border-pink-500 text-pink-600",
  "PE・メザニンファンド": "border-fuchsia-600 text-fuchsia-700",
  独立系ブティックAM: "border-blue-500 text-blue-600",
  国内アセットマネジメント: "border-blue-600 text-blue-700",
  外資系伝統運用会社: "border-violet-600 text-violet-700",
  日系証券銀行: "border-sky-600 text-sky-700",
  事業会社: "border-emerald-600 text-emerald-700",
  その他: "border-gray-400 text-gray-600",
};

export default function DealTypeBadge({ dealType }: { dealType: DealType }) {
  if (!dealType) return null;
  return (
    <span
      title={DEAL_TYPE_DESCRIPTIONS[dealType]}
      className={`inline-flex items-center rounded-full border bg-white px-2.5 py-0.5 text-xs font-semibold ${
        COLOR_MAP[dealType] ?? "border-gray-400 text-gray-600"
      }`}
    >
      {dealType}
    </span>
  );
}

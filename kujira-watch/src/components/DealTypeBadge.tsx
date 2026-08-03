import type { DealType } from "@/types/article";
import { DEAL_TYPE_DESCRIPTIONS } from "@/lib/dealTypeInfo";

const COLOR_MAP: Record<DealType, string> = {
  個人: "bg-amber-600 text-amber-800",
  創業家の資産管理会社: "bg-orange-500 text-orange-700",
  "公益/一般財団法人": "bg-teal-600 text-teal-700",
  プライムブローカー: "bg-indigo-500 text-indigo-700",
  アクティビスト: "bg-rose-600 text-rose-700",
  VC: "bg-pink-500 text-pink-600",
  "PE・メザニンファンド": "bg-fuchsia-600 text-fuchsia-700",
  独立系ブティックAM: "bg-blue-500 text-blue-600",
  国内アセットマネジメント: "bg-blue-600 text-blue-700",
  外資系伝統運用会社: "bg-violet-600 text-violet-700",
  日系証券銀行: "bg-sky-600 text-sky-700",
  事業会社: "bg-emerald-600 text-emerald-700",
  その他: "bg-gray-400 text-foreground/60",
};

export default function DealTypeBadge({ dealType }: { dealType: DealType }) {
  if (!dealType) return null;
  const [dotColor, textColor] = (COLOR_MAP[dealType] ?? COLOR_MAP.その他).split(" ");
  return (
    <span
      title={DEAL_TYPE_DESCRIPTIONS[dealType]}
      className={`kicker inline-flex items-center gap-1.5 ${textColor}`}
    >
      <span aria-hidden className={`h-1.5 w-1.5 shrink-0 rounded-full ${dotColor}`} />
      {dealType}
    </span>
  );
}

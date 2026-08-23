import { DEAL_TYPE_COLORS } from "@/lib/dealTypeInfo";
import type { DealType } from "@/types/article";

// 投資家カード用の分類アイコン。投資家（提出者）のロゴは取得元・商標の問題で使えないため、
// 投資家分類を絵文字＋分類色（DEAL_TYPE_COLORSのドット色を薄めた背景）で代替する。
// サーバー・クライアントどちらのコンポーネントからも使えるよう素のspanのみで描く。
const DEAL_TYPE_EMOJI: Record<DealType, string> = {
  個人: "👤",
  創業家の資産管理会社: "👑",
  "公益/一般財団法人": "🎗️",
  プライムブローカー: "🏦",
  アクティビスト: "📣",
  VC: "🚀",
  "PE・メザニンファンド": "💼",
  独立系ブティックAM: "🎯",
  国内アセットマネジメント: "🏯",
  外資系伝統運用会社: "🌍",
  日系証券銀行: "🏛️",
  事業会社: "🏭",
  その他: "🗂️",
  自社株買い: "🔄",
};

const SIZE_CLASSES = {
  sm: "h-5 w-5 text-[11px]",
  md: "h-7 w-7 text-[15px]",
} as const;

export default function DealTypeIcon({
  dealType,
  size = "md",
}: {
  dealType: DealType;
  size?: keyof typeof SIZE_CLASSES;
}) {
  const emoji = DEAL_TYPE_EMOJI[dealType] ?? DEAL_TYPE_EMOJI["その他"];
  const color = (DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS["その他"]).dot;
  return (
    <span
      aria-hidden
      title={dealType}
      className={`flex shrink-0 select-none items-center justify-center rounded-full leading-none ${SIZE_CLASSES[size]}`}
      // 分類色の背景（16進2桁のアルファ＝約12%）。ドット・ラベルと同じ色系で分類を示す。
      style={{ backgroundColor: `${color}1f` }}
    >
      {emoji}
    </span>
  );
}

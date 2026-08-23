import type { ReactNode } from "react";
import { DEAL_TYPE_COLORS } from "@/lib/dealTypeInfo";
import type { DealType } from "@/types/article";

// 投資家カード用の分類アイコン。投資家（提出者）のロゴは取得元・商標の問題で使えないため、
// 投資家分類を自作のSVGラインアイコン＋分類色（DEAL_TYPE_COLORSと同系色）で代替する。
// 絵文字はOS・ブラウザごとに見た目がバラつくため、線画で統一する。
// サーバー・クライアントどちらのコンポーネントからも使えるよう素のspan+svgのみで描く。
// 各グリフは24x24グリッドのストローク描画（fillは点などの一部のみ）。
const DEAL_TYPE_GLYPHS: Record<DealType, ReactNode> = {
  // 人物
  個人: (
    <>
      <circle cx="12" cy="8" r="3.4" />
      <path d="M5.2 20a6.8 6.8 0 0113.6 0" />
    </>
  ),
  // 王冠
  創業家の資産管理会社: (
    <path d="M4.5 17.5l-1-9 5 3.6L12 6l3.5 6.1 5-3.6-1 9z" />
  ),
  // ハート
  "公益/一般財団法人": (
    <path d="M12 20.2S4.5 15.4 4.5 10a4.1 4.1 0 017.5-2.3A4.1 4.1 0 0119.5 10c0 5.4-7.5 10.2-7.5 10.2z" />
  ),
  // 銀行（柱）
  プライムブローカー: (
    <>
      <path d="M3.5 9L12 4l8.5 5z" />
      <path d="M5.5 9v7.5M9.8 9v7.5M14.2 9v7.5M18.5 9v7.5" />
      <path d="M4 16.5h16M3 20h18" />
    </>
  ),
  // メガホン
  アクティビスト: (
    <>
      <path d="M3.5 10.2v3.6h3.2l6.8 4.4V5.8l-6.8 4.4z" />
      <path d="M16.8 9.3a4.3 4.3 0 010 5.4" />
      <path d="M19.3 7.2a7.6 7.6 0 010 9.6" />
    </>
  ),
  // ロケット
  VC: (
    <>
      <path d="M12 2.5c2.8 2 4.3 5.1 4.3 8.6 0 2.1-.6 4-1.6 5.4H9.3c-1-1.4-1.6-3.3-1.6-5.4 0-3.5 1.5-6.6 4.3-8.6z" />
      <circle cx="12" cy="9.5" r="1.6" />
      <path d="M7.8 12.5l-2.3 4h3.4M16.2 12.5l2.3 4h-3.4" />
      <path d="M12 16.5v4" />
    </>
  ),
  // 積層（レイヤー）
  "PE・メザニンファンド": (
    <>
      <path d="M12 3.5l8.5 4.3L12 12.1 3.5 7.8z" />
      <path d="M3.5 12.2l8.5 4.3 8.5-4.3" />
      <path d="M3.5 16.4l8.5 4.3 8.5-4.3" />
    </>
  ),
  // 的（ターゲット）
  独立系ブティックAM: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="4.6" />
      <circle cx="12" cy="12" r="1.3" fill="currentColor" stroke="none" />
    </>
  ),
  // 鳥居
  国内アセットマネジメント: (
    <>
      <path d="M3.2 6.8c5.7-2.2 11.9-2.2 17.6 0" />
      <path d="M5 11h14" />
      <path d="M6.8 6.2v14.3M17.2 6.2v14.3" />
      <path d="M12 6.2V11" />
    </>
  ),
  // 地球（緯線）
  外資系伝統運用会社: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M4.8 9.2h14.4M4.8 14.8h14.4" />
    </>
  ),
  // ¥カード
  日系証券銀行: (
    <>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <path d="M8.8 8l3.2 4 3.2-4M12 12v4.8M9.6 13.6h4.8M9.6 15.6h4.8" />
    </>
  ),
  // 工場
  事業会社: (
    <>
      <path d="M3.5 20.5V10l5.5 3.4V10l5.5 3.4V4.5h6v16z" />
      <path d="M3.5 20.5h17" />
    </>
  ),
  // 3点リーダー
  その他: (
    <>
      <circle cx="5.5" cy="12" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="18.5" cy="12" r="1.4" fill="currentColor" stroke="none" />
    </>
  ),
  // 循環矢印（自己株式の取得）
  自社株買い: (
    <>
      <path d="M20 12a8 8 0 11-2.34-5.66" />
      <path d="M17.8 2.8v3.7h-3.7" />
    </>
  ),
};

const SIZES = {
  sm: { circle: "h-5 w-5", svg: 12 },
  md: { circle: "h-7 w-7", svg: 16 },
} as const;

export default function DealTypeIcon({
  dealType,
  size = "md",
}: {
  dealType: DealType;
  size?: keyof typeof SIZES;
}) {
  const glyph = DEAL_TYPE_GLYPHS[dealType] ?? DEAL_TYPE_GLYPHS["その他"];
  const colors = DEAL_TYPE_COLORS[dealType] ?? DEAL_TYPE_COLORS["その他"];
  const { circle, svg } = SIZES[size];
  return (
    <span
      aria-hidden
      title={dealType}
      className={`flex shrink-0 select-none items-center justify-center rounded-full leading-none ${circle}`}
      // 背景=分類のドット色を約12%アルファ（16進2桁）で薄めたもの、線=分類の文字色。
      style={{ backgroundColor: `${colors.dot}1f`, color: colors.text }}
    >
      <svg
        viewBox="0 0 24 24"
        width={svg}
        height={svg}
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {glyph}
      </svg>
    </span>
  );
}

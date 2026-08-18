import { DEAL_TYPES } from "@/types/article";
import { DEAL_TYPE_EN } from "@/lib/dealTypeInfo";
import type { Locale } from "@/lib/i18n";

export type NavLink = { href: string; label: string };

// ヘッダーの上部タブとハンバーガーメニュー「主要ページ」欄が共用する主要ナビゲーション。
// 2箇所で別々に定義するとページ改名時にどちらかが取り残されるため（実例:
// /weekly改名後もメニューだけ「今週のまとめ」のままだった）、ここで一元管理する。
export function mainNavLinks(locale: Locale): NavLink[] {
  if (locale === "en") {
    return [
      { href: "/en", label: "Top" },
      ...DEAL_TYPES.map((dealType) => ({
        href: `/en/category/${DEAL_TYPE_EN[dealType].slug}`,
        label: DEAL_TYPE_EN[dealType].label,
      })),
    ];
  }
  return [
    { href: "/", label: "TOP" },
    { href: "/disclosures", label: "開示速報" },
    { href: "/trending", label: "急増銘柄" },
    { href: "/ranking", label: "月間ランキング" },
    { href: "/weekly", label: "週次トレンド" },
    { href: "/activists", label: "アクティビストの動き" },
    { href: "/investors", label: "投資家一覧" },
    { href: "/stocks", label: "銘柄一覧" },
    // 月別アーカイブは回遊の起点というより過去分の入口なので一番右に置く。
    { href: "/monthly", label: "月別アーカイブ" },
  ];
}

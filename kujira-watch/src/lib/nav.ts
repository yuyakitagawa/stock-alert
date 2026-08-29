
export type NavLink = { href: string; label: string };

// ヘッダーの上部タブとハンバーガーメニュー「主要ページ」欄が共用する主要ナビゲーション。
// 2箇所で別々に定義するとページ改名時にどちらかが取り残されるため（実例:
// /weekly改名後もメニューだけ「今週のまとめ」のままだった）、ここで一元管理する。
export function mainNavLinks(): NavLink[] {
  return [
    { href: "/", label: "TOP" },
    { href: "/trending", label: "銘柄ランキング" },
    { href: "/ranking/returns", label: "投資家ランキング" },
    { href: "/weekly", label: "週次トレンド" },
    // ページタイトルは「アクティビスト注目銘柄」のままだが、上部タブでは長くて
    // 他タブを押し出すため短縮ラベルにする（2026-08-23）。
    { href: "/activists", label: "アクティビスト" },
    { href: "/buybacks", label: "自社株買い" },
    { href: "/investors", label: "投資家一覧" },
    { href: "/stocks", label: "銘柄一覧" },
    // 月別アーカイブは回遊の起点というより過去分の入口なので一番右に置く。
    { href: "/monthly", label: "月別アーカイブ" },
  ];
}

// データ/一覧ページ同士の横移動。ヘッダーのタブには全部並んでいるが、GA4の実測（28日）で
// TOPへの内部到達が398件＝他ページからTOPへ戻る動きが多く、データページ間を直接渡り歩けて
// いなかった。各ページの末尾に「次に見る」として兄弟ページを2〜3件出す。
const DATA_PAGES: NavLink[] = [
  { href: "/trending", label: "銘柄ランキング" },
  { href: "/ranking/returns", label: "投資家ランキング" },
  { href: "/weekly", label: "週次トレンド" },
  { href: "/activists", label: "アクティビスト注目銘柄" },
  { href: "/buybacks", label: "自社株買い" },
  { href: "/investors", label: "投資家一覧" },
  { href: "/stocks", label: "銘柄一覧" },
  { href: "/monthly", label: "月別アーカイブ" },
];

/**
 * 現在のページを除いた兄弟ページを、DATA_PAGESの並び順で指定件数だけ返す。
 * 並びを固定にしているのは、同じページを開くたびに違う候補が出ると
 * 「前に見たあれ」を辿れなくなるため。
 */
export function siblingDataPages(currentHref: string, limit = 3): NavLink[] {
  const others = DATA_PAGES.filter((page) => page.href !== currentHref);
  // 自分の次のページを起点にして、そこから順に拾う（全ページで先頭3件が並ぶと、
  // どのページの末尾も同じ顔になる）。currentを抜いた others では、自分の直後の
  // ページが「自分のindex」に来る（+1すると1つ飛ばしになる）。
  // リストに無いページ（-1）から呼ばれたときは先頭から拾う。
  const start = Math.max(0, DATA_PAGES.findIndex((page) => page.href === currentHref));
  return Array.from({ length: Math.min(limit, others.length) }, (_, i) =>
    others[(start + i) % others.length]
  );
}

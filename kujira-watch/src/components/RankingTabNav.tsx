import FilterButtonNav from "./FilterButtonNav";

// 月間ランキング（/ranking・/ranking/[slug]・/ranking/trending）共通のタブ切り替え。
// アクティビストランキング（/ranking/activist）はタブには含めず、各ページ下部の
// 関連リンクから遷移する（タブは投資家軸のランキングに絞る）。
const TABS = [
  { href: "/ranking", label: "3ヶ月リターンランキング" },
  { href: "/ranking/buys", label: "買い増しランキング" },
  { href: "/ranking/sells", label: "売却ランキング" },
  { href: "/ranking/filings", label: "報告書件数ランキング" },
  { href: "/ranking/trending", label: "開示急増投資家" },
];

export default function RankingTabNav({ current }: { current: string }) {
  return (
    <FilterButtonNav
      ariaLabel="ランキングの種類を切り替える"
      items={TABS.map((tab) => ({ ...tab, selected: tab.href === current }))}
    />
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import WatchlistView from "@/components/WatchlistView";

export const metadata: Metadata = {
  title: "ウォッチリスト",
  description:
    "投資家ページ・銘柄ページで保存した「ウォッチ中」の投資家・銘柄の一覧です。この端末のブラウザにのみ保存されます。",
  // 中身が閲覧者ごとに異なる端末ローカルのページのため、検索結果には出さない
  robots: { index: false },
};

export default function WatchlistPage() {
  return (
    <div>
      <nav aria-label="パンくずリスト" className="mb-4 text-xs text-foreground/50">
        <Link href="/" className="hover:text-brand-blue">トップ</Link>
        {" / "}
        <span className="text-foreground/70">ウォッチリスト</span>
      </nav>
      <h1 className="mb-6 text-2xl font-bold text-brand-navy sm:text-3xl">ウォッチリスト</h1>
      <WatchlistView />
    </div>
  );
}

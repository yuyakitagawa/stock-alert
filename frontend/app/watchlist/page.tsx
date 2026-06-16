import type { Metadata } from "next";
import { fetchRankings, fetchSectorMap } from "@/lib/data";
import type { Ranking } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import WatchlistClient from "@/components/WatchlistClient";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "ウォッチリスト — StockSignal",
  description: "気になる日本株をブックマークして、AIネットスコア・株価・PER/PBRをまとめて監視。ログイン不要のマイ・ウォッチリスト。",
};

export default async function WatchlistPage() {
  let rankMap: Record<string, Ranking> = {};
  let sectorMap: Record<string, string> = {};
  let asOf = "";
  try {
    const [{ date, rows }, sectors] = await Promise.all([fetchRankings(), fetchSectorMap()]);
    asOf = date;
    rankMap = Object.fromEntries(rows.map((r) => [String(r.code), r]));
    sectorMap = sectors;
  } catch {
    // 取得失敗時もブックマークUI（空/件数）は表示する
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-3xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">ウォッチリスト</h1>
          <p className="text-sm text-gray-600 mt-1">
            気になる銘柄をブックマークして、AIネットスコアと一緒に監視できる「マイ・ウォッチリスト」。
          </p>
        </div>

        <WatchlistClient rankMap={rankMap} sectorMap={sectorMap} asOf={asOf} />
      </main>

      <Footer />
    </div>
  );
}

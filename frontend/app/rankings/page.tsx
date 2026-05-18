import type { Metadata } from "next";
import { fetchRankings } from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import RankingsTable from "@/components/RankingsTable";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "ランキング",
  description: "日本株AIシグナルのランキング一覧。S買い・A買い・高値警戒・売り検討でフィルタリング可能。",
};

function formatDate(date: string) {
  if (!date) return "—";
  return new Date(date).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric", weekday: "short",
  });
}

export default async function RankingsPage() {
  const { date, rows } = await fetchRankings();
  const dateLabel = formatDate(date);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar dateLabel={dateLabel} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">シグナルランキング</h1>
          <p className="text-sm text-gray-600 mt-1">
            {date ? `${dateLabel} 時点のAIスコア順` : "データを取得中..."}
          </p>
        </div>

        {rows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-32 text-gray-600 space-y-3">
            <span className="text-5xl">📊</span>
            <p className="text-lg font-medium text-gray-500">本日のデータはまだありません</p>
            <p className="text-sm">平日16時以降に更新されます</p>
          </div>
        ) : (
          <RankingsTable rows={rows} />
        )}
      </main>

      <Footer />
    </div>
  );
}

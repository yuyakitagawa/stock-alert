import type { Ranking } from "@/lib/types";
import RankingsTable from "@/components/RankingsTable";
import PushButton from "@/components/PushButton";

export const revalidate = 3600;

const SB_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SB_KEY = process.env.SUPABASE_SERVICE_KEY!;

function sbHeaders() {
  return {
    apikey: SB_KEY,
    Authorization: `Bearer ${SB_KEY}`,
  };
}

async function fetchRankings(): Promise<{ date: string; rows: Ranking[] }> {
  const latestRes = await fetch(
    `${SB_URL}/rest/v1/web_rankings?select=date&order=date.desc&limit=1`,
    { headers: sbHeaders(), next: { revalidate: 3600 } }
  );
  if (!latestRes.ok) return { date: "", rows: [] };
  const latest = await latestRes.json();
  if (!latest.length) return { date: "", rows: [] };

  const date = latest[0].date;
  const rowsRes = await fetch(
    `${SB_URL}/rest/v1/web_rankings?date=eq.${date}&order=rank.asc`,
    { headers: sbHeaders(), next: { revalidate: 3600 } }
  );
  if (!rowsRes.ok) return { date, rows: [] };
  const rows = await rowsRes.json();
  return { date, rows: rows as Ranking[] };
}

function SummaryCard({
  label,
  count,
  color,
}: {
  label: string;
  count: number;
  color: string;
}) {
  return (
    <div className={`bg-gray-800 rounded-lg p-4 border-l-4 ${color}`}>
      <div className="text-2xl font-bold">{count}</div>
      <div className="text-sm text-gray-400 mt-1">{label}</div>
    </div>
  );
}

export default async function HomePage() {
  const { date, rows } = await fetchRankings();

  const sBuy   = rows.filter((r) => r.recommend === "S買い").length;
  const aBuy   = rows.filter((r) => r.recommend === "A買い").length;
  const sell   = rows.filter((r) => r.recommend === "売り検討").length;
  const hold   = rows.filter((r) => r.recommend === "買い継続" || r.recommend === "買い増し").length;

  const dateLabel = date
    ? new Date(date).toLocaleDateString("ja-JP", {
        year: "numeric", month: "long", day: "numeric",
      })
    : "—";

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* ヘッダー */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-green-400">📈 StockSignal</h1>
            <p className="text-xs text-gray-500">{dateLabel}</p>
          </div>
          <PushButton />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        {rows.length === 0 ? (
          <div className="text-center py-20 text-gray-500">
            <p className="text-4xl mb-4">📊</p>
            <p>本日のデータはまだありません</p>
            <p className="text-sm mt-2">平日16時以降に更新されます</p>
          </div>
        ) : (
          <>
            {/* サマリーカード */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              <SummaryCard label="S買い"   count={sBuy} color="border-green-500" />
              <SummaryCard label="A買い"   count={aBuy} color="border-green-700" />
              <SummaryCard label="保有継続" count={hold} color="border-blue-600" />
              <SummaryCard label="売り検討" count={sell} color="border-red-600" />
            </div>

            {/* ランキングテーブル */}
            <RankingsTable rows={rows} />
          </>
        )}
      </main>

      <footer className="mt-12 pb-8 text-center text-xs text-gray-700">
        本サービスは投資助言ではありません。投資判断はご自身の責任で行ってください。
      </footer>
    </div>
  );
}

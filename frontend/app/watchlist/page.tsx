import type { Metadata } from "next";
import Link from "next/link";
import { fetchRankings } from "@/lib/data";
import { PRICING_POWER_WATCHLIST } from "@/lib/watchlist";
import type { Ranking } from "@/lib/types";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "値上げ力ウォッチリスト — StockSignal",
  description: "シェアを独占しインフレ下で値上げを通せる、toC独占ブランド銘柄のウォッチリスト",
};

function NetBadge({ net }: { net: number | null }) {
  if (net == null) return <span className="text-gray-600 text-sm">—</span>;
  const up = net >= 0;
  return (
    <span className={`font-mono text-sm font-semibold ${up ? "text-green-400" : "text-red-400"}`}>
      {up ? "+" : ""}{net.toFixed(1)}%
    </span>
  );
}

function OverseasBar({ ratio }: { ratio: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 rounded-full bg-gray-800 overflow-hidden">
        <div className="h-full bg-blue-400" style={{ width: `${Math.min(ratio, 100)}%` }} />
      </div>
      <span className="font-mono text-xs text-gray-400 tabular-nums">{ratio}%</span>
    </div>
  );
}

export default async function WatchlistPage() {
  let rankMap: Record<string, Ranking> = {};
  let asOf = "";
  try {
    const { date, rows } = await fetchRankings();
    asOf = date;
    rankMap = Object.fromEntries(rows.map(r => [String(r.code), r]));
  } catch {
    // ランキング取得失敗時も静的リストは表示する
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-white">値上げ力ウォッチリスト</h1>
          <p className="text-sm text-gray-600 mt-1">
            シェアを独占し、インフレ下でも値上げを通せる toC ブランド銘柄。将来の買い候補として監視します。
          </p>
        </div>

        {/* デスクトップ: テーブル */}
        <div className="hidden md:block bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-800">
                <th className="text-left font-medium px-4 py-3">銘柄</th>
                <th className="text-left font-medium px-4 py-3">独占商品</th>
                <th className="text-left font-medium px-4 py-3">国内/世界シェア</th>
                <th className="text-left font-medium px-4 py-3">海外比率</th>
                <th className="text-right font-medium px-4 py-3">現在のネットスコア</th>
              </tr>
            </thead>
            <tbody>
              {PRICING_POWER_WATCHLIST.map((s) => {
                const r = rankMap[s.code];
                return (
                  <tr key={s.code} className="border-b border-gray-800 last:border-0 hover:bg-gray-800/40 transition-colors">
                    <td className="px-4 py-3">
                      <Link href={`/stocks/${s.code}`} className="group">
                        <div className="font-medium text-white group-hover:text-green-400 transition-colors">{s.name}</div>
                        <div className="text-xs text-gray-600 font-mono">{s.code} · {s.category}</div>
                      </Link>
                    </td>
                    <td className="px-4 py-3">
                      <div className="text-gray-300">{s.product}</div>
                      <div className="text-xs text-gray-600 mt-0.5">{s.note}</div>
                    </td>
                    <td className="px-4 py-3 text-gray-300">{s.domesticShare}</td>
                    <td className="px-4 py-3"><OverseasBar ratio={s.overseasRatio} /></td>
                    <td className="px-4 py-3 text-right">
                      <NetBadge net={r?.net ?? null} />
                      {r?.recommend && <div className="text-xs text-gray-600 mt-0.5">{r.recommend}</div>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* モバイル: カード */}
        <div className="md:hidden space-y-3">
          {PRICING_POWER_WATCHLIST.map((s) => {
            const r = rankMap[s.code];
            return (
              <Link
                key={s.code}
                href={`/stocks/${s.code}`}
                className="block bg-gray-900 border border-gray-800 rounded-xl p-4 hover:bg-gray-800/40 transition-colors"
              >
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="font-medium text-white">{s.name}</div>
                    <div className="text-xs text-gray-600 font-mono">{s.code} · {s.category}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <NetBadge net={r?.net ?? null} />
                    {r?.recommend && <div className="text-xs text-gray-600 mt-0.5">{r.recommend}</div>}
                  </div>
                </div>
                <div className="text-sm text-gray-300 mt-2">{s.product}</div>
                <div className="text-xs text-gray-600 mt-1">{s.note}</div>
                <div className="flex items-center justify-between mt-3 text-xs">
                  <span className="text-gray-400">{s.domesticShare}</span>
                  <OverseasBar ratio={s.overseasRatio} />
                </div>
              </Link>
            );
          })}
        </div>

        <p className="text-xs text-gray-700 leading-relaxed">
          ※ シェア率・海外売上比率は公知情報をもとにした概算で、決算期により変動します。投資判断の際は各社IRで最新値をご確認ください。
          ネットスコアは {asOf || "—"} 時点のAIモデル値（上昇確率−下落確率）。
        </p>
      </main>

      <Footer />
    </div>
  );
}

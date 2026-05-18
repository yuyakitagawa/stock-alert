import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import {
  fetchStockRanking,
  fetchStockMeta,
  fetchEarnings,
  fetchAiAnalysis,
} from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import RecommendBadge from "@/components/RecommendBadge";
import { signalStyle } from "@/lib/signals";

export const revalidate = 3600;

interface Props {
  params: Promise<{ code: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { code } = await params;
  const ranking = await fetchStockRanking(code);
  if (!ranking) return { title: `${code} — StockSignal` };
  return {
    title: `${ranking.name} (${code})`,
    description: `${ranking.name}のAIシグナル: ${ranking.recommend}。ネットスコア ${ranking.net.toFixed(1)}%。`,
  };
}

function fmt(n: number | null, digits = 1) {
  if (n == null) return "—";
  return n.toFixed(digits);
}
function signFmt(n: number | null) {
  if (n == null) return "—";
  return (n >= 0 ? "+" : "") + (n as number).toFixed(1) + "%";
}
function formatDate(date: string | null | undefined) {
  if (!date) return "—";
  return new Date(date).toLocaleDateString("ja-JP", {
    year: "numeric", month: "long", day: "numeric",
  });
}

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  colorClass?: string;
}

function MetricCard({ label, value, sub, colorClass = "text-white" }: MetricCardProps) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-1">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold font-mono ${colorClass}`}>{value}</p>
      {sub && <p className="text-xs text-gray-600">{sub}</p>}
    </div>
  );
}

export default async function StockDetailPage({ params }: Props) {
  const { code } = await params;

  const [ranking, meta, earnings, ai] = await Promise.all([
    fetchStockRanking(code),
    fetchStockMeta(code),
    fetchEarnings(code),
    fetchAiAnalysis(code),
  ]);

  if (!ranking) notFound();

  const s = signalStyle(ranking.recommend);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 space-y-8">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-sm text-gray-600">
          <Link href="/" className="hover:text-gray-400 transition-colors">ホーム</Link>
          <span>/</span>
          <Link href="/rankings" className="hover:text-gray-400 transition-colors">ランキング</Link>
          <span>/</span>
          <span className="text-gray-400 font-mono">{code}</span>
        </nav>

        {/* Header */}
        <header className={`bg-gray-900 border ${s.border} rounded-2xl p-6`}>
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-3 flex-wrap">
                <h1 className="text-2xl font-bold text-white">{ranking.name}</h1>
                <RecommendBadge value={ranking.recommend} size="md" />
              </div>
              <div className="flex items-center gap-3 text-sm text-gray-500">
                <span className="font-mono font-semibold text-gray-400">{ranking.code}</span>
                {meta?.sector && <span>{meta.sector}</span>}
                {meta?.market && (
                  <>
                    <span className="text-gray-700">·</span>
                    <span>{meta.market}</span>
                  </>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold font-mono text-white">
                ¥{ranking.close?.toLocaleString() ?? "—"}
              </div>
              <div className={`font-mono font-bold ${ranking.net >= 0 ? "text-green-400" : "text-red-400"}`}>
                ネット {signFmt(ranking.net)}
              </div>
            </div>
          </div>
        </header>

        {/* Score metrics */}
        <section>
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">AIスコア</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <MetricCard
              label="ネットスコア"
              value={signFmt(ranking.net)}
              colorClass={ranking.net >= 0 ? "text-green-400" : "text-red-400"}
            />
            <MetricCard
              label="上昇確率"
              value={`${fmt(ranking.rise_prob)}%`}
              colorClass="text-green-400"
            />
            <MetricCard
              label="下落確率"
              value={`${fmt(ranking.drop_prob)}%`}
              colorClass="text-red-400"
            />
            <MetricCard
              label="日経比 20日"
              value={signFmt(ranking.rel20)}
              colorClass={ranking.rel20 >= 0 ? "text-blue-400" : "text-orange-400"}
            />
            <MetricCard
              label="ボラティリティ"
              value={ranking.vol != null ? `${fmt(ranking.vol)}%` : "—"}
            />
          </div>
        </section>

        {/* Valuation */}
        <section>
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">バリュエーション</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="PER"
              value={ranking.per != null ? `${fmt(ranking.per)}x` : "—"}
            />
            <MetricCard
              label="PBR"
              value={ranking.pbr != null ? `${fmt(ranking.pbr)}x` : "—"}
            />
            {earnings && (
              <div className="col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-1">
                <p className="text-xs text-gray-500 uppercase tracking-wide">次回決算日</p>
                <p className="text-lg font-bold font-mono text-white">
                  {formatDate(earnings.next_date)}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Stop loss */}
        {ranking.stop_loss != null && (
          <section className="bg-red-950/20 border border-red-900/30 rounded-xl p-4">
            <div className="flex items-center gap-3">
              <span className="text-red-400 font-bold text-sm uppercase tracking-wide">損切りライン</span>
              <span className="font-mono text-white font-bold text-lg">
                ¥{ranking.stop_loss.toLocaleString()}
              </span>
            </div>
          </section>
        )}

        {/* AI Analysis */}
        {ai ? (
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">AI解析</h2>
              <span className="text-xs text-gray-700 font-mono">{formatDate(ai.date)}</span>
            </div>

            {/* Summary */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <p className="text-gray-300 text-sm leading-relaxed">{ai.summary}</p>
            </div>

            {/* Bull / Bear */}
            <div className="grid sm:grid-cols-2 gap-4">
              {/* Bull points */}
              {ai.bull_points?.length > 0 && (
                <div className="bg-green-950/20 border border-green-900/30 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-bold text-green-500 uppercase tracking-wide">強気ポイント</h3>
                  <ul className="space-y-2">
                    {ai.bull_points.map((pt, i) => (
                      <li key={i} className="flex gap-2 text-sm text-gray-300">
                        <span className="text-green-500 mt-0.5 shrink-0">↑</span>
                        <span>{pt}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Bear points */}
              {ai.bear_points?.length > 0 && (
                <div className="bg-red-950/20 border border-red-900/30 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-bold text-red-500 uppercase tracking-wide">弱気ポイント</h3>
                  <ul className="space-y-2">
                    {ai.bear_points.map((pt, i) => (
                      <li key={i} className="flex gap-2 text-sm text-gray-300">
                        <span className="text-red-400 mt-0.5 shrink-0">↓</span>
                        <span>{pt}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <p className="text-xs text-gray-700">model: {ai.model_version}</p>
          </section>
        ) : (
          <section className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 text-center">
            <p className="text-gray-600 text-sm">この銘柄のAI解析はまだありません</p>
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
}

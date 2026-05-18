import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import {
  fetchStockRanking,
  fetchStockMeta,
  fetchEarnings,
  fetchAiAnalysis,
  fetchCompanyProfile,
  fetchRecentEarnings,
  fetchDailyQuote,
} from "@/lib/data";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import RecommendBadge from "@/components/RecommendBadge";
import StockChart from "@/components/StockChart";
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
    title: `${ranking.name} (${code}) — 日本株`,
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
function fmtVolume(v: number | null): string {
  if (v == null) return "—";
  if (v >= 100_000_000) return `${(v / 100_000_000).toFixed(1)}億株`;
  if (v >= 10_000)      return `${(v / 10_000).toFixed(0)}万株`;
  return `${v.toLocaleString()}株`;
}

function fmtJPY(n: number | null): string {
  if (n == null) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(1)}兆円`;
  if (abs >= 1e8)  return `${(n / 1e8).toFixed(0)}億円`;
  if (abs >= 1e4)  return `${(n / 1e4).toFixed(0)}万円`;
  return `${n.toLocaleString()}円`;
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

  const [ranking, meta, earnings, ai, profile, recentEarnings, quote] = await Promise.all([
    fetchStockRanking(code),
    fetchStockMeta(code),
    fetchEarnings(code),
    fetchAiAnalysis(code),
    fetchCompanyProfile(code),
    fetchRecentEarnings(code),
    fetchDailyQuote(code),
  ]);

  if (!ranking) notFound();

  const s = signalStyle(ranking.recommend);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 space-y-8">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-2 text-sm text-gray-600">
          <Link href="/" className="hover:text-gray-400 transition-colors">TOP</Link>
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
              <div className="flex items-center gap-3 text-sm text-gray-500 flex-wrap">
                <span className="font-mono font-semibold text-gray-400">{ranking.code}</span>
                {meta?.sector && <span>{meta.sector}</span>}
                {meta?.market && (
                  <>
                    <span className="text-gray-700">·</span>
                    <span>{meta.market}</span>
                  </>
                )}
                {profile.employees && (
                  <span className="text-gray-600">
                    {profile.employees.toLocaleString()}人
                  </span>
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

        {/* Price Chart — central element */}
        <section>
          <StockChart code={code} />
        </section>

        {/* Today's market data */}
        {quote && (
          <section>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">本日の市場データ</h2>
              {quote.date && (
                <span className="text-xs text-gray-600 font-mono">{quote.date}</span>
              )}
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
              {/* Day change highlight */}
              {quote.changePct != null && (
                <div className={`px-5 py-3 border-b border-gray-800 flex items-center gap-3 ${
                  quote.changePct >= 0 ? "bg-green-950/20" : "bg-red-950/20"
                }`}>
                  <span className={`font-mono text-xl font-bold ${
                    quote.changePct >= 0 ? "text-green-400" : "text-red-400"
                  }`}>
                    {quote.changePct >= 0 ? "+" : ""}{quote.changePct.toFixed(2)}%
                  </span>
                  {quote.change != null && (
                    <span className={`font-mono text-sm ${
                      quote.change >= 0 ? "text-green-600" : "text-red-600"
                    }`}>
                      ({quote.change >= 0 ? "+" : ""}¥{quote.change.toLocaleString("ja-JP", { maximumFractionDigits: 0 })})
                    </span>
                  )}
                  {quote.prevClose != null && (
                    <span className="text-xs text-gray-600 ml-auto">
                      前日終値 ¥{quote.prevClose.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}
                    </span>
                  )}
                </div>
              )}

              {/* OHLCV grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-y divide-gray-800/60">
                {[
                  { label: "始値",  val: quote.open  != null ? `¥${quote.open.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—" },
                  { label: "高値",  val: quote.high  != null ? `¥${quote.high.toLocaleString("ja-JP",  { maximumFractionDigits: 0 })}` : "—", color: "text-green-400" },
                  { label: "安値",  val: quote.low   != null ? `¥${quote.low.toLocaleString("ja-JP",   { maximumFractionDigits: 0 })}` : "—", color: "text-red-400" },
                  { label: "出来高", val: fmtVolume(quote.volume) },
                ].map(({ label, val, color }) => (
                  <div key={label} className="px-4 py-3">
                    <div className="text-xs text-gray-500 mb-1">{label}</div>
                    <div className={`font-mono text-sm font-bold ${color ?? "text-white"}`}>{val}</div>
                  </div>
                ))}
              </div>

              {/* 52-week range */}
              {(quote.fiftyTwoWeekLow != null || quote.fiftyTwoWeekHigh != null) && (
                <div className="px-5 py-3 border-t border-gray-800/60 flex items-center gap-4 text-xs">
                  <span className="text-gray-500">52週レンジ</span>
                  <span className="font-mono text-red-400">
                    安値 {quote.fiftyTwoWeekLow != null
                      ? `¥${quote.fiftyTwoWeekLow.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}`
                      : "—"}
                  </span>
                  {quote.fiftyTwoWeekLow != null && quote.fiftyTwoWeekHigh != null && quote.price != null && (
                    <div className="flex-1 flex items-center gap-1">
                      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{
                            width: `${Math.max(0, Math.min(100,
                              ((quote.price - quote.fiftyTwoWeekLow) /
                               (quote.fiftyTwoWeekHigh - quote.fiftyTwoWeekLow)) * 100
                            ))}%`
                          }}
                        />
                      </div>
                    </div>
                  )}
                  <span className="font-mono text-green-400">
                    高値 {quote.fiftyTwoWeekHigh != null
                      ? `¥${quote.fiftyTwoWeekHigh.toLocaleString("ja-JP", { maximumFractionDigits: 0 })}`
                      : "—"}
                  </span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Company Overview */}
        {(profile.description || profile.website) && (
          <section className="space-y-4">
            <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide">会社概要</h2>

            {profile.description && (
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
                <p className="text-gray-400 text-sm leading-relaxed">{profile.description}</p>
              </div>
            )}

            {/* IR Links */}
            <div>
              <p className="text-xs text-gray-600 mb-2 font-semibold">IR・開示情報</p>
              <div className="flex flex-wrap gap-2">
                <a
                  href={`https://irbank.net/${code}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
                >
                  IR Bank
                  <span className="text-gray-600">↗</span>
                </a>
                <a
                  href={`https://www.release.tdnet.info/inbs/I_main_00.html?code=${code}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
                >
                  TDnet 適時開示
                  <span className="text-gray-600">↗</span>
                </a>
                {profile.website && (
                  <a
                    href={profile.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
                  >
                    会社HP
                    <span className="text-gray-600">↗</span>
                  </a>
                )}
              </div>
            </div>
          </section>
        )}

        {/* IR links only if no profile description */}
        {!profile.description && !profile.website && (
          <section>
            <p className="text-xs text-gray-600 mb-2 font-semibold">IR・開示情報</p>
            <div className="flex flex-wrap gap-2">
              <a
                href={`https://irbank.net/${code}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
              >
                IR Bank
                <span className="text-gray-600">↗</span>
              </a>
              <a
                href={`https://www.release.tdnet.info/inbs/I_main_00.html?code=${code}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-xs text-gray-300 hover:bg-gray-700 hover:text-white transition-colors font-medium"
              >
                TDnet 適時開示
                <span className="text-gray-600">↗</span>
              </a>
            </div>
          </section>
        )}

        {/* Score metrics */}
        <section>
          <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">AIスコア</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <MetricCard
              label="ネットスコア"
              value={signFmt(ranking.net)}
              colorClass={ranking.net >= 0 ? "text-green-400" : "text-red-400"}
            />
            <MetricCard label="上昇確率"     value={`${fmt(ranking.rise_prob)}%`} colorClass="text-green-400" />
            <MetricCard label="下落確率"     value={`${fmt(ranking.drop_prob)}%`} colorClass="text-red-400" />
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
            <MetricCard label="PER" value={ranking.per != null ? `${fmt(ranking.per)}x` : "—"} />
            <MetricCard label="PBR" value={ranking.pbr != null ? `${fmt(ranking.pbr)}x` : "—"} />
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

        {/* Recent Earnings */}
        {recentEarnings.length > 0 && (
          <section>
            <h2 className="text-sm font-bold text-gray-500 uppercase tracking-wide mb-3">
              最新決算（直近{recentEarnings.length}四半期）
            </h2>
            <div className="overflow-hidden rounded-xl border border-gray-800">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-900/80 text-gray-500 text-left border-b border-gray-800">
                    <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide">期間</th>
                    <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-right">売上高</th>
                    <th className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-right">純利益</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/60">
                  {recentEarnings.map((q, i) => (
                    <tr key={i} className="bg-gray-900">
                      <td className="px-4 py-2.5 font-mono text-gray-300 text-xs">{q.period}</td>
                      <td className="px-4 py-2.5 font-mono text-gray-200 text-sm text-right">
                        {fmtJPY(q.revenue)}
                      </td>
                      <td className={`px-4 py-2.5 font-mono text-sm font-bold text-right ${
                        q.netIncome == null ? "text-gray-500"
                        : q.netIncome >= 0 ? "text-green-400" : "text-red-400"
                      }`}>
                        {fmtJPY(q.netIncome)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-gray-700 px-4 py-2 bg-gray-900/50 border-t border-gray-800">
                出典: Yahoo Finance
              </p>
            </div>
          </section>
        )}

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

            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <p className="text-gray-300 text-sm leading-relaxed">{ai.summary}</p>
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
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
